"""Custom SciFact sentence-selection diagnostics, not official SciFact scoring."""

import argparse
import importlib.metadata
import platform
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
import sys
from pathlib import Path
from statistics import mean

from ranking_evaluation.evidence_selection import EvidenceSpan, select_evidence
from ranking_evaluation.scifact import load_scifact, sha256

POLICIES = {'sentence_top3': (1, 3), 'sentence_top6': (1, 6), 'context2_top3': (2, 3)}



def covered_sentence_indices(sentences, intervals):
    """Map the union of selected character intervals to original sentences."""
    complete, partial, offset = [], [], 0
    for index, text in enumerate(sentences):
        positions = [offset + j for j, char in enumerate(text) if not char.isspace()]
        covered = [any(start <= pos < end for start, end in intervals) for pos in positions]
        if covered and all(covered):
            complete.append(index)
        elif any(covered):
            partial.append(index)
        offset += len(text) + 1
    return complete, partial


def frontend_spans(selector, claim, doc_id, sentences):
    body = ' '.join(sentences)
    output = []
    for span in selector(claim, doc_id, body, top_k=3):
        match = re.search(r':abstract:(\d+):(\d+):[0-9a-f]{12}$', span.passage_id)
        if not match or span.paper_id != doc_id:
            raise ValueError('Frontend passage lacks valid source identity/offsets')
        start, end = map(int, match.groups())
        if not 0 <= start < end <= len(body) or body[start:end] != span.text:
            raise ValueError('Frontend passage does not match original source text')
        indices, _ = covered_sentence_indices(sentences, [(start, end)])
        output.append(EvidenceSpan(doc_id, tuple(indices), start, end, span.text,
                                   float(span.score), span.passage_id))
    return output


def prf(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * tp / (predicted + gold) if predicted + gold else 0.0}


def score_selection(predicted: set[tuple[str, int]], evidence: dict) -> dict:
    """Gold union plus an explicit any-complete-rationale diagnostic.

    Rationale alternatives are used only for scoring; no gold-based selector
    choice occurs. Union recall requires all annotated sentences and can be
    stricter than finding one sufficient rationale. Never average NEI as zero.
    """
    union = {(doc, index) for doc, rationales in evidence.items()
             for rationale in rationales for index in rationale.sentences}
    complete = sum(any({(doc, i) for i in rationale.sentences} <= predicted
                       for rationale in rationales) for doc, rationales in evidence.items())
    if not union:
        return {'eligible': False, 'predicted': len(predicted), 'gold': 0, 'tp': 0,
                'precision': None, 'recall': None, 'f1': None,
                'gold_documents': 0, 'complete_gold_documents': 0,
                'any_complete_rationale': None, 'all_gold_documents_complete': None}
    tp = len(predicted & union)
    return {'eligible': True, 'predicted': len(predicted), 'gold': len(union), 'tp': tp,
            **prf(tp, len(predicted), len(union)),
            'gold_documents': len(evidence), 'complete_gold_documents': complete,
            'any_complete_rationale': complete > 0,
            'all_gold_documents_complete': complete == len(evidence)}


def summarize(records: list[dict]) -> dict:
    eligible = [r for r in records if r['eligible']]
    excluded = [r for r in records if not r['eligible']]
    totals = {name: sum(r[name] for r in eligible) for name in
              ('tp', 'predicted', 'gold', 'gold_documents', 'complete_gold_documents')}
    return {'total_claims': len(records), 'evaluated_claims': len(eligible),
            'excluded_no_gold': len(excluded), 'counts': totals,
            'micro': prf(totals['tp'], totals['predicted'], totals['gold']) if eligible else None,
            'macro': {name: mean(r[name] for r in eligible) if eligible else None
                      for name in ('precision', 'recall', 'f1')},
            'complete_rationale_document_recall': totals['complete_gold_documents'] / totals['gold_documents']
                if totals['gold_documents'] else None,
            'claims_any_complete_rationale_rate': mean(r['any_complete_rationale'] for r in eligible)
                if eligible else None,
            'claims_all_gold_documents_complete_rate': mean(r['all_gold_documents_complete'] for r in eligible)
                if eligible else None,
            'no_gold_claims_predicted_sentences': sum(r['predicted'] for r in excluded),
            'no_gold_note': 'Excluded from P/R/F1; selections are counted, not independently labelled errors'}


def evaluate(docs, claims, rankings, *, depths=(5, 10), policy_names=None, frontend_selector=None):
    """Oracle mode isolates selection; retrieved mode includes missing gold docs."""
    cache = {}
    rows, per_claim = [], []
    policies = dict(POLICIES)
    if frontend_selector is not None:
        policies["frontend_current"] = (0, 3)
    names = list(policies) if policy_names is None else policy_names
    if not names or len(names) != len(set(names)) or set(names) - policies.keys():
        raise ValueError("Unknown/duplicate selector policy or missing frontend selector")
    for policy in names:
        window_size, top_k = policies[policy]
        configurations = [('oracle_gold_papers', None, None)]
        configurations += [(method, depth, ranks) for method, ranks in rankings.items() for depth in depths]
        for method, depth, ranks in configurations:
            records = []
            for claim_id, claim in claims.items():
                # Gold documents are deliberately supplied only in oracle mode.
                ids = list(claim.evidence) if ranks is None else ranks[claim_id][:depth]
                predicted, partial_ids, spans = set(), set(), []
                for doc_id in ids:
                    key = (claim_id, doc_id, policy)
                    if key not in cache:
                        if policy == "frontend_current":
                            cache[key] = frontend_spans(frontend_selector, claim.claim, doc_id, docs[doc_id].abstract)
                        else:
                            cache[key] = select_evidence(claim.claim, doc_id, docs[doc_id].abstract,
                                                        window_size=window_size, top_k=top_k)
                    for span in cache[key]:
                        if ' '.join(docs[doc_id].abstract)[span.start:span.end] != span.text:
                            raise ValueError('Selected text does not match source span')
                        spans.append(asdict(span))
                    complete, partial = covered_sentence_indices(docs[doc_id].abstract,
                        [(span.start, span.end) for span in cache[key]])
                    predicted.update((doc_id, i) for i in complete)
                    partial_ids.update((doc_id, i) for i in partial)
                row = {'claim_id': claim_id, 'policy': policy, 'ranking_method': method,
                       'paper_depth': depth, **score_selection(predicted, claim.evidence),
                       'selected_papers': ids, 'partially_covered_sentence_ids': sorted(partial_ids),
                       'gold_union': sorted({(doc, i) for doc, sets in claim.evidence.items()
                                             for rationale in sets for i in rationale.sentences}),
                       'predicted_sentence_ids': sorted(predicted), 'passages': spans}
                records.append(row)
                per_claim.append(row)
            rows.append({'policy': policy, 'ranking_method': method, 'paper_depth': depth,
                         **summarize(records)})
            print(f'Completed {policy}: {method} depth={depth}', flush=True)
    return rows, per_claim


def load_rankings(paths, *, dataset_version, claims, docs):
    rankings, hashes = {}, {}
    for path in paths:
        report = json.loads(path.read_bytes())
        if report['dataset']['dataset_version'] != dataset_version or report['dataset']['synthetic']:
            raise ValueError('Ranking report dataset/version mismatch or synthetic input')
        depth = report['configuration']['top_n']
        for method, ranks in report['rankings'].items():
            name = f'rrf_ce_top{depth}' if method == 'rrf_cross_encoder' else method
            if set(ranks) != set(claims):
                raise ValueError('Rankings must cover every split claim')
            for ids in ranks.values():
                if len(ids) != len(set(ids)) or set(ids) - docs.keys():
                    raise ValueError('Duplicate or unknown ranked document')
            if name in rankings and rankings[name] != ranks:
                raise ValueError('Conflicting copies of the same ranking method')
            rankings[name] = ranks
        hashes[str(path.resolve())] = sha256(path)
    return rankings, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'dev'], required=True)
    parser.add_argument('--ranking-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--paper-depths', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frontend-root', type=Path)
    parser.add_argument('--policies', nargs='+', default=list(POLICIES),
                        choices=[*POLICIES, 'frontend_current'])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a new path')
    if not args.paper_depths or any(k <= 0 for k in args.paper_depths) or len(set(args.paper_depths)) != len(args.paper_depths):
        parser.error('Paper depths must be positive and unique')
    frontend_selector, frontend_sources = None, {}
    if args.frontend_root:
        sys.path.insert(0, str(args.frontend_root.resolve()))
        from explanation_frontend.passage_selection import select_abstract
        frontend_selector = select_abstract
        for name in ("passage_selection.py", "models.py", "_upstream/retrieval/bm25.py", "_upstream/claim_verifier/schemas.py"):
            frontend_sources[name] = sha256(args.frontend_root / "explanation_frontend" / name)
    if "frontend_current" in args.policies and frontend_selector is None:
        parser.error("frontend_current requires --frontend-root")
    docs, claims = load_scifact(args.corpus, args.claims, split=args.split)
    version = f'corpus:{sha256(args.corpus)};claims:{sha256(args.claims)}'
    rankings, input_hashes = load_rankings(args.ranking_reports, dataset_version=version, claims=claims, docs=docs)
    args.output_dir.mkdir(parents=True)
    protocol = {
        'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_version': version, 'split': args.split, 'corpus_documents': len(docs),
        'purpose': 'Development diagnostics, not untouched final testing; no selector parameter tuning',
        'policies': args.policies, 'policy_settings': POLICIES, 'paper_depths': args.paper_depths,
        'frontend_source_sha256': frontend_sources,
        'python': platform.python_version(),
        'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
        'ranking_report_sha256': input_hashes,
        'source_sha256': {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('evidence_selection.py'), Path(__file__).with_name('scifact.py'))},
        'metric': 'Custom exact (document ID, original sentence index) union P/R/F1; not official SciFact evidence scoring',
        'limitations': [
            'Gold union covers all alternative rationales; also report any-complete-rationale recovery',
            'Context neighbours and unannotated relevant sentences count as false positives against annotated union',
            'Gold-paper mode supplies document identities and is not end-to-end retrieval',
            'Retrieved-paper mode includes all selected non-gold papers and unretrieved gold as misses',
            'No NLI verdict evaluation, token truncation evaluation, PDF extraction or live-search quality claims',
            'Three sentence passages vs three two-sentence passages have different text budgets; sentence_top6 is an additional budget control',
            'New lexical baselines differ from frontend_current; that optional policy evaluates unchanged frontend source in its own environment',
            'Character coverage is merged across all selected spans; partial original sentences receive no full-sentence credit',
        ],
    }
    manifest = args.output_dir / 'manifest.json'
    manifest.write_text(json.dumps(protocol, indent=2)+'\n')
    try:
        rows, records = evaluate(docs, claims, rankings, depths=tuple(args.paper_depths),
                                 policy_names=args.policies, frontend_selector=frontend_selector)
        with (args.output_dir/'per_claim.jsonl').open('x') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        (args.output_dir/'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
        lines = ['# SciFact evidence-selection development evaluation', '',
                 'Custom sentence-index diagnostics; not official SciFact evidence scoring or final held-out evaluation.', '',
                 '| Selector | Papers | K | Micro P | Micro R | Micro F1 | Complete rationale doc recall |',
                 '| --- | --- | --- | --- | --- | --- | --- |']
        for row in rows:
            m = row['micro']
            lines.append(f"| {row['policy']} | {row['ranking_method']} | {row['paper_depth'] or 'gold'} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {row['complete_rationale_document_recall']:.4f} |")
        lines += ['', '## Interpretation limits', '', *['- '+note for note in protocol['limitations']]]
        (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
        protocol['status'] = 'complete'
    except Exception as exc:
        protocol.update(status='failed', error=str(exc))
        raise
    finally:
        manifest.write_text(json.dumps(protocol, indent=2)+'\n')
    print(args.output_dir/'report.md')


if __name__ == '__main__':
    main()
