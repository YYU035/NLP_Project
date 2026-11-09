"""Archive historical live search, hybrid ranking and exact abstract evidence.

This is an operational pilot, not relevance/evidence-gold evaluation or a
feasibility classifier. It never consumes labels or generated gold explanations.
"""

import argparse
import asyncio
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from paper_search.config import Settings
from paper_search.keybert_queries import KeyBERTQueryGenerator
from paper_search.retrieval import RetrievalService, RetrievalUnavailable
from paper_search.retrieval_models import ProviderPlan, RetrievalResult
from ranking_evaluation.pipeline import ModelDescription, RankingConfig, rank_claim
from ranking_evaluation.reranker import LocalCrossEncoder
from ranking_evaluation.run_csfcube import ordered_ids
from ranking_evaluation.schemas import RankingInput
from ranking_evaluation.semantic_evidence import LocalSemanticSelector, SelectionConfig, person4_passages
from retrieval.bm25 import BM25Retriever
from retrieval.sbert import SBERTRetriever
from retrieval.schema import Paper
from shared.datasets.matter_discovery import discover_case
from shared.datasets.setup import file_hash


def select_cases(payload, limit):
    if payload.get('dataset') != 'Matter-of-Fact' or not payload.get('version'):
        raise ValueError('Expected prepared Matter-of-Fact inference input')
    cases = payload['cases']
    if type(limit) is not int or limit <= 0:
        raise ValueError('Limit must be positive')
    if len({row['case_id'] for row in cases}) != len(cases):
        raise ValueError('Duplicate inference case IDs')
    for row in cases:
        if set(row) != {'case_id', 'claim_text', 'evidence_not_after'}:
            raise ValueError('Only leakage-safe inference fields are permitted')
    # Opaque hash order, chosen without labels, topics or generated metadata.
    return sorted(cases, key=lambda row: row['case_id'])[:limit]


def process_case(case, discovery, *, selector, scorer, model, encoder_id, paper_depth=5):
    if discovery['case_id'] != case['case_id'] or discovery['evidence_not_after'] != case['evidence_not_after']:
        raise ValueError('Discovery case/cutoff mismatch')
    candidates = RetrievalResult.model_validate(discovery['candidates'])
    if candidates.original_claim != case['claim_text']:
        raise ValueError('Original claim mismatch')
    from datetime import date
    cutoff = date.fromisoformat(case['evidence_not_after'])
    for entry in candidates.papers:
        dates = [entry.paper.publication_date, *[r.publication_date for r in entry.source_records]]
        if any(value is None or value > cutoff for value in dates):
            raise ValueError('Unknown or future publication date reached ranking')
    started = perf_counter()
    ids = [entry.identity for entry in candidates.papers]
    papers = [Paper(paper_id=entry.identity, title=entry.paper.title or '',
                    abstract=[entry.paper.abstract] if entry.paper.abstract else [], source=entry.paper.source)
              for entry in candidates.papers]
    lexical = BM25Retriever(stemmer='simple')
    dense = SBERTRetriever(model_name=encoder_id, encoder=selector.encoder, cache_dir=None, device=model.device)
    lexical_seconds = dense_seconds = 0.0
    bm25_ids, sbert_ids = [], []
    if papers:
        begin = perf_counter()
        lexical.index(papers)
        bm25_ids = ordered_ids(lexical.score_all(case['claim_text']), ids)
        lexical_seconds = perf_counter() - begin
        begin = perf_counter()
        dense.index(papers, use_cache=False)
        sbert_ids = ordered_ids(dense.score_all(case['claim_text']), ids)
        dense_seconds = perf_counter() - begin
    inputs = RankingInput(claim_id=case['case_id'], candidates=candidates,
        bm25_ranking=bm25_ids, sbert_ranking=sbert_ids,
        bm25_seconds=lexical_seconds, sbert_seconds=dense_seconds)
    ranking = rank_claim(inputs, config=RankingConfig(rrf_k=60, rerank_enabled=True,
        rerank_top_n=20), scorer=scorer, model=model)
    evidence, missing_abstracts = [], 0
    for row in ranking.papers[:paper_depth]:
        entry = row.entry
        abstract = entry.paper.abstract or ''
        missing_abstracts += int(not abstract.strip())
        spans = selector.select_abstract(case['claim_text'], entry.identity, abstract)
        for span in spans:
            if abstract[span.start:span.end] != span.text:
                raise ValueError('Selected passage is not an exact source substring')
        evidence.append({'paper_id': entry.identity, 'rank': row.rank,
            'title': entry.paper.title, 'published_date': entry.paper.publication_date.isoformat(),
            'reference': str(entry.paper.landing_url) if entry.paper.landing_url else entry.identity,
            'passages': person4_passages(spans), 'source_spans': [asdict(span) for span in spans],
            'status': 'selected' if spans else 'missing_abstract' if not abstract.strip() else 'below_threshold'})
    return {'case_id': case['case_id'], 'ranking': ranking.model_dump(mode='json'),
            'evidence': evidence, 'selection_config': asdict(selector.config),
            'operational': {'candidate_count': len(papers),
                'papers_with_abstract': sum(bool((entry.paper.abstract or '').strip()) for entry in candidates.papers),
                'examined_papers': len(evidence), 'examined_without_abstract': missing_abstracts,
                'selected_passages': sum(len(row['passages']) for row in evidence),
                'search_seconds': candidates.seconds,
                'post_search_seconds': perf_counter() - started},
            'verification_status': 'not_run',
            'limitations': ['No feasibility verdict or evidence correctness gold',
                'Abstract fallback only; exact source offsets retained',
                'Not token-budget-validated for Person 4; not a completed verifier handoff',
                'Metadata dates checked; publication-version history and model training exposure unverified']}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


async def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.paper_depth <= 0 or not 1 <= args.top_k <= 100:
        raise ValueError('Invalid paper depth or provider result depth')
    payload = json.loads(args.input.read_text())
    cases = select_cases(payload, args.limit)
    selector = LocalSemanticSelector(args.encoder_path,
        config=SelectionConfig(top_k=3, min_score=0.5, context_radius=0), device='cpu')
    if args.cpu_float64:
        selector.encoder.double()
    scorer = LocalCrossEncoder(args.reranker_path, device='cpu', batch_size=16,
                               cpu_float64=args.cpu_float64)
    model = ModelDescription(model_id=args.reranker_id, revision=args.reranker_revision,
        device='cpu', dtype='float64' if args.cpu_float64 else 'default', max_length=512, batch_size=16)
    from keybert import KeyBERT
    generator = KeyBERTQueryGenerator(args.encoder_path, extractor=KeyBERT(model=selector.encoder))
    cache = {}
    def planner(provider, claim):
        if claim not in cache:
            alternatives = generator.generate(claim, max_queries=2)
            if len(alternatives) < 2:
                raise ValueError('No keyword query generated')
            cache[claim] = alternatives[1]
        return ProviderPlan(provider=provider, strategy='keybert', query=cache[claim], notes=[
            f'Experimental encoder override: {args.encoder_id}@{args.encoder_revision}',
            'Reuses the offline ranking encoder; not the default Person 1 pinned query encoder'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {'dataset_version': payload['version'], 'status': 'running',
        'task': 'historical-search operational pilot, not gold relevance or feasibility evaluation',
        'selection': 'First N cases sorted by opaque case ID; no label selection',
        'total_available_cases': len(payload['cases']), 'selected_cases': len(cases), 'cases': [],
        'configuration': {'providers': ['openalex', 'openaire'], 'top_k_per_provider': args.top_k,
            'rrf_k': 60, 'cross_encoder_top_n': 20, 'paper_depth': args.paper_depth,
            'evidence': asdict(selector.config), 'encoder_id': args.encoder_id,
            'encoder_revision': args.encoder_revision, 'reranker': model.model_dump(),
            'query_encoder_policy': 'Explicit override to reuse ranking encoder',
            'hardware_note': 'CPU; synchronous stage timings, not GPU throughput'},
        'provenance': {'input_sha256': file_hash(args.input), 'runner_sha256': file_hash(Path(__file__)),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'sentence-transformers', 'keybert', 'httpx')},
            'models': {name: {p.relative_to(path).as_posix(): file_hash(p)
                for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(path).parts}
                for name, path in [('encoder', args.encoder_path), ('reranker', args.reranker_path)]}}}
    write(args.output_dir / 'summary.json', summary)
    try:
        async with RetrievalService(settings=Settings(timeout_seconds=12,openaire_enabled=True),
                planner=planner, providers=('openalex', 'openaire'), archive=args.output_dir) as service:
            for case in cases:
                directory = args.output_dir / case['case_id']
                directory.mkdir()
                try:
                    discovery = await discover_case(case, service, top_k=args.top_k)
                except RetrievalUnavailable as exc:
                    write(directory / 'search-failure.json', exc.result.model_dump(mode='json'))
                    summary['cases'].append({'case_id': case['case_id'], 'status': 'search_failed'})
                else:
                    write(directory / 'discovery.json', discovery)
                    result = process_case(case, discovery, selector=selector, scorer=scorer,
                        model=model, encoder_id=args.encoder_id, paper_depth=args.paper_depth)
                    write(directory / 'pipeline.json', result)
                    summary['cases'].append({'case_id': case['case_id'], 'status': 'processed',
                        'providers': {key: value['status'] for key, value in discovery['candidates']['providers'].items()},
                        'rejected_date_records': len(discovery['audit']['rejected']), **result['operational']})
                write(args.output_dir / 'summary.json', summary)
                print(json.dumps(summary['cases'][-1]), flush=True)
        summary['status'] = 'complete_with_failures' if any(row['status'] != 'processed' for row in summary['cases']) else 'complete'
        rows = [row for row in summary['cases'] if row['status'] == 'processed']
        summary['operational_metrics'] = {'attempted_cases': len(cases), 'processed_cases': len(rows),
            'cases_with_candidates': sum(row['candidate_count'] > 0 for row in rows),
            'cases_with_selected_passages': sum(row['selected_passages'] > 0 for row in rows),
            'candidate_occurrences': sum(row['candidate_count'] for row in rows),
            'selected_passages': sum(row['selected_passages'] for row in rows),
            'note': 'Counts/coverage are not relevance precision, recall or evidence accuracy'}
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        write(args.output_dir / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--paper-depth', type=int, default=5)
    parser.add_argument('--encoder-path', type=Path, required=True)
    parser.add_argument('--encoder-id', required=True)
    parser.add_argument('--encoder-revision', required=True)
    parser.add_argument('--reranker-path', type=Path, required=True)
    parser.add_argument('--reranker-id', required=True)
    parser.add_argument('--reranker-revision', required=True)
    parser.add_argument('--cpu-float64', action='store_true')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
