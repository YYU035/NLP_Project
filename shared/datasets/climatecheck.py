"""Adapt human-annotated ClimateCheck abstract pools for ranking evaluation.

This is an explicitly restricted judged-pool reranking protocol, not search
against the complete 394,269-paper corpus. No sentence rationale gold exists.
"""

import argparse
import hashlib
import json
from pathlib import Path

from shared.datasets.setup import verify_download

LABELS = {'Supports', 'Refutes', 'Not Enough Information'}


def from_rows(rows, *, split, manifest):
    if split not in ('train', 'test'):
        raise ValueError('Expected train or test split')
    documents, queries, grades, pair_labels = {}, {}, {}, {}
    claim_variants = {}
    for row in rows:
        claim, abstract, label = row['claim'], row['abstract'], row['annotation']
        if (not isinstance(claim, str) or not claim.strip() or
                not isinstance(abstract, str) or not abstract.strip()):
            raise ValueError('Expected nonempty original claim/abstract text')
        if type(row['claim_id']) is not int or type(row['abstract_id']) is not int:
            raise ValueError('Expected integer source IDs')
        if label not in LABELS:
            raise ValueError('Unknown claim-abstract annotation')
        # Some official IDs refer to different claim texts. Keep them distinct,
        # preserving every exact text and pairing rather than merging their gold.
        digest = hashlib.sha256(claim.encode()).hexdigest()[:24]
        qid, pid = f"cc-{row['claim_id']}-{digest}", f"cc-abstract-{row['abstract_id']}"
        claim_variants.setdefault(row['claim_id'], set()).add(claim)
        document = {'paper_id': pid, 'title': '', 'abstract': [abstract]}
        if pid in documents and documents[pid] != document:
            raise ValueError('Abstract ID has conflicting source texts')
        documents[pid] = document
        if qid in queries and queries[qid]['text'] != claim:
            raise ValueError('Claim hash collision')
        queries.setdefault(qid, {'query_id': qid, 'source_claim_id': row['claim_id'],
                                 'text': claim, 'candidate_ids': []})
        if pid in grades.setdefault(qid, {}):
            raise ValueError('Duplicate exact claim-abstract pair')
        queries[qid]['candidate_ids'].append(pid)
        grades[qid][pid] = 0 if label == 'Not Enough Information' else 2
        pair_labels.setdefault(qid, {})[pid] = label
    if not queries:
        raise ValueError('No annotated queries')
    protocol = {'split': split, 'candidate_scope': 'per-exact-claim human-judged abstract pool only',
        'query_policy': 'unchanged exact released claim; source ID plus text hash prevents ID collision',
        'positive_policy': 'Supports OR Refutes; NEI is non-evidentiary for this task, not necessarily off-topic',
        'title_policy': 'not available in pair files; abstract-only ranking',
        'task': 'judged abstract reranking; not full-corpus retrieval or sentence selection',
        'sentence_evidence_gold': False, 'source_claim_ids_with_multiple_texts': sum(len(v) > 1 for v in claim_variants.values())}
    version = hashlib.sha256(json.dumps({'manifest': manifest, 'protocol': protocol}, sort_keys=True).encode()).hexdigest()
    return {'dataset': 'ClimateCheck', 'version': version, 'manifest': manifest,
            'protocol': protocol, 'documents': documents, 'queries': queries, 'grades': grades,
            'pair_labels': pair_labels}


def load(directory: Path, *, split='test'):
    manifest = verify_download(directory, 'climatecheck')
    if split not in ('train', 'test'):
        raise ValueError('Expected train or test split')
    import pyarrow.parquet as pq
    rows = pq.read_table(directory / 'data' / f'{split}-00000-of-00001.parquet').to_pylist()
    return from_rows(rows, split=split, manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/climatecheck'))
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    args = parser.parse_args()
    dataset = load(args.data, split=args.split)
    print(json.dumps({'dataset': dataset['dataset'], 'version': dataset['version'],
        'documents': len(dataset['documents']), 'exact_claims': len(dataset['queries']),
        'pairs': sum(len(v) for v in dataset['grades'].values()), 'protocol': dataset['protocol']}, indent=2))


if __name__ == '__main__':
    main()
