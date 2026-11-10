"""RECIPER source-paper retrieval with a complete fixed paper collection.

Only original titles/abstracts and unchanged questions become scoring text.
Generated answers/recipes and source-paper targets are excluded from features.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .setup import verify_download


def from_rows(papers, questions, *, manifest):
    documents = {}
    for paper in papers:
        identity = paper['paper_id']
        if not isinstance(identity, str) or not identity or identity in documents:
            raise ValueError('Paper IDs must be nonempty and unique')
        title, abstract = paper['title'], paper['abstract']
        if not isinstance(title, str) or not title.strip() or not isinstance(abstract, str) or not abstract.strip():
            raise ValueError('Original title and abstract are required')
        metadata = paper.get('metadata', {})
        source = metadata.get('arxiv_id')
        landing = f'https://arxiv.org/abs/{source}' if source else None
        if not landing and metadata.get('doi'):
            landing = 'https://doi.org/' + metadata['doi'].removeprefix('https://doi.org/')
        authors = metadata.get('authors') or []
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            raise ValueError('Expected source author strings')
        documents[identity] = {'paper_id': identity, 'title': title, 'abstract': [abstract],
                               'landing_url': landing, 'authors': authors}
    if not documents:
        raise ValueError('Empty paper collection')
    candidate_ids = list(documents)
    queries, qrels, seen_texts = {}, {}, set()
    for index, row in enumerate(questions):
        text, target = row['question'], row['paper_id']
        if not isinstance(text, str) or not text.strip() or text in seen_texts:
            raise ValueError('Question texts must be nonempty and unique')
        if target not in documents:
            raise ValueError('Question references a paper outside the corpus')
        seen_texts.add(text)
        identity = f'reciper-{index:05d}-' + hashlib.sha256(text.encode()).hexdigest()[:16]
        queries[identity] = {'query_id': identity, 'text': text,
                             'candidate_ids': candidate_ids.copy()}
        qrels[identity] = [target]
    if not queries:
        raise ValueError('Empty question set')
    protocol = {'split': 'entire released QA set; frozen zero-shot settings, no training or tuning',
        'candidate_scope': 'complete released paper collection for every question',
        'scoring_text': 'original title plus abstract only; full text and recipes not indexed',
        'query_policy': 'unchanged question text; no generated answer, topic, source ID or paper title appended',
        'judgment': 'single source-paper ID per question; not exhaustive relevance judgments',
        'metric_unit': 'unique paper IDs', 'generated_qa': True,
        'evidence_sentence_gold': False, 'verification_labels': False,
        'official_parity': 'Adapted title/abstract paper retrieval; original RECIPER indexes paragraph/recipe streams'}
    version = hashlib.sha256(json.dumps({'manifest': manifest, 'protocol': protocol}, sort_keys=True).encode()).hexdigest()
    return {'dataset': 'RECIPER', 'version': version, 'manifest': manifest,
            'protocol': protocol, 'documents': documents, 'queries': queries, 'qrels': qrels}


def load(directory: Path):
    manifest = verify_download(directory, 'reciper')
    papers = json.loads((directory / 'data/rag_database/rag_database.json').read_text(encoding='utf-8'))
    questions = [json.loads(line) for line in (directory / 'data/rag_database/qa_dataset.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return from_rows(papers, questions, manifest=manifest)


def prepare(dataset, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {key: dataset[key] for key in ('dataset', 'version', 'protocol', 'documents', 'queries')}
    (directory / 'retrieval-input.json').write_text(json.dumps(inputs, indent=2) + '\n')
    (directory / 'judgments.json').write_text(json.dumps({'version': dataset['version'],
        'qrels': dataset['qrels'], 'warning': 'Evaluator only; do not provide targets to scoring'}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/reciper'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    dataset = load(args.data)
    prepare(dataset, args.output_dir)
    print(json.dumps({'papers': len(dataset['documents']), 'queries': len(dataset['queries']),
                      'version': dataset['version']}, indent=2))


if __name__ == '__main__':
    main()
