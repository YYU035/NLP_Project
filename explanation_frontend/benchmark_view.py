"""Offline published benchmark reporting; never launch experiments during render."""
import csv
import hashlib
import json
import math
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent / 'benchmarks'


def checked_path(path, root=ROOT):
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Benchmark path escapes snapshot')
    return target


def load_results(root=ROOT):
    source = json.loads((root/'SOURCE.json').read_text(encoding='utf-8'))
    for name, expected in source['sha256'].items():
        if hashlib.sha256(checked_path(name, root).read_bytes()).hexdigest() != expected:
            raise ValueError('Benchmark asset checksum mismatch')
    catalog = json.loads(checked_path('shared/datasets/evaluation_catalog.json', root).read_text())
    if catalog['active_dataset_ids'] != ['scifact', 'csfcube', 'reciper']:
        raise ValueError('Unexpected dataset catalog')
    result = []
    for item in catalog['datasets']:
        with checked_path(item['ranking_summary_path'], root).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        keys = set()
        for row in rows:
            key = tuple(row[c] for c in item['method_key_columns'])
            if key in keys:
                raise ValueError('Duplicate benchmark method/configuration')
            keys.add(key)
            for metric in item['ranking_metrics']:
                value = float(row[metric])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid benchmark metric')
                row[metric] = value
        evidence = json.loads(checked_path(item['evidence_summary_path'], root).read_text()) if item.get('evidence_summary_path') else None
        result.append({'metadata': item, 'rows': rows, 'evidence': evidence, 'source': source})
    return result


def render_benchmarks():
    st.title('Three-domain evaluation')
    st.write('Published teammate experiments: SciFact, CSFCube and RECIPER. These fixed benchmarks do not restrict normal live searches.')
    st.info('Development/benchmark results, not a three-domain scientific accuracy score. Frontend integration validates the result files; it does not independently reproduce the experiments.')
    try:
        datasets = load_results()
    except Exception:
        st.error('Benchmark snapshot failed integrity or schema validation.')
        return
    names = [x['metadata']['name'] for x in datasets]
    name = st.selectbox('Dataset', names, key='benchmark_dataset')
    dataset = datasets[names.index(name)]
    item = dataset['metadata']
    st.subheader(item['name'] + ' — ' + item['domain'])
    st.text('Task: '+item['task'])
    st.text('Split: '+item['split'])
    st.text(f"{item['total_queries']} {item['query_unit']} inputs; collection: {item['corpus_documents']} documents")
    if item.get('ranking_excluded_no_positive') is not None:
        st.text(f"Ranking denominator: {item['ranking_eligible_queries']}; excluded without positive paper labels: {item['ranking_excluded_no_positive']}")
    for note in item['limitations']:
        st.text('Limit: '+note)
    st.caption('Scores below are fractions in [0, 1]. Graded nDCG uses graded labels; binary nDCG is a separate column. No pooled score is computed across datasets.')
    columns = list(dict.fromkeys(item['method_key_columns']+item['ranking_metrics']))
    st.dataframe([{k: row[k] for k in columns} for row in dataset['rows']], hide_index=True)
    if dataset['evidence'] is not None:
        st.subheader('SciFact evidence-selection diagnostics')
        st.caption('Custom sentence/rationale metrics. Oracle-paper and retrieved-paper conditions are separate; these are not final verdict accuracy or official SciFact score parity.')
        st.dataframe([{'selector': row['policy'], 'paper_input': row['ranking_method'], 'paper_depth': row['paper_depth'],
            'precision': row['micro']['precision'], 'recall': row['micro']['recall'], 'F1': row['micro']['f1'],
            'complete_rationale_document_recall': row['complete_rationale_document_recall'],
            'evaluated_claims': row['evaluated_claims'], 'excluded_no_gold': row['excluded_no_gold']}
            for row in dataset['evidence']], hide_index=True)
    else:
        st.warning('No expert sentence-evidence or final-verdict gold is supplied here. Evidence F1 and verification accuracy are not available for this dataset.')
    st.subheader('What remains to validate')
    st.write('Final verification quality needs labelled claim/evidence inference. Human explanation and PDF review remain pending. Materials Science is the third domain; it is not a Physics dataset.')
    with st.expander('Source revision, protocol and original summaries'):
        st.text('YYU035/NLP_Project @ '+dataset['source']['revision'])
        st.json(item)
        for key in ['provenance_path', 'verification_audit_path', 'report_path']:
            if item.get(key):
                raw = checked_path(item[key]).read_bytes()
                label = {'provenance_path':'Download experiment provenance',
                         'verification_audit_path':'Download source-paper ranking audit',
                         'report_path':'Download protocol report'}[key]
                st.download_button(label, raw, Path(item[key]).name, key=f'benchmark:{name}:{key}')
        st.download_button('Download original ranking CSV', checked_path(item['ranking_summary_path']).read_bytes(),
                           name+'-ranking.csv', 'text/csv')
