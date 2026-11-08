"""Install an opt-in local preview adapter; never pushes teammate branches."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def install(frontend_root: Path):
    module = frontend_root/'explanation_frontend'
    target = module/'passage_selection.py'
    text = target.read_text()
    signature = 'def select_abstract(claim, identity, abstract, *, top_k=3):\n'
    marker = '# Person 3 opt-in semantic evidence bridge'
    if signature not in text:
        raise ValueError('Unsupported frontend selector interface')
    existing_pipeline = (module/'pipeline.py').read_text()
    if ('# Person 3 input budget diagnostics' not in existing_pipeline
            and '        passages = {pid: select_abstract(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}' not in existing_pipeline):
        raise ValueError('Unsupported frontend passage construction boundary')
    package = module/'_person3'
    package.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    hashes = {}
    for name in ['evidence_selection.py','semantic_evidence.py','frontend_bridge.py','verifier_budget.py']:
        path = source/name
        shutil.copyfile(path, package/name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (package/'__init__.py').write_text('"""Pinned local Person 3 evidence runtime."""\n')
    (package/'SOURCE.json').write_text(json.dumps({'source_sha256':hashes,
        'activation':'PERSON3_EVIDENCE_CONFIG and PERSON3_EVIDENCE_MODEL_PATH; default legacy preserved'},indent=2)+'\n')
    if marker not in text:
        backup = module/'outputs/person3-integration/original-passage-selection.py'
        backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_text(text)
        hook = '''    # Person 3 opt-in semantic evidence bridge
    import os
    if os.environ.get('PERSON3_EVIDENCE_CONFIG'):
        from ._person3.frontend_bridge import select_abstract as select_person3
        return [EvidencePassage.model_validate(row) for row in
                select_person3(claim, identity, abstract, top_k=top_k)]
'''
        target.write_text(text.replace(signature, signature+hook, 1))
    pipeline = module/'pipeline.py'
    text = pipeline.read_text()
    if 'import os\n' not in text:
        text = text.replace('import json\n','import json\nimport os\n',1)
    text = text.replace('"passages": "frontend_abstract_windows_v1"',
        '"passages": "person3_semantic_train_frozen" if os.environ.get("PERSON3_EVIDENCE_CONFIG") else "frontend_abstract_windows_v1"')
    text = text.replace('"passage_selector": "exact abstract windows; offsets encoded in passage IDs"',
        '"passage_selector": "Person 3 frozen semantic selection; source offsets retained" if os.environ.get("PERSON3_EVIDENCE_CONFIG") else "exact abstract windows; offsets encoded in passage IDs"')
    original_passages = '        passages = {pid: select_abstract(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}'
    if '# Person 3 input budget diagnostics' not in text:
        bounded_passages = """        # Person 3 input budget diagnostics
        budget_selections = {}
        if os.environ.get('PERSON3_EVIDENCE_CONFIG'):
            from ._person3.frontend_bridge import select_abstract_with_diagnostics
            from ._upstream.claim_verifier.schemas import EvidencePassage
            budget_selections = {pid: select_abstract_with_diagnostics(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}
            passages = {pid: [EvidencePassage.model_validate(row) for row in value['passages']] for pid, value in budget_selections.items()}
        else:
            passages = {pid: select_abstract(claim, pid, entry_map[pid].paper.abstract) for pid in ordered}"""
        if original_passages not in text:
            raise ValueError('Unsupported frontend passage construction boundary')
        text = text.replace(original_passages, bounded_passages, 1)
        diagnostics = """        if os.environ.get('PERSON3_EVIDENCE_CONFIG'):
            data['stage_statuses']['passages'] = 'person3_semantic_bounded_train_frozen'
            data['filters']['verifier_budget_json'] = json.dumps({pid: {'checks': value['budget_checks'], 'rejected': value['rejected_passages'], 'contract': value['verifier_contract']} for pid, value in budget_selections.items()}, sort_keys=True)
"""
        text = text.replace('        if source.result.status == "failed":', diagnostics+'        if source.result.status == "failed":', 1)
    if '# Person 3 budget rejection is operational' not in text:
        operational = """        # Person 3 budget rejection is operational, not a scientific verdict.
        rejected_papers = [pid for pid, value in budget_selections.items() if value['rejected_passages']]
        for pid in rejected_papers:
            data['verification']['processing_failures'].append({'stage': 'passage_selection', 'code': 'VERIFIER_INPUT_LIMIT', 'paper_id': pid})
        if rejected_papers and not any(passages.values()):
            data['stage_statuses']['passages'] = 'person3_verifier_input_limit'
            data['provisional'] = AggregationResponse(claim=claim, aggregated_verdict=None, status='failed', sufficiency_cause='VERIFIER_INPUT_LIMIT').model_dump(mode='json')
        elif rejected_papers and data['provisional']['status'] != 'failed':
            data['provisional']['status'] = 'partial'
"""
        text = text.replace('        result = RunSnapshot.model_validate(data)', operational+'        result = RunSnapshot.model_validate(data)', 1)
        aggregate_guard = """        if snapshot.stage_statuses.get('passages') == 'person3_verifier_input_limit':
            if self._snapshot is None or snapshot.fingerprint != self._snapshot.fingerprint:
                raise ValueError('No matching original response')
            if set(paper_ids) - set(snapshot.papers):
                raise ValueError('Unknown selected paper')
            return snapshot.provisional
"""
        text = text.replace('        if snapshot.stage_statuses.get("discovery") == "failed":', aggregate_guard+'        if snapshot.stage_statuses.get("discovery") == "failed":', 1)
    pipeline.write_text(text)
    print(package)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-root',type=Path,required=True)
    install(parser.parse_args().frontend_root)
