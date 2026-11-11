import csv
import json
import zipfile

from evaluation.run import run_study
from evaluation import review_bundle
from research_verifier.explanations.reviews import load_review_packets


def test_review_bundle_is_blinded_escaped_and_never_invents_human_scores(tmp_path, monkeypatch):
    result = run_study([{'case_id': 'practice', 'fixture': 'supported'}], mode='fixture')
    result['review_packets'][0]['raw_output'] = '<script>alert(1)</script>'
    for path in [tmp_path / 'outputs' / 'qwen_qc', tmp_path / 'outputs' / 'development_qc' / 'paired_review']:
        path.mkdir(parents=True)
        for filename, key in [('review_packets.jsonl', 'review_packets'), ('private_key.jsonl', 'private_key'),
                              ('records.jsonl', 'records')]:
            path.joinpath(filename).write_text('\n'.join(json.dumps(r) for r in result[key]), encoding='utf-8')
    monkeypatch.setattr(review_bundle, 'ROOT', tmp_path)
    review_bundle.main()
    output = tmp_path / 'outputs' / 'review_packet'
    packets = load_review_packets(output.joinpath('review_packets.jsonl').read_bytes())
    assert len(packets) == 4
    assert len({p.output_id for p in packets}) == 4
    page = output.joinpath('index.html').read_text(encoding='utf-8')
    assert '&lt;script&gt;' in page and '<script>alert' not in page
    with output.joinpath('review.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(not value for row in rows for key, value in row.items() if key not in {'output_id', 'reviewer'})
    assert output.joinpath('private/private_key.jsonl').is_file()
    with zipfile.ZipFile(tmp_path / 'outputs' / 'Kireeti_QC_Review_Packet.zip') as bundle:
        assert not any(name.startswith('private/') for name in bundle.namelist())
    with zipfile.ZipFile(tmp_path / 'outputs' / 'Kireeti_QC_Coordinator_Packet.zip') as bundle:
        assert 'private/private_key.jsonl' in bundle.namelist()
