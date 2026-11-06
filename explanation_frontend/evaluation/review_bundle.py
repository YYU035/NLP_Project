"""Prepare local offline review materials; never fill in human scores."""
import csv
import html
import json
from pathlib import Path
import zipfile

from .run import REVIEW_FIELDS
from ..reviews import load_review_packets

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'outputs' / 'review_packet'
    private = target / 'private'
    private.mkdir(parents=True, exist_ok=True)
    packets, keys, records = [], [], []
    for label, source in [('stress', ROOT / 'outputs' / 'qwen_qc'),
                          ('real', ROOT / 'outputs' / 'development_qc' / 'paired_review')]:
        for filename, destination in [('review_packets.jsonl', packets), ('private_key.jsonl', keys),
                                       ('records.jsonl', records)]:
            for line in (source / filename).read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                row['output_id'] = label + '-' + row['output_id']
                destination.append(row)
    raw = '\n'.join(json.dumps(row, ensure_ascii=False) for row in packets)
    checked = load_review_packets(raw)
    (target / 'review_packets.jsonl').write_text(raw + '\n', encoding='utf-8')
    for name, rows in [('private_key.jsonl', keys), ('generation_records.jsonl', records)]:
        (private / name).write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
    with (target / 'review.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for packet in checked:
            for reviewer in ['reviewer-1', 'reviewer-2']:
                writer.writerow({'output_id': packet.output_id, 'reviewer': reviewer})
    escaped = html.escape
    cards = []
    for packet in checked:
        evidence = ''.join(f'<li><strong>{escaped(p.passage_id)}</strong><pre>{escaped(p.text)}</pre></li>' for p in packet.evidence)
        cards.append(f'<details><summary>{escaped(packet.output_id)} | {escaped(packet.case_id)}</summary>'
            f'<h3>Claim</h3><p>{escaped(packet.claim)}</p><p>Verifier output: {escaped(str(packet.authoritative_verdict))}</p>'
            f'<h3>Supplied primary evidence</h3><ul>{evidence}</ul>'
            f'<h3>Untrusted raw explanation for review</h3><pre>{escaped(packet.raw_output)}</pre></details>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Human review packet</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#18251e;background:#fafaf7}details{padding:16px;margin:12px 0;background:white;border:1px solid #cbd5ce;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}li{margin:14px 0}</style>
<h1>Explanation review packet</h1><p>__CASE_COUNTS__ These are development QC materials, not untouched held-out evidence.</p>
<p>Review the raw output against the supplied passages. A verifier output is not ground truth. Do not infer scientific support from quotation alone. Methods have A/B IDs, but writing style can reveal them.</p>
<p>Upload <a href="review_packets.jsonl">review_packets.jsonl</a> in the app's Evaluate explanations workspace, or fill <a href="review.csv">review.csv</a>. Read the rubric in evaluation/rubric.md. All 48 human-review rows start blank; no score has been invented. Export submitted scores before closing the app.</p>
<p>The coordinator stores the private method key, generation statuses and displayed fallback records separately. The public review ZIP excludes that private directory; writing style can still reveal methods.</p>'''
    page = page.replace('__CASE_COUNTS__', f'{len(checked)//2} cases, {len(checked)} paired outputs across synthetic stress and SciFact development oracle-document QC.')
    (target / 'index.html').write_text(page + ''.join(cards) + '</html>', encoding='utf-8')
    archive = ROOT / 'outputs' / 'Kireeti_QC_Review_Packet.zip'
    coordinator = ROOT / 'outputs' / 'Kireeti_QC_Coordinator_Packet.zip'
    for destination, include_private in [(archive, False), (coordinator, True)]:
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for file in target.rglob('*'):
                relative = file.relative_to(target)
                if file.is_file() and (include_private or 'private' not in relative.parts):
                    bundle.write(file, relative.as_posix())
    print(json.dumps({'outputs': len(checked), 'pending_human_rows': 2 * len(checked),
        'index': str(target / 'index.html'), 'zip': str(archive), 'coordinator_zip': str(coordinator)}), flush=True)


if __name__ == '__main__':
    main()
