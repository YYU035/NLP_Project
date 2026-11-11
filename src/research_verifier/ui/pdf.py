"""Explicit local PDF verification; never fetch arbitrary URLs."""
from research_verifier.runtime.paths import PROJECT_ROOT
import json
import hashlib
from pathlib import Path

import streamlit as st

from research_verifier.contracts.models import Paper, RunSnapshot, make_context, digest
from research_verifier.verification.adapter import VerifierAdapter
from research_verifier.pipeline import local_verifier, safe_aggregate, _MODEL_LOCK
from research_verifier.evidence.semantic import local_selector
from research_verifier.evidence.pdf import extract_pdf, select_full_text
from research_verifier.explanations.templates import build_template
from research_verifier.explanations.grounding import validate_explanation
from research_verifier.explanations.replay import export_snapshot
from research_verifier.runtime.session import start_job, poll_job

ROOT = PROJECT_ROOT


def verify_pdf(raw, claim):
    if not raw.startswith(b'%PDF-') or len(raw) > 20*1024*1024:
        raise ValueError('Expected a PDF of at most 20 MiB')
    if not claim.strip() or len(claim)>20_000:
        raise ValueError('Expected a bounded original claim')
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    path = ROOT / '.cache' / 'tmp' / 'pdf_uploads' / (sha+'.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    identity = 'pdf:'+sha
    document = extract_pdf(path, identity, max_pages=200)
    if sum(len(p['text']) for p in document['pages']) > 2_000_000:
        raise ValueError('PDF text exceeds the local processing limit')
    with _MODEL_LOCK:
        verifier = local_verifier()
        payload = select_full_text(claim, document, local_selector(), verifier.tokenizer, top_k=3)
        from research_verifier.verification.schemas import EvidencePassage
        passages = [EvidencePassage.model_validate(p) for p in payload['passages']]
        native = verifier.verify(claim, [{'paper_id':identity,'title':'Local uploaded PDF','evidence_type':'full_text'}],
                                 {identity:passages})
    adapter = VerifierAdapter(lambda response,subset_of_paper_ids: safe_aggregate(verifier,response,subset_of_paper_ids))
    snapshot = adapter.snapshot(native,[Paper(paper_id=identity,title='Local uploaded PDF',source_version=sha,
        has_fulltext=True,source_status='local_upload_unverified')],
        abstained_paper_ids=[identity] if payload['status']=='no_passage_above_threshold' else [])
    data = snapshot.model_dump(mode='json')
    refs = {p['passage_id']:p for p in payload['source_spans']}
    for field in [data['verification']['evidence_passages'],data['provisional']['primary_evidence']]:
        for row in field:
            row.update(page=refs[row['passage_id']]['page'],source_version=sha)
    data.update(source_snapshot_id='pdf:'+sha,stage_statuses={'passages':'person3_local_pdf_bounded',
        'verification':'local_nli','pdf_review':'human_review_pending'},
        filters={'source_sha256':sha,'source':'Explicit local PDF upload',
                 'page_count':str(len(document['pages'])),'pdf_evidence_json':json.dumps(payload),
                 'retrieval_timestamp':'unknown; local import time used for bookkeeping'})
    if payload['rejected_passages']:
        data['verification']['processing_failures'].append({'stage':'passage_selection',
            'code':'VERIFIER_INPUT_LIMIT','paper_id':identity})
        if data['provisional']['status']!='failed':
            data['provisional']['status']='partial'
    return RunSnapshot.model_validate(data)


def render_pdf():
    ss = st.session_state
    st.title('Verify evidence from a local PDF')
    st.write('Upload a readable English research PDF. Source identity uses the file hash; page references and exact extracted text are retained. No PDF URL is fetched.')
    st.caption('Limits: 20 MiB, 200 pages, 2 million extracted characters. Scanned/encrypted documents, OCR and image-table interpretation are unsupported. Layout and scientific interpretation need human review.')
    upload = st.file_uploader('Research PDF',type=['pdf'],key='pdf_upload')
    if upload is not None and upload.size > 20*1024*1024:
        st.error('PDF exceeds the 20 MiB limit.')
        return
    claim = st.text_area('Original claim for this PDF',key='pdf_claim',max_chars=20_000)
    key = digest({'claim':claim,'pdf':hashlib.sha256(upload.getvalue()).hexdigest() if upload else None})
    if st.button('Verify local PDF',disabled=upload is None or not claim.strip() or 'pdf_job' in ss):
        ss.pop('pdf_snapshot',None)
        ss.pop('pdf_error',None)
        ss.pdf_job = start_job(lambda b=upload.getvalue(),c=claim:verify_pdf(b,c),key,timeout=300)
        st.rerun()

    @st.fragment(run_every=.5 if 'pdf_job' in ss else None)
    def poll():
        if 'pdf_job' not in ss:
            return
        job = ss.pdf_job
        status, result = poll_job(job)
        if status=='pending':
            st.info('Extracting exact PDF spans and verifying locally…')
            return
        if not job.future.done():
            st.warning('PDF work exceeded the display deadline; waiting to release the worker. Its late result is discarded.')
            return
        ss.pop('pdf_job')
        if status=='done' and job.key==key:
            ss.pdf_snapshot, ss.pdf_result_key = result,key
        else:
            ss.pdf_error = 'PDF processing failed, timed out or became stale. Check local model setup, document readability and input limits.'
        st.rerun()
    poll()
    if ss.get('pdf_error'):
        st.error(ss.pdf_error)
    snapshot = ss.get('pdf_snapshot')
    if snapshot is not None and ss.get('pdf_result_key')==key:
        st.subheader('Model verdict: '+(snapshot.provisional.aggregated_verdict or 'PROCESSING FAILURE'))
        st.text('Status: '+snapshot.provisional.status)
        st.text('Source SHA256: '+snapshot.filters['source_sha256'])
        st.warning('Local full-text extraction; PDF layout and source authenticity are not independently verified. Human PDF review is pending.')
        context = make_context(snapshot,list(snapshot.papers),snapshot.provisional)
        explanation = build_template(context)
        if not validate_explanation(explanation,context).displayable:
            st.error('Explanation failed the grounding gate.')
            return
        for passage in snapshot.provisional.primary_evidence:
            st.text(f'Page {passage.page} | {passage.passage_id}')
            st.text(passage.text)
        for note in explanation.limitations:
            st.text(note)
        with st.expander('Exact source spans, token checks and processing failures'):
            st.json(json.loads(snapshot.filters['pdf_evidence_json']))
            st.json([p.model_dump() for p in snapshot.verification.processing_failures])
        st.download_button('Download PDF verification replay',export_snapshot(context,explanation),
                           'pdf-verification.json','application/json')
