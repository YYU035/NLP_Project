"""Optional model attribution on a frozen run; never a scientific support score."""

import streamlit as st

from .pipeline import local_verifier, _MODEL_LOCK
from .pipeline_setup import NLI_ID, NLI_REVISION
from .state import start_job, poll_job


def explain_passage(snapshot, passage_id, baseline="mask"):
    import os
    from pathlib import Path
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.model-cache/matplotlib'))
    from ._upstream.claim_verifier.schemas import EvidencePassage
    settings = snapshot.verification.model_settings
    if (settings.checkpoint, settings.revision) != (NLI_ID, NLI_REVISION):
        raise ValueError('Attribution model does not match the frozen verifier')
    passage = next(p for p in snapshot.verification.evidence_passages if p.passage_id == passage_id)
    with _MODEL_LOCK:
        verifier = local_verifier()
        if len(verifier.tokenizer.encode(passage.text, snapshot.claim, add_special_tokens=True, truncation=False)) > 512:
            raise ValueError('Attribution input exceeds the original verifier budget')
        native = EvidencePassage(passage_id=passage.passage_id,paper_id=passage.paper_id,
            section=passage.section,text=passage.text,score=passage.score or 0.0)
        from .xai_runtime import explain_checked
        result = explain_checked(verifier,snapshot.claim,native,baseline=baseline)
    return result


def render_attribution():
    ss = st.session_state
    st.title('Inspect model attribution')
    st.write('Naila’s Integrated Gradients shows which words influenced the NLI model. Positive/negative attribution is not evidence that a scientific claim is true or false.')
    from pathlib import Path
    report_path=Path(__file__).parent/'evaluation/xai-results.json'
    if report_path.is_file():
        import json
        with st.expander('Recorded XAI validation and limitations'):
            report=json.loads(report_path.read_text(encoding='utf-8'))
            st.caption('Recorded exploratory results, not a new experiment. Convergence does not establish scientific correctness.')
            st.json(report, expanded=False)
            st.download_button('Download XAI validation summary',json.dumps(report,indent=2),'xai-validation.json','application/json')

    sources = {label: ss[name] for label,name in [('Verification run','snapshot'),('Local PDF run','pdf_snapshot')] if name in ss}
    if not sources:
        st.info('Verify a claim, load a real-model replay, or verify a local PDF first.')
        return
    if ss.get('attribution_source') not in sources:
        ss.attribution_source = next(iter(sources))
    snapshot = sources[st.selectbox('Frozen run',list(sources),key='attribution_source')]
    settings = snapshot.verification.model_settings
    if (settings.checkpoint, settings.revision) != (NLI_ID, NLI_REVISION):
        st.info('This run does not use the supported pinned NLI model. Synthetic/demo predictions cannot be explained as real-model predictions.')
        return
    entries = {p.passage_id:p for p in snapshot.verification.evidence_passages}
    if not entries:
        st.info('This run has no passages to explain.')
        return
    if ss.get('attribution_passage') not in entries:
        ss.attribution_passage = next(iter(entries))
    pid = st.selectbox('Passage', list(entries), key='attribution_passage')
    baseline = st.selectbox('Attribution baseline', ['mask','pad'], key='attribution_baseline',
        help='Replaces content tokens along the embedding path; special tokens and original attention mask stay fixed. Baseline choice changes the explanation.')
    key = (snapshot.fingerprint, pid, baseline)
    st.text(snapshot.claim)
    st.text(entries[pid].text)
    st.caption('Opt-in CPU computation: 32 integration steps, retrying at 128 and 512 if completeness is poor. Explains the pinned model argmax before decision thresholding; the verification result is not changed.')
    if st.button('Compute word attribution', disabled='attribution_job' in ss):
        ss.pop('attribution_result', None)
        ss.pop('attribution_error', None)
        ss.attribution_job = start_job(lambda s=snapshot,p=pid,b=baseline: explain_passage(s,p,b), key, timeout=600)
        st.rerun()

    @st.fragment(run_every=.5 if 'attribution_job' in ss else None)
    def poll():
        if 'attribution_job' not in ss:
            return
        job = ss.attribution_job
        status, result = poll_job(job)
        if status == 'pending':
            st.info('Computing local model gradients…')
            return
        # Keep a timed-out running worker registered; do not admit repeated work.
        if not job.future.done():
            st.warning('Attribution exceeded the display deadline; waiting for the worker to release the model. No result will be displayed.')
            return
        ss.pop('attribution_job')
        if status == 'done' and job.key == key:
            ss.attribution_result = (key, result)
        else:
            ss.attribution_error = 'Attribution failed, timed out or became stale. No explanation result was substituted.'
        st.rerun()
    poll()
    if ss.get('attribution_error'):
        st.warning(ss.attribution_error)
    saved = ss.get('attribution_result')
    if saved and saved[0] == key:
        result = saved[1]
        st.text(f"Explained label: {result['target_label']} | raw model class: {result['target_class']}")
        st.text(f"Completeness delta: {result['convergence_delta']:.6g}; baseline: {result['settings'].get('baseline','pad')}")
        st.caption('Larger absolute deltas indicate a less reliable numerical approximation. Attribution is model behaviour, not an independent factual/citation evaluation.')
        st.caption('Green means a positive contribution toward the explained class relative to this baseline; red means a negative contribution. Neither colour measures factual correctness. The target is the raw model class, which may differ from a thresholded verdict.')
        if result.get('attempts'):
            st.dataframe(result['attempts'], hide_index=True)
        # Text and numeric table instead of executing supplied HTML.
        if result['numerical_status']=='passed':
            rows = [dict(w, direction='🟢 toward target' if w['attribution'] > 0 else '🔴 away from target' if w['attribution'] < 0 else 'zero') for w in result['words']]
            import pandas as pd
            def colour(value):
                return 'background-color: #d9f2dd; color: #153b1c' if value > 0 else 'background-color: #f8dada; color: #541b1b' if value < 0 else ''
            styled = pd.DataFrame(rows).style.map(colour, subset=['attribution','normalized'])
            st.dataframe(styled, hide_index=True)
        else:
            st.warning('Attribution did not pass the completeness checks (relative ≤5% and absolute ≤0.001). Token importance is not presented as reliable; diagnostics can still be downloaded. The scientific verdict is unchanged.')
        import json
        st.download_button('Download attribution', json.dumps(result, indent=2), 'model-attribution.json', 'application/json')
