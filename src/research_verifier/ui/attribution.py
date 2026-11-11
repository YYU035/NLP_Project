"""Optional model attribution on a frozen run; never a scientific support score."""

from research_verifier.runtime.paths import PROJECT_ROOT, CACHE_DIR
import streamlit as st

from research_verifier.pipeline import local_verifier, _MODEL_LOCK
from research_verifier.runtime.setup_verifier import NLI_ID, NLI_REVISION
from research_verifier.runtime.session import start_job, poll_job


def explain_passage(snapshot, passage_id, baseline="mask", method="Word occlusion (fast)"):
    import os
    from pathlib import Path
    os.environ.setdefault('MPLCONFIGDIR',str(CACHE_DIR/'matplotlib'))
    from research_verifier.verification.schemas import EvidencePassage
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
        if method=='Word occlusion (fast)':
            from research_verifier.interpretability.occlusion import explain_occlusion
            return explain_occlusion(verifier,snapshot.claim,native)
        from research_verifier.interpretability.service import explain_checked
        result = explain_checked(verifier,snapshot.claim,native,baseline=baseline)
    return result


def render_attribution():
    ss = st.session_state
    st.title('Inspect model attribution')
    st.write('Inspect which words influence the NLI model with direct word occlusion or Naila’s experimental Integrated Gradients. Positive/negative attribution is not evidence that a scientific claim is true or false.')
    from pathlib import Path
    report_path=PROJECT_ROOT/'evaluation/xai-results.json'
    if report_path.is_file():
        import json
        with st.expander('Recorded XAI validation and limitations'):
            report=json.loads(report_path.read_text(encoding='utf-8'))
            st.caption('Recorded exploratory results, not a new experiment. Convergence does not establish scientific correctness.')
            st.json(report, expanded=False)
            st.download_button('Download XAI validation summary',json.dumps(report,indent=2),'xai-validation.json','application/json')

    occlusion_report=PROJECT_ROOT/'evaluation/occlusion-results.json'
    if occlusion_report.is_file():
        with st.expander('Word-occlusion exploratory evaluation (four cases)'):
            import json
            study=json.loads(occlusion_report.read_text(encoding='utf-8'))
            st.caption(study['protocol'])
            cols=st.columns(2)
            cols[0].metric('Sentence AP',f"{study['mean_ap']:.3f}",f"random {study['mean_ap_random']:.3f}",delta_color='off')
            cols[1].metric('Top sentence rationale hit',f"{study['top1_hit_rate']:.0%}",f"random {study['top1_random']:.1%}",delta_color='off')
            st.json(study['faithfulness'])
            st.warning('Exploratory four-case dev result; not IG, representative accuracy or a significance test. Gold-target attribution differs from explaining a wrong predicted class.')
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
        decisive=[p.passage_id for p in snapshot.provisional.primary_evidence if p.passage_id in entries]
        ss.attribution_passage = decisive[0] if decisive else next(iter(entries))
    pid = st.selectbox('Passage', list(entries), key='attribution_passage',
        format_func=lambda value: snapshot.papers[entries[value].paper_id].title[:60]+' | '+entries[value].text[:65])
    method=st.selectbox('Explanation method',['Word occlusion (fast)','Integrated Gradients (experimental)'],key='attribution_method')
    if method=='Word occlusion (fast)' and ss.get('attribution_baseline')=='pad':ss.attribution_baseline='mask'
    baseline = st.selectbox('Attribution baseline', ['mask'] if method=='Word occlusion (fast)' else ['mask','pad'], key='attribution_baseline',
        help='MASK replaces one whole word at a time for occlusion. IG interpolates embeddings from the named baseline. The original attention mask and special tokens stay fixed.')
    key = (snapshot.fingerprint, pid, baseline, method)
    st.text(snapshot.claim)
    st.text(entries[pid].text)
    if method=='Integrated Gradients (experimental)':st.caption('Opt-in CPU computation: 32 integration steps, retrying at 128 and 512 if completeness is poor. Explains the pinned model argmax before decision thresholding; the verification result is not changed.')
    if method=='Word occlusion (fast)':
        st.caption('Directly measures target-probability change when each word is replaced by MASK. Up to96 words, balanced between evidence and claim; coverage is reported. Scores need not sum to the model output.')
    if method=='Word occlusion (fast)':
        from research_verifier.interpretability.cache import load_recorded
        try:
            recorded=load_recorded(snapshot,pid)
        except (ValueError,OSError,KeyError):
            recorded=None
            st.warning('Saved attribution could not be validated. No saved scores were substituted; fresh computation remains available.')
        if recorded is not None and st.button('Load recorded word sensitivity'):
            ss.attribution_result=(key,recorded)
            st.rerun()
    if st.button('Compute word attribution', disabled='attribution_job' in ss):
        ss.pop('attribution_result', None)
        ss.pop('attribution_error', None)
        ss.attribution_job = start_job(lambda s=snapshot,p=pid,b=baseline,m=method: explain_passage(s,p,b,m), key, timeout=600)
        st.rerun()

    @st.fragment(run_every=.5 if 'attribution_job' in ss else None)
    def poll():
        if 'attribution_job' not in ss:
            return
        job = ss.attribution_job
        status, result = poll_job(job)
        if status == 'pending':
            st.info('Measuring local model responses…' if method=='Word occlusion (fast)' else 'Computing local model gradients…')
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
        if result.get('recorded'):st.info(result['recording_note'])
        st.text(f"Explained label: {result['target_label']} | raw model class: {result['target_class']}")
        if result.get("kind")=="word_occlusion":
            st.caption("Measured word-MASK occlusion; not Integrated Gradients. "+str(result["coverage"]["scored_words"])+"/"+str(result["coverage"]["total_words"])+" words scored.")
            if not result['coverage']['complete']:st.warning('Only the scored words are displayed; unscored words have unknown importance, not zero importance.')
        else:
            st.text(f"Completeness delta: {result['convergence_delta']:.6g}; baseline: {result['settings'].get('baseline','pad')}")
        if result.get('kind')!='word_occlusion':st.caption('Larger absolute deltas indicate a less reliable numerical approximation. Attribution is model behaviour, not an independent factual/citation evaluation.')
        st.caption('Green means a positive contribution toward the explained class relative to this baseline; red means a negative contribution. Neither colour measures factual correctness. The target is the raw model class, which may differ from a thresholded verdict.')
        if result.get('attempts'):
            st.dataframe(result['attempts'], hide_index=True)
        # Text and numeric table instead of executing supplied HTML.
        if result.get('kind')=='word_occlusion' or result.get('numerical_status')=='passed':
            rows = [dict(w, direction='🟢 toward target' if w['attribution'] > 0 else '🔴 away from target' if w['attribution'] < 0 else 'zero') for w in result['words']]
            import pandas as pd
            def colour(value):
                if abs(value)<.001:return ''
                return 'background-color: #d9f2dd; color: #153b1c' if value > 0 else 'background-color: #f8dada; color: #541b1b' if value < 0 else ''
            styled = pd.DataFrame(rows).style.map(colour, subset=['attribution','normalized'])
            st.dataframe(styled, hide_index=True)
        else:
            st.warning('Attribution did not pass the completeness checks (relative ≤5% and absolute ≤0.001). Token importance is not presented as reliable; diagnostics can still be downloaded. The scientific verdict is unchanged.')
        import json
        st.download_button('Download attribution', json.dumps(result, indent=2), 'model-attribution.json', 'application/json')
