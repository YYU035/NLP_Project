"""Project presentation and honest, reproducible actual-model replay gallery."""
import hashlib,json
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).parent


def navigate(workspace):st.session_state.workspace=workspace


def load_showcase(name):
    root=ROOT/'showcase'
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Replay path escape')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=manifest['sha256'][name]
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Replay checksum mismatch')
    from .export import import_snapshot
    return import_snapshot(raw)


def install_showcase(name):
    bundle=load_showcase(name);ss=st.session_state
    for key in ('generation','operation_error','pending_discovery','attribution_result','attribution_error'):
        ss.pop(key,None)
    ss.workspace='Verify evidence';ss.mode='Cached replay';ss.claim=bundle.snapshot.claim
    ss.snapshot=bundle.snapshot;ss.provider=None;ss.replay=bundle
    ss.selection_snapshot=bundle.snapshot.fingerprint
    ss.paper_selection={pid:pid in bundle.selected_paper_ids for pid in bundle.snapshot.papers}
    for pid,value in ss.paper_selection.items():ss[f'paper:{bundle.snapshot.fingerprint}:{pid}']=value
    ss.aggregation_cache={};ss.aggregation_errors=set()


def start_showcase(name):
    from .discovery import DiscoverySnapshot
    from .state import install_discovery_state
    bundle=load_showcase(name)
    data=dict(bundle.snapshot.discovery_context);data['ranking_context']=None
    source=DiscoverySnapshot.model_validate(data)
    ss=st.session_state
    for key in ('snapshot','replay','provider','generation','pending_discovery','operation_error'):
        ss.pop(key,None)
    install_discovery_state(ss,source)
    ss.discovery_source='Upload search'
    ss.workspace='Find papers'


def render_overview():
    st.title('Every verdict, open to inspection')
    st.write('Follow a scientific claim from paper discovery to ranked sources, exact evidence, a local model decision and a source-linked explanation.')
    st.caption('Five team contributions · Three evaluation domains · Local model inference · Replayable evidence')
    cols=st.columns(3)
    for col,title,text,dest in zip(cols,['Explore a claim','Inspect the research','Explain the model'],
        ['Find papers, verify passages and compare selected sources.','Compare retrieval methods, evidence metrics and uncertainty.','Inspect signed word contributions with numerical checks.'],
        ['Find papers','Benchmark results','Model attribution']):
        with col:
            with st.container(border=True):
                st.subheader(title);st.write(text);st.button({'Find papers':'Start a search','Benchmark results':'Compare methods','Model attribution':'Inspect attribution'}[dest],on_click=navigate,args=(dest,))
    st.subheader('The integrated research pipeline')
    steps=st.columns(5)
    for col,label,text in zip(steps,['01 Discover','02 Retrieve','03 Select evidence','04 Verify','05 Explain'],
        ['Public catalogues and provenance','BM25 / semantic search / RRF','Frozen semantic selector and token budget','Pinned DeBERTa with cached aggregation','Grounded citations, replay and attribution']):
        with col:st.markdown('**'+label+'**');st.caption(text)
    from .benchmark_view import load_results
    try:
        datasets=load_results()
    except (ValueError,OSError,KeyError):
        st.error('Benchmark assets failed validation; navigation remains available, but no benchmark scores are displayed.')
        return
    validation_root=ROOT/'evaluation/validation'
    validation_summary=validation_root/'results/validation-summary.json'
    if validation_summary.is_file():
        with st.expander('Full-pipeline validation · 10 November 2026'):
            try:
                measured=json.loads(validation_summary.read_text(encoding='utf-8'))
                study=measured['science']['conditions']['retrieved_papers_production_selector']
                left,right=st.columns(2)
                left.metric('SciFact offline claim accuracy',f"{study['all']['accuracy']:.1%}")
                right.metric('Processing failures /300',str(study['operational_failures']))
                st.caption('All 300 previously inspected dev claims; all 5,183 corpus papers; automatic RRF top 5. Frozen descriptive audit, not untouched holdout or public-web recall. Accuracy still needs work; confidence is not a correctness guarantee.')
                st.download_button('Download full validation report',(validation_root/'FINAL_TEST_REPORT.md').read_bytes(),file_name='FINAL_TEST_REPORT.md',mime='text/markdown')
                st.download_button('Download error catalogue',(validation_root/'results/error_catalogue.csv').read_bytes(),file_name='error_catalogue.csv',mime='text/csv')
            except (OSError,ValueError,KeyError,TypeError):
                st.error('Validation report could not be loaded; no validation scores are inferred.')
    st.subheader('Measured retrieval results')
    cards=st.columns(3)
    for col,dataset in zip(cards,datasets):
        row=next(r for r in dataset['rows'] if r['method']=='rrf')
        with col:
            st.metric(dataset['metadata']['name']+' · RRF Recall@10',f"{row['Recall@10']:.1%}")
            st.caption(dataset['metadata']['split']+' · '+str(dataset['metadata'].get('ranking_eligible_queries',dataset['metadata']['total_queries']))+' evaluated tasks')
    st.caption('Published owner experiments; different tasks and candidate universes. These are not three-domain verdict accuracy or pooled scores. Full protocols are in Benchmark results.')
    report=ROOT/'evaluation/release-report.json'
    if report.is_file():
        release=json.loads(report.read_text(encoding='utf-8'))
        with st.expander('Validation coverage and limitations'):
            st.json(release,expanded=False)
            st.download_button('Download validation report',json.dumps(release,indent=2),'release-validation.json','application/json')
    manifest=ROOT/'showcase/manifest.json'
    if manifest.is_file():
        gallery=json.loads(manifest.read_text(encoding='utf-8'))
        st.subheader('Inspect an actual saved run')
        st.write('These examples use actual local models on public SciFact text. The known annotated paper was included with two seeded distractors. They demonstrate the integration; they are not unbiased retrieval or verdict-quality estimates.')
        for example in gallery['examples']:
            with st.container(border=True):
                st.markdown('**'+example['verdict']+' · claim '+str(example['claim_id'])+'**')
                st.write(example['claim'])
                st.caption(str(example['passages'])+' exact passages · model-generated verdict · source-linked explanation · reproducible replay')
                st.button('Run from candidates',key='run:'+example['file'],on_click=start_showcase,args=(example['file'],))
                st.button('Inspect this run',key='showcase:'+example['file'],on_click=install_showcase,args=(example['file'],))
    st.subheader('Demonstration evidence for the rubric')
    tabs=st.tabs(['Efficiency','Reusability','Presentation','Five-person video'])
    with tabs[0]:
        timing=ROOT/'evaluation/efficiency-results.json'
        if timing.is_file():
            measured=json.loads(timing.read_text(encoding='utf-8'))
            cols=st.columns(3)
            cols[0].metric('Cached subset median',f"{measured['cached_subset_p50_ms']:.2f} ms")
            cols[1].metric('Extra NLI calls /100 changes',str(measured['cached_subset_extra_neural_forwards']))
            cols[2].metric('Replay load median',f"{measured['replay_p50_ms']:.2f} ms")
            st.caption(measured['condition'])
        st.write('Models load once per process. Selection changes reuse cached predictions; rendering/replay/benchmark inspection do not run neural inference. Background work is bounded, stale results are rejected, and token budgets reject whole spans.')
    with tabs[1]:
        st.write('One set of typed contracts connects discovery, ranking, selection and verification. The same selector/verifier adapter serves imported searches and local PDFs. Grounding, export/replay, job handling and scope summaries are reused by the UI, evaluation runners and tests.')
        st.code('snapshot = pipeline.run_from_discovery(source, paper_ids)\nsubset = pipeline.aggregate(snapshot, selected_ids)\ncontext = make_context(snapshot, selected_ids, subset)\nexplanation = build_template(context)\nexport_snapshot(context, explanation)',language='python')
    with tabs[2]:
        st.write('Run from candidates performs actual local inference. Inspect this run opens a saved actual-model result instantly. Keep the same claim through ranking, evidence, verdict, subset comparison and replay, then show independent domain-specific evaluation.')
    with tabs[3]:
        st.dataframe([{'Presenter':name,'Own the explanation of':scope} for name,scope in [
            ('Raghavi','Discovery, provider provenance and candidate handoff'),('Naila','BM25/SBERT retrieval and XAI methods'),
            ('Yu Yue','RRF, semantic evidence, token budget and benchmarks'),('Chong Yao','NLI, abstention and cached aggregation'),
            ('Kireeti','Grounded explanations, interactive comparison, replay and integrated QC')]],hide_index=True)
        st.caption('Recording script and shot checklist: DEMO_RUBRIC_GUIDE.md. Each person records their own segment; participation is not inferred from software checks.')
    with st.expander('Suggested demonstration'):
        st.write('Open a saved run, inspect its passages and citations, deselect a paper, export/reload it, then compare the three-domain benchmarks. For new work, use Find papers or upload a text PDF. Optional attribution may take minutes and is displayed only after its numerical checks pass.')
