"""Report all frozen internal holdout results; never tune the candidate on them."""
import json,csv,hashlib,shutil,zipfile,math
from pathlib import Path
from sklearn.metrics import accuracy_score,f1_score
from .independent_audit import ROOT,OUT,load
from .validation.metrics import classification,evidence_score,bootstrap
RESULTS=ROOT/'evaluation/independent_results_20261111'

def citation_clusters(claims):
    parent={str(c['id']):str(c['id']) for c in claims};owner={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for c in claims:
        cid=str(c['id'])
        for pid in {str(p) for p in [*c.get('evidence',{}),*c.get('cited_doc_ids',[])]}:
            if pid in owner:parent[find(cid)]=find(owner[pid])
            else:owner[pid]=cid
    return {cid:find(cid) for cid in parent}

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])

def main():
    RESULTS.mkdir(exist_ok=True);protocol=load(OUT/'PROTOCOL.json');cohort=load(OUT/'cohort.json');rows=[];work=[]
    assert hashlib.sha256((ROOT/'evaluation/independent_audit.py').read_bytes()).hexdigest()==protocol['source_sha256']
    for shard in range(3):
        work.append(load(OUT/f'shard-{shard}/complete.json'));rr=load(OUT/f'shard-{shard}/rows.json');assert len(rr)==work[-1]['rows'];rows+=rr
    assert len(rows)==2*protocol['n'] and len({(r['claim_id'],r['condition']) for r in rows})==len(rows)
    train=list(map(json.loads,(ROOT/'outputs/context_study_20261111/claims_train.jsonl').read_text(encoding='utf-8').splitlines()))
    dev=list(map(json.loads,(ROOT/'outputs/xai/data/claims_dev.jsonl').read_text(encoding='utf-8').splitlines()))
    used=set(load(ROOT/'evaluation/context_study/TRAIN_FROZEN.json')['train_ids'])|{str(c['id']) for c in train[:200]}
    seen=[c for c in train if str(c['id']) in used]+dev
    seen_text={c['claim'].strip().casefold() for c in seen}
    seen_papers={str(p) for c in seen for p in [*c.get('evidence',{}),*c.get('cited_doc_ids',[])]}
    assert all(str(c['id']) not in used and c['claim'].strip().casefold() not in seen_text and not {str(p) for p in [*c.get('evidence',{}),*c.get('cited_doc_ids',[])]}&seen_papers for c in cohort)
    expected=set(protocol['claim_ids']);groups={name:{r['claim_id']:r for r in rows if r['condition']==name} for name in ['baseline','yuyue_frozen']}
    assert all(set(g)==expected for g in groups.values());clusters=citation_clusters(cohort)
    summaries={}
    for name,g in groups.items():
        rr=list(g.values());m=classification(rr);positive=[r for r in rr if r['gold']!='INSUFFICIENT']
        assert abs(m['accuracy']-accuracy_score([r['gold'] for r in rr],[r['prediction'] for r in rr]))<1e-12
        assert abs(m['macro_f1']-f1_score([r['gold'] for r in rr],[r['prediction'] for r in rr],labels=['SUPPORTED','CONTRADICTED','INSUFFICIENT'],average='macro',zero_division=0))<1e-12
        summaries[name]={'all':m,'evidence':evidence_score(*(sum(r['evidence'][k] for r in positive) for k in ['tp','fp','fn']))}
    ids=list(groups['baseline']);base=groups['baseline'];candidate=groups['yuyue_frozen']
    delta=[int(candidate[c]['correct'])-int(base[c]['correct']) for c in ids];paired=bootstrap(delta,[clusters[c] for c in ids]);claim_paired=bootstrap(delta)
    transitions={'fixes':sum(not base[c]['correct'] and candidate[c]['correct'] for c in ids),'regressions':sum(base[c]['correct'] and not candidate[c]['correct'] for c in ids)}
    finite=0
    for shard in range(3):
        for p in (OUT/f'shard-{shard}/nli').glob('*.json'):
            data=load(p);probs=data['raw_probabilities'];assert len(probs)==3 and all(math.isfinite(v) and 0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5
            assert not data['truncated'] and data['operational_status']=='OK';finite+=1
        for p in (OUT/f'shard-{shard}').glob('*-trace.json'):
            data=load(p);assert all(c['accepted'] and c['pair_tokens']<=512 for c in data['budget'])
    failures=sum(r['status']!='ok' for r in rows)
    passed=paired['low']>0 and summaries['yuyue_frozen']['all']['macro_f1']>summaries['baseline']['all']['macro_f1'] and failures==0
    report={'date':'2026-11-11','protocol':protocol,'metrics':summaries,'paired_accuracy_delta_cluster_ci':paired,'paired_accuracy_delta_claim_ci':claim_paired,
        'citation_clusters':len(set(clusters.values())),'transitions':transitions,'operational_failures':failures,'finite_nli_predictions':finite,'work':work,
        'promotion_criteria_met':passed,'decision':'Eligible for primary abstract-provider promotion under the predeclared overall-accuracy rule; class tradeoffs remain explicit.' if passed else 'Retain baseline default; predeclared improvement criteria not established.',
        'scope':'Internal unused/cited-paper-disjoint published-train audit, not official hidden-test performance or independent external-domain validation. No guarantee about all historic teammate/model-pretraining exposure.'}
    if (OUT/'PROVENANCE.json').exists():shutil.copy2(OUT/'PROVENANCE.json',RESULTS/'PROVENANCE.json')
    (RESULTS/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');shutil.copy2(OUT/'PROTOCOL.json',RESULTS/'PROTOCOL.json')
    write_csv(RESULTS/'per_claim.csv',rows)
    changes=[{'claim_id':c,'claim':base[c]['claim'],'gold':base[c]['gold'],'baseline':base[c]['prediction'],'candidate':candidate[c]['prediction'],'correct_before':base[c]['correct'],'correct_after':candidate[c]['correct']} for c in ids if base[c]['prediction']!=candidate[c]['prediction']]
    write_csv(RESULTS/'changed_claims.csv',changes)
    b=summaries['baseline'];a=summaries['yuyue_frozen'];lines=['# Frozen-candidate internal holdout audit — 11 November 2026','',
      f"**{protocol['n']} previously unused claims**, with cited/evidence paper IDs disjoint from the logged selection-training and inspected development claims. All eligible claims were retained. This is an internal audit from the published SciFact training release, not an official hidden test or a new external dataset.",'',
      '## Frozen comparison','',
      'Same full5,183-paper corpus, production MiniLM retrieval, Porter BM25, RRF60/top100 branches/top5 papers, pretrained NLI weights and0.5 NLI threshold. Candidate: exact bounded abstract context plus frozen0.65 paper gate. No settings chosen or changed from these results.','',
      '| Metric | Original baseline | Yu Yue frozen candidate |','|---|---:|---:|',
      f"| Accuracy | {b['all']['accuracy']:.1%} | {a['all']['accuracy']:.1%} |",f"| Macro F1 | {b['all']['macro_f1']:.3f} | {a['all']['macro_f1']:.3f} |"]
    for label in ['SUPPORTED','CONTRADICTED','INSUFFICIENT']:
        for metric in ['precision','recall']:lines.append(f"| {label} {metric} | {b['all']['per_class'][label][metric]:.1%} | {a['all']['per_class'][label][metric]:.1%} |")
    lines += [f"| Input evidence F1 | {b['evidence']['f1']:.3f} | {a['evidence']['f1']:.3f} |",'',
      f"Paired accuracy change: {paired['estimate']:+.1%};95% cited-paper-cluster bootstrap interval [{paired['low']:+.1%}, {paired['high']:+.1%}], {report['citation_clusters']} connected citation clusters. Claim-level interval is retained separately; clustering is the primary assessment to reflect related claims.",'',
      f"Fixes:{transitions['fixes']}; regressions:{transitions['regressions']}; operational failures:{failures}. All{len(rows)} condition rows are retained. All{finite} cached NLI predictions have finite valid class probabilities, no truncation and operational status OK. Exact source/budget checks and grounded template gates ran during inference.",'',
      '## Decision','',report['decision'],'',
      'The comparison assesses the abstract verification route. Public-web recall, human explanation usefulness, PDF extraction order and external-domain verdict accuracy are separate questions. CSFCube/RECIPER retrieval scores do not establish verdict accuracy. The no-positive label is closed-benchmark INSUFFICIENT, not a universal claim about literature.','',
      '## Reproducibility and limitations','',
      'The frozen cohort excludes prior SHA-selected200 training IDs, conservatively also the first200 raw training IDs, every inspected dev claim text (normalized), and all cited/evidence paper IDs of those claims. It does not prove no past teammate ever inspected any remaining published training record; pretrained-model exposure cannot be ruled out. Related uncited scientific statements can remain.','',
      'Protocol, complete per-claim outcomes and every changed verdict are in this folder. Local exact model-input traces and real-model replay examples are under outputs/independent_audit_20261111. Models/corpora remain local and ignored. No gold papers were injected into retrieval.','',
      'Reproduce ranking/cohort once with `python -m explanation_frontend.evaluation.independent_audit prepare`; the protocol refuses overwrite. Run shards0/1/2 with `... independent_audit run --shard N`, then `python -m explanation_frontend.evaluation.independent_report`. Existing exact-input model predictions can be reused after interruption. No threshold search is allowed on this cohort.']
    activation=RESULTS/'ACTIVATION.json'
    if activation.exists():
        deployed=load(activation)
        lines+=['','## Application activation','',
            'The primary abstract provider is now enabled for new Live/Selected search jobs and the default pipeline_smoke command. PDF selection and direct LocalPipeline baseline behaviour are unchanged. Older replays retain their original metadata/results.',
            '',f"Software suite: {deployed['software_tests']['passed']} passed; dependency check passed. Actual-model subset test: {deployed['real_model_cache_check']['subset_changes']} changes with {deployed['real_model_cache_check']['extra_neural_forwards']} extra neural forwards; replay roundtrip passed. Fresh live primary smoke status: {deployed['live_primary_smoke']['status']}; its claim is not independently adjudicated.",'',
            'Use EXPLANATION_ABSTRACT_POLICY=baseline for the original app provider; pipeline_smoke --baseline retains the original CLI route. Primary exports include frozen context configuration and source commit.']
    for a,b in {'full5,183':'full 5,183','RRF60/top100':'RRF 60/top 100','/top5':'/top 5','and0.5':'and 0.5','frozen0.65':'frozen 0.65',';95%':'; 95%','Fixes:':'Fixes: ','; regressions:':'; regressions: ','; operational failures:':'; operational failures: ','All418':'All 418','All2010':'All 2010','SHA-selected200':'SHA-selected 200','first200':'first 200','shards0/1/2':'shards 0/1/2'}.items():lines=[line.replace(a,b) for line in lines]
    (RESULTS/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    replay=RESULTS/'replays';replay.mkdir(exist_ok=True)
    for shard in range(3):
        for p in (OUT/f'shard-{shard}').glob('*-replay.json'):shutil.copy2(p,replay/p.name)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.rglob('*') if p.is_file() and p.name!='ARTIFACTS.json'}
    (RESULTS/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'review-kit.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in RESULTS.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'accuracy':{n:m['all']['accuracy'] for n,m in summaries.items()},'paired_cluster_ci':paired,'criteria_met':passed,'transitions':transitions}),flush=True)
if __name__=='__main__':main()
