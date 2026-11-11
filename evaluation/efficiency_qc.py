"""Instrument actual neural-forward reuse rather than inferring it from fast timing."""
import json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    from research_verifier.pipeline import LocalPipeline, local_verifier
    from research_verifier.explanations.replay import import_snapshot
    from research_verifier.discovery.service import DiscoverySnapshot
    bundle=import_snapshot((ROOT/'examples/replays/scifact-230.json').read_bytes())
    data=dict(bundle.snapshot.discovery_context);data['ranking_context']=None
    source=DiscoverySnapshot.model_validate(data)
    v=local_verifier();forwards=[0]
    def count(*args):forwards[0]+=1
    hook=v.model.register_forward_pre_hook(count)
    try:
        engine=LocalPipeline(verifier=v);t=time.perf_counter()
        s=engine.run_from_discovery(source,list(bundle.snapshot.papers));verify_ms=(time.perf_counter()-t)*1000
        verify_forwards=forwards[0]
        if verify_forwards<=0:raise RuntimeError('Expected actual neural inference')
        before=forwards[0];times=[]
        for index in range(100):
            ids=list(s.papers)[:1] if index%2==0 else list(s.papers)[1:]
            t=time.perf_counter();engine.aggregate(s,ids);times.append((time.perf_counter()-t)*1000)
        additional=forwards[0]-before
        if additional!=0:raise RuntimeError('Subset aggregation unexpectedly reran neural inference')
        replay_times=[];raw=(ROOT/'examples/replays/scifact-230.json').read_bytes()
        for _ in range(20):
            t=time.perf_counter();import_snapshot(raw);replay_times.append((time.perf_counter()-t)*1000)
        report={'date':'2026-11-09','condition':'Actual local CPU models already loaded; case230 fixed3-paper input. Timings are local observations under concurrent research jobs, not a controlled SLA.',
            'first_verification_ms':verify_ms,'first_verification_neural_forwards':verify_forwards,
            'cached_subset_runs':100,'cached_subset_extra_neural_forwards':additional,
            'cached_subset_p50_ms':statistics.median(times),'cached_subset_p95_ms':sorted(times)[94],
            'replay_loads':20,'replay_p50_ms':statistics.median(replay_times),'replay_p95_ms':sorted(replay_times)[18],
            'replay_extra_neural_forwards':forwards[0]-verify_forwards,
            'limits':'Verification and subset aggregation are different workloads. No speedup factor, production throughput or model accuracy is inferred.'}
        (ROOT/'evaluation/efficiency-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
        return report
    finally:hook.remove()
if __name__=='__main__':main()
