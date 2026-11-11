# End-to-end verification - 6 November 2026

## Automated validation

**113 tests passed**, no skips reported, on the module Python 3.12 environment. pip check reports no broken requirements. Tests include the default Streamlit factory workflow, actual owner cached aggregator on synthetic data, duplicate/foreign IDs, operational failures, partial discovery, source-span offsets and negation, internal/across-paper conflicts, conflict cap preserving both sides, cached subset inference avoidance and export/replay lineage.

The injected classifier and embedding doubles in orchestration tests are synthetic. They establish integration contracts, not ML performance. Separate real-model smoke runs below exercise the downloaded local weights.

## Real local-model smoke runs

Both used real local MiniLM and MIT DeBERTa weights, CPU, threshold 0.50, max length 512 and immutable model revision b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7. HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 were set for inference. The fresh-search run explicitly called anonymous public discovery; the recorded run made no discovery requests.

| Check | Recorded real discovery | Fresh public discovery |
| --- | --- | --- |
| Retrieved candidates | 32 cached | 9 live |
| Selected papers | 3 | 3 |
| Selected exact passages | 9 | 9 |
| Grounding/display gate | Passed | Passed |
| Export/replay fingerprint | Matched | Matched |
| Total wall time, including cold local model load | 28.20 s | 31.44 s, including discovery |
| First single-paper cached subset aggregation | 49.98 ms | 10.43 ms |
| Additional neural inference for subset | None | None |

These single observations are not latency percentiles or quality/calibration benchmarks. Reports retain original claims, model outputs and provenance in ignored outputs/integration/recorded-report.json and live-report.json; replay artifacts are recorded-run.json and live-run.json. The recorded discovery source is Raghavi commit 259d2a9, live_search/artifacts/depth-approved-run/c01.json. The fresh-search claim was the generic cardiovascular-mortality/particulate-matter example in pipeline_smoke.py.

## Reproduce from repository root

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.pipeline_smoke
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.pipeline_smoke --live --limit 3
```

Run setup-pipeline.ps1 first on a fresh machine before setting offline flags. Live results and timings can vary; provider outages remain explicit. A grounded exact quotation is not proof of paper authenticity, relevance, causal interpretation or scientific truth. Human/held-out scientific evaluation and full-text verification remain pending.
