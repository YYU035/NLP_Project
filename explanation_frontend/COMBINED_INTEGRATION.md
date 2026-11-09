# Combined team integration - 9 November 2026

All changes are in explanation_frontend on kireeti. Main and teammate folders/branches were not overwritten. A blanket merge of yuyue would change other owners' folders, so this integration uses pinned runtime copies and published result snapshots inside the permitted folder.

## What is connected

| Workspace / stage | Implementation and source |
| --- | --- |
| Find papers / Live verification | Existing Raghavi discovery, Naila BM25/SBERT and Yu Yue RRF; latest RRF/reranker runtime pinned at 7ed93fa4be69c661422b33b6c88b949dd2636e12. Original claims, canonical IDs and provenance remain intact. |
| Evidence selection | Yu Yue's actual semantic selector and verifier-budget code from the same commit, replacing the temporary lexical selector as the default. Train-frozen settings: top 3, cosine threshold 0.5, context radius 0, MiniLM revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Whole claim/evidence pairs are checked against the actual 512-token NLI tokenizer before inference. |
| Verification and aggregation | Existing pinned Chong Yao classifier and cached native responses. A successful selector abstention is labelled NO_PASSAGE_ABOVE_THRESHOLD, not missing-text processing failure. Missing text and input-budget failures remain operational; no scientific verdict is fabricated. Conflict-side preservation and grounding gates remain. |
| Local PDF | Yu Yue's local PDF extractor and full-text selector, using the same local model and pair budget. One uploaded PDF is verified with exact page/character references and source hashes. No arbitrary URL fetching, OCR or visual table interpretation. |
| Model attribution | Naila's Integrated Gradients code at 56f46c12d8ef1cc36202f046b4bbaa7825fc6ade, packaged with relative imports. Opt-in, local CPU and frozen run identity. Model argmax attribution is distinct from calibrated verdict, factual support and citation correctness. |
| Benchmark results | Offline, checksum-checked snapshots from Yu Yue's evaluation catalog, CSV summaries, provenance and reports for SciFact, CSFCube and RECIPER. No raw corpus or model download is needed to view them. |
| Explanation review | Existing grounded templates, experimental Qwen generation and explicit human review workspace. Invalid generation is still withheld. No human scores are auto-filled. |

## Usage

On this machine, run explanation_frontend/run.ps1. On a fresh machine:

```powershell
.\explanation_frontend\setup.ps1
.\explanation_frontend\setup-pipeline.ps1
.\explanation_frontend\run.ps1
```

setup-pipeline explicitly installs the pinned local PDF/Captum dependencies, NLI weights and exact Person 3 semantic model. -SkipExtras leaves optional PDF/attribution dependencies uninstalled. No installation/download happens while displaying results or verifying a claim. All weights/caches remain ignored inside this folder.

The old lexical selector remains available to trusted Python callers through LocalPipeline(selector_mode='legacy') for regression/research comparison. Production factory defaults to the actual semantic selector. Missing new-model setup fails visibly rather than silently falling back to a different selector.

Normal live queries still use public catalogue APIs. The three benchmark corpora are fixed evaluation data, not a router restricting live queries to those domains.

## Dataset reporting boundaries

- **SciFact:** 300 dev claims, 188 ranking-eligible, 112 without positive paper gold, 5,183 corpus papers. Paper-ranking and sentence/rationale metrics are separate from verifier accuracy. Multiple reranker depths/constants retain distinct row keys.
- **CSFCube:** 24 fold1-dev query/facet tasks over released judged candidate pools; 4,207 collection records. Expert paper grades, binary-positive threshold >=2 and graded nDCG have separate meanings. This is not unrestricted full-corpus search or sentence-evidence gold.
- **RECIPER:** 1,024 generated questions against all 343 source papers, titles/abstracts only. Recall measures recovery of the single known source paper, not all scientifically relevant papers. This differs from original paragraph/recipe retrieval and supplies no expert sentence/verdict gold.
- Materials Science is the third domain; the app does not label it Physics. No overall accuracy is averaged across incompatible tasks. Published teammate experiments are labelled as such, not independently reproduced by this UI integration.

All referenced artifacts are under benchmarks/, preserving their original relative paths and owner manifests. SOURCE.json records the exact revision and file hashes. The reader refuses modified assets and path escapes. CSV/JSON downloads do not launch experiments.

For independent reruns without changing teammate folders, evaluation.team_qc yuyue_current stages pinned ranking/shared/retrieval sources under .tmp/team_qc/yuyue_current. Run the documented owner commands from that isolated workspace with the module Python and explicit module-local data/model/output locations. Long benchmark jobs are intentionally not launched by the reporting page.

## PDF and source constraints

Only explicit local uploads, up to 20 MiB, 200 pages and 2 million extracted characters. Filenames are replaced by file hashes in storage/exports; arbitrary filesystem paths and remote URLs are not accepted. Page offsets refer to exact extracted page strings, not PDF byte positions. Scanned/encrypted/empty documents fail explicitly. File/layout safety is bounded local capstone handling, not a hardened document sandbox.

Full-text evidence preserves page numbers and source SHA in both verification and primary evidence. Export/replay carries full exact selected passages and source offsets; source authenticity, PDF reading order and scientific interpretation require human review. Known early PDF replays can gain only the canonical missing PDF-review note, with snapshot hashes, claims and evidence unchanged.

## Attribution quality and failure handling

Captum 0.9.0 is installed without hosted/paid optional extras. The wrapper shares the supported pinned NLI model under the same lock as verification, explains one frozen passage and discards stale/late results. It starts at 32 integration steps and retries once at 128. Nonfinite output is rejected; relative completeness error above 5% withholds the normal word-score display, with diagnostic download available.

The real NLI/PDF smoke case had approximately 38.1% relative completeness error even at 128 steps. **That case does not validate reliable token attribution.** It is explicitly flagged as unstable; neither attribution quality nor scientific correctness is inferred from the 12 passing tiny-model plumbing tests. The default claim explanation remains the grounded template. No source HTML is executed to render word scores.

## Validation performed

- Combined frontend suite: **136 passed**, including Naila's 12 tiny-model attribution tests, result integrity/path checks, dataset UI, semantic abstention, budget rejection, stale-state/legacy replay and prior regressions.
- Current Yu Yue suite: **163 passed**, external network blocked in the isolated source tests; this is a fresh run against 7ed93fa.
- Actual pretrained-model recorded search: 3 papers / 9 bounded semantic passages; grounded explanation and replay passed.
- Actual fresh public search: 9 candidates, 3 verified papers / 9 bounded semantic passages; grounding and replay passed; one cached subset about 3.40 ms in this single run. Not a latency benchmark.
- Synthetic text PDF, actual local selector/NLI: extracted exact page evidence, produced a supported model verdict and exported/replayed successfully. This is a constructed plumbing check, not a real scientific PDF quality study.
- Real NLI attribution returned finite values but failed the completeness gate, as disclosed above.
- pip check passed; no hosted/proprietary inference or paid API dependency introduced.

Raw reports, PDF, replay and attribution diagnostics remain ignored under outputs/integration. Owner test logs are under outputs/team_qc. The previous TEAM_QC_REPORT.md remains a dated 6 November baseline; its missing-selector finding is superseded by this integration. Scientific quality limitations, untouched-test protocol, human PDF/explanation reviews and NLI calibration work remain pending.

## Additions and packaging changes

- benchmark_view.py and benchmarks/: lightweight three-dataset reporting and source integrity.
- _upstream/person3/: unchanged evidence_selection, semantic_evidence, verifier_budget and pdf_evidence files plus the train-frozen configuration.
- semantic_setup.py / semantic_integration.py: explicit safe-weight provisioning, manifest checks and original-span/native-verifier handoff.
- _upstream/ranking_evaluation/: refreshed unchanged runtime and source manifest at 7ed93fa.
- _upstream/xai/: Naila explain_ig and eval_ig_dev with package-relative imports; no model logic rewritten.
- pdf_view.py: local PDF input, bounded background work, page-aware verification and replay.
- attribution_view.py: bounded optional gradient work, finite/completeness checks and text-safe display.
- person4.py: explicit successful selector-abstention mapping; native inference is not fabricated or repeated.
- app.py / verification_view.py / templates.py / export.py: new workspaces, complete budget diagnostics, source-review limits and strict known-note replay migration.
- setup-pipeline.ps1 and optional requirement files: reproducible local dependencies/model setup; all caches stay in the owned folder.
- tests/test_combined_integration.py and packaged Naila tests: contracts, reporting, UI and failure gates. Historical lexical tests explicitly retain legacy mode instead of pretending they exercise semantic selection.
