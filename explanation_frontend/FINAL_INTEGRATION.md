# Full-pipeline validation — 10 November 2026

The full automated offline route completed all **300 SciFact dev claims** with **zero processing failures**, **56.7% claim accuracy** and **0.584 macro F1**. This is a frozen descriptive audit of previously inspected dev data, not an untouched test set. Retrieval is stronger than final verification: RRF Recall@10 is 95.6% on eligible SciFact claims, 39.2% on CSFCube test and 93.7% on RECIPER. These are different retrieval tasks and protocols.

Read the [complete test report](evaluation/validation/FINAL_TEST_REPORT.md), [per-condition outputs](evaluation/validation/results/test_results.csv), [error catalogue](evaluation/validation/results/error_catalogue.csv) and [machine-readable metrics](evaluation/validation/results/validation-summary.json). Representative real-model replays are in `evaluation/validation/results/replays`; the local review ZIP is `outputs/validation/review-kit.zip`. Open Project overview for report/error downloads. Run `validate.cmd` to reproduce checks and reuse completed frozen studies.

Clean setup: 164 frontend tests pass; all 452 pinned owner tests pass. Seven PDF cases, 30 retrieval repeat checks and five repeated NLI pairs pass. Synthetic scientific-language expectations pass 12/15; population/generalisation errors remain. IG convergence is still unresolved. **Software passes do not imply high scientific accuracy.** Older release notes below describe their original narrower scopes.

# Final integration and demo release — 9 November 2026

The integrated app now opens on **Project overview**. It connects all five contributions through one set of contracts, provides actual-model runs and saved replays, exposes three-domain results and supplies measured evidence for the rubric. This does not guarantee a grade or scientific correctness.

## Start and demonstrate

Double-click `start.cmd`, or run it from a terminal. It launches the existing script with a process-only PowerShell policy override, without changing the machine's persistent policy. Open the printed localhost URL.

1. In Project overview, choose **Run from candidates** for claim 230. The candidate pool is explicitly labelled SciFact, with its known annotated paper and two seeded distractors.
2. Inspect candidates/provenance, prepare the selected papers, then verify. This performs actual local retrieval, fusion, semantic selection and NLI. Alternatively **Inspect this run** opens an actual-model saved replay instantly.
3. Inspect the source quotation and model-label counts. Change paper selection and inspect the scoped result. Download/reload the JSON.
4. Open Model attribution. Word occlusion is the default: whole words are masked and the target probability is measured again. The decisive passage is selected first. For claim 230, **Load recorded word sensitivity** opens a checksum-validated computation without neural inference, explicitly labelled saved. Fresh computation scored all 37 words in about 5.25 seconds in the recorded local run.
5. Compare the three-domain benchmark charts and protocols. Open the rubric tabs for measured efficiency and the reusable API map.
6. Load the actual-model explanation review set. Eight real-output packets and a blank review CSV are prepared; zero human ratings are invented.

The five-person timed script, role handoffs, exact UI actions and recording checklist are in [DEMO_RUBRIC_GUIDE.md](DEMO_RUBRIC_GUIDE.md).

## Integration and verification by owner

| Contribution | Connected and exercised | Evidence |
| --- | --- | --- |
| Raghavi | Public/recorded discovery, canonical identities, provider/source records and exact-claim handoff. |268 original owner tests passed. Recorded real32-candidate run passed through3-paper/9-passage NLI, grounding and replay. Fresh search also passed the remaining pipeline, correctly marked partial after an OpenAlex429. |
| Naila | BM25/SBERT and local embeddings feed actual RRF. Original IG remains available as experimental. |13 original retrieval tests passed; original12 tiny-model IG tests are part of frontend tests. A frontend stable-cutoff tie fix is separately documented in retrieval/SOURCE.json and tested. |
| Yu Yue | Pinned RRF/semantic/PDF/budget runtime; latest comprehensive dataset artifacts at90aaca1. |166 current original owner tests passed offline. CSFCube26-task test is distinct from24-task dev. Ablations, uncertainty, evidence diagnostics and latency are integrated. |
| Chong Yao | Actual pinned local NLI, typed mappings, conflict-preserving integration, subset cache and PDF verification. |5 original model tests passed with explicitly offline initialization. Full-abstract audit has181 nontruncated annotated pairs,28 excluded; class-specific metrics are in evaluation/verifier-metrics.json. |
| Kireeti | Seven workspaces, grounded templates/citation validation, exact evidence, scope summaries, replay, actual-output review and presentation assets. |Eight seeded actual-model acceptance cases passed, each with three subset checks and replay. PDF page-preserving constructed test passed. Efficiency/model-call instrumentation and browser walkthrough passed. |

The frontend suite passes **156 tests**, and pip check passes. The owner suites total **452 tests** across separate isolated runs. Frontend tests also include source-level integration/plumbing tests, so these counts must not be advertised as independent scientific examples.

## Measured results suitable for the presentation

- Published RRF Recall@10: **SciFact 95.9%**, **CSFCube test 39.2%**, **RECIPER 93.7%**, each under its own task/candidate universe and denominator. No pooled three-domain accuracy. Owner experiments are source-pinned, validated artifacts; full scientific benchmark reruns were not independently repeated by this frontend.
- Actual cache instrumentation: first case 230 verification used3 neural forwards;100 subsequent subset changes and20 replay loads used **zero additional neural forwards**. Observed subset median 1.14 ms/p95 1.33 ms; replay median 3.30 ms. Conditions and limitations are in evaluation/efficiency-results.json.
- Four-case word-occlusion pilot: sentence AP0.833 versus random0.438, top-sentence hit75% versus17.5%; comprehensiveness0.661 versus0.112 (higher desired), sufficiency0.135 versus0.564 (lower desired). These are gold-target, short-dev, exploratory word-deletion diagnostics, not IG, significance claims or broad scientific accuracy. Four pairs come from three documents; the incorrect case is retained and grouped separately.
- IG still failed the original four-case study, targeted512-step retry and new bounded adaptive study. A selected-passage adaptive diagnostic also failed the declared checks. No thresholds were relaxed; failed scores stay withheld. Adaptive IG is a separately documented research CLI option, not the normal UI path.

## Reuse and readability

Production logic is independent of Streamlit where possible: PipelineProvider contracts, LocalPipeline dependency injection, Person4Adapter, source/ranking schemas, template/grounding/export APIs, scope counts and background-job helpers. The same APIs are exercised by the UI, acceptance runner, PDF bridge and tests. The candidate installer is shared by live/imported/discovery/demo workflows. Benchmark integrity loading is shared by overview and results pages.

The owned pinned retrieval copy includes a small integration correction: selecting top-k now deterministically includes the lowest-index ties at the cutoff, validates finite scores and retains O(n + k log k) selection. The original owner branch is unchanged. Other pinned teammate files are not silently rewritten.

## Additions ledger

- Project overview and guided actual-model gallery, with distinct fresh-computation and saved-replay paths; eight public-text exports plus checksums.
- Latest comprehensive benchmark snapshot, grouped method charts, RRF-versus-BM25 deltas and protocol/uncertainty/ablation/latency downloads.
- Evidence-balance cards; shared scope-count and discovery-state helpers; tested state restoration.
- Word-level MASK occlusion, bounded batching/coverage, finite checks, precise offsets, coloured numeric tables and explicitly saved exact-input results.
- Experimental adaptive Captum integration, parity checks and unchanged strict gates; negative outcomes preserved.
- Actual-output reviewer packets and blank CSV; no fabricated scores.
- Efficiency, release acceptance, occlusion and model-metric reports/runners; local text-PDF and live/recorded search tests.
- Process-only launcher, viewer toolbar and consistent20MiB PDF upload ceiling; replay uploads retain their own10MB guard.
- Rubric recording script and five-person participation checklist.

All source changes remain under explanation_frontend on kireeti. No paid/hosted inference or proprietary API dependency was introduced. Models, corpora, caches and raw research outputs remain ignored; small pinned reports and demonstration exports are tracked.

## Human work still required

Record each member's own segment, review/approve the assembled video, and perform the explanation/PDF scientific reviews. The software can prepare, validate and demonstrate these artifacts; it cannot truthfully supply participation or human ratings. Broader calibration/XAI and source-authenticity studies remain scientific limitations rather than hidden “completed” tasks.

The reproducible `test.cmd` wrapper creates the ignored temporary parent needed on a clean checkout before running pytest and pip check. An initial Desktop verification exposed this missing-directory test-command issue; the wrapper fixes it without changing application behaviour.
