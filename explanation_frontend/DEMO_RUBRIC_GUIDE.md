# Rubric-ready demonstration and five-person recording guide

Use `start.cmd`, then select **Project overview**. All demonstrations below use public scientific text and actual local models. Saved runs are visibly identified as replays. The fixed-candidate examples deliberately include an annotated paper, so their success is an integration check, not an unbiased retrieval accuracy estimate.

## Evidence for each highest-band criterion

| Rubric front | What the demo proves | Inspectable evidence |
| --- | --- | --- |
| Efficiency without losing readability | One neural pass per evidence pair, cached subset aggregation, reusable loaded models, bounded jobs and token budgets. | Project overview > Efficiency; `evaluation/efficiency-results.json`; `evaluation/efficiency_qc.py` instruments the model rather than assuming no inference. |
| Reusability | Typed discovery/ranking/verification contracts; injectable engines; shared template/grounding/replay and async helpers; same verification adapter for search and PDF. | Project overview > Reusability; `providers.PipelineProvider`, `LocalPipeline`, `Person4Adapter`, `install_discovery_state`, `evidence_counts`, `build_template`, `validate_explanation`, export/import APIs. |
| Impressive, intuitive presentation | One-click actual-model workflow and instant replay; source quotations, evidence counts, scoped comparisons; separate three-domain charts and limitations. | Project overview, Verify evidence, Benchmark results, Model attribution; all example outcomes retained in `evaluation/release-acceptance.json`. |
| All members and good video flow | Each person explains their own boundary while carrying the same claim through the system. | Five-person script below; actual personal footage/voice must still be recorded. A script is not evidence that someone participated. |

Current measured efficiency:100 cached subset changes caused **zero extra NLI forwards**. Local observed median 1.14 ms,95th percentile1.33ms; replay median 3.30 ms. Timing conditions include concurrent research jobs and preloaded models. Verification and subset aggregation are different workloads; do not turn this into a misleading universal speedup or production throughput claim.

## One coherent story (about6 minutes40 seconds)

Primary example: overview > SUPPORTED example, SciFact claim 230. Keep the exact claim unchanged. Use **Run from candidates**, then **Prepare selected papers for verification** and **Verify claim**. This runs actual retrieval/ranking/selection/NLI locally. **Inspect this run** is a separate, explicitly cached route, useful as a fallback.

| Time | Presenter | Screen and action | Suggested explanation and handoff |
| --- | --- | --- | --- |
| 0:00–0:20 | Kireeti | Project overview; show the five-stage pipeline. | “Our system turns a scientific claim into inspectable evidence and a source-linked verdict. We will follow one claim through everyone's component.” |
| 0:20–1:25 | Raghavi | Run from candidates; Find papers, metadata, source records and selected identities. | “Our discovery contract accepts public searches and imported scientific records. For a repeatable demo we use this declared SciFact candidate pool, containing its annotated paper and two distractors. These candidates have not yet been verified.” Show original claim and provenance. Hand off: “Naila's retrieval compares these candidates against that exact claim.” |
| 1:25–2:25 | Naila | Prepare/verify; Run details > original ranking output/provenance, retrieval scores. | Explain lexical BM25 versus semantic SBERT and why raw scores are not comparable truth probabilities. Point to actual branch orderings. Hand off: “Yu Yue combines these ordered lists and selects the evidence the verifier is allowed to see.” |
| 2:25–3:40 | Yu Yue | RRF output; exact passage IDs and verifier-budget diagnostics. Briefly open Benchmark results. | Explain rank fusion, train-frozen threshold/top-k, exact source spans and rejecting oversized spans. Show three domains and CSFCube's separate test group. State that sentence evidence gold exists here only for SciFact; no invented evidence F1 for other datasets. Hand off: “Chong Yao now classifies the bounded claim–passage pairs.” |
| 3:40–4:50 | Chong Yao | Verify evidence > verdict, Papers & passages, model labels. Deselect the paper containing the supporting passage. | Explain SUPPORT/CONTRADICT/NEUTRAL, confidence threshold and aggregation. Distinguish insufficient evidence from a processing error. Show that a changed selection reuses predictions. State that model confidence is not scientific truth probability. Hand off: “Kireeti makes this decision, its scope and the exact supporting sources inspectable.” |
| 4:50–6:20 | Kireeti | Restore selection; inspect citation; export/replay; overview Efficiency; Model attribution Word occlusion; benchmark method comparison. | Show exact quotation-to-source link, scoped verdict and cached replay. Show zero extra model calls for100 subset changes. Word occlusion measures what changes when one word is masked; it is not IG and has no completeness guarantee. IG remains an experimental option whose failed numerical checks are disclosed. Highlight that RRF and reranking do not win uniformly across datasets. |
| 6:20–6:40 | All five, brief closing | Five faces/names or each person's recorded closing cut; overview. | One sentence each: discovery provenance; complementary retrieval; bounded evidence; cached verification; inspectable explanations. Close with measured strengths and limitations rather than a claim of perfect scientific accuracy. |

If a computation wait is edited out, label the cut. Do not present cached inference as fresh inference. Do not claim the full pipeline independently reproduced all owner benchmarks; this integration validates/presents pinned owner artifacts and independently exercises the connected software and model flows.

## Before pressing record

1. Launch `start.cmd`; use the owned `kireeti` checkout and installed local models.
2. Walk the primary example once. Inspect this run is an offline fallback. Keep the fixed-candidate scope visible.
3. Check the source quote and selected-paper controls. Export/reload one result. Load the actual-model review set and confirm no human scores have been filled in.
4. Open Benchmark results for all3 datasets; show26 CSFCube test tasks separately from24 development tasks. Keep grading pools/generated targets and188 SciFact ranking denominator visible.
5. Prefer recorded/imported candidates for the main take. A fresh catalogue request can be rate-limited; the latest live test correctly retained a partial-result warning when OpenAlex returned429. Do not hide that limitation or depend on an external API for the recording.
6. Use Word occlusion for the routine explanation segment. Do not wait for IG in the live recording or describe failed IG scores as reliable.
7. Record each member's own voice/face. Use the same terminology, claim and pointer style. Name the next person at each handoff.
8. Record at1080p with readable app text; turn off unrelated notifications, check microphone levels and keep the cursor still while explaining a result. Use existing recording software; no paid service is required.
9. Review the combined video for continuity: exact claim, candidate IDs, displayed source scope, selected papers, experiment splits and timing labels must remain consistent.

## Reuse demonstration (no GUI required)

The same production APIs power tests and evaluation runners:

```python
pipeline = LocalPipeline()  # encoder/verifier/selector/tokenizer can be injected
snapshot = pipeline.run_from_discovery(source, source.selected_paper_ids)
aggregate = pipeline.aggregate(snapshot, selected_ids)
context = make_context(snapshot, selected_ids, aggregate)
explanation = build_template(context)
assert validate_explanation(explanation, context).displayable
saved = export_snapshot(context, explanation)
restored = import_snapshot(saved)
```

`evaluation/release_acceptance.py` executes this against8 seeded real SciFact cases; all cases and outcomes are retained. `pipeline_smoke.py` separately exercises recorded real catalogue discovery and fresh public discovery. This is one connected implementation with source-pinned adapters, not five unrelated interfaces pasted together.

## Recording completion checklist

- [ ] Raghavi's actual segment recorded and reviewed.
- [ ] Naila's actual segment recorded and reviewed.
- [ ] Yu Yue's actual segment recorded and reviewed.
- [ ] Chong Yao's actual segment recorded and reviewed.
- [ ] Kireeti's actual segment recorded and reviewed.
- [ ] One continuous claim/story, clear handoffs and readable screen capture.
- [ ] No fabricated reviews, hidden failures, pooled incompatible metrics or unsupported guarantees.

Software, replay assets, measurements, reviewer packets and this guide are prepared. Real member participation, human review scores and final recording approval remain human actions; they are not automatically marked complete.
