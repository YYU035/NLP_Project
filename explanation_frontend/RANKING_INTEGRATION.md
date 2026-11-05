# Ranking integration

Person 1 discovery → Person 2 branch rankings → Yu Yue RRF → Person 5 ranked browser and selected-paper handoff is connected. RRF runs locally; it makes no API calls and loads no model.

1. Start the frontend using the existing README instructions. Load a recorded, uploaded or live search.
2. Expand **Connect BM25/SBERT rankings and Person 3 RRF**. Download **Person 2 ranking input** and send the file to the retrieval owner. The empty `bm25_ranking` and `sbert_ranking` lists are placeholders, not computed results.
3. The owner fills those lists with ordered, unique canonical `identity` values from `candidates.papers`, most relevant first. Preserve the entire candidates object and exact original claim. Branch times may remain null when unknown. Do not substitute alias IDs.
4. Upload that completed `RankingInput` and click **Connect ranking**. Alternatively upload Yu Yue's RRF-only `RankingOutput` for exactly the same candidates. The frontend validates output rows by replaying the supplied branch lists through the pinned runtime.
5. Inspect final rank, RRF score and both branch ranks. Only output papers are selectable; filters retain rank order. Connecting a ranking resets selection to its output papers. Restoring discovery order resets selection to all discovery candidates.
6. Download the connected output or search snapshot, or prepare selected papers for verification. The `discovery-selection-1.0` handoff contains `ranking_context` and full selected records in ranking order. A successful verification adapter also retains ranking inside `RunSnapshot.discovery_context`.

The six unchanged runtime files in `_upstream/ranking_evaluation` come from [Yu Yue's commit](https://github.com/YYU035/NLP_Project/commit/a91e257ab260ded8df8c2cc2f214c4c7fd16af6c). They import Raghavi's `paper_search` schemas, so use the existing `setup-discovery.ps1` dependency setup before connecting rankings. Offline template/demo verification still works without that optional package. No teammate directory is changed. Their ML dependency pins are not installed or substituted for the frontend environment.

Live search payloads require provider/query provenance. Yu Yue's synthetic and fixed-corpus payloads with empty `providers` are not accepted as live search imports. Use upstream's own evaluation tooling for those benchmarks; never invent provider diagnostics to pass validation.

## Remaining owner handoffs

- Person 2 **[PENDING]**: real BM25/SBERT retrieval code and ordered branch results. This frontend does not manufacture those results from discovery order.
- Yu Yue **[CONNECTED]**: RRF schema/runtime, ranked metadata display, import/export, selection and downstream lineage. **[PENDING]**: validated optional cross-encoder environment; reranker outputs are explicitly rejected for now because the shared environment has a reported CPU NaN issue and different dependency pins.
- Person 4 **[AVAILABLE, NOT CONNECTED]**: `claim_verifier` on `chong_yao_claim_verification` at `1ac06b6` now publishes verify/aggregate callables. **[PENDING — Kireeti + Person 4]**: schema/status mapping, isolated imports, pinned local model setup and real conflict/failure/subset acceptance tests; then configure `EXPLANATION_VERIFY_DISCOVERY_FACTORY`. **[PENDING — evidence owner]**: real passage extraction remains missing. See the README adapter-readiness table. No verification verdict is produced by RRF.
- Team **[PENDING]**: confirm passage-selection ownership because the task split and new ranking README differ; run the full chain on real outputs and perform human explanation evaluation.

Ranking-stage timing excludes discovery, branch retrieval, model loading and serialization. Scores are retrieval relevance signals, not probabilities that a claim is true. Saved ranking data is validated structurally and against its branch lists, not authenticated as an owner's scientific output.
