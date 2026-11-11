# Ranking integration - current status, 6 November 2026

The built-in pipeline now computes Naila BM25/SBERT branch rankings automatically and invokes Yu Yue RRF. You do not need to exchange ranking files to run the app. Manual imports below remain useful for teammate handoffs and inspection. Naila is pinned at 2bc7d81292313498fd70dd13656024d00cf3dda4. See INTEGRATION_CHANGELOG.md for the schema bridge and local encoder policy.

1. Start the frontend using the existing README instructions. Load a recorded, uploaded or live search.
2. Expand **Connect BM25/SBERT rankings and Person 3 RRF**. Download **Person 2 ranking input** and send the file to the retrieval owner. The empty `bm25_ranking` and `sbert_ranking` lists are placeholders, not computed results.
3. The owner fills those lists with ordered, unique canonical `identity` values from `candidates.papers`, most relevant first. Preserve the entire candidates object and exact original claim. Branch times may remain null when unknown. Do not substitute alias IDs.
4. Upload that completed `RankingInput` and click **Connect ranking**. Alternatively upload Yu Yue's RRF-only `RankingOutput` for exactly the same candidates. The frontend validates output rows by replaying the supplied branch lists through the pinned runtime.
5. Inspect final rank, RRF score and both branch ranks. Only output papers are selectable; filters retain rank order. Connecting a ranking resets selection to its output papers. Restoring discovery order resets selection to all discovery candidates.
6. Download the connected output or search snapshot, or prepare selected papers for verification. The `discovery-selection-1.0` handoff contains `ranking_context` and full selected records in ranking order. A successful verification adapter also retains ranking inside `RunSnapshot.discovery_context`.

The six unchanged runtime files in `_upstream/ranking_evaluation` come from [Yu Yue's commit](https://github.com/YYU035/NLP_Project/commit/a91e257ab260ded8df8c2cc2f214c4c7fd16af6c). They import Raghavi's `paper_search` schemas, so use the existing `setup-discovery.ps1` dependency setup before connecting rankings. Offline template/demo verification still works without that optional package. No teammate directory is changed. Their ML dependency pins are not installed or substituted for the frontend environment.

Live search payloads require provider/query provenance. Yu Yue's synthetic and fixed-corpus payloads with empty `providers` are not accepted as live search imports. Use upstream's own evaluation tooling for those benchmarks; never invent provider diagnostics to pass validation.

## Remaining owner handoffs

- Person 2: connected to the actual runtime and local embeddings; frozen retrieval-quality evaluation remains pending.
- Yu Yue: RRF connected; optional cross-encoder now passes numerical QC here and improved the small development sample, but remains disabled pending broader quality/protocol review.
- Chong Yao: real local NLI and cached aggregation connected, with a documented frontend conflict policy. Team acceptance and broader validation remain pending.
- Passage owner: Person 3 owns this stage per the current task document and must approve/replace Kireeti's explicit abstract-span fallback. Full-text evidence is not implemented.
- Team: human scientific explanation evaluation, held-out benchmarks and full-text/status handling remain pending.

Ranking-stage timing excludes discovery, branch indexing, model loading and serialization. Naila query times exclude index construction. The pipeline separately records its own integration duration. These must not be labelled interchangeably as end-to-end timings.
