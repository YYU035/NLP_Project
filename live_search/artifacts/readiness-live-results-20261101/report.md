# Live retrieval depth and arXiv fallback — 1 November 2026

**Recommendation: retain `top_k=30` per provider for Person 2 candidate generation, and keep arXiv fallback disabled until its 14 new candidates are assessed.** Depth 30 adds already-labelled potentially useful candidates, while fallback currently demonstrates broader coverage only. Neither experiment establishes improved accuracy or an optimal depth.

## Execution and evidence

Executed the two documented commands with the unchanged frozen plans and runner: [depth run](../depth-approved-run/plan.json), [fallback run](../fallback-approved-run/plan.json). Plans match the originals structurally. Ten original claims; dates **2020-01-01–2026-10-30**; current policy **OpenAlex/OpenAIRE → KeyBERT, arXiv → spaCy**. Experimental retries were zero, existing provider pacing remained active, and no production routing/default changed.

- **Depth: 30/30 permitted HTTP attempts**, ten/provider, all successful; no 429s or provider failures. Each provider fetched top 30 once/claim; 10/20/30 are nested response prefixes, not separate requests.
- **Fallback: 15/18 permitted HTTP attempts**, ten primary plus five conditional relaxations; all successful, no 429s/failures. No second relaxation was needed/available in these frozen cases.
- **All 45 response bodies are archived and SHA-256 verified** under each run's `responses/`. Per-claim `c01.json`…`c10.json` retain normalized records, exact request parameters, timings and outcomes. [query_diagnostics.csv](query_diagnostics.csv) collects all 45 logical queries and actual HTTP costs. [results.json](results.json) contains detailed summaries, label reconciliation and input/response hashes.

Reconstructed historical DOI/arXiv-ID/conservative-title groups before appending new records, preserving frozen numeric identities. Labels join on **(claim ID, canonical identity)**; the existing 336 nonconflicting provisional labels are reused, and the 12 historical conflicts remain unknown. Source labels/results are unchanged. These are AI-assisted provisional judgments, not validated ground truth. Unknown never means irrelevant.

## Retrieval-depth comparison

R = known label 1 or 2; E = known label 2, included in R; U = missing/conflicting label. Counts are distinct claim–paper pairs except returned-record totals. Metadata refers to this run's records; PDF means a URL, not verified content.

| Provider | Prefix | Returned / unique | R | E | U | Abstract/description pairs | PDF URL pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAlex | 10 | 100 / 100 | 93 | 67 | 3 | 86 | 75 |
| OpenAlex | 20 | 200 / 200 | 117 | 83 | 77 | 173 | 154 |
| OpenAlex | 30 | 300 / 300 | 133 | 93 | 161 | 254 | 234 |
| OpenAIRE | 10 | 81 / 81 | 64 | 39 | 10 | 81 | 6 |
| OpenAIRE | 20 | 131 / 131 | 67 | 41 | 57 | 131 | 9 |
| OpenAIRE | 30 | 171 / 171 | 73 | 46 | 91 | 170 | 11 |
| arXiv | 10 | 40 / 40 | 29 | 21 | 5 | 40 | 40 |
| arXiv | 20 | 80 / 80 | 31 | 22 | 43 | 80 | 80 |
| arXiv | 30 | 120 / 120 | 33 | 24 | 81 | 120 | 120 |
| **Cross-provider union** | **10** | **221 / 209** | **174** | **116** | **18** | **198** | **119** |
| **Cross-provider union** | **20** | **411 / 387** | **194** | **129** | **174** | **366** | **239** |
| **Cross-provider union** | **30** | **591 / 555** | **209** | **140** | **327** | **518** | **359** |

The top-10 set is entirely contained in top 20, which is contained in top 30. After cross-provider deduplication:

- **10 → 20:** +178 candidates, including **20 known R / 13 known E**, and 156 unknown.
- **20 → 30:** +168 candidates, including **15 known R / 11 known E**, and 153 unknown.
- **10 → 30:** +346 candidates, including **35 known R / 24 known E**. New-to-history papers have no invented labels; the known gains come from previously assessed papers retrieved by other historical arms.

The same ten requests/provider support all three prefixes: **zero additional requests between prefixes**. Top-30 provider-call latency sums were **19.65 s OpenAlex, 25.12 s OpenAIRE, 21.91 s arXiv** (includes pacing). Summed per-claim service latency was **27.96 s** with providers concurrent; this excludes process startup and analysis. HTTP-attempt duration sum was 26.18 s. These shared top-30 timings **do not measure independent top-10 or top-20 latency**, payload overhead or Person 2 ranking cost.

### Per-claim candidate counts

Each cell is **top-10 / top-20 / top-30 unique candidates**. Within each query, unique equals returned records in this run. `*` marks a prefix exactly at its limit, hence potentially capped; it does not mean the provider has only that many matches.

| Claim / topic | OpenAlex KeyBERT | OpenAIRE KeyBERT | arXiv spaCy |
|---|---|---|---|
| c01 TNF/COVID/older adults | 10*/20*/30* | 2/2/2 | 0/0/0 |
| c02 battery capacity | 10*/20*/30* | 10*/20*/30* | 10*/20*/30* |
| c03 SiC/silicon efficiency | 10*/20*/30* | 6/6/6 | 0/0/0 |
| c04 CO2/ocean acidification | 10*/20*/30* | 10*/10/10 | 0/0/0 |
| c05 dark matter/galaxy rotation | 10*/20*/30* | 10*/20*/30* | 10*/20*/30* |
| c06 irrigation/wheat | 10*/20*/30* | 10*/20*/30* | 0/0/0 |
| c07 vitamin D/infections | 10*/20*/30* | 10*/17/17 | 0/0/0 |
| c08 molecular prediction | 10*/20*/30* | 3/3/3 | 10*/20*/30* |
| c09 platinum/hydrogen evolution | 10*/20*/30* | 10*/13/13 | 0/0/0 |
| c10 exercise/memory | 10*/20*/30* | 10*/20*/30* | 10*/20*/30* |

OpenAlex/OpenAIRE have **0/10 zero-result claims** at every prefix. arXiv has **6/10**: c01, c03, c04, c06, c07, c09; increasing depth alone does not address these zero queries. OpenAIRE c01 and c08 remain below five candidates, with c03 at six. Short returned pools are observed; corpus coverage versus individual query constraints was not isolated.

Live variability check against the previous current-policy top-10 results: overlap is **100/100 OpenAlex, 81/81 OpenAIRE, 39/40 arXiv**. arXiv c10 shares nine of ten; this is result-order/response variation, not a new strategy effect. [historical_top10_overlap.csv](historical_top10_overlap.csv) records every comparison. Across providers, top-30 record count 591 becomes 555 pairs, removing 36 duplicate occurrences. Prefix counts must not be summed as independent retrieved collections.

## arXiv primary versus bounded relaxation

| Claim | Primary unique | Relaxation candidates | Primary + fallback unique | Extra requests | Extra call latency |
|---|---:|---:|---:|---:|---:|
| c01 | 0 | Not available | 0 | 0 | 0 |
| c02 | 10* | Not triggered | 10* | 0 | 0 |
| c03 | 0 | 10* | 10* | 1 | 6.21 s |
| c04 | 0 | 0 | 0 | 1 | 4.26 s |
| c05 | 10* | Not triggered | 10* | 0 | 0 |
| c06 | 0 | 2 | 2 | 1 | 7.99 s |
| c07 | 0 | 1 | 1 | 1 | 6.14 s |
| c08 | 10* | Not triggered | 10* | 0 | 0 |
| c09 | 0 | 1 | 1 | 1 | 2.35 s |
| c10 | 10* | Not triggered | 10* | 0 | 0 |
| **Total** | **40** | **14** | **54** | **5** | **26.96 s** |

Primary and relaxed result sets have **zero overlap** in these cases: relaxation was triggered only on empty primaries. The 54-pair union also fits the service's per-claim top-10 cap, so raw and retained union counts agree. All 54 have abstracts and PDF URL metadata. Known relevance remains **29 R / 21 E**; unknown increases **5 → 19**, because all **14 additional candidates lack labels**. Zero-result claims fall **6/10 → 2/10** (remaining c01/c04); c06/c07/c09 still have fewer than five candidates.

The frozen policy retained every selected concept token and the same dates; **no optional phrase was removed**. It relaxed exact phrase order, for example:

```text
c03 primary:  (all:"SiC MOSFETs" AND all:"silicon devices")
c03 fallback: (all:SiC AND all:MOSFETs AND all:silicon AND all:devices)
c06 primary:  (all:"deficit irrigation" AND all:"water use")
c06 fallback: (all:deficit AND all:irrigation AND all:water AND all:use)
Each appends: AND submittedDate:[202001010000 TO 202610302359]
```

c01's required concepts are already single atoms, so the frozen safe relaxation plan has no distinct alternative. c04 stayed empty even after removing phrase boundaries. Rescued cases demonstrate that phrase constraints affected candidate retrieval; they do not establish relevance or prove corpus absence for the remaining zeros. Outcomes/conditions omitted by the existing original concept extractor were not restored or silently changed.

Primary calls cost **10 requests / 50.44 s**; relaxation adds **5 requests / 26.96 s**, for **15 requests / 77.41 s summed service time**. Query-call sum is 77.39 s, with tiny service overhead. Timings include pacing and variable network/server delays, not just HTTP duration (69.75 s). Primary timing differs from the depth run and is not a controlled latency improvement comparison.

## Assessment exports, overlap and next decision

Across the two experiments: **645 occurrences / 569 unique claim–paper pairs**; depth/fallback overlap is **40**, counted once in the union. There are **209 known R / 140 known E**, 19 known label-0 pairs, and **341 unknown**. The extra fallback 14 are distinct from the entire depth-30 union as well as historical retrieval.

- [new_unresolved_for_assessment.csv](new_unresolved_for_assessment.csv): **341 pairs**, deduplicated across both runs: **318 new-to-history**, **15 existing unlabelled**, **8 conflicting existing labels**. All new review fields are blank; existing nonconflicting labels are never overwritten.
- [new_pairs_for_assessment.csv](new_pairs_for_assessment.csv): the **318 genuinely new pairs** only (**304 depth + 14 fallback**).
- [fallback_new_for_assessment.csv](fallback_new_for_assessment.csv): **14 fallback additions**, the smallest immediate review target for deciding whether to enable relaxation.
- [retrieved_pairs.csv](retrieved_pairs.csv): all **569 pairs**, including original claims, titles, full available abstracts/descriptions, canonical numeric identities, strong-ID aliases, existing-label status and provider/query/rank provenance. Numeric identities are tied to this frozen historical grouping, not universal paper IDs.
- [depth_per_claim.csv](depth_per_claim.csv): complete per-claim/provider/prefix counts, R/E/U, metadata and shared costs. Run archives contain the exact request syntax and dates; assessment CSVs do not conceal provenance, so hide those columns for blinded manual judging.

**Depth decision:** keep the existing handoff default of **30 per provider**, because it retains 24 additional already-labelled evidence candidates versus depth 10 without an additional page/request in these clients. Top 20 is an available candidate-budget compromise, but it would exclude 11 known evidence pairs present in ranks 21–30. Thirty is a defensible candidate-generation choice, **not a measured relevance optimum**; it grows the deduplicated pool from an average 20.9 to 55.5 papers/claim. Assess the unlabelled deeper candidates and measure Person 2's ranking costs before reducing or increasing that depth.

**Fallback decision:** keep it **opt-in/off by default**. First assess the 14 additions against their original claims; only then consider enabling this bounded mechanism. Reduced zero counts alone are insufficient evidence. No additional live calls are needed to assess these saved candidates.

Limitations: ten claims, provisional/uneven labels enriched around earlier top-10 lists, many unassessed deeper papers, provider-index/date semantics, potential result truncation and live ordering variability. OpenAIRE descriptions are not guaranteed abstracts; PDFs were not downloaded. No Precision/Recall, significance or accuracy improvement is claimed. The existing 47-paper validation sample and unresolved historical labels remain separate pending work.

Offline validation: **263 pytest tests passed; Ruff passed**. The new analysis helper has focused tests for conflict handling, claim-scoped deduplication counts and unknown labels. [execution_manifest.json](execution_manifest.json) records the exact commands, Python version, saved-run completion timestamps and source-code fingerprints. The runner and production routing were not modified. To reproduce the offline analysis without additional requests, use a fresh output directory:

```powershell
.venv\Scripts\python.exe scripts/analyze_readiness_runs.py --depth artifacts/depth-approved-run --fallback artifacts/fallback-approved-run --output artifacts/readiness-reanalysis
```
