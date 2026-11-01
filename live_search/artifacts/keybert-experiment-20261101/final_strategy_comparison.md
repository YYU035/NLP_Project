# Final query-strategy comparison with provisional relevance labels

**Recommendation: use both complementary concept queries, with KeyBERT first and spaCy retained for broader coverage; deduplicate identical queries and papers before Person 2 ranking.** If constrained to one query, KeyBERT is the stronger provisional choice on OpenAlex, but replacing spaCy everywhere would discard useful candidates. This is a practical integration decision from a small provisional dataset, not a validated accuracy claim.

## Evidence, identity and label reconciliation

- Retrieval is the saved [KeyBERT experiment](results.json): ten claims, dates **2020-01-01?2026-10-30**, top 10, **10 requests per method per provider** (40 total), counterbalanced order, no failures or retries. `calls.jsonl` uses `original` for the **spaCy concept query**, not a full-claim search; `alternative` means KeyBERT. The unchanged original claim anchors relevance assessment.
- Recomputed the existing DOI/arXiv-ID/conservative-title canonical groups from frozen historical records followed by fresh records. Matched labels by **(claim_id, canonical identity)** and verified every fresh CSV identity against saved result positions. Deduplication is within a claim; labels are not assigned by provider, query, rank or title alone.
- [Bulk 250-row labels](../arxiv-concepts-20261031/manual_assessment_provisional_250.csv): 234 distinct pairs, 226 labelled, eight blank. Duplicate occurrences have no conflicting bulk labels.
- [New 120-row labels](keybert_new_pairs_provisional_labels.csv): 120 distinct valid pairs, **all labelled** (9 irrelevant, 40 topical-only, 71 evidence). No duplicate IDs, missing export rows, or altered claim/title/abstract/provenance fields were found. No overlap/conflict with the bulk cohort: these are historically new identities.
- [30-example check](../gemini-relevance-20261031/manual_validation_30_filled.csv): 27 labels and three blanks. Opaque IDs were mapped through its frozen manifest. **Twelve labels conflict with the bulk cohort**, 13 agree, and two supply previously blank label-0 judgments. The field name `human_label` does not make this validated human ground truth.
- **Main tables accept existing labels only when all available assessments for that pair agree; conflicting pairs stay unresolved.** No source silently overrides another. All sources remain unchanged. The 94 earlier [OpenAIRE new-pair assessment rows](../openaire-experiment-20261101/new_pairs_for_assessment.csv) still have blank relevance fields; the preview is also unlabelled. Gemini judgments are blank too. Consequently OpenAIRE spaCy coverage remains heavily unassessed.
- Across historical and fresh collections there are 336 nonconflicting labelled pairs. Counts below are known provisional relevant/evidence yields. **Unknown includes missing and conflicting labels and is never counted as irrelevant.** These AI-assisted labels are not validated ground truth.

## Why 120 labelled rows versus 119 KeyBERT pairs?

The export covers **all newly retrieved pairs from both fresh methods**: 118 KeyBERT-only, one shared, and one spaCy-only. Thus **119 occur in KeyBERT**, while the full assessment legitimately has **120**. None was dropped.

- The spaCy-only row is **c04 / identity 377**, *Seaweed responses to ocean acidification: global impacts on growth, biochemical composition, and CO2 mitigation potential*, provisionally label **2**.
- The shared row is **c10 / identity 437**, *Contemporary rap narratives: Pedagogical exercise, memory and social representation*, label **0**. c10 has identical generated queries; its shared result is not a KeyBERT-specific gain.
- Of the 119 historically new KeyBERT pairs, **110 are provisionally relevant (1+2), including 70 evidence (2), and nine irrelevant**. The 118 KeyBERT-only new pairs include the same 110 relevant/70 evidence counts; the shared row is irrelevant.

## Equal-budget provider comparison

R = labels 1+2; E = label 2 (a subset of R); U = unresolved/unlabelled. Counts are unique claim?paper pairs except returned-record totals. Each row costs ten API requests. Unknowns prevent a complete relevance ranking, especially on OpenAIRE.

| Provider | Method | Total / unique | R | E | U (conflicts included) | Abstract/description records | PDF URL records | R / E pairs with PDF URL | API latency sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openalex | spacy | 100 / 100 | 90 | 47 | 6 (3) | 81 | 59 | 54 / 33 | 16.28 s |
| openalex | keybert | 100 / 100 | 93 | 67 | 3 (1) | 86 | 75 | 69 / 50 | 14.20 s |
| openaire | spacy | 100 / 100 | 17 | 10 | 83 (0) | 85 | 8 | 2 / 2 | 28.74 s |
| openaire | keybert | 81 / 81 | 64 | 39 | 10 (0) | 81 | 6 | 6 / 5 | 28.81 s |

Latency is summed client-call time including pacing, not local query-generation time or independent end-to-end trials. Local inference time and RAM were not benchmarked, so no CPU-speed comparison is asserted. OpenAIRE descriptions are not guaranteed abstracts; PDF availability is URL metadata, not verified downloads.

**OpenAlex:** KeyBERT has 67 known evidence candidates versus 47 for spaCy, with three versus six unresolved. Even all six unresolved spaCy candidates being label 2 would give 53; within this provisional cohort, unknowns alone cannot reverse that evidence-yield comparison. Topical yields (93 versus 90) are much closer. This is not proof of generalized superiority.

**OpenAIRE:** KeyBERT has 64 known relevant/39 evidence candidates versus spaCy?s 17/10, but **83 spaCy candidates remain unresolved** versus ten for KeyBERT. Do not interpret that difference as a complete relevance comparison. KeyBERT also returns fewer candidates (81 versus 100), with smaller biomedical/device/molecular pools.

## Complementarity and combined retrieval

Additional means unique to that method **relative to the other fresh method for the same claim/provider**, not merely new versus history. Unresolved exclusive candidates are shown separately.

| Provider | Pair overlap | spaCy-only R / E | spaCy-only U | KeyBERT-only R / E | KeyBERT-only U | Union total records / unique | Union R / E | Union U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openalex | 22 | 69 / 31 | 6 | 72 / 51 | 3 | 200 / 178 | 162 / 98 | 9 |
| openaire | 13 | 13 / 7 | 74 | 60 / 36 | 1 | 181 / 168 | 77 / 46 | 84 |
| combined | 33 | 71 / 32 | 79 | 125 / 80 | 3 | 381 / 324 | 218 / 129 | 92 |

Combining both providers with both strategies yields **324 unique pairs from 381 records**, with **218 known relevant and 129 evidence**, plus 92 unresolved. Relative to KeyBERT alone across providers, spaCy contributes **71 additional known relevant pairs, including 32 evidence**. Relative to spaCy alone, KeyBERT contributes **125 additional known relevant pairs, including 80 evidence**. These combined counts deduplicate provider overlap; do not add the two provider rows naively.

**The combination has meaningful provisional complementarity:** 31 additional OpenAlex evidence papers from spaCy are not in its KeyBERT results, while KeyBERT adds 51 not in spaCy. This supports keeping both rather than treating either list as redundant. The full union costs **40 observed requests**, versus **20** for either method across both providers, and supplies more candidates for ranking. No equal-cost superiority of the union is claimed.

## Per-claim retrieval and known relevance

T/U = returned records / unique pairs; R/E = known relevant / evidence; ? = unresolved. Every call succeeded and **none returned zero papers**. All 20 OpenAlex lists returned ten; OpenAIRE KeyBERT c01/c03/c08 returned 2/6/3, with archived totals equal to those counts. Of 37 lists returning ten, 36 had more than ten reported matches; the remaining list reported exactly ten. A ten-result list is not evidence that only ten papers exist.

| Claim | Provider | spaCy T/U | spaCy R/E | spaCy ? | KeyBERT T/U | KeyBERT R/E | KeyBERT ? |
|---|---|---:|---:|---:|---:|---:|---:|
| c01 | openalex | 10/10 | 9/2 | 1 | 10/10 | 8/2 | 1 |
| c01 | openaire | 10/10 | 3/1 | 7 | 2/2 | 1/0 | 1 |
| c02 | openalex | 10/10 | 5/0 | 3 | 10/10 | 8/3 | 0 |
| c02 | openaire | 10/10 | 0/0 | 10 | 10/10 | 8/7 | 0 |
| c03 | openalex | 10/10 | 9/4 | 1 | 10/10 | 10/9 | 0 |
| c03 | openaire | 10/10 | 0/0 | 10 | 6/6 | 6/1 | 0 |
| c04 | openalex | 10/10 | 9/6 | 1 | 10/10 | 10/8 | 0 |
| c04 | openaire | 10/10 | 0/0 | 10 | 10/10 | 8/6 | 2 |
| c05 | openalex | 10/10 | 10/5 | 0 | 10/10 | 10/10 | 0 |
| c05 | openaire | 10/10 | 4/2 | 6 | 10/10 | 9/7 | 0 |
| c06 | openalex | 10/10 | 9/0 | 0 | 10/10 | 9/8 | 1 |
| c06 | openaire | 10/10 | 1/0 | 9 | 10/10 | 10/6 | 0 |
| c07 | openalex | 10/10 | 10/9 | 0 | 10/10 | 10/5 | 0 |
| c07 | openaire | 10/10 | 3/3 | 7 | 10/10 | 10/5 | 0 |
| c08 | openalex | 10/10 | 10/6 | 0 | 10/10 | 10/8 | 0 |
| c08 | openaire | 10/10 | 2/1 | 8 | 3/3 | 3/2 | 0 |
| c09 | openalex | 10/10 | 10/9 | 0 | 10/10 | 9/8 | 1 |
| c09 | openaire | 10/10 | 1/1 | 9 | 10/10 | 6/3 | 0 |
| c10 | openalex | 10/10 | 9/6 | 0 | 10/10 | 9/6 | 0 |
| c10 | openaire | 10/10 | 3/2 | 7 | 10/10 | 3/2 | 7 |

Claims: c01 TNF/COVID/older adults; c02 battery capacity; c03 SiC/silicon efficiency; c04 coastal acidification; c05 dark matter/galaxy rotation; c06 irrigation/wheat; c07 vitamin D/respiratory infections; c08 molecular prediction; c09 platinum/hydrogen evolution; c10 exercise/memory.

KeyBERT?s OpenAlex evidence yields are notably higher for c03 (9 versus 4), c05 (10 versus 5), c06 (8 versus 0), and c08 (8 versus 6); spaCy retains an advantage for c07 (9 versus 5) and c09 (9 versus 8). c01, c03 and c08 show that a focused query can still supply too few OpenAIRE candidates in particular cases. Literal constraints, provider content and ranking may contribute; this frozen experiment does not isolate individual causes. There are no API failures explaining these small pools.

## Inconsistencies and sensitivity

The 12 cross-source disagreements are preserved below as unresolved rather than silently relabelled. In particular, two exercise/memory papers received bulk label 2 but sample label 0 despite using ?exercise/memory? in educational/computational senses. This illustrates why label-2 counts are not validated scientific evidence.

| Claim / identity | Bulk label | 30-example check label | Title |
|---|---:|---:|---|
| c01 / 59 | 2 | 1 | Anti-TNF-α Agents in Inflammatory Bowel Disease and Course of COVID-19 |
| c02 / 7 | 0 | 1 | Cerberus: A Deep Learning Hybrid Model for Lithium-Ion Battery Aging Estimation and Prediction Based on Relaxation Voltage Curves |
| c03 / 86 | 2 | 1 | Efficient Power Voltage Management of SiC MOSFET at Low Frequencies |
| c03 / 98 | 2 | 1 | A Comparison between Si and SiC MOSFETs |
| c04 / 105 | 2 | 1 | Ocean Acidification and Human Health |
| c05 / 15 | 2 | 1 | A note on shock wave in the dark matter medium |
| c05 / 16 | 2 | 1 | Ubiquitous Corotation of Dark Matter Halos: Implications for Direct Detection |
| c06 / 149 | 2 | 1 | Wheat growth, applied water use efficiency and flag leaf metabolome under continuous and pulsed deficit irrigation |
| c08 / 22 | 2 | 1 | KA-GNN: Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction |
| c09 / 202 | 1 | 2 | A platinum nanoparticle doped self-assembled peptide bolaamphiphile hydrogel as an efficient electrocatalyst for the hydrogen evolution reaction |
| c10 / 30 | 2 | 0 | BamaER: A Behavior-Aware Memory-Augmented Model for Exercise Recommendation |
| c10 / 43 | 2 | 0 | Auditing LLM Program Search for Bermudan Stopping with Primal--Dual Certificates |

For sensitivity, using only the two systematic bulk AI-assisted CSVs (without the separate 30-example check) gives the following R/E counts. This is an alternative explicit policy, not the main result:

| Provider | spaCy R / E | KeyBERT R / E |
|---|---:|---:|
| openalex | 93 / 50 | 94 / 68 |
| openaire | 17 / 10 | 64 / 39 |

The recommendation to retain complementary queries does not depend on silently choosing the bulk labels over disputed sample labels. It also does not assume that missing OpenAIRE judgments are irrelevant.

## Final integration decision

1. **For OpenAlex/OpenAIRE, use both tested concept strategies: KeyBERT first and spaCy as a complementary broader query.** Reuse the tested spaCy concept-discovery path, not the older conservative reporting-frame-only alternative. Keep the existing arXiv spaCy path unchanged: this experiment did not test KeyBERT on arXiv. Keep the unchanged original scientific claim for Person 2?s relevance/ranking input; do not substitute the discovery query for the claim. Do not add an untested third query automatically.
2. **Deduplicate identical provider query text before sending and canonical claim?paper pairs afterward.** c10 is identical between methods. The experiment deliberately requested both to keep budgets equal; suppressing that duplicate in production would avoid one request per provider for that claim. This is a proposed integration optimization, not a measured 38-request rerun.
3. **Retain provenance for Person 2:** claim ID/original claim, canonical paper identity, title, abstract/description, DOI/date/URLs and the list of provider/strategy/query/rank occurrences. Ranking operates on the deduplicated union. Keep provisional assessment labels separate as evaluation metadata.
4. **Budget trade-off:** both methods require up to two searches per provider/claim and expand the ranking pool. If a strict single-query budget is required, prefer KeyBERT on OpenAlex provisionally; keep spaCy as a fallback when KeyBERT has insufficient candidates, particularly OpenAIRE c01/c03/c08. Fallback-only use saves requests but would miss some complementary evidence even when KeyBERT returns ten.
5. **Operational trade-off:** KeyBERT requires pinned local MiniLM weights and extra PyTorch/Sentence Transformers dependencies; load once and reuse/cache generated queries. spaCy is already available and remains a lightweight, supported alternative. No hosted embedding or paid LLM service is needed. Public API calls were free in this run, but provider quota, latency and ranking work are real costs.
6. **Stop query-strategy experimentation and integrate the two-query union now.** Resolve the 12 disputed labels and remaining missing assessments as evaluation cleanup, without making further retrieval experiments a prerequisite for Person 2 integration. Revisit the policy only if downstream validation or operational limits give a concrete reason.

## Limits

Only ten hand-selected claims and one capped live snapshot were compared. AI-assisted labels are not independent validated ground truth, can overvalue topical abstracts and may apply different tolerances across assessment cohorts. There is no significance test, complete corpus recall, full-text claim verification or proof of relevance generalization. OpenAIRE?s incomplete labels prevent a definitive standalone method-quality ranking there. Metadata can be duplicated, multilingual or non-article content; conservative matching may retain near-duplicates. More label-2 papers does not establish stronger or correct scientific support.

Only this Markdown report was written. No API calls, new retrieval experiments, label edits or retrieval-logic changes were made.
