# Pre-retrieval query review

Local KeyBERT/MiniLM versus the exact saved spaCy concept text. No retrieval results were consulted to tune these queries.

| Claim | spaCy | KeyBERT | Retained phrases |
|---|---|---|---|
| c01 | TNF-α COVID-19 | expression of TNF-α risk of COVID-19 older adults | expression of TNF-α; risk of COVID-19; older adults |
| c02 | lithium-ion batteries capacity | lithium-ion batteries 90% capacity -5 °C | lithium-ion batteries; 90% capacity; -5 °C |
| c03 | SiC MOSFETs silicon devices | efficiency of SiC MOSFETs silicon devices 100 kHz | efficiency of SiC MOSFETs; silicon devices; 100 kHz |
| c04 | elevated atmospheric CO2 concentrations ocean acidification | atmospheric CO2 concentrations ocean acidification coastal waters | atmospheric CO2 concentrations; ocean acidification; coastal waters |
| c05 | dark matter rotational velocity | dark matter rotational velocity of spiral galaxies | dark matter; rotational; velocity of spiral galaxies |
| c06 | deficit irrigation water use | deficit irrigation wheat yield drought conditions | deficit irrigation; wheat yield; drought conditions |
| c07 | vitamin D supplementation respiratory infections | vitamin D supplementation respiratory infections older adults | vitamin D supplementation; respiratory infections; older adults |
| c08 | graph neural networks molecular property prediction | graph neural networks molecular property prediction random forests | graph neural networks; molecular property prediction; random forests |
| c09 | platinum nanoparticles hydrogen evolution | platinum nanoparticles hydrogen evolution acidic solutions | platinum nanoparticles; hydrogen evolution; acidic solutions |
| c10 | Exercise memory | Exercise memory | Exercise; memory |

## Qualitative caveats

KeyBERT ranks literal candidate phrases, not scientific entities or evidence. Compared with spaCy, it can retain different outcomes/conditions, overemphasize a measurement or omit a population/comparator. Neither discovery query is equivalent to the original claim. Negation and numeric conditions may be omitted; all omitted source spans and phrase cosine scores are in queries.json. Those scores are not relevance scores for retrieved papers.

## Inspection before retrieval

- c01: adds expression/risk phrases and older adults; polarity still omitted.
- c02: retains 90% capacity and -5 ?C, omits 500 cycles; likely more restrictive.
- c03: retains efficiency and complete 100 kHz; no unit splitting remains.
- c04: retains coastal waters, removes elevated; adds geographic specificity.
- c05: retains spiral galaxies; adjacent scored chunks combine into the literal
  rotational-velocity phrase in the final unquoted query.
- c06: retains wheat yield and drought conditions, drops water use; changes the
  balance of outcome constraints compared with spaCy.
- c07: adds older adults, potentially excluding studies in other populations.
- c08: adds random forests, retaining the claim's comparator.
- c09: adds acidic solutions but not the 25 ?C condition.
- c10: identical to spaCy; duplicate fresh requests serve as an equal-budget
  repeated-query control, not a distinct reformulation.

Initial preparation exposed a split unit, retained reporting language and an
empty punctuation token. These candidate-generation bugs were corrected before
any retrieval. queries_initial.json and queries_candidate_fix.json preserve
those unrun drafts; only queries.json is the final retrieval plan. No search
outcomes were used to choose these corrections or tune the final queries.
