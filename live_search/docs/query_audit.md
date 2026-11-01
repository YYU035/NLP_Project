# Scientific query audit: actual before/after outputs

Captured baseline outputs are in `query_audit_before.json`; the old
generator is not reimplemented by this report. These strings were
generated locally using spaCy 3.8.18.

**14 claims: 8 focused alternatives, 6 original-only decisions.**

The first query after this audit is the original input unchanged.
No retrieval evaluation was performed. Risks below are qualitative,
not measured recall losses. Retained conditions are existing claim
constraints, not constraints introduced by the focused query.

Removing reporting words can broaden matches and change provider
ordering; under a fixed top-k budget, this can still lose relevant
papers from the returned list. No recall guarantee follows from
shortening the query. Exact symbol matching is backend-dependent.

## 1. biomedical

Original / primary query:

```text
The results of this study show that the expression of TNF-α does not increase the risk of COVID-19 in older adults.
```

Before: additional focused query:

```text
"results" of "this study" show that "expression" of "TNF-α" does not increase "risk" of "COVID-19" in "older adults".
```

After: additional focused query:

```text
the expression of TNF-α does not increase the risk of COVID-19 in older adults.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: expression of TNF-α; risk of COVID-19; older adults.

Recall caveat: The older-adult population and exact TNF-α/COVID-19 terminology remain restrictive; backend symbol tokenization may differ.

## 2. immunology

Original / primary query:

```text
The CRISPR-Cas9 system may reduce IL-6 in HER2+ cells.
```

Before: additional focused query:

```text
"CRISPR-Cas9 system" may reduce "IL-6" in "HER2+" "cells".
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `no_useful_focused_alternative`.

Selected concepts: CRISPR-Cas9 system; IL-6; HER2+ cells.

Recall caveat: No alternative emitted. Semantic Scholar still cannot enforce exact hyphenated identifier matching.

## 3. battery_engineering

Original / primary query:

```text
In this study, we demonstrate that lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C.
```

Before: additional focused query:

```text
In "this study", we demonstrate that "lithium-ion batteries" retain "90% capacity" after "500 cycles" at -5 "°C".
```

After: additional focused query:

```text
lithium-ion batteries retain 90% capacity after 500 cycles at -5 °C.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: lithium-ion batteries; 90% capacity; 500 cycles; -5 °C.

Recall caveat: The retained 90%, 500-cycle and -5 °C conditions may exclude papers reporting other units or experimental settings.

## 4. power_electronics

Original / primary query:

```text
Our experiments show that the efficiency of SiC MOSFETs is not lower than that of silicon devices at 100 kHz.
```

Before: additional focused query:

```text
"Our experiments" show that "efficiency" of "SiC MOSFETs" is not lower than that of "silicon devices" at "100 kHz".
```

After: additional focused query:

```text
the efficiency of SiC MOSFETs is not lower than that of silicon devices at 100 kHz.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: efficiency of SiC MOSFETs; silicon devices; 100 kHz.

Recall caveat: The silicon comparator and 100 kHz condition remain restrictive; comparison language is not interpreted logically.

## 5. climate

Original / primary query:

```text
This study demonstrates that elevated atmospheric CO2 concentrations increase ocean acidification in coastal waters.
```

Before: additional focused query:

```text
"This study" demonstrates that "elevated atmospheric CO2 concentrations" increase "ocean acidification" in "coastal waters".
```

After: additional focused query:

```text
elevated atmospheric CO2 concentrations increase ocean acidification in coastal waters.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: elevated atmospheric CO2 concentrations; ocean acidification; coastal waters.

Recall caveat: Coastal waters remains a population/environment constraint; relevant open-ocean studies may not match.

## 6. astronomy

Original / primary query:

```text
The presence of dark matter increases the rotational velocity of spiral galaxies.
```

Before: additional focused query:

```text
"presence" of "dark matter" increases "rotational velocity" of "spiral galaxies".
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `no_useful_focused_alternative`.

Selected concepts: presence of dark matter; rotational velocity; spiral galaxies.

Recall caveat: No alternative emitted. Terminology variation is not addressed through unsupported synonyms.

## 7. agriculture

Original / primary query:

```text
Experimental results show that deficit irrigation reduces water use without decreasing wheat yield under drought conditions.
```

Before: additional focused query:

```text
"Experimental results" show that "deficit irrigation" reduces "water use" without decreasing "wheat yield" under "drought conditions".
```

After: additional focused query:

```text
deficit irrigation reduces water use without decreasing wheat yield under drought conditions.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: deficit irrigation; water use; wheat yield; drought conditions.

Recall caveat: Both drought conditions and the without-decreasing qualification are retained; a paper may express them differently.

## 8. nutrition_uncertainty

Original / primary query:

```text
Preliminary findings suggest that vitamin D supplementation may reduce respiratory infections in older adults.
```

Before: additional focused query:

```text
"Preliminary findings" suggest that "vitamin D supplementation" may reduce "respiratory infections" in "older adults".
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `no_useful_focused_alternative`.

Selected concepts: Preliminary findings; vitamin D supplementation; respiratory infections; older adults.

Recall caveat: No alternative emitted: the preliminary, suggest, and may qualifiers must not be erased.

## 9. computer_science

Original / primary query:

```text
Our results show that graph neural networks improve molecular property prediction compared with random forests.
```

Before: additional focused query:

```text
"Our results" show that "graph neural networks" improve "molecular property prediction" compared with "random forests".
```

After: additional focused query:

```text
graph neural networks improve molecular property prediction compared with random forests.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: graph neural networks; molecular property prediction; random forests.

Recall caveat: The random-forest comparator remains restrictive; removing reporting terms can change the provider's capped result order.

## 10. catalysis

Original / primary query:

```text
It has been demonstrated that platinum nanoparticles accelerate hydrogen evolution at 25 °C in acidic solutions.
```

Before: additional focused query:

```text
It has been demonstrated that "platinum nanoparticles" accelerate "hydrogen evolution" at "25 °C" in "acidic solutions".
```

After: additional focused query:

```text
platinum nanoparticles accelerate hydrogen evolution at 25 °C in acidic solutions.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: platinum nanoparticles; hydrogen evolution; 25 °C; acidic solutions.

Recall caveat: The 25 °C and acidic-solution conditions remain restrictive; related studies may describe different temperatures or environments.

## 11. mathematics

Original / primary query:

```text
Every continuous function on a compact interval is uniformly continuous.
```

Before: additional focused query:

```text
"Every continuous function" on "compact interval" is uniformly continuous.
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `no_useful_focused_alternative`.

Selected concepts: Every continuous function; compact interval.

Recall caveat: No alternative emitted. Universal quantifiers and the compact-interval condition remain intact.

## 12. general_science

Original / primary query:

```text
Exercise improves memory.
```

Before: additional focused query:

```text
"Exercise" improves "memory".
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `insufficient_meaningful_concepts`.

Selected concepts: Exercise; memory.

Recall caveat: No alternative emitted: the original is already short, and generic noun lists add no useful focus.

## 13. chemistry_numbers

Original / primary query:

```text
At 1e-3 M, NaCl does not alter protein stability at pH 7.4.
```

Before: additional focused query:

```text
At "1e-3 M", "NaCl" does not alter "protein stability" at "pH" "7.4".
```

After: additional focused query:

```text
[none — original query only]
```

Decision: `no_useful_focused_alternative`.

Selected concepts: 1e-3 M; NaCl; protein stability; pH; 7.4.

Recall caveat: No alternative emitted. Scientific exponents, negative/polarity context, and pH values remain provider-sensitive.

## 14. biomedical_population

Original / primary query:

```text
This study shows that vitamin A supplementation does not reduce mortality in children under five years of age.
```

Before: additional focused query:

```text
"This study" shows that vitamin "A supplementation" does not reduce "mortality" in "children" "under five years of age".
```

After: additional focused query:

```text
vitamin A supplementation does not reduce mortality in children under five years of age.
```

Decision: `concept_anchored_assertion_without_reporting_frame`.

Selected concepts: vitamin A supplementation; mortality; children; under five years of age.

Recall caveat: The under-five age constraint remains restrictive. General English NER may mislabel the age span as DATE.
