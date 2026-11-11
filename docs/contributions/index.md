# Contributions and source provenance

One application is organized by function. This record preserves authorship and distinguishes source contributions from integration adaptations.

| Contributor | Integrated responsibility | Recorded source revision |
|---|---|---|
| Raghavi | Public discovery, query generation, normalized metadata, identity/deduplication, provenance and abstract-quality rules | Discovery `259d2a9e90fb8e9d5988f030a156733f4f3576d1`; quality `0ee95651cdd0c9d40a0ffb09a6bba896db3be455` |
| Naila | Custom NumPy BM25, MiniLM/SBERT retrieval and experimental Captum IG helpers | Retrieval `2bc7d81292313498fd70dd13656024d00cf3dda4`; IG `56f46c12d8ef1cc36202f046b4bbaa7825fc6ade` |
| Yu Yue | RRF, semantic evidence, exact spans, PDF/budget support, three-domain evaluation and bounded context | Core `7ed93fa4be69c661422b33b6c88b949dd2636e12`; context `ba8cdd6e7766306447913d8087b23f62195fa05a` |
| Chong Yao | Local DeBERTa classification, paper verdicts, threshold and decoupled verify/aggregate API | `1ac06b675e3d0719a63cb8efe98eda4f2c5fd576` |
| Kireeti | Streamlit integration, contracts, grounded templates, citation gates, scope controls, caching/replay, PDF/XAI bridges, quality UI, evaluation and release consolidation | Pre-consolidation `11ba8dcadbbdd5ab2d672f9e42d61b53a7b592ce`; later release commits record migration |

The latest owner runtime work was represented in the pre-consolidation integration. Selected source snapshots were integrated; not every research file from every owner branch became runtime code.

Source records retain original revisions/hashes and adaptations, including relative imports, stable cutoff ties, isolated XAI imports and empty quality-report handling. Discovery provenance documents local bundling and omitted non-runtime experiments. The migration manifest maps old paths to current functional names.

## Remaining evidence and review

| Responsibility | Open review |
|---|---|
| Discovery | Shared-query provider recall, source eligibility and human validation of warnings |
| Retrieval/evidence | Retrieval misses, rationale coverage, PDF reading order and contradiction-recall tradeoffs |
| Verification | Population/generalisation errors, calibration and broader labelled-domain evaluation |
| Interpretability | Broader word-sensitivity plausibility/faithfulness and reliable real-case IG |
| Explanations/integration | Genuine human explanation/citation/PDF ratings and retained regression review |
| Whole team | Reproduce the release on teammates' machines and communicate benchmark scope |

These are responsibilities supported by evidence, not estimated completion percentages. Reorganizing folders does not complete scientific or human review.
