# Pinned team runtime sources

- discovery_quality: Raghavi `0ee95651cdd0c9d40a0ffb09a6bba896db3be455`; independently pinned advisory abstract checks and optional offline batch report. Type-only/import isolation plus an explicitly recorded empty-result CSV fix; classification logic unchanged. The installed discovery dependency remains pinned at259d2a9. See discovery_quality/SOURCE.json.

- ranking_evaluation: Yu Yue 7ed93fa4be69c661422b33b6c88b949dd2636e12; six unchanged runtime files.
- claim_verifier: Chong Yao 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576; schemas and verifier, with bare schemas import changed to relative .schemas for isolation.
- retrieval: Naila 2bc7d81292313498fd70dd13656024d00cf3dda4; six runtime files; retriever._top_k has a documented frontend stable-cutoff fix (SOURCE.json).

Each package stays inside Kireeti's folder. No model weights or full corpora are tracked; compact published benchmark artifacts are tracked under ../benchmarks/. Discovery remains the separately installed pinned Raghavi paper_search dependency. See ../INTEGRATION_CHANGELOG.md for the exact bridge and behavior additions rather than assuming the whole pipeline is unchanged teammate code.

- person3: Yu Yue 7ed93fa4be69c661422b33b6c88b949dd2636e12; unchanged semantic/evidence/PDF/budget modules and frozen configuration.
- xai: Naila 56f46c12d8ef1cc36202f046b4bbaa7825fc6ade; attribution and evaluation helper with isolated package imports. See SOURCE.json and ../../COMBINED_INTEGRATION.md.


Context addon: `person3/contextual_evidence.py` is byte-exact from Yu Yue commit ba8cdd6, with a separate companion SOURCE record. Existing baseline Person3 modules remain pinned and unchanged. The primary provider uses bounded abstract context and the frozen 0.65 relevance gate; original policy and historical study code remain available.
