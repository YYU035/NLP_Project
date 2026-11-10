# Pinned team runtime sources

- ranking_evaluation: Yu Yue 7ed93fa4be69c661422b33b6c88b949dd2636e12; six unchanged runtime files.
- claim_verifier: Chong Yao 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576; schemas and verifier, with bare schemas import changed to relative .schemas for isolation.
- retrieval: Naila 2bc7d81292313498fd70dd13656024d00cf3dda4; six runtime files; retriever._top_k has a documented frontend stable-cutoff fix (SOURCE.json).

Each package stays inside Kireeti's folder. No model weights or full corpora are tracked; compact published benchmark artifacts are tracked under ../benchmarks/. Discovery remains the separately installed pinned Raghavi paper_search dependency. See ../INTEGRATION_CHANGELOG.md for the exact bridge and behavior additions rather than assuming the whole pipeline is unchanged teammate code.

- person3: Yu Yue 7ed93fa4be69c661422b33b6c88b949dd2636e12; unchanged semantic/evidence/PDF/budget modules and frozen configuration.
- xai: Naila 56f46c12d8ef1cc36202f046b4bbaa7825fc6ade; attribution and evaluation helper with isolated package imports. See SOURCE.json and ../../COMBINED_INTEGRATION.md.
