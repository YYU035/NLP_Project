# Pinned team runtime sources

- ranking_evaluation: Yu Yue a91e257ab260ded8df8c2cc2f214c4c7fd16af6c; six unchanged runtime files.
- claim_verifier: Chong Yao 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576; schemas and verifier, with bare schemas import changed to relative .schemas for isolation.
- retrieval: Naila 2bc7d81292313498fd70dd13656024d00cf3dda4; six unchanged runtime files.

Each package stays inside Kireeti's folder. No weights or datasets are tracked. Discovery remains the separately installed pinned Raghavi paper_search dependency. See ../INTEGRATION_CHANGELOG.md for the exact bridge and behavior additions rather than assuming the whole pipeline is unchanged teammate code.
