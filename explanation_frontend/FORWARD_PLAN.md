# Forward plan - 6 November 2026

## Completed

- Discovery and canonical provenance connected to a working Streamlit search browser.
- Naila BM25/SBERT and Yu Yue RRF integrated inside this folder, using explicit local embeddings.
- Temporary exact-span abstract/description selector added with source offsets and stable IDs.
- Chong Yao classifier/verify methods connected to pinned local MIT weights and cached native responses.
- Explicit frontend conflict-side preservation, operational failure handling, subset comparison, grounded explanations and JSON replay connected.
- Real recorded/fresh-search smoke checks and regression/UI tests completed; see evaluation/end-to-end-smoke.md.

## Pending team acceptance / quality work

1. Passage owner approves/replaces the provisional selector; preserve source text and dataset sentence indices in any benchmark implementation.
2. Chong Yao reviews the frontend internal-conflict and five-passage-cap policy with both evidence sides preserved.
3. Person 1/team supplies full-text acquisition and eligibility/retraction/freshness/source-version policy.
4. Freeze real held-out cases and collect human scientific explanation/citation reviews. Smoke checks are not accuracy benchmarks.
5. Optional cross-encoder remains disabled until numerical/dependency validation; optional local Qwen comparison still needs completed setup and evaluation.
6. Team reviews/merges kireeti into the shared branch after inspecting INTEGRATION_CHANGELOG.md. No main-branch merge was performed by this integration work.

All implementation changes stay in explanation_frontend. No hosted/proprietary inference or paid API dependency is introduced.
