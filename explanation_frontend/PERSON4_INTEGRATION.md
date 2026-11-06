# Person 4 integration in Kireeti branch

The actual schemas and runtime from Chong Yao commit `1ac06b675e3d0719a63cb8efe98eda4f2c5fd576` are included under `_upstream/claim_verifier`. Only the verifier import was changed to a relative `.schemas` import so it cannot collide with other modules. No teammate folder, model weights or dataset is included.

`person4.Person4Adapter` accepts an actual `VerificationResponse` (typed object or dictionary), frontend `Paper` metadata and the owner's aggregation callable. It converts the response to a validated `RunSnapshot`, caches the original typed response and delegates subsequent paper subsets to the real owner aggregator. Unknown IDs and stale snapshots fail closed. The built-in pipeline now uses this adapter for both Live and Selected search, with actual local NLI and cached subset aggregation.

```python
from explanation_frontend.person4 import Person4Adapter

# verifier must already be explicitly set up with approved local model weights.
# passages_by_paper must come from the agreed evidence selector.
native_response = verifier.verify(original_claim, native_papers, passages_by_paper)
adapter = Person4Adapter(verifier.aggregate)
snapshot = adapter.snapshot(native_response, frontend_papers, mode="live")
subset_result = adapter.aggregate(snapshot, selected_paper_ids)
```

`frontend_papers` is a list of the frontend Paper models, preserving discovery canonical IDs and original metadata. The adapter does not manufacture passage text or metadata. For standalone response imports, retrieval time/search completeness remain unknown. The built-in pipeline enriches the result with the original discovery/ranking context, known retrieval time when supplied, and explicit unknown-timestamp bookkeeping otherwise. The adapter and original response remain provider-scoped.

## Connected and tested

- Native lists to frontend checked ID maps, enum/status conversion and model settings mapping.
- Original typed VerificationResponse reused by the actual cached aggregator; no neural inference on checkbox changes.
- Empty subset, unknown IDs, duplicate labels, foreign paper IDs and total truncation behavior.
- Total absence of operationally usable passages becomes failed processing, with no scientific verdict.
- Source exceptions/failure strings are not exposed in public snapshot failure fields.

## Current status and remaining review

- **[CONNECTED]** Pinned local MIT NLI weights, offline model loading, real verify() execution and actual cached aggregate() delegation.
- **[CONNECTED]** Built-in selected-search/live factories preserve original claims, canonical IDs, discovery metadata and ranking outputs. No repeated inference on paper toggles.
- **[ADDED BY KIREETI]** safe_aggregate preserves internal and across-paper conflict sides before applying a five-passage cap. This explicit policy needs Chong Yao/team review. The base adapter still rejects unresolved conflicts.
- **[ADDED BY KIREETI]** Exact abstract-span fallback provides missing evidence inputs without claiming full-text inspection.
- **[VERIFIED]** Recorded and fresh public-search smoke checks passed with actual NLI; synthetic regression tests cover operational failures and UI/adapter contracts.
- **[PENDING]** Final passage-owner agreement, policy acceptance, held-out scientific evaluation, full-text handling, optional cross-encoder and Qwen comparisons. No scientific-quality benchmark is inferred from smoke checks.
