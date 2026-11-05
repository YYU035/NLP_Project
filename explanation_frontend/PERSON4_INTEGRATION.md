# Person 4 integration in Kireeti branch

The actual schemas and runtime from Chong Yao commit `1ac06b675e3d0719a63cb8efe98eda4f2c5fd576` are included under `_upstream/claim_verifier`. Only the verifier import was changed to a relative `.schemas` import so it cannot collide with other modules. No teammate folder, model weights or dataset is included.

`person4.Person4Adapter` accepts an actual `VerificationResponse` (typed object or dictionary), frontend `Paper` metadata and the owner's aggregation callable. It converts the response to a validated `RunSnapshot`, caches the original typed response and delegates subsequent paper subsets to the real owner aggregator. Unknown IDs and stale snapshots fail closed. This is a programmatic adapter; it is not yet configured as the Streamlit selected-search factory.

```python
from explanation_frontend.person4 import Person4Adapter

# verifier must already be explicitly set up with approved local model weights.
# passages_by_paper must come from the agreed evidence selector.
native_response = verifier.verify(original_claim, native_papers, passages_by_paper)
adapter = Person4Adapter(verifier.aggregate)
snapshot = adapter.snapshot(native_response, frontend_papers, mode="live")
subset_result = adapter.aggregate(snapshot, selected_paper_ids)
```

`frontend_papers` is a list of the frontend Paper models, preserving discovery canonical IDs and original metadata. The adapter does not manufacture passage text or metadata. The imported response does not supply a retrieval timestamp: retrieved_at records the import time and the snapshot explicitly marks retrieval time/search completeness as unknown. To retain ranking/discovery lineage, the future selected-search factory must use the existing FromDiscoveryProvider seam and keep this adapter session-scoped.

## Connected and tested

- Native lists to frontend checked ID maps, enum/status conversion and model settings mapping.
- Original typed VerificationResponse reused by the actual cached aggregator; no neural inference on checkbox changes.
- Empty subset, unknown IDs, duplicate labels, foreign paper IDs and total truncation behavior.
- Total absence of operationally usable passages becomes failed processing, with no scientific verdict.
- Source exceptions/failure strings are not exposed in public snapshot failure fields.

## Still pending

- **[PENDING — Kireeti + Person 4]** Explicit local model installation with immutable revision, compatible dependencies and offline runtime. The upstream constructor can download a default main revision; this adapter never invokes it.
- **[PENDING — Person 4]** Resolve internal-conflict aggregation and cases where the five-passage cap drops one conflict side. Adapter rejects internal conflicts and the frontend validation rejects decisive aggregates without corresponding primary evidence. It does not replace the owner's scientific algorithm.
- **[PENDING — evidence owner]** Real ranked passage inputs and ownership agreement.
- **[PENDING — Kireeti]** Configure the full selected-search factory once those dependencies exist, retaining discovery/ranking provenance and original exact claim.
- **[PENDING — team]** Real model smoke test, measured subset latency and full-chain/human evaluation. Current integration tests use synthetic responses, not new scientific accuracy evidence.
