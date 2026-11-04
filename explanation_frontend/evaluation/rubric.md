# Paired explanation review

Use frozen, identical evidence and verdict snapshots for both methods. Default cases are synthetic stress tests, not SciFact or real model evidence. Use the agreed held-out split for the final study and tune prompts only on development data. Keep live/PDF examples separate.

Run `--mode fixture` to validate tooling without model weights. The scripted fixture generator intentionally injects failures. Run `--mode llm` after setup-local-llm.ps1: inference uses the pinned Apache-2.0 Qwen model on this computer, with no hosted API, credentials or per-call fee. Record the generated settings, model, prompt version, evidence hash and split in the study report.

review_packets.jsonl contains randomly assigned A/B outputs with claim, authoritative verdict and evidence. Keep private_key.jsonl and records.jsonl away from reviewers until scoring. Method blinding is imperfect because styles can reveal the method. Two reviewers independently score every output in review.csv; leave unreviewed rows blank and adjudicate disagreements separately. Do not invent scores for missing reviews.

For each raw output, split scientific assertions into atomic statements. A partly supported conjunction counts as unsupported unless all clauses are supported. Compare to the exact cited passages, including numbers, units, qualifications, population/benchmark scope, uncertainty and attribution. Exact quotation can still be irrelevant to the explanation; do not equate string matching with support.

| Field | Meaning |
| --- | --- |
| factual_supported / factual_total | Count statements wholly supported by supplied evidence / all scientific statements |
| citation_supported / citation_total | Count statement–citation pairs with actual support / all cited pairs |
| citation_complete / citation_required | Count scientific statements with adequate supporting citations / all statements needing citations |
| verdict_consistent | 1 if explanation agrees with verifier including conflict/uncertainty, otherwise 0 |
| scope_preserved | 1 if numbers, units, qualifiers, attribution and uncertainty are preserved throughout, otherwise 0 |
| correction_supported / correction_total | Count explicit replacement assertions supported / all proposed corrections |
| usefulness | 1 misleading/unusable; 2 difficult/major omissions; 3 understandable but limited; 4 clear and traceable; 5 clear, complete and directly useful for inspecting evidence |
| notes | Explain failure, missing evidence, first failing stage or reviewer disagreement |

Zero denominator means not applicable, never automatic 100%. For invalid JSON or failed generation, annotate the operational failure and rate usefulness conservatively; do not silently exclude the case. Review raw model output and final displayed fallback separately. The generated raw review packets target raw quality; records.jsonl contains displayed output and validator outcome for a separate displayed-system review if desired.

`--reviews review.csv` reports pooled and per-method counts with denominators, pending-review totals and paired reviewer agreement on verdict consistency. It joins private_key.jsonl beside the review CSV (or use --key). Document adjudication and further agreement analysis separately. Report generation acceptance, rejection, timeout/fallback, latency and human support metrics separately. Do not tune on final-test reviews.

Real manifests can reference a versioned exported snapshot instead of a fixture:

```json
{"case_id":"heldout-001","snapshot_file":"snapshots/heldout-001.json","dataset":"SciFact-agreed-version","split":"team-heldout"}
```

The path is relative to the manifest. Keep private real manifests, snapshots and generations under outputs/ (ignored), not in committed synthetic fixtures. For the cleanest pairing retain the saved selection and aggregation; changing a saved selection uses the documented replay policy rather than Person 4's live aggregator.
