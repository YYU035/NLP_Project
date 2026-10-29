# Evaluation protocol

Freeze the SciFact release, record its origin and checksum, and retain original corpus document IDs, claim IDs, splits, labels, and rationale sentence indices. Implement loading in `shared/datasets/scifact.py`. Do not tune on held-out evaluation labels. If test labels are unavailable, report held-out development results and document the tuning split separately.

Retrieval: report Recall@K, MRR and nDCG at configured K values. Define relevance from SciFact evidence documents and document treatment of claims without relevant documents and multiple relevant documents. State the MRR cutoff, nDCG gain definition and all metric denominators.

Compare BM25, SBERT, RRF, and RRF + Cross-Encoder on identical queries and corpus. Ablate each retrieval branch, reranking, and candidate depth. Record seeds, model revisions, configuration, hardware and package versions.

Verification: report Accuracy and Macro-F1 over Supported / Contradicted / Insufficient Evidence. Document the SciFact label conversion, evidence aggregation rule and treatment of conflicting evidence. Distinguish gold-evidence verification from retrieved-evidence end-to-end evaluation.

Latency: record per-stage and end-to-end wall-clock time, warm/cold model state, sample count, median and p95. Separate network search/download time from fixed-corpus retrieval. Keep failed requests visible.

Store run metadata, per-claim outputs and aggregate metrics under ignored `outputs/`. Live search is an integration demonstration, not a reproducible benchmark substitute.
