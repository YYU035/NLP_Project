# Person 3 — complete checklist audit (v2)

**Status: all implementation and automatic experiments below are complete. Human PDF review is pending; the task is not fully signed off.**

## Screenshot completion criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every passage is locatable in source | Implemented and automatically checked | Original sentence IDs/character offsets; PDF page/offset/hash; separate page images |
| Evidence precision, recall, F1 measured against abstract annotations | Complete | Full SciFact dev; alternatives/no-gold/missing documents handled |
| Separate manually reviewed PDF sample | **Human review pending** | 2 versioned PDFs, 6 constructed claims, 12 exact passages, original page images and blank ratings |
| Failure analysis: irrelevant passages, missing context, incomplete evidence | Complete | 35 / 23 / 3 annotated claim-document cases; 28 missing-paper cases; actual source text included |
| Fits actual verifier input limits without silent truncation | Complete | Actual P4 pair tokenizer; 39,308 checks; max accepted 484/512; whole over-budget input rejected |

## Person 4 source and real inference

- Branch: chong_yao_claim_verification, pinned commit 1ac06b675e3d0719a63cb8efe98eda4f2c5fd576.
- The Git CLI fetch required unavailable credentials; the connected GitHub tool read the same pinned source. No teammate branch was merged or pushed.
- Actual source signature: evidence premise + original claim hypothesis, special tokens included, 512 total pair tokens.
- Real original verify/classify methods ran on all 12 accepted PDF passages; zero NLI truncation. Local initializer explicitly used pinned offline CPU weights.
- The separate real-token stress input was 907 tokens and was rejected before inference. 512/513 boundary and overlong-claim cases are tested.

## Failure analysis

Frozen RRF top-5 semantic sentence/context predictions, after actual token preflight:

- relevant_paper_irrelevant_passages: 35 claim-document cases.
- incomplete_evidence: 3 claim-document cases.
- missing_context_rescued: 23 claim-document cases.
- missing_gold_paper: 28 claim-document cases.

These categories may overlap. Context recovery means recovering an annotated rationale; it does not claim general pronoun resolution. Unannotated sentences can be penalized by the reference.

## Optional passage Cross-Encoder ablation

Same candidate pool and identical top-1 per-paper output budget in both arms; not directly comparable to top-3 output metrics. Both Cross-Encoder and NLI tokenizers are checked without truncation. No dev retuning.

| Paper input | Passage reranker | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| oracle_gold_papers | reranker_off | 0.5572 | 0.3060 | 0.3951 |
| oracle_gold_papers | reranker_on | 0.6617 | 0.3634 | 0.4691 |
| rrf top 5 | reranker_off | 0.1427 | 0.2705 | 0.1868 |
| rrf top 5 | reranker_on | 0.1700 | 0.3224 | 0.2226 |
| rrf_ce_top20 top 5 | reranker_off | 0.1598 | 0.2842 | 0.2045 |
| rrf_ce_top20 top 5 | reranker_on | 0.1874 | 0.3333 | 0.2399 |

## Manual review steps

1. Forward local cluster port 18504 and open http://localhost:18504, or unzip PDF_Manual_Review.zip and open index.html.
2. Compare each of the twelve quotes with its original PDF page. Fill reviewer name and source/relevance/context/input-limit judgements; negative/uncertain ratings are allowed.
3. Download human-review-completed.csv. The page does not send data automatically.
4. Return the actual completed CSV and run validate_pdf_review with --record-in-packet. Blank forms cannot satisfy this criterion.

AI visual prechecks are separately marked. Human names, labels and approvals have not been invented.

## Full text and frontend

- Local PDF and upstream page-text inputs are supported. Scanned/encrypted sources fail explicitly; OCR is not implemented.
- Positional PDF sorting interleaved columns in BERT. Packet v2 uses content-stream order and keeps a separate visual review requirement.
- Semantic frontend now requires PERSON3_VERIFIER_TOKENIZER_PATH, stores every budget check/rejection and treats all-over-limit input as operational failure, not a scientific verdict.
- Actual recorded-search frontend smoke passed with all accepted pairs within budget and no NLI truncation. The new preview on 18503 has the budget guard enabled.

## Shared results

shared-results-v1/retrieval-results.csv distinguishes method owner from experiment runner. The separate owner-submissions.csv is ready for each owner to add their own run and confirmation. Those confirmations remain pending; no messages or signatures were fabricated.

## Validation

- Person 3 tests: 126 passed in the complete evidence runtime (including actual PDF extraction).
- Frontend tests after budget integration: 117 passed, including all-budget-rejected operational failure and cached aggregation.
- Separate PDF and token-budget jobs completed successfully; passage-CE matched-budget ablation completed successfully.

## Boundaries

This is development evaluation, not untouched test accuracy. The PDFs are a manual-review sample, not a second independently labelled benchmark. Paper extraction order, table/figure interpretation and scientific relevance remain reviewable limitations. Human review and teammate result confirmations are the remaining external signoffs.
