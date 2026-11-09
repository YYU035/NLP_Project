# Active Three-Dataset Evaluation Plan

Updated: 2026-11-09. All authored code, experiments and documentation are English.

## Selected datasets

The user selected RECIPER after the completed
[EEE/materials screening](datasets/EEE_MATERIALS_SCREENING.md).
The active set is **SciFact + CSFCube + RECIPER**. Earlier Matter-of-Fact and
ClimateCheck choices/results are retained as archived exploratory work.

| Dataset | Domain | Evaluation role | Boundary |
|---|---|---|---|
| SciFact | Primarily biomedical | Paper retrieval/ranking and gold sentence-evidence selection | Existing measured GPU development results; low evidence precision remains a limitation |
| CSFCube | Computer science | Expert graded paper ranking | Measured four-method fold1-dev comparison; no sentence/verdict gold |
| RECIPER | Materials science with EEE-related content | Source-paper retrieval over all 343 papers for 1,024 questions | Generated QA; single-source targets, no exhaustive relevance or expert evidence-sentence gold |

These are independent datasets with task-specific metrics. Do not average their
scores into an overall system accuracy or present them as three identical
end-to-end verification benchmarks. The known source paper is not necessarily
the only scientifically relevant paper for a question.

## Current implementation

- Pinned downloads/checksums and adapters are in `shared/datasets/`.
- P3 evaluation/runners are in `ranking_evaluation/` and call actual P2 BM25/SBERT
  and P3 fusion/Cross-Encoder boundaries.
- RECIPER's first protocol indexes original titles and abstracts, matching the
  existing P2 representation. Generated answers, topics and gold source labels
  are excluded from features. Every question faces the same complete collection.
- RRF k=60, Cross-Encoder top 20 and model revisions were fixed before scoring.
  No RECIPER training or parameter selection is performed.
- RECIPER's original paragraph/recipe retrieval setup differs from this paper-level
  baseline. The published scores are not reused as project results.
- See [RECIPER_REPORT.md](datasets/RECIPER_REPORT.md) for exact commands, state,
  measured numbers, provenance and limitations.

## Instructor feedback

Report Recall@K, MRR, nDCG and latency for paper-ranking tasks. SciFact also
provides sentence-rationale P/R/F1 and complete-rationale recovery; distinguish
oracle-paper and retrieved-paper conditions. RECIPER and CSFCube cannot yield
expert sentence-evidence F1 without new independent annotations. Final verdict
Accuracy/Macro-F1 requires actual verification inference on a suitably labeled
task and remains distinct from the ranking experiments.

The live application still searches scholarly APIs independently of these
fixed evaluation resources. No mandatory dataset router, Person 1 redesign or
frontend dataset selector is needed. Frontend remains out of scope.

## Measured result and remaining work

1. The full RECIPER run and independent checks are complete: RRF+CE source-paper
   Recall@10 0.943359, MRR@10 0.830227. Full predictions and compact provenance
   have been preserved with their representation/source-label limitations.
2. Complete final held-out CSFCube experiments if needed; the existing pilot is
   development evaluation.
3. Run/review Person 4 final-verdict evaluation separately and confirm label,
   evidence-budget and data-split compatibility.
4. Present task-specific tables, source-linked errors and evidence limitations.
   Further datasets or metrics alone do not guarantee stronger model quality.

No Git push is authorized by this implementation step. Code/results stay local
until the user requests the planned push.
