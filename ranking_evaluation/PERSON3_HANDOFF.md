# Person 3 handoff

**Status:** Implementation and automated verification are complete. Human PDF
review is pending; do not mark the full task closed or invent reviewer approval.

Implemented scope: hybrid ranking, optional reranking, source-linked sentence and
context selection, evaluation/ablations, Person 4 handoff and local frontend
integration. Models are pretrained; selector settings are selected on train only.

## Delivered interfaces

- `pipeline.rank_claim`: BM25/SBERT ordered identities to source-preserving RRF
  output; optional explicitly supplied Cross-Encoder.
- `semantic_evidence.select_scored_sentences`: original source sentences and
  finite relevance scores to stable evidence spans, with threshold abstention
  and optional neighbouring context.
- `semantic_evidence.LocalSemanticSelector`: offline MiniLM sentence scoring;
  dataset callers pass original sentence lists; live abstracts use heuristic
  boundaries while preserving exact original character offsets.
- `semantic_evidence.select_ranked_evidence`: RankingOutput to source-linked
  evidence with original claim, paper metadata and explicit per-paper status.
- `semantic_evidence.person4_passages`: exact native EvidencePassage fields:
  passage_id, paper_id, section, text and score.
- `select_evidence` CLI: validated RankingOutput plus frozen config/local model
  to a complete JSON handoff. New output paths are required.

## Completed checks

| Item | Evidence |
| --- | --- |
| BM25/SBERT integration | Pinned Naila commit 2bc7d81; complete-corpus retrieval, all dev claims retained |
| Ranking metrics | Recall@K, MRR/MRR@K, nDCG@K, stage latency and per-claim records |
| Ablations | BM25, SBERT, RRF; rerank depths 10/20/50; RRF constants 20/60/100 |
| GPU inference | A40 and Pro 6000 smoke checks; complete dev comparison on Pro 6000 |
| Evidence baselines | Three-sentence, six-sentence, two-sentence-window and original frontend selector |
| Semantic evidence | Train-only 200-claim deterministic configuration search; frozen dev sentence/context ablations |
| Gold/reference handling | Original sentence IDs; all rationale alternatives retained; no-gold claims excluded with counts |
| Source integrity | Exact character spans, paper IDs, original claims and metadata preserved |
| Person 4 contract | Real dev-claim payload validated against native schema and original text |
| Local frontend | Opt-in semantic hook; recorded/live real-model checks; explicit partial-provider state retained |
| Tests | 126 module tests; 117 frontend tests after adapter installation |

## Measured development results

The complete SciFact dev split contains 300 claims: 188 with gold evidence and
112 without. These are development results, not untouched final testing.

| Retrieval | MRR@10 | Recall@10 | nDCG@10 |
| --- | --- | --- | --- |
| BM25 | 0.7844 | 0.9164 | 0.8139 |
| SBERT | 0.7314 | 0.9117 | 0.7743 |
| RRF k=60 | 0.8164 | 0.9591 | 0.8458 |
| RRF + CE top 20 | 0.8561 | 0.9498 | 0.8736 |
| RRF + CE top 50 | 0.8550 | 0.9605 | 0.8761 |

Train-selected evidence settings: three anchors, cosine threshold 0.5, no added
context by default. Context radius 1 remains a measured alternative.

| Selector, paper input | Sentence precision | Recall | F1 |
| --- | --- | --- | --- |
| Original frontend, gold papers | 0.2532 | 0.6421 | 0.3632 |
| Semantic sentence, gold papers | 0.3748 | 0.5437 | 0.4437 |
| Original frontend, RRF top 5 | 0.0492 | 0.5656 | 0.0906 |
| Semantic sentence, RRF top 5 | 0.1093 | 0.4918 | 0.1788 |
| Semantic sentence, CE top 20 then top 5 | 0.1229 | 0.5109 | 0.1982 |
| Semantic context, RRF top 5 | 0.0706 | 0.6776 | 0.1279 |

The gain in strict F1 comes with lower recall. Context raises recall with more
non-rationale text. These are custom exact-sentence union diagnostics, not the
official SciFact evidence scorer or NLI accuracy. Complete-rationale recovery
is separately reported. Annotation incompleteness can penalize relevant text.

## Reproduction and frontend activation

See README.md for every run command. Provision semantic weights explicitly on
SSD, with caches redirected there:

```sh
python -m ranking_evaluation.scripts.setup_semantic_model \
  --config ranking_evaluation/experiments/semantic-train-frozen.json \
  --model-dir /path/to/new-local-model
python -m ranking_evaluation.scripts.install_frontend_adapter \
  --frontend-root /path/to/kireeti-checkout
export PERSON3_EVIDENCE_CONFIG=/absolute/path/to/semantic-train-frozen.json
export PERSON3_EVIDENCE_MODEL_PATH=/path/to/new-local-model
```

Launch the frontend using its environment. Optional script
`scripts/run_frontend_person3.sh` additionally takes FRONTEND_ROOT,
FRONTEND_PYTHON and optional FRONTEND_PORT. Default port is 18503.
No remote branch is pushed by setup or adapter installation. Without the config
variable the previous frontend selector remains available.

## Artifacts on this workspace

- Final readable report: `outputs/ranking_evaluation/person3-final/PERSON3_REPORT.md`.
- Native handoff example and source/schema validation: same directory.
- Retrieval experiment: `outputs/ranking_evaluation/scifact-dev-276357/comparison/`.
- Compact constant ablation: `outputs/ranking_evaluation/rrf-constant-ablation/`.
- Evidence baselines: `evidence-dev-277879/results/` and `evidence-dev-277880/results/` under the same output root.
- Train search, frozen configuration and semantic dev diagnostics:
  `outputs/ranking_evaluation/semantic-evidence-277884/results/`.
- Models, data, environment files and large per-claim outputs remain on SSD;
  source references and hashes make the runs reproducible.

## Remaining team boundaries

NLI calibration/accuracy belongs to Person 4; human explanation review to Person
5/reviewers. PDF acquisition and a multiple-dataset grading decision require
team agreement. Dev has been used across the project; the team must agree an
untouched final protocol before claiming held-out results. These do not prevent
using the completed Person 3 interfaces and reproducing the reported experiments.

The latest real Live smoke found six candidates and verified three papers/nine
passages. OpenAlex/arXiv succeeded while OpenAIRE had a transport failure. The
application correctly recorded partial status; it was not replaced by replay.


## Checklist correction and v2 additions

The earlier delivery covered abstracts and was not the complete task checklist.
Full-text/PDF extraction and page references are now implemented; three requested
failure categories have actual annotated cases; the entire frozen prediction
set was checked against the actual 512-token Person 4 pair tokenizer. A separate
PDF packet has 2 papers, 6 claims and 12 source-linked passages tested through
Chong Yao's original verifier, with zero truncation. It remains **human-review
pending** until a named reviewer completes its blank ratings.

See outputs/ranking_evaluation/pdf-review-v2/index.html and shared-results-v1/.
The shared table is ready; teammate independent runs/confirmations are not
fabricated. AI visual prechecks are not counted as human approval.


### Additional v2 results and limits

The complete budget audit checked 39,308 passage instances with the actual NLI
tokenizer: maximum accepted length 484/512, zero benchmark rejections. Synthetic
512/513 boundary tests and a 907-token real-tokenizer stress input verify rejection.
The frontend now records budget checks and gives an operational failure if all
passages are rejected. Neither original claim nor source evidence is shortened.

Optional **passage-level** Cross-Encoder on/off is now implemented and measured,
separately from document reranking. With the same semantic candidates and top-1
per-paper output budget, RRF top-5 sentence F1 was 0.1868 off / 0.2226 on;
with document rerank top-20 then top-5 papers it was 0.2045 / 0.2399. These are
fixed development ablations, not a dev-selected production setting.

Actual failure analysis includes 35 relevant-paper/non-rationale-selection
cases, 3 incomplete-rationale cases, 23 cases whose annotated rationale was
recovered by context, and 28 missed gold-document cases. Categories overlap.
See token-budget-278004/failure-analysis/ under repository outputs.
