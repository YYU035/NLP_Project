# Full-pipeline context comparison: execution handoff

## Current update: local CPU inference completed

The pinned NLI weights are now available in the ignored local cache `.cache/ranking/verifier-model`. All 30 condition runs completed on CPU float32 without operational failures. New baseline verdicts matched the archive on all 15 claims. Baseline accuracy was 3/15 and candidate accuracy 4/15 on this failure-enriched diagnostic sample. See `CPU_REPLAY_RESULTS.md` for actual metrics and timing. Full 300-claim evaluation still requires the complete frozen ranking lists. The following preparation status describes the earlier handoff, before local model provisioning.

## Original preparation status

The runner `pipeline_context.py` is implemented. Real semantic inference prepared baseline and train-selected candidate evidence for all 15 archived replay claims (30 condition rows), using fixed selected papers, exact source spans and actual pinned verifier tokenization. All 15 newly encoded baseline evidence scores match the archived full-route results. Prepared inputs are in `prepared-replays/`, with data, policy, tokenizer and input hashes. These cases are a selected diagnostic sample, not full-dataset results.

No new NLI inference has run. Therefore there are no new verdict Accuracy, Macro-F1 or model latency numbers. The local environment has the tokenizer and semantic encoder but lacks full pinned DeBERTa weights. A batch-mode read-only SSH check found authentication failure on the configured cluster and connection refusal on the other configured GPU host. No remote files were changed and no jobs submitted. No production/frontend configuration was changed or pushed.

## Run on the evaluation host

Use the existing model/evaluation environment, with this branch's experiment files available. Do not run neural evaluation on a cluster login node; obtain a compute allocation according to the host's normal procedure.

For a first diagnostic pass, use the already prepared 15-case inputs. Both baseline and candidate must rerun on the same device. The NLI directory must contain the pinned checkpoint's `model.safetensors`, config and tokenizer files; the runner verifies the known weight SHA-256 before inference.

```bash
python ranking_evaluation/analysis/kireeti_validation_20261110/pipeline_context.py infer \
  --prepared ranking_evaluation/analysis/kireeti_validation_20261110/prepared-replays/prepared.jsonl \
  --nli-model explanation_frontend/.models/claim-verifier \
  --device cuda \
  --output outputs/ranking_evaluation/context-replay-nli
```

Use a new output directory for each run. Model loading is offline; no automatic large downloads occur. The runner keeps float32, seed 42, threshold 0.5, the owner's verification methods and the frontend's conflict-preserving aggregation. It writes every condition's prediction, selected evidence, passage/paper verdicts, budget checks and failures. Operational failures remain in accuracy accounting.

## Full 300-claim comparison

Locate the original evaluation-host `rankings.json` created by `explanation_frontend.evaluation.validation.scifact`, containing `bm25`, `sbert`, and `rrf` maps for all 300 claim IDs. These rankings are not in the checked-in 15-case replay archive. Use the original frozen retrieval run and preserve its source/protocol provenance; the runner records its file hash and validates claim coverage, paper identities and ranking uniqueness. Shape validation alone cannot prove ranking provenance, so retain the original evaluation-host manifest with the results.

Prepare evidence for all 300 claims, then run actual NLI:

```bash
python ranking_evaluation/analysis/kireeti_validation_20261110/pipeline_context.py prepare \
  --rankings /path/to/original/scifact_full/rankings.json \
  --encoder /path/to/verified/person3-semantic \
  --tokenizer explanation_frontend/.models/claim-verifier \
  --device cuda \
  --output outputs/ranking_evaluation/context-full-prepared

python ranking_evaluation/analysis/kireeti_validation_20261110/pipeline_context.py infer \
  --prepared outputs/ranking_evaluation/context-full-prepared/prepared.jsonl \
  --nli-model explanation_frontend/.models/claim-verifier \
  --device cuda \
  --output outputs/ranking_evaluation/context-full-nli
```

The encoder directory must have a `download_manifest.json` for pinned semantic revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; every listed model file is verified. Provision SciFact under `data/scifact/` with frozen corpus/dev hashes. Both baseline and candidate use RRF top five. This experiment isolates the context policy; it does not change paper depth or live discovery.

Outputs include evidence Precision/Recall/F1 on annotated-positive claims, complete rationale document counts, all-claim and positive-only Accuracy/Macro-F1, processing failures, and measured verification/aggregation time. Timing excludes preparation, retrieval, model loading and frontend rendering. Both GPU conditions rerun rather than comparing GPU latency against historical CPU latency. The runner records device and package versions; GPU floating-point results may differ from historical CPU predictions.

## Checks completed locally

Sixteen selector, budget and ranking-validation tests passed; one optional integration test skipped. Newly added validation rejects incomplete rankings, duplicate ranked IDs and unknown paper IDs. Original production code and frozen settings remain unchanged. Full NLI execution is untested until weights and a compute host are available.
