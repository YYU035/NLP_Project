Current final-integration/demo update: see [FINAL_INTEGRATION.md](FINAL_INTEGRATION.md) and [DEMO_RUBRIC_GUIDE.md](DEMO_RUBRIC_GUIDE.md). Later changes include latest benchmarks, actual-model replays, efficiency instrumentation, word occlusion and additional negative IG studies. Earlier observations below remain historical.

# XAI validation and reproducible evaluation — 9 November 2026

Naila's attribution implementation stays pinned unchanged apart from isolated imports. `xai_runtime.py` adds numerical checks; `evaluation/xai_qc.py` adds gated evaluation. CY's model weights, classifier and threshold are unchanged.

## Interpretation

Positive/green attribution contributes toward the named target class relative to the chosen baseline; negative/red contributes away. It is not scientific truth, causal proof or an independent evidence score. The UI targets the raw argmax, which can differ from the thresholded verdict. Evaluation explicitly targets the gold class; on a misclassified case that does not explain the predicted class.

Exact IG completeness concerns the output difference from a baseline. Captum numerically approximates the integral; word scores need not sum exactly in practice. Repeatability needs fixed model, baseline, precision and execution settings. No universal speed advantage over LIME or SHAP is claimed.

References: [Captum](https://captum.ai/api/integrated_gradients.html), [IG paper](https://arxiv.org/abs/1703.01365), [ERASER](https://aclanthology.org/2020.acl-main.408/).

## Numerical investigation

Actual pinned CPU float32 DeBERTa, eval mode, identical synthetic claim/passage: “Plants require water to survive.” Original passage/claim order, special tokens and original attention mask are preserved. Content embeddings are interpolated from the named baseline.

| Baseline | Steps | Absolute error | Relative error | Combined gate |
| --- | ---: | ---: | ---: | --- |
| PAD | 32 | 0.227387 | 34.66% | fail |
| PAD | 128 | 0.250182 | 38.14% | fail |
| PAD | 512 | 0.021587 | 3.29% | fail: absolute error |
| MASK | 32 | 0.029505 | 3.91% | fail: absolute error |
| MASK | 128 | 0.000672 | 0.089% | pass |

This shows baseline/quadrature sensitivity; no implementation bug or general solution was established. A separate synthetic derivative audit compared autograd with central finite differences at six path points for each baseline: all 12 checks passed the declared diagnostic tolerance, and input-ID versus explicit-embedding forward probabilities matched exactly. This supports the gradient plumbing on that synthetic case; it does not prove global convergence. See `evaluation/xai-gradient-check.json`; rerun with `python -m explanation_frontend.evaluation.xai_gradient_check`. Additional MASK512 work was stopped to avoid competing with the verifier audit; no result claimed. Timings from these experiments are not a speed comparison. Details: `evaluation/xai-diagnosis.json`.

Policy fixed before the real pilot: MASK default, explicit PAD option; retries 32/128/512. Accept only finite, nontruncated results with **relative error <=5% AND absolute probability error <=0.001**. The absolute criterion is a conservative project policy, not a Captum standard. Failed cases retain diagnostic downloads but no word-score display. Scores are not renormalised to conceal error. Convergence is necessary here, not sufficient for useful explanations.

## Evaluation protocol

- Exact SciFact corpus/dev files staged from pinned Git revision, verified by SHA-256. No runtime model download or hosted inference.
- One annotated claim/document pair, excluding missing documents and mixed-label annotations. Gold rationale is the union of annotated sentence sets for that document.
- Seeded sample balanced by gold SUPPORT/CONTRADICT among length-eligible full abstracts. This is an exploratory short-input dev pilot, not representative held-out performance.
- Gold-target sentence average precision and top-1 rationale hit; 50 random sentence rankings and analytical random top-1 comparator.
- Faithfulness deletes/retains evidence subword tokens only; claim/special tokens fixed. Top signed contributions at 1/5/10/20/50 percent versus 10 seeded random selections with matched token budgets. Report AOPC comprehensiveness (higher desired) and sufficiency (lower desired); negative values are retained.
- Unstable/failed rows excluded from XAI means, with coverage always reported. Separate correct/incorrect raw predictions and gold labels. Random averages are not significance tests.
- Selected original sentence indices and actual selected-passage predictions are separate diagnostics. Missing rationale suggests selector investigation; present rationale with all selected predictions NEUTRAL suggests classifier investigation. Neither proves causation. Full-abstract errors are not blamed on selection.

## Full-abstract verifier audit

Actual native verifier, revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, CPU float32, max length512, threshold0.50:

| Gold | Predicted SUPPORT | Predicted CONTRADICT | Predicted NEUTRAL |
| --- | ---: | ---: | ---: |
| SUPPORT | 91 | 9 | 21 |
| CONTRADICT | 8 | 40 | 12 |

209 annotated pairs processed, 181 usable, 28 truncated excluded. Correct:131/181 (72.38%). SUPPORT predicted NEUTRAL:21/121 (17.36%) of usable SUPPORT pairs. This is pair-level classification on gold-cited full abstracts, not retrieval or deployed-pipeline accuracy. The quoted116/216 experiment is not reproduced by this protocol; its original split/configuration/confusion matrix remain required.

Machine-readable report:`evaluation/xai-verifier-audit.json`. Raw pair records are ignored at `outputs/xai/pilot/verifier-audit.json`. Initial IG sampling with a96-token cap yielded no eligible pairs, so no XAI metrics are claimed from that attempt; the actual pilot uses192 tokens.

## Reproduce

From the repository root with setup-pipeline dependencies/models already installed:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.xai_data
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.xai_qc --data-dir explanation_frontend/outputs/xai/data --out-dir explanation_frontend/outputs/xai/pilot192 --limit 4 --max-tokens 192 --steps 32 128 --baseline mask
```

Run the verifier audit without IG:

```powershell
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.xai_qc --data-dir explanation_frontend/outputs/xai/data --out-dir explanation_frontend/outputs/xai/verifier-audit --audit-only
```

`--audit-all` performs both. The recorded pilot caps at128 steps to bound CPU work; UI permits a further512 retry. Expand using a new output directory and explicit sample/token/step limits. Preserve the original pilot and do not silently tune baselines against rationale scores. All generated data, caches and raw attributions remain ignored inside this folder. Data staging requires the pinned source commit to exist locally; a shallow checkout may need to fetch that commit first.

## Remaining responsibilities

- [PENDING — Naila + team] Larger frozen-protocol long-input study, baseline sensitivity and stability; decide latency/coverage tradeoff.
- [PENDING — Naila + CY] Review correct/incorrect cases, raw versus thresholded NEUTRAL, and probability saturation; supply original116/216 provenance.
- [PENDING — Yu Yue + CY] Joint rationale coverage and selected-passage error analysis.
- [PENDING — Kireeti + reviewers] Human usefulness and colour/readability review. Numerical convergence does not establish scientific correctness.

## Recorded four-case pilot

Seed42, MASK,32/128 steps, full-input limit192 tokens. Eighteen of209 annotated pairs were length-eligible; four were sampled, two per gold label. Runtime566 seconds excluding initial model loading. **0/4 passed both numerical gates; no plausibility/faithfulness means are reported.** This is an important negative result, not a reason to relax the thresholds after seeing outcomes.

| Claim ID | Gold | Raw prediction | Absolute error at128 | Relative error | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| 298 | SUPPORT | SUPPORT | 0.065349 | 9.03% | unstable |
| 1088 | CONTRADICT | SUPPORT | 0.007814 | 4.92% | unstable |
| 53 | SUPPORT | SUPPORT | 0.206468 | 28.37% | unstable |
| 51 | CONTRADICT | CONTRADICT | 0.630366 | 79.43% | unstable |

Claim1088/document37549932 illustrates why input protocols matter: the full abstract's raw prediction was SUPPORT for CONTRADICT gold; a selected passage was classified CONTRADICT, with another NEUTRAL. Its gold rationale was selected. This does not establish the cause of the full-abstract error or overall pipeline correctness. Its attribution is withheld at128 steps. Raw inputs and attributions remain in the ignored `outputs/xai/pilot192` artifacts.

A targeted512-step follow-up on claim1088 is explicitly post-hoc (chosen for the smallest initial absolute error). Its outcome must be reported separately, not mixed into the frozen four-case pilot. Machine-readable summary is `evaluation/xai-results.json` and is downloadable in the attribution workspace.

## Software validation and additions ledger

145 tests passed; the final UI/report/driver subset was rerun after UI text and report-display changes (17 passed). pip check and Git whitespace checks passed. Tests include finite/tiny-gap/truncated rejection, probability parity and unchanged verifier output on a tiny local model, deterministic sampling, stable-only metric denominators, a complete tiny-model evaluation-driver run, colour-table rendering, and hiding stale results after a baseline change. These software tests are not scientific XAI validation.

Added: `xai_runtime.py` (reliability/retries); `evaluation/xai_data.py` (verified local data staging); `evaluation/xai_qc.py` (gated pilot, targeted diagnostics and optional verifier audit); `evaluation/xai_gradient_check.py` (derivative audit); numerical/audit/pilot reports; this document; UI baseline controls, safe coloured tables and downloadable validation summary; nine frontend tests. Naila/CY upstream runtime files, model weights and teammate directories were not changed. No paid/hosted API dependency was added.

## Targeted512-step follow-up outcome

Claim1088, same MASK baseline and gold CONTRADICT target: absolute error **0.00123970**, relative error **0.781%**, approximately407 seconds. Relative error improved from4.923% at128 steps, but absolute error remained narrowly above the pre-set0.001 limit. It therefore **failed the project gate**, and no plausibility/faithfulness averages were produced. This is not evidence that every reasonable numerical tolerance would reject the result; the absolute criterion is our deliberately conservative policy. We did not relax it after seeing the outcome. The case was selected post-hoc for low initial error; it cannot estimate population-level reliability.

Reproduce separately:

```powershell
.\explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend.evaluation.xai_qc --data-dir explanation_frontend/outputs/xai/data --out-dir explanation_frontend/outputs/xai/followup1088 --claim-id 1088 --limit 1 --max-tokens 192 --steps 512 --baseline mask
```

**Report conclusion:** code, UI, numerical safeguards, data provenance and evaluation plumbing are tested. General real-model attribution reliability and quantitative plausibility/faithfulness claims remain open. Next owner work is to agree a pre-registered numerical tolerance and compute budget, test convergence/baseline sensitivity on a broader fixed sample (preferably accelerated hardware), then run and report XAI quality with pass coverage and uncertainty. Do not present existing token highlights or the small-model tests as scientific validation.
