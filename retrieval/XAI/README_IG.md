# Integrated Gradients explanations for the claim verifier

This add-on explains the verifier's DeBERTa NLI predictions. For each claim–passage pair, it shows which words in the evidence and in the claim pushed the model toward SUPPORT, CONTRADICT or NEUTRAL.

It **does not modify** `verifier.py` or `schemas.py`. It wraps a `ClaimVerifier` and reuses its tokenizer, model, label mapping, max length and input order (premise = passage, hypothesis = claim). The explanation is therefore of exactly the prediction the verifier makes.

## Files (all in `claim_verifier/`)

| File | Purpose |
|---|---|
| `explain_ig.py` | `IGExplainer`, the output schemas, and `to_html()` for Streamlit |
| `eval_ig_dev.py` | Evaluation on SciFact dev: plausibility against human rationales, and ERASER faithfulness |
| `test_explain_ig.py` | 12 tests using a tiny local model (no download, no GPU) |

## Setup

```bash
pip install captum            # in addition to torch, transformers, pydantic
python -m pytest -q test_explain_ig.py
```

## Usage

```python
from verifier import ClaimVerifier
from explain_ig import IGExplainer, to_html

verifier = ClaimVerifier()
explainer = IGExplainer(verifier)                 # n_steps=50 by default

# One claim-passage pair. Default target = the model's predicted label.
exp = explainer.explain(claim, passage)           # EvidencePassage or plain string
exp.top_words(5, segment="evidence")              # words pushing hardest toward the label
exp.top_words(5, segment="claim", positive=False) # words pushing against it

# The passages that decided each paper's verdict, after verify():
response = verifier.verify(claim, papers, passages_by_paper)
explanations = explainer.explain_response(response)

# Streamlit (Kireeti): green = pushes toward the label, red = pushes away
st.markdown(to_html(exp), unsafe_allow_html=True)
```

### Output (`PassageExplanation`)

| Field | Meaning |
|---|---|
| `words` | List of `{text, segment ("evidence" or "claim"), start, end, attribution, normalized}`. `start`/`end` are character offsets into the passage text or the claim, for highlighting. `normalized` is in [-1, 1]. |
| `target_label`, `target_class` | The label being explained, e.g. `CONTRADICT` / `"contradiction"` |
| `predicted_label` | The model's top class before the verifier's confidence threshold |
| `target_probability`, `baseline_probability` | P(label) for the real input, and for the all-[PAD] baseline |
| `convergence_delta` | IG error check. Attributions should sum to `target_probability - baseline_probability`; a delta near 0 means the explanation is numerically reliable |
| `sentence_scores` | Per-sentence totals, when `sentences=[...]` is passed |
| `truncated` | Same rule as the verifier: over 512 tokens |

## Method

- **Attribution:** Captum `LayerIntegratedGradients` on the input word-embedding layer. It explains the softmax probability of the target class with 50 integration steps.
- **Baseline:** the same input with every content token replaced by `[PAD]`, keeping `[CLS]` and `[SEP]`. By IG's completeness property, attributions then add up exactly to how much the words raise the label's probability over an empty input.
- **Precision:** on GPU the verifier runs in float16, where gradients are unreliable. IG therefore runs on a separate float32 copy of the model, using about 1.7 GB of extra GPU memory for DeBERTa-v3-large. The verifier's own model is left untouched.
- **Word level:** sub-word token attributions are summed into whole words.

## Evaluation on SciFact dev

```bash
python eval_ig_dev.py --limit 50     # quick check
python eval_ig_dev.py                # all dev claim-abstract pairs
```

Each example uses the **full abstract** as evidence, with the gold label as the target.

**Plausibility:** do IG's top sentences match the rationale sentences humans annotated?
- Mean average precision (AP) of sentence scores
- Top-1 hit rate
- Both compared with random rankings

**Faithfulness** ([ERASER](https://aclanthology.org/2020.acl-main.408/), DeYoung et al., 2020): does the model really depend on the high-attribution tokens? Tokens are deleted, and the score is averaged over 1/5/10/20/50% of tokens (AOPC).
- *Comprehensiveness:* the drop in P(label) when the top tokens are removed. Should be **higher than random**.
- *Sufficiency:* the drop in P(label) when only the top tokens are kept. Should be **lower than random**.

Results are split by all examples, correct vs incorrect predictions, and SUPPORT vs CONTRADICT. Outputs go to `ig_outputs/`: `summary.md` (table for the report), `summary.json`, and `per_example.jsonl` (for picking qualitative examples).

**Runtime:** use a GPU (a Colab T4 is fine). Expect a few seconds per example with DeBERTa-v3-large, so a full dev run takes tens of minutes. If you run out of memory, lower `--internal-batch-size` to 4 or 2. If the reported relative delta is above about 0.05, raise `--n-steps` to 100.

## Limitations

- IG explains what the model relied on, not whether that reasoning is scientifically correct. Plausibility and faithfulness are reported separately for that reason.
- Attributions depend on the baseline choice. `[PAD]` is standard; `IGExplainer(verifier, baseline="mask")` is available as an ablation.
- Truncated passages are explained but excluded from the evaluation, matching how the verifier excludes them from aggregation.
- Sentence scores are sums over tokens, so longer sentences can collect more attribution.
