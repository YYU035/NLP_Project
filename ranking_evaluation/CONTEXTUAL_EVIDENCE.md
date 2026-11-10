# Optional bounded abstract context for verification

`BoundedAbstractSelector` is an opt-in adapter around the existing semantic sentence selector. It is intended to avoid removing qualifying context before NLI verification. It does not change model weights, relevance thresholds, retrieval rankings, the original claim, the verifier, or frontend defaults.

The wrapped selector first supplies its usual relevance anchors. If it abstains, the adapter also abstains. If the exact original abstract fits the actual verifier's evidence-premise/claim-hypothesis pair budget, the adapter returns one source-preserving whole-abstract passage. If it does not fit, the adapter returns the original baseline passages. The ordinary downstream verifier-budget checks must still run. No truncation or label-based selection occurs.

## Backend use

```python
from ranking_evaluation.contextual_evidence import BoundedAbstractSelector
import json
from pathlib import Path
from ranking_evaluation.semantic_evidence import (
    LocalSemanticSelector, SelectionConfig, select_ranked_evidence,
)

base = LocalSemanticSelector(
    model_path,
    config=SelectionConfig(top_k=3, min_score=0.5, context_radius=0),
)
frozen = json.loads(Path(
    'ranking_evaluation/experiments/contextual-accuracy-train-frozen.json'
).read_text())
selector = BoundedAbstractSelector(
    base, verifier.tokenizer, min_paper_score=frozen['min_paper_score'],
)
result = select_ranked_evidence(
    ranking, selector, tokenizer=verifier.tokenizer, paper_depth=5,
)
# Persist this metadata: base selection config alone does not describe context.
result['context_configuration'] = selector.context_configuration
```

Use the pinned semantic encoder and actual pinned verifier tokenizer from the experiment, not another model's tokenizer. The inherited `config` describes only the underlying anchor selector; `context_configuration` explicitly identifies the context policy, budget and fallback and must accompany exported runs.

This is a verifier-context policy. Returning an entire abstract is not a claim that every sentence is annotated evidence. Evaluate full input evidence precision/recall honestly; do not score only the short anchors while supplying the longer abstract to the verifier. Any UI highlighting or token attribution is a separate task.

The optional `min_paper_score` is an additional paper gate over the maximum
relevance of budget-eligible passages. Below-threshold papers abstain before
verification. Its default is `None`, retaining the wrapped selector's ordinary
gate. The experimental frozen configuration selects 0.65 from training data;
it is not an NLI confidence threshold or truth probability. Record this setting
explicitly and retain the original baseline when integrating the adapter.

## Validation and experiment records

The adapter is tested against the evaluated policy at the exact 512-token boundary and the 513-token fallback, including original whitespace, abstention and source offsets. The original frozen selector remains available and unchanged.

The training/development comparison runner, source hashes, individual case records and report are under `ranking_evaluation/analysis/kireeti_validation_20261110/`. Use the completed `FULL_ACCURACY_RESULTS.md` report for measured accuracy rather than assuming more context improves every claim. Development data is already inspected; independent test evaluation remains necessary for generalization claims. No frontend integration is performed by this adapter.
