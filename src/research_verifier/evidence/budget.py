"""Preflight the exact Person 4 evidence-premise/original-claim token pair."""

from dataclasses import dataclass

VERIFIER_CONTRACT = {
    'owner': 'Chong Yao',
    'source_commit': '1ac06b675e3d0719a63cb8efe98eda4f2c5fd576',
    'source_file': 'claim_verifier/verifier.py',
    'max_pair_tokens': 512,
    'tokenizer_checkpoint': 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    'tokenizer_revision': 'b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7',
    'orientation': 'premise=evidence text; hypothesis=original claim',
    'special_tokens': True,
    'policy': 'reject complete over-budget passages; never trim source text or claim',
}


@dataclass(frozen=True)
class BudgetDecision:
    accepted: tuple
    rejected: tuple[dict, ...]
    checked: tuple[dict, ...]
    max_pair_tokens: int


def enforce_verifier_budget(claim, spans, tokenizer, *, max_pair_tokens=512):
    """No neural inference or truncating encode is allowed at this boundary."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError('Original claim must be nonempty')
    if type(max_pair_tokens) is not int or not 0 < max_pair_tokens <= 512:
        raise ValueError('Pair budget must be between 1 and the verified limit 512')
    if tokenizer is None:
        raise ValueError('The actual verifier tokenizer is required')
    spans = list(spans)
    claim_tokens = len(tokenizer.encode('', claim, add_special_tokens=True, truncation=False))
    if claim_tokens > max_pair_tokens and not spans:
        failure = {'passage_id': None, 'pair_tokens': claim_tokens, 'max_pair_tokens': max_pair_tokens,
                   'accepted': False, 'reason': 'original_claim_exceeds_pair_budget',
                   'text_modified': False, 'claim_modified': False}
        return BudgetDecision((), (failure,), (failure,), max_pair_tokens)
    accepted, rejected, checked = [], [], []
    for span in spans:
        count = len(tokenizer.encode(span.text, claim, add_special_tokens=True, truncation=False))
        reason = ('original_claim_exceeds_pair_budget' if claim_tokens > max_pair_tokens
                  else 'passage_pair_exceeds_budget' if count > max_pair_tokens else None)
        record = {'passage_id': span.passage_id, 'pair_tokens': count,
                  'max_pair_tokens': max_pair_tokens, 'accepted': reason is None,
                  'reason': reason, 'text_modified': False, 'claim_modified': False}
        checked.append(record)
        if reason: rejected.append(record)
        else: accepted.append(span)
    return BudgetDecision(tuple(accepted), tuple(rejected), tuple(checked), max_pair_tokens)
