"""Opt-in whole-abstract verifier context with the existing relevance gate.

This adapter does not change the frozen sentence selector or frontend defaults.
It retains source text and falls back to the base selector for long abstracts.
"""
from hashlib import sha256
from dataclasses import asdict
import re
from math import isfinite

from research_verifier.evidence.spans import EvidenceSpan
from research_verifier.evidence.budget import enforce_verifier_budget


class BoundedAbstractSelector:
    """Wrap a semantic selector while preserving its configured relevance gate.

    The actual verifier tokenizer is required. Each selected whole abstract must
    fit the evidence-premise/original-claim pair budget, including special tokens.
    Existing downstream budget checks still apply to baseline fallback passages.
    """

    def __init__(self, base_selector, tokenizer, *, max_pair_tokens=512,
                 min_paper_score=None):
        if tokenizer is None:
            raise ValueError('The actual verifier tokenizer is required')
        if type(max_pair_tokens) is not int or not 0 < max_pair_tokens <= 512:
            raise ValueError('Pair budget must be between 1 and 512')
        if min_paper_score is not None and (isinstance(min_paper_score, bool) or
                not isfinite(min_paper_score) or not -1 <= min_paper_score <= 1):
            raise ValueError('Paper relevance gate must be a finite cosine threshold')
        self.base_selector = base_selector
        self.tokenizer = tokenizer
        self.max_pair_tokens = max_pair_tokens
        self.min_paper_score = min_paper_score
        self.config = base_selector.config

    @property
    def context_configuration(self):
        """Record separately from the wrapped sentence selector configuration."""
        return {'policy': 'bounded_abstract', 'base_selection': asdict(self.config),
                'max_pair_tokens': self.max_pair_tokens,
                'min_paper_score': self.min_paper_score,
                'over_budget_policy': 'unchanged_baseline_passages',
                'text_policy': 'exact_whole_original_abstract'}

    def select_abstract(self, claim, paper_id, abstract):
        """Return original-source context, abstain, or unchanged baseline spans."""
        baseline = self.base_selector.select_abstract(claim, paper_id, abstract)
        if not baseline:
            return []
        if any(abstract[s.start:s.end] != s.text for s in baseline):
            raise ValueError('Base selector must preserve exact source spans')
        count = len(self.tokenizer.encode(abstract, claim, add_special_tokens=True,
                                          truncation=False))
        if count > self.max_pair_tokens:
            return self._gate(claim, baseline)
        cuts = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9])', abstract)], len(abstract)]
        segment_count = sum(bool(abstract[a:b].strip()) for a, b in zip(cuts, cuts[1:]))
        passage_id = f'{paper_id}:abstract:0:{len(abstract)}:{sha256(abstract.encode()).hexdigest()[:12]}'
        return self._gate(claim, [EvidenceSpan(
            paper_id, tuple(range(segment_count)), 0, len(abstract), abstract,
            max(s.score for s in baseline), passage_id)])

    def _gate(self, claim, proposed):
        if self.min_paper_score is None:
            return proposed
        eligible = enforce_verifier_budget(
            claim, proposed, self.tokenizer, max_pair_tokens=self.max_pair_tokens).accepted
        if max((s.score for s in eligible), default=-2.) < self.min_paper_score:
            return []
        return proposed
