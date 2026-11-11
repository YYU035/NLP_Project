"""Local semantic sentence ranking, optional source-preserving context and handoff."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
import re

from research_verifier.evidence.spans import EvidenceSpan


@dataclass(frozen=True)
class SelectionConfig:
    top_k: int = 1
    min_score: float = 0.3
    context_radius: int = 0

    def __post_init__(self):
        if type(self.top_k) is not int or not 1 <= self.top_k <= 10:
            raise ValueError('top_k must be between 1 and 10')
        if type(self.context_radius) is not int or self.context_radius not in (0, 1):
            raise ValueError('context_radius must be 0 or 1')
        if isinstance(self.min_score, bool) or not isfinite(self.min_score) or not -1 <= self.min_score <= 1:
            raise ValueError('min_score must be a finite cosine threshold in [-1, 1]')


def select_scored_sentences(paper_id, sentences, scores, *, config=SelectionConfig()):
    """Select original sentences from finite relevance scores, never gold labels.

    Context radius 1 adds one original sentence on each side. Overlapping
    intervals are merged; scores are relevance values, not truth probabilities.
    Below-threshold papers yield no passages rather than a fabricated verdict.
    """
    if not paper_id or isinstance(sentences, str) or any(not isinstance(s, str) for s in sentences):
        raise ValueError('Expected paper identity and original sentence list')
    if len(scores) != len(sentences) or any(isinstance(s, bool) or not isfinite(float(s)) for s in scores):
        raise ValueError('Expected one finite score per sentence')
    anchors = sorted((i for i, text in enumerate(sentences)
                      if text.strip() and float(scores[i]) >= config.min_score),
                     key=lambda i: (-float(scores[i]), i))[:config.top_k]
    intervals = sorted((max(0, i-config.context_radius),
                        min(len(sentences), i+config.context_radius+1), float(scores[i])) for i in anchors)
    merged = []
    for start, end, score in intervals:
        if merged and start < merged[-1][1]:
            old = merged[-1]
            merged[-1] = (old[0], max(end, old[1]), max(score, old[2]))
        else:
            merged.append((start, end, score))
    offsets, pos = [], 0
    for sentence in sentences:
        offsets.append(pos)
        pos += len(sentence)+1
    body = ' '.join(sentences)
    output = []
    for first, after, score in merged:
        start, end = offsets[first], offsets[after-1]+len(sentences[after-1])
        text = body[start:end]
        pid = f'{paper_id}:abstract:{start}:{end}:{sha256(text.encode()).hexdigest()[:12]}'
        output.append(EvidenceSpan(paper_id, tuple(range(first, after)), start, end, text, score, pid))
    return sorted(output, key=lambda span: (-span.score, span.start))


class LocalSemanticSelector:
    """Reuse one offline CPU encoder across claims; weights provisioned explicitly."""
    def __init__(self, model_path, *, config=SelectionConfig(), device='cpu'):
        if not Path(model_path).is_dir():
            raise ValueError('An existing local model directory is required')
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(str(Path(model_path).resolve()), device=device,
                                          local_files_only=True, trust_remote_code=False)
        self.encoder.max_seq_length = 256
        self.config = config

    def select(self, claim, paper_id, sentences):
        if not claim.strip():
            raise ValueError('Claim must be nonempty')
        if not sentences:
            return []
        vectors = self.encoder.encode([claim, *sentences], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        scores = vectors[1:] @ vectors[0]
        return select_scored_sentences(paper_id, sentences, scores, config=self.config)

    def select_abstract(self, claim, paper_id, abstract):
        """Live abstracts only: heuristic boundaries, exact original character spans.

        Benchmark callers must instead supply the dataset's original sentences.
        All whitespace, numbers and qualifiers remain unchanged in output text.
        """
        if not abstract or not abstract.strip():
            return []
        boundaries = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9])', abstract)], len(abstract)]
        source = []
        for a, b in zip(boundaries, boundaries[1:]):
            raw = abstract[a:b]
            start, end = a+len(raw)-len(raw.lstrip()), a+len(raw.rstrip())
            if start < end:
                source.append((start, end))
        sentences = [abstract[a:b] for a, b in source]
        output = []
        for span in self.select(claim, paper_id, sentences):
            start = source[span.sentence_indices[0]][0]
            end = source[span.sentence_indices[-1]][1]
            text = abstract[start:end]
            pid = f'{paper_id}:abstract:{start}:{end}:{sha256(text.encode()).hexdigest()[:12]}'
            output.append(EvidenceSpan(paper_id, span.sentence_indices, start, end, text, span.score, pid))
        return output


def person4_passages(spans):
    """Native Person 4 EvidencePassage dictionaries; source IDs remain embedded."""
    identities = [span.passage_id for span in spans]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate evidence passage identity')
    return [{'passage_id': s.passage_id, 'paper_id': s.paper_id, 'section': 'abstract',
             'text': s.text, 'score': float(s.score)} for s in spans]


def select_ranked_evidence(ranking, selector, *, tokenizer, paper_depth=5):
    """Preserve upstream metadata and exact original claim for team handoff."""
    if type(paper_depth) is not int or paper_depth <= 0:
        raise ValueError('paper_depth must be positive')
    from research_verifier.evidence.budget import enforce_verifier_budget, VERIFIER_CONTRACT
    claim = ranking.inputs.candidates.original_claim
    records = []
    for row in ranking.papers[:paper_depth]:
        entry = row.entry
        if entry.original_claim != claim:
            raise ValueError('Claim mismatch in ranked paper')
        spans = selector.select_abstract(claim, entry.identity, entry.paper.abstract)
        decision = enforce_verifier_budget(claim, spans, tokenizer)
        spans = list(decision.accepted)
        records.append({'paper_id': entry.identity, 'title': entry.paper.title or '',
                        'rank': row.rank, 'source_entry': entry.model_dump(mode='json'),
                        'evidence_type': 'abstract', 'status': 'selected' if spans else 'verifier_budget_rejected' if decision.rejected else 'no_passage_above_threshold',
                        'budget_checks': list(decision.checked), 'rejected_passages': list(decision.rejected),
                        'passages': person4_passages(spans), 'source_spans': [asdict(s) for s in spans]})
    return {'schema_version': 'person3-evidence-1', 'claim': claim,
            'configuration': asdict(selector.config), 'verifier_contract': VERIFIER_CONTRACT, 'papers': records,
            'limitations': ['Abstract only; no PDF extraction',
                           'Live sentence boundaries are heuristic; dataset sentence lists must be supplied explicitly',
                           'Cosine scores are relevance values, not verification probabilities',
                           'Over-budget passages are rejected whole before handoff; rejections are explicit']}
