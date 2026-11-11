"""
explain_ig.py
Integrated Gradients (IG) explanations for the ClaimVerifier's DeBERTa NLI model.

Answers: "Which words in the evidence and the claim pushed the model toward
SUPPORT / CONTRADICT / NEUTRAL?"

Design rules
- Does NOT modify verifier.py or schemas.py. It reuses the verifier's own
  tokenizer, model, label mapping, max length and input order
  (premise = passage text, hypothesis = claim), so it explains exactly the
  prediction the verifier makes.
- On GPU the verifier runs in float16. Gradients in float16 are unreliable,
  so by default IG runs on a float32 copy of the model (about +1.7 GB on GPU
  for DeBERTa-v3-large). The copy's prediction is reported so any fp16/fp32
  disagreement is visible rather than hidden.
- Attribution is computed with Captum's LayerIntegratedGradients on the
  word-embedding layer. Baseline = the same input with every non-special
  token replaced by [PAD]; [CLS]/[SEP] are kept, so attributions sum to
  P(target | input) - P(target | baseline)  (the IG completeness axiom).

Quick use
    from ..claim_verifier.verifier import ClaimVerifier
    from explain_ig import IGExplainer, to_html

    verifier = ClaimVerifier()
    explainer = IGExplainer(verifier)
    exp = explainer.explain(claim, passage)          # EvidencePassage or str
    exp.words                                        # word-level attributions
    st.markdown(to_html(exp), unsafe_allow_html=True)

    # Explain the passages that drove each paper's verdict:
    explanations = explainer.explain_response(verification_response)
"""

from __future__ import annotations

import copy
import html
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from pydantic import BaseModel, Field

from research_verifier.verification.schemas import EvidencePassage, PassageLabel, VerificationResponse
from research_verifier.verification.engine import ClaimVerifier

EVIDENCE, CLAIM = "evidence", "claim"


# --------------------------------------------------------------------------- schemas
class WordAttribution(BaseModel):
    text: str
    segment: str                      # "evidence" | "claim"
    start: Optional[int] = None       # char offsets into the passage text or the claim
    end: Optional[int] = None
    attribution: float                # signed; > 0 pushes toward the target label
    normalized: float                 # attribution / max |attribution| in this pair, in [-1, 1]


class PassageExplanation(BaseModel):
    passage_id: str
    paper_id: str
    claim: str
    evidence_text: str
    target_label: PassageLabel        # label being explained
    target_class: str                 # model's raw class name, e.g. "contradiction"
    predicted_label: PassageLabel     # fp32 model argmax (before the verifier's threshold)
    target_probability: float         # P(target | input)
    baseline_probability: float       # P(target | all-[PAD] baseline)
    convergence_delta: float          # sum(attr) - (P_input - P_baseline); near 0 = reliable
    truncated: bool
    words: List[WordAttribution]
    sentence_scores: Optional[List[float]] = None   # set when sentences were passed in
    settings: Dict[str, Any] = Field(default_factory=dict)

    def top_words(self, n: int = 5, segment: Optional[str] = None,
                  positive: bool = True) -> List[WordAttribution]:
        ws = [w for w in self.words if segment is None or w.segment == segment]
        return sorted(ws, key=lambda w: w.attribution, reverse=positive)[:n]


# --------------------------------------------------------------------------- explainer
class IGExplainer:
    def __init__(
        self,
        verifier: ClaimVerifier,
        n_steps: int = 50,
        internal_batch_size: int = 8,
        explain_in_fp32: bool = True,
        baseline: str = "pad",          # "pad" | "mask" | "unk"
    ):
        self.verifier = verifier
        self.tokenizer = verifier.tokenizer
        self.device = verifier.device
        self.max_length = verifier.max_length
        self.n_steps, self.internal_batch_size, self.baseline = n_steps, internal_batch_size, baseline

        model = verifier.model
        if explain_in_fp32 and next(model.parameters()).dtype != torch.float32:
            model = copy.deepcopy(model).float()
        self.model = model.eval()
        self.model_dtype = str(next(self.model.parameters()).dtype)

        # PassageLabel -> model class index (first index wins if several map to one label)
        self.label_to_idx: Dict[PassageLabel, int] = {}
        for idx in sorted(verifier.label_mapping):
            self.label_to_idx.setdefault(verifier.label_mapping[idx], idx)
        missing = set(PassageLabel) - set(self.label_to_idx)
        if missing:
            raise ValueError(f"Model has no class for {missing}; id2label={self.model.config.id2label}")

        self.baseline_token_id = {
            "pad": self.tokenizer.pad_token_id,
            "mask": self.tokenizer.mask_token_id,
            "unk": self.tokenizer.unk_token_id,
        }[baseline]
        if self.baseline_token_id is None:
            raise ValueError(f"Tokenizer has no {baseline} token")

        self.lig = LayerIntegratedGradients(self._forward, self.model.get_input_embeddings())

    def settings(self) -> Dict[str, Any]:
        return {
            "method": "LayerIntegratedGradients (Captum)",
            "layer": "input word embeddings",
            "baseline": f"{self.baseline} for non-special tokens; special tokens kept",
            "output_explained": "softmax probability of target class",
            "n_steps": self.n_steps,
            "explain_dtype": self.model_dtype,
            "checkpoint": self.verifier.checkpoint,
            "max_length": self.max_length,
        }

    # ------------------------------------------------------------------ model plumbing
    def _forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return F.softmax(self.model(**kwargs).logits.float(), dim=-1)

    def _encode(self, premise: str, claim: str) -> Dict[str, Any]:
        """Tokenise exactly as ClaimVerifier._classify_passage does."""
        fast = getattr(self.tokenizer, "is_fast", False)
        enc = self.tokenizer(premise, claim, max_length=self.max_length, truncation=True,
                             return_tensors="pt", return_offsets_mapping=fast)
        offsets = enc.pop("offset_mapping")[0].tolist() if "offset_mapping" in enc else None
        ids = enc["input_ids"][0].tolist()
        if fast:
            seq_ids = enc.sequence_ids(0)
            word_ids = enc.word_ids(0)
        else:
            seq_ids, word_ids = self._segments_from_special_mask(ids), None
        full_len = len(self.tokenizer.encode(premise, claim, add_special_tokens=True, truncation=False))
        return {
            "tensors": {k: v.to(self.device) for k, v in enc.items()},
            "ids": ids, "seq_ids": seq_ids, "word_ids": word_ids, "offsets": offsets,
            "truncated": full_len > self.max_length,
        }

    def _segments_from_special_mask(self, ids: List[int]) -> List[Optional[int]]:
        special = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        seg, out, seen_content = 0, [], False
        for s in special:
            if s:
                out.append(None)
                if seen_content:
                    seg, seen_content = 1, False
            else:
                out.append(seg)
                seen_content = True
        return out

    def _target_index(self, probs: torch.Tensor, target: Optional[PassageLabel]) -> Tuple[int, PassageLabel]:
        pred_idx = int(probs.argmax())
        pred_label = self.verifier.label_mapping.get(pred_idx, PassageLabel.NEUTRAL)
        if target is None:
            return pred_idx, pred_label
        return self.label_to_idx[PassageLabel(target)], pred_label

    # ------------------------------------------------------------------ public API
    def explain(
        self,
        claim: str,
        passage: Union[EvidencePassage, str],
        target: Optional[PassageLabel] = None,
        sentences: Optional[Sequence[str]] = None,
    ) -> PassageExplanation:
        """Explain one claim-passage pair.

        target: label to explain. Default = the model's predicted label.
        sentences: optional list of evidence sentences. If given, the premise is
            " ".join(sentences) and per-sentence scores are returned (used for
            the plausibility check against SciFact rationales).
        """
        if isinstance(passage, str):
            passage = EvidencePassage(passage_id="p0", paper_id="", section="abstract", text=passage)
        premise = " ".join(sentences) if sentences is not None else passage.text
        if not premise.strip() or not claim.strip():
            raise ValueError("Empty passage or claim")

        run = self._run(premise, claim, target)
        return self._build_explanation(run, passage, claim, premise, sentences)

    def explain_with_faithfulness(
        self,
        claim: str,
        passage: Union[EvidencePassage, str],
        target: Optional[PassageLabel] = None,
        sentences: Optional[Sequence[str]] = None,
        **faith_kwargs,
    ) -> Tuple[PassageExplanation, Dict[str, Any]]:
        """explain() plus ERASER faithfulness metrics, sharing one IG pass."""
        if isinstance(passage, str):
            passage = EvidencePassage(passage_id="p0", paper_id="", section="abstract", text=passage)
        premise = " ".join(sentences) if sentences is not None else passage.text
        run = self._run(premise, claim, target)
        exp = self._build_explanation(run, passage, claim, premise, sentences)
        return exp, self._faithfulness(run, **faith_kwargs)

    def _run(self, premise: str, claim: str, target: Optional[PassageLabel]) -> Dict[str, Any]:
        if not premise.strip() or not claim.strip():
            raise ValueError("Empty passage or claim")
        enc = self._encode(premise, claim)
        t = enc["tensors"]
        input_ids = t["input_ids"]
        baseline_ids = torch.tensor(
            [[tok if s is None else self.baseline_token_id for tok, s in zip(enc["ids"], enc["seq_ids"])]],
            device=self.device)
        extra = (t["attention_mask"], t.get("token_type_ids"))

        with torch.no_grad():
            probs = self._forward(input_ids, *extra)[0]
            base_probs = self._forward(baseline_ids, *extra)[0]
        target_idx, pred_label = self._target_index(probs, target)

        attrs, delta = self.lig.attribute(
            inputs=input_ids, baselines=baseline_ids, additional_forward_args=extra,
            target=target_idx, n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size, return_convergence_delta=True)
        return {
            "enc": enc, "target_idx": target_idx, "pred_label": pred_label,
            "token_attr": attrs.sum(dim=-1)[0].detach().float().cpu().tolist(),
            "p": float(probs[target_idx]), "p_base": float(base_probs[target_idx]),
            "delta": float(delta.reshape(-1)[0]),
        }

    def _build_explanation(self, run, passage, claim, premise, sentences) -> PassageExplanation:
        enc, tidx = run["enc"], run["target_idx"]
        return PassageExplanation(
            passage_id=passage.passage_id, paper_id=passage.paper_id, claim=claim,
            evidence_text=premise,
            target_label=self.verifier.label_mapping.get(tidx, PassageLabel.NEUTRAL),
            target_class=self.model.config.id2label[tidx],
            predicted_label=run["pred_label"],
            target_probability=run["p"], baseline_probability=run["p_base"],
            convergence_delta=run["delta"], truncated=enc["truncated"],
            words=self._words(enc, run["token_attr"], premise, claim),
            sentence_scores=(self._sentence_scores(enc, run["token_attr"], sentences)
                             if sentences is not None else None),
            settings=self.settings(),
        )

    def explain_response(
        self,
        response: VerificationResponse,
        max_passages_per_paper: int = 3,
        include_neutral: bool = False,
    ) -> List[PassageExplanation]:
        """Explain the passages that decided each paper's verdict, using the
        label the verifier gave each passage as the target. Skips truncated or
        unusable passages, matching the verifier's own aggregation rules."""
        passages = {p.passage_id: p for p in response.evidence_passages}
        labels = {v.passage_id: v for v in response.per_passage_labels}
        out: List[PassageExplanation] = []
        for pv in response.per_paper_verdicts:
            ids = list(pv.supporting_passage_ids) + list(pv.contradicting_passage_ids)
            if include_neutral and not ids:
                ids = [v.passage_id for v in response.per_passage_labels
                       if v.paper_id == pv.paper_id]
            for pid in ids[:max_passages_per_paper]:
                v = labels.get(pid)
                if pid not in passages or v is None or v.truncated:
                    continue
                out.append(self.explain(response.claim, passages[pid], target=v.label))
        return out

    # ------------------------------------------------------------------ grouping
    def _words(self, enc, token_attr, premise, claim) -> List[WordAttribution]:
        texts = {0: premise, 1: claim}
        groups: List[Dict[str, Any]] = []
        tokens = self.tokenizer.convert_ids_to_tokens(enc["ids"])
        sp_style = any(t.startswith(("▁", "Ġ")) for t in tokens)   # sentencepiece / BPE markers
        for i, seg in enumerate(enc["seq_ids"]):
            if seg is None:
                continue
            if enc["word_ids"] is not None:
                key = (seg, enc["word_ids"][i])
                new_word = not groups or groups[-1]["key"] != key
            else:
                tok = tokens[i]
                key = (seg, None)
                starts_word = (tok.startswith(("▁", "Ġ")) if sp_style else not tok.startswith("##"))
                new_word = not groups or groups[-1]["key"][0] != seg or starts_word
            if new_word:
                groups.append({"key": key, "seg": seg, "idx": [], "start": None, "end": None})
            g = groups[-1]
            g["idx"].append(i)
            if enc["offsets"] is not None:
                s, e = enc["offsets"][i]
                g["start"] = s if g["start"] is None else min(g["start"], s)
                g["end"] = e if g["end"] is None else max(g["end"], e)

        words = []
        for g in groups:
            attr = sum(token_attr[i] for i in g["idx"])
            if g["start"] is not None:
                raw = texts[g["seg"]][g["start"]:g["end"]]
                start = g["start"] + (len(raw) - len(raw.lstrip()))
                text, end = raw.strip(), g["end"]
            else:
                text = self.tokenizer.convert_tokens_to_string([tokens[i] for i in g["idx"]]).strip()
                start = end = None
            words.append(WordAttribution(text=text, segment=EVIDENCE if g["seg"] == 0 else CLAIM,
                                         start=start, end=end, attribution=attr, normalized=0.0))
        scale = max((abs(w.attribution) for w in words), default=0.0) or 1.0
        for w in words:
            w.normalized = w.attribution / scale
        return words

    def _sentence_scores(self, enc, token_attr, sentences) -> List[float]:
        """Sum of signed token attributions per evidence sentence."""
        spans, pos = [], 0
        for s in sentences:
            spans.append((pos, pos + len(s)))
            pos += len(s) + 1  # the joining space
        scores = [0.0] * len(sentences)
        ev_tokens = [i for i, s in enumerate(enc["seq_ids"]) if s == 0]
        if enc["offsets"] is not None:
            for i in ev_tokens:
                start = enc["offsets"][i][0]
                for j, (a, b) in enumerate(spans):
                    if a <= start < b + 1:
                        scores[j] += token_attr[i]
                        break
        else:  # approximate: allocate tokens by each sentence's own token count
            counts = [len(self.tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
            k = 0
            for j, c in enumerate(counts):
                for i in ev_tokens[k:k + c]:
                    scores[j] += token_attr[i]
                k += c
        return scores

    # ------------------------------------------------------------------ faithfulness
    @torch.no_grad()
    def _probs_for_variants(self, variants: List[List[int]], enc, target_idx: int) -> List[float]:
        """Run the model on token-index subsets of the original input (deletion, not masking)."""
        ids, tti = enc["ids"], enc["tensors"].get("token_type_ids")
        tti = tti[0].tolist() if tti is not None else None
        pad = self.tokenizer.pad_token_id or 0
        out: List[float] = []
        bs = max(1, self.internal_batch_size)
        for b in range(0, len(variants), bs):
            chunk = variants[b:b + bs]
            L = max(len(v) for v in chunk)
            batch_ids = [[ids[i] for i in v] + [pad] * (L - len(v)) for v in chunk]
            mask = [[1] * len(v) + [0] * (L - len(v)) for v in chunk]
            kwargs = {"input_ids": torch.tensor(batch_ids, device=self.device),
                      "attention_mask": torch.tensor(mask, device=self.device)}
            if tti is not None:
                kwargs["token_type_ids"] = torch.tensor(
                    [[tti[i] for i in v] + [0] * (L - len(v)) for v in chunk], device=self.device)
            probs = self._forward(**kwargs)[:, target_idx]
            out.extend(float(p) for p in probs)
        return out

    def faithfulness(self, claim: str, passage: Union[EvidencePassage, str],
                     target: Optional[PassageLabel] = None, **kwargs) -> Dict[str, Any]:
        """Faithfulness metrics alone (runs its own IG pass)."""
        return self.explain_with_faithfulness(claim, passage, target, **kwargs)[1]

    def _faithfulness(
        self,
        run: Dict[str, Any],
        ratios: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50),
        segments: Sequence[str] = (EVIDENCE, CLAIM),
        n_random: int = 3,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """ERASER-style comprehensiveness and sufficiency (DeYoung et al., 2020).

        comprehensiveness@r = P(t|x) - P(t|x without top-r tokens)    higher = more faithful
        sufficiency@r       = P(t|x) - P(t|only top-r tokens kept)    lower  = more faithful
        Tokens are DELETED (not masked) and ranked by signed attribution toward the
        target. Each score is paired with the same number of RANDOM tokens, so a
        faithful explanation should beat random on both.
        """
        enc, token_attr, target_idx = run["enc"], run["token_attr"], run["target_idx"]
        seg_ok = {0: EVIDENCE in segments, 1: CLAIM in segments}
        cand = [i for i, s in enumerate(enc["seq_ids"]) if s is not None and seg_ok[s]]
        all_idx = list(range(len(enc["ids"])))
        ranked = sorted(cand, key=lambda i: token_attr[i], reverse=True)
        rng = random.Random(seed)

        variants, keys = [all_idx], [None]
        for r in ratios:
            k = max(1, round(r * len(cand)))
            sets = [("ig", set(ranked[:k]))] + [
                (f"rand{j}", set(rng.sample(cand, k))) for j in range(n_random)]
            for name, chosen in sets:
                variants.append([i for i in all_idx if i not in chosen])                  # removed
                keys.append(("comp", r, name))
                variants.append([i for i in all_idx if i not in cand or i in chosen])    # kept only
                keys.append(("suff", r, name))
        probs = self._probs_for_variants(variants, enc, target_idx)
        p_full = probs[0]

        res: Dict[str, Any] = {"p_full": p_full, "n_candidate_tokens": len(cand),
                               "comprehensiveness": {}, "sufficiency": {},
                               "comprehensiveness_random": {}, "sufficiency_random": {}}
        for (kind, r, name), p in zip(keys[1:], probs[1:]):
            metric = "comprehensiveness" if kind == "comp" else "sufficiency"
            if name == "ig":
                res[metric][r] = p_full - p
            else:
                res[metric + "_random"].setdefault(r, []).append(p_full - p)
        for b in ("comprehensiveness_random", "sufficiency_random"):
            res[b] = {r: sum(v) / len(v) for r, v in res[b].items()}
        for m in ("comprehensiveness", "sufficiency", "comprehensiveness_random", "sufficiency_random"):
            res[f"aopc_{m}"] = sum(res[m].values()) / len(res[m])
        return res


# --------------------------------------------------------------------------- display
_COLORS = {"pos": (34, 139, 34), "neg": (200, 40, 40)}


def to_html(exp: PassageExplanation, show_claim: bool = True, min_abs: float = 0.05) -> str:
    """Highlight words: green pushes toward the explained label, red pushes away.
    Safe for st.markdown(..., unsafe_allow_html=True): all text is escaped."""
    def render(segment: str) -> str:
        parts = []
        for w in exp.words:
            if w.segment != segment:
                continue
            txt = html.escape(w.text)
            if abs(w.normalized) < min_abs:
                parts.append(txt)
                continue
            r, g, b = _COLORS["pos" if w.normalized > 0 else "neg"]
            parts.append(f'<span title="{w.attribution:+.4f}" style="background: rgba({r},{g},{b},'
                         f'{0.15 + 0.6 * abs(w.normalized):.2f}); border-radius: 3px; '
                         f'padding: 0 2px">{txt}</span>')
        return " ".join(parts)

    head = (f"<div><b>Explaining:</b> {html.escape(exp.target_label.value)} "
            f"(p = {exp.target_probability:.2f})"
            + (" · <i>passage was truncated</i>" if exp.truncated else "") + "</div>")
    body = f"<div><b>Evidence:</b> {render(EVIDENCE)}</div>"
    if show_claim:
        body += f"<div><b>Claim:</b> {render(CLAIM)}</div>"
    return head + body
