"""
verifier.py
Core claim verification engine and multi-level aggregation module.
"""

from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .schemas import (
    EvidencePassage,
    PassageVerdict,
    PaperVerdict,
    VerificationResponse,
    AggregationResponse,
    PassageLabel,
    PaperVerdictEnum,
    CorpusVerdictEnum,
    SufficiencyCause,
    OperationalStatus,
    ModelSettings,
)


class ClaimVerifier:
    def __init__(
        self,
        checkpoint: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        revision: str = "main",
        tokenizer_max_length: int = 512,
        threshold: float = 0.50,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.checkpoint = checkpoint
        self.revision = revision
        self.max_length = tokenizer_max_length
        self.threshold = threshold
        self.seed = seed

        # Set reproducible seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device assignment
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load Tokenizer & Model (use FP16 on GPU for speed and memory savings)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, revision=revision
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            revision=revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        # Build dynamic mapping from model config to standard PassageLabel
        self.label_mapping = self._build_label_mapping()

    def _build_label_mapping(self) -> Dict[int, PassageLabel]:
        """Resolves model id2label mapping to SUPPORT, CONTRADICT, NEUTRAL."""
        mapping = {}
        for idx, label_str in self.model.config.id2label.items():
            l_lower = label_str.lower()
            if "entail" in l_lower or "support" in l_lower:
                mapping[int(idx)] = PassageLabel.SUPPORT
            elif "contradict" in l_lower:
                mapping[int(idx)] = PassageLabel.CONTRADICT
            else:
                mapping[int(idx)] = PassageLabel.NEUTRAL
        return mapping

    def _classify_passage(
        self, claim: str, passage: EvidencePassage
    ) -> PassageVerdict:
        """
        Classifies a single passage against the claim.
        Input discipline: Premise = Passage text, Hypothesis = Claim.
        """
        if not passage.text or passage.text.strip() == "":
            return PassageVerdict(
                passage_id=passage.passage_id,
                paper_id=passage.paper_id,
                label=PassageLabel.NEUTRAL,
                confidence=0.0,
                operational_status=OperationalStatus.NO_PASSAGE_AVAILABLE,
            )

        # 1. Truncation Guard: Tokenize without truncation first to detect overflow
        untruncated_tokens = self.tokenizer.encode(
            passage.text, claim, add_special_tokens=True, truncation=False
        )
        is_truncated = len(untruncated_tokens) > self.max_length

        # 2. Tokenize with enforcement
        inputs = self.tokenizer(
            passage.text,
            claim,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

        # Extract mapped probabilities
        raw_probs: Dict[str, float] = {}
        label_probs: Dict[PassageLabel, float] = {
            PassageLabel.SUPPORT: 0.0,
            PassageLabel.CONTRADICT: 0.0,
            PassageLabel.NEUTRAL: 0.0,
        }

        for idx, prob in enumerate(probs):
            prob_val = float(prob.item())
            mapped_label = self.label_mapping.get(idx, PassageLabel.NEUTRAL)
            raw_probs[self.model.config.id2label[idx]] = prob_val
            label_probs[mapped_label] += prob_val

        # Find predicted label and max probability
        predicted_label = max(label_probs, key=lambda l: label_probs[l])
        confidence = label_probs[predicted_label]

        # 4. Calibration & Uncertainty Discipline:
        # If model predicts SUPPORT or CONTRADICT but confidence < threshold, downgrade to NEUTRAL
        op_status = OperationalStatus.OK
        if is_truncated:
            op_status = OperationalStatus.TRUNCATED

        if (
            predicted_label in [PassageLabel.SUPPORT, PassageLabel.CONTRADICT]
            and confidence < self.threshold
        ):
            final_label = PassageLabel.NEUTRAL
        else:
            final_label = predicted_label

        return PassageVerdict(
            passage_id=passage.passage_id,
            paper_id=passage.paper_id,
            label=final_label,
            confidence=confidence,
            raw_probabilities=raw_probs,
            truncated=is_truncated,
            operational_status=op_status,
        )

    def _aggregate_single_paper(
        self,
        paper_id: str,
        paper_title: str,
        verdicts: List[PassageVerdict],
        evidence_type: str = "abstract",
        max_votes: int = 3,
    ) -> PaperVerdict:
        """
        Aggregates passage verdicts into a single paper verdict.
        Rules:
        - Exclude TRUNCATED passages from confident aggregation.
        - Cap top votes per paper (e.g. max 3 passages).
        - Detect internal conflict (support vs contradiction).
        """
        # Exclude truncated or broken passages from confident aggregation
        usable_verdicts = [
            v for v in verdicts if v.operational_status == OperationalStatus.OK
        ]

        if not usable_verdicts:
            # Determine sufficiency cause
            cause = SufficiencyCause.NO_PASSAGE_ABOVE_THRESHOLD
            if any(
                v.operational_status == OperationalStatus.NO_PASSAGE_AVAILABLE
                for v in verdicts
            ):
                cause = SufficiencyCause.NO_RETRIEVAL
            elif any(v.truncated for v in verdicts):
                cause = SufficiencyCause.PROCESSING_FAILURE

            return PaperVerdict(
                paper_id=paper_id,
                title=paper_title,
                verdict=PaperVerdictEnum.INSUFFICIENT,
                confidence=0.0,
                evidence_type=evidence_type,
                sufficiency_cause=cause,
            )

        # Sort by confidence descending and cap votes
        usable_verdicts.sort(key=lambda v: v.confidence, reverse=True)
        top_verdicts = usable_verdicts[:max_votes]

        supporting = [
            v for v in top_verdicts if v.label == PassageLabel.SUPPORT
        ]
        contradicting = [
            v for v in top_verdicts if v.label == PassageLabel.CONTRADICT
        ]

        sup_ids = [v.passage_id for v in supporting]
        con_ids = [v.passage_id for v in contradicting]

        # Check for conflicting evidence within the same paper
        has_conflict = len(supporting) > 0 and len(contradicting) > 0

        if has_conflict:
            # Paper has both supporting and contradicting evidence
            max_conf = max(
                max(v.confidence for v in supporting),
                max(v.confidence for v in contradicting),
            )
            return PaperVerdict(
                paper_id=paper_id,
                title=paper_title,
                verdict=PaperVerdictEnum.INSUFFICIENT,
                confidence=max_conf,
                evidence_type=evidence_type,
                sufficiency_cause=SufficiencyCause.MODEL_UNCERTAINTY,
                conflict_flag=True,
                supporting_passage_ids=sup_ids,
                contradicting_passage_ids=con_ids,
            )

        if len(supporting) > len(contradicting):
            avg_conf = sum(v.confidence for v in supporting) / len(supporting)
            return PaperVerdict(
                paper_id=paper_id,
                title=paper_title,
                verdict=PaperVerdictEnum.SUPPORTED,
                confidence=avg_conf,
                evidence_type=evidence_type,
                conflict_flag=False,
                supporting_passage_ids=sup_ids,
            )
        elif len(contradicting) > len(supporting):
            avg_conf = sum(v.confidence for v in contradicting) / len(contradicting)
            return PaperVerdict(
                paper_id=paper_id,
                title=paper_title,
                verdict=PaperVerdictEnum.CONTRADICTED,
                confidence=avg_conf,
                evidence_type=evidence_type,
                conflict_flag=False,
                contradicting_passage_ids=con_ids,
            )
        else:
            # All neutral or below threshold
            return PaperVerdict(
                paper_id=paper_id,
                title=paper_title,
                verdict=PaperVerdictEnum.INSUFFICIENT,
                confidence=0.0,
                evidence_type=evidence_type,
                sufficiency_cause=SufficiencyCause.NO_PASSAGE_ABOVE_THRESHOLD,
            )

    # --- Decoupled Public APIs ---

    def verify(
        self,
        claim: str,
        papers: List[Dict[str, Any]],
        passages_by_paper: Dict[str, List[EvidencePassage]],
    ) -> VerificationResponse:
        """
        API 1: Full neural inference pass. Run once per search.
        Cached for UI interaction and lightweight re-aggregation.
        """
        all_passage_verdicts: List[PassageVerdict] = []
        all_passages_flat: List[EvidencePassage] = []
        paper_verdicts: List[PaperVerdict] = []
        processing_failures: List[str] = []

        for p_info in papers:
            p_id = p_info.get("paper_id", "")
            p_title = p_info.get("title", "")
            evidence_type = p_info.get("evidence_type", "abstract")

            passages = passages_by_paper.get(p_id, [])
            paper_pv: List[PassageVerdict] = []

            if not passages:
                processing_failures.append(f"No passages found for paper {p_id}")
                pv_dummy = PassageVerdict(
                    passage_id=f"{p_id}_empty",
                    paper_id=p_id,
                    label=PassageLabel.NEUTRAL,
                    confidence=0.0,
                    operational_status=OperationalStatus.NO_PASSAGE_AVAILABLE,
                )
                paper_pv.append(pv_dummy)
            else:
                for passage in passages:
                    all_passages_flat.append(passage)
                    pv = self._classify_passage(claim, passage)
                    paper_pv.append(pv)
                    all_passage_verdicts.append(pv)

            paper_verdict = self._aggregate_single_paper(
                paper_id=p_id,
                paper_title=p_title,
                verdicts=paper_pv,
                evidence_type=evidence_type,
            )
            paper_verdicts.append(paper_verdict)

        model_settings = ModelSettings(
            checkpoint=self.checkpoint,
            revision=self.revision,
            tokenizer_max_length=self.max_length,
            threshold=self.threshold,
            seed=self.seed,
        )

        return VerificationResponse(
            claim=claim,
            per_paper_verdicts=paper_verdicts,
            per_passage_labels=all_passage_verdicts,
            processing_failures=processing_failures,
            evidence_passages=all_passages_flat,
            model_settings=model_settings,
        )

    def aggregate(
        self,
        verification_response: VerificationResponse,
        subset_of_paper_ids: Optional[List[str]] = None,
    ) -> AggregationResponse:
        """
        API 2: Fast UI subset re-aggregation.
        MUST NOT run neural inference. Guaranteed execution < 100ms.
        """
        papers_to_use = verification_response.per_paper_verdicts

        if subset_of_paper_ids is not None:
            id_set = set(subset_of_paper_ids)
            papers_to_use = [p for p in papers_to_use if p.paper_id in id_set]

        if not papers_to_use:
            return AggregationResponse(
                claim=verification_response.claim,
                aggregated_verdict=CorpusVerdictEnum.INSUFFICIENT,
                primary_evidence=[],
                global_confidence=0.0,
                verdict_delta_reason="No papers selected in subset.",
            )

        supported_papers = [
            p for p in papers_to_use if p.verdict == PaperVerdictEnum.SUPPORTED
        ]
        contradicted_papers = [
            p for p in papers_to_use if p.verdict == PaperVerdictEnum.CONTRADICTED
        ]

        passage_lookup = {
            p.passage_id: p for p in verification_response.evidence_passages
        }

        # Case 1: Across papers, both SUPPORT and CONTRADICT exist
        if supported_papers and contradicted_papers:
            # Conflicting papers present
            all_opposing_ids = []
            for p in supported_papers:
                all_opposing_ids.extend(p.supporting_passage_ids)
            for p in contradicted_papers:
                all_opposing_ids.extend(p.contradicting_passage_ids)

            primary_ev = [
                passage_lookup[pid]
                for pid in all_opposing_ids
                if pid in passage_lookup
            ]
            global_conf = max(
                max(p.confidence for p in supported_papers),
                max(p.confidence for p in contradicted_papers),
            )
            return AggregationResponse(
                claim=verification_response.claim,
                aggregated_verdict=CorpusVerdictEnum.CONFLICTING,
                primary_evidence=primary_ev[:5],
                global_confidence=global_conf,
                verdict_delta_reason="Corpus contains contradictory evidence across papers.",
            )

        # Case 2: Unanimous SUPPORT
        if supported_papers:
            supported_papers.sort(key=lambda p: p.confidence, reverse=True)
            top_paper = supported_papers[0]
            primary_ev = [
                passage_lookup[pid]
                for pid in top_paper.supporting_passage_ids
                if pid in passage_lookup
            ]
            return AggregationResponse(
                claim=verification_response.claim,
                aggregated_verdict=CorpusVerdictEnum.SUPPORTED,
                primary_evidence=primary_ev,
                global_confidence=top_paper.confidence,
            )

        # Case 3: Unanimous CONTRADICT
        if contradicted_papers:
            contradicted_papers.sort(key=lambda p: p.confidence, reverse=True)
            top_paper = contradicted_papers[0]
            primary_ev = [
                passage_lookup[pid]
                for pid in top_paper.contradicting_passage_ids
                if pid in passage_lookup
            ]
            return AggregationResponse(
                claim=verification_response.claim,
                aggregated_verdict=CorpusVerdictEnum.CONTRADICTED,
                primary_evidence=primary_ev,
                global_confidence=top_paper.confidence,
            )

        # Case 4: Everything is INSUFFICIENT
        return AggregationResponse(
            claim=verification_response.claim,
            aggregated_verdict=CorpusVerdictEnum.INSUFFICIENT,
            primary_evidence=[],
            global_confidence=0.0,
            verdict_delta_reason="No papers provided decisive evidence above threshold.",
        )