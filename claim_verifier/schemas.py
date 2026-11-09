"""
schemas.py
Contract definitions for Person 4 (Claim Verification & Evidence Aggregation).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Scientific & Operational Enums ---

class PassageLabel(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"


class PaperVerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class CorpusVerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"


class SufficiencyCause(str, Enum):
    NO_RETRIEVAL = "NO_RETRIEVAL"
    NO_PASSAGE_ABOVE_THRESHOLD = "NO_PASSAGE_ABOVE_THRESHOLD"
    ABSTRACT_ONLY_LIMITATION = "ABSTRACT_ONLY_LIMITATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    MODEL_UNCERTAINTY = "MODEL_UNCERTAINTY"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"


class OperationalStatus(str, Enum):
    OK = "OK"
    TRUNCATED = "TRUNCATED"
    NO_PASSAGE_AVAILABLE = "NO_PASSAGE_AVAILABLE"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"


# --- Schemas ---

class EvidencePassage(BaseModel):
    """Input schema provided by Person 3 (Yu Yue)."""
    passage_id: str
    paper_id: str
    section: str
    text: str
    score: float = 0.0


class PassageVerdict(BaseModel):
    """Output for each evaluated passage."""
    passage_id: str
    paper_id: str
    label: PassageLabel
    confidence: float
    raw_probabilities: Dict[str, float] = Field(default_factory=dict)
    truncated: bool = False
    operational_status: OperationalStatus = OperationalStatus.OK


class PaperVerdict(BaseModel):
    """Aggregated verdict for a single paper."""
    paper_id: str
    title: str = ""
    verdict: PaperVerdictEnum
    confidence: float
    evidence_type: str = "abstract"  # 'abstract' or 'full_text'
    sufficiency_cause: Optional[SufficiencyCause] = None
    conflict_flag: bool = False
    supporting_passage_ids: List[str] = Field(default_factory=list)
    contradicting_passage_ids: List[str] = Field(default_factory=list)


class ModelSettings(BaseModel):
    checkpoint: str
    revision: str = "main"
    tokenizer_max_length: int = 512
    threshold: float
    seed: int = 42


class VerificationResponse(BaseModel):
    """Output of verify(): Full inference results, cached for UI/re-aggregation."""
    claim: str
    per_paper_verdicts: List[PaperVerdict]
    per_passage_labels: List[PassageVerdict]
    processing_failures: List[str] = Field(default_factory=list)
    evidence_passages: List[EvidencePassage] = Field(default_factory=list)
    model_settings: ModelSettings


class AggregationResponse(BaseModel):
    """Output of aggregate(): Lightweight UI re-aggregation over selected papers."""
    claim: str
    aggregated_verdict: CorpusVerdictEnum
    primary_evidence: List[EvidencePassage] = Field(default_factory=list)
    global_confidence: float
    verdict_delta_reason: Optional[str] = None