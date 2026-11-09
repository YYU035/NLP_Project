"""
test_verifier.py
Unit tests for Person 4 verification and aggregation.
"""

import pytest
import time
from schemas import (
    EvidencePassage,
    PassageLabel,
    PaperVerdictEnum,
    CorpusVerdictEnum,
    OperationalStatus,
)
from verifier import ClaimVerifier


@pytest.fixture(scope="module")
def verifier():
    # Initializes once for all tests using CUDA if available
    return ClaimVerifier(threshold=0.60)


def test_label_mapping_support(verifier):
    claim = "Regular exercise improves cardiovascular health."
    passage = EvidencePassage(
        passage_id="pass_1",
        paper_id="paper_1",
        section="abstract",
        text="Clinical trials demonstrate that routine aerobic exercise significantly improves cardiovascular endurance and heart health.",
    )
    result = verifier._classify_passage(claim, passage)
    assert result.label == PassageLabel.SUPPORT
    assert result.confidence > 0.60


def test_label_mapping_contradict(verifier):
    claim = "Smoking cigarettes has no negative impact on lung capacity."
    passage = EvidencePassage(
        passage_id="pass_2",
        paper_id="paper_2",
        section="results",
        text="Cigarette smoking causes severe chronic obstructive pulmonary disease and permanently reduces lung capacity.",
    )
    result = verifier._classify_passage(claim, passage)
    assert result.label == PassageLabel.CONTRADICT
    assert result.confidence > 0.60


def test_truncation_detection(verifier):
    claim = "A short test claim."
    # Create an artificially long passage exceeding 512 tokens
    long_text = "This is a repeated sentence designed to test the model's tokenizer truncation boundaries. " * 70
    passage = EvidencePassage(
        passage_id="pass_long",
        paper_id="paper_long",
        section="discussion",
        text=long_text,
    )
    result = verifier._classify_passage(claim, passage)
    assert result.truncated is True
    assert result.operational_status == OperationalStatus.TRUNCATED


def test_paper_conflict_detection(verifier):
    # Tests internal conflict within one paper
    claim = "Vitamin D prevents viral infections."
    papers = [{"paper_id": "paper_conflicting", "title": "Vitamin D Study"}]
    passages = {
        "paper_conflicting": [
            EvidencePassage(
                passage_id="p_sup",
                paper_id="paper_conflicting",
                section="intro",
                text="High Vitamin D supplementation prevents upper respiratory viral infections effectively.",
            ),
            EvidencePassage(
                passage_id="p_con",
                paper_id="paper_conflicting",
                section="results",
                text="The randomized controlled trial showed Vitamin D has no protective effect against viral infections.",
            ),
        ]
    }
    resp = verifier.verify(claim, papers, passages)
    pv = resp.per_paper_verdicts[0]
    assert pv.conflict_flag is True
    assert pv.verdict == PaperVerdictEnum.INSUFFICIENT
    assert len(pv.supporting_passage_ids) > 0
    assert len(pv.contradicting_passage_ids) > 0


def test_decoupled_fast_aggregation(verifier):
    claim = "Aspirin reduces cardiovascular event risk."
    papers = [
        {"paper_id": "paper_A", "title": "Aspirin Heart Study"},
        {"paper_id": "paper_B", "title": "Cardio Prevention"},
    ]
    passages = {
        "paper_A": [
            EvidencePassage(
                passage_id="p1",
                paper_id="paper_A",
                section="conclusion",
                text="Low-dose aspirin daily significantly reduces recurrent cardiovascular event risk.",
            )
        ],
        "paper_B": [
            EvidencePassage(
                passage_id="p2",
                paper_id="paper_B",
                section="conclusion",
                text="Daily aspirin was found completely ineffective in reducing cardiovascular mortality.",
            )
        ],
    }

    # 1. Run the heavy inference pass once
    resp = verifier.verify(claim, papers, passages)

    # 2. Fast aggregation should take < 50ms and require no neural inference
    t0 = time.time()
    agg_all = verifier.aggregate(resp)
    elapsed = time.time() - t0

    assert elapsed < 0.1, f"Aggregation took too long: {elapsed:.4f}s"
    assert agg_all.aggregated_verdict == CorpusVerdictEnum.CONFLICTING

    # 3. Filter down to only Paper A via UI subset selection
    agg_subset = verifier.aggregate(resp, subset_of_paper_ids=["paper_A"])
    assert agg_subset.aggregated_verdict == CorpusVerdictEnum.SUPPORTED