"""
demo_for_person5.py
Clean demonstration for Person 5 to integrate ClaimVerifier into Streamlit.
"""

from schemas import EvidencePassage
from verifier import ClaimVerifier

def run_demo():
    print("Initializing ClaimVerifier (Person 4 module)...")
    verifier = ClaimVerifier()

    claim = "Exposure to fine particulate matter air pollution increases cardiovascular mortality."

    # Mock candidate papers (from Person 1 / Person 2)
    papers = [
        {"paper_id": "paper_101", "title": "Air Pollution & Mortality Study", "evidence_type": "abstract"},
        {"paper_id": "paper_102", "title": "Atmospheric Particles Overview", "evidence_type": "abstract"}
    ]

    # Mock ranked passages (from Person 3)
    passages_by_paper = {
        "paper_101": [
            EvidencePassage(
                passage_id="p101_1",
                paper_id="paper_101",
                section="abstract",
                text="Long-term exposure to PM2.5 significantly increases cardiovascular disease mortality in urban cohorts."
            )
        ],
        "paper_102": [
            EvidencePassage(
                passage_id="p102_1",
                paper_id="paper_102",
                section="abstract",
                text="Particulate matter contains organic carbon, sulfate, and elemental traces of various diameters."
            )
        ]
    }

    # STEP 1: Full Neural Pass (Streamlit calls this once per search)
    print("\n--- 1. Running verify() [Neural Pass] ---")
    verification_response = verifier.verify(claim, papers, passages_by_paper)

    for pv in verification_response.per_paper_verdicts:
        print(f"Paper: {pv.paper_id} | Verdict: {pv.verdict.value} | Confidence: {pv.confidence:.2f}")

    # STEP 2: Fast UI Aggregation (Streamlit calls this whenever the user checks/unchecks papers)
    print("\n--- 2. Running aggregate() [Sub-second UI Re-aggregation] ---")
    
    # Aggregation over all papers
    agg_all = verifier.aggregate(verification_response)
    print(f"Global Verdict (All Papers): {agg_all.aggregated_verdict.value} (Confidence: {agg_all.global_confidence:.2f})")

    # Aggregation if the user filters to only Paper 102
    agg_filtered = verifier.aggregate(verification_response, subset_of_paper_ids=["paper_102"])
    print(f"Filtered Verdict (Only Paper 102): {agg_filtered.aggregated_verdict.value}")

if __name__ == "__main__":
    run_demo()