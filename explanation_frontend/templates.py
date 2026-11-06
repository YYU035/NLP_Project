"""Conservative deterministic explanation baseline."""
from .models import Explanation, ExplanationContext, Statement

AUTOMATIC_SELECTION_LIMITATION = "Passages were selected automatically from unverified provider abstracts/descriptions; full text and source authenticity were not checked."


def build_template(context: ExplanationContext) -> Explanation:
    limitations = ["This result concerns only the selected retrieved evidence, not all scientific literature."]
    if context.snapshot.stage_statuses.get("passages") == "frontend_abstract_windows_v1":
        limitations.append(AUTOMATIC_SELECTION_LIMITATION)
    if context.snapshot.mode == "demo":
        limitations.append("Synthetic demonstration: these are invented test papers and findings.")
    if not context.snapshot.search_complete:
        limitations.append("Search is incomplete; relevant evidence may be missing.")
    if context.aggregation.status == "failed":
        limitations.append("Processing failed; no scientific verdict is available.")
    if context.aggregation.sufficiency_cause:
        limitations.append("Insufficiency cause: " + context.aggregation.sufficiency_cause)
    if any(context.snapshot.verification.per_paper_verdicts.get(pid) and
           context.snapshot.verification.per_paper_verdicts[pid].evidence_type == "abstract_only"
           for pid in context.selected_paper_ids):
        limitations.append("Some selected evidence is abstract only; full-text findings were not inspected.")
    if any(label.truncated and label.paper_id in context.selected_paper_ids
           for label in context.snapshot.verification.per_passage_labels.values()):
        limitations.append("TRUNCATED passages are inspectable but excluded from explanation evidence.")
    return Explanation(context_key=context.key, verdict=context.aggregation.aggregated_verdict,
                       statements=[Statement(text=p.text, citation_ids=[p.passage_id])
                                   for p in context.aggregation.primary_evidence], limitations=limitations)
