"""Reusable scope-aware inspection counts, independent of Streamlit."""

def evidence_counts(snapshot, paper_ids):
    selected=set(paper_ids)
    if len(selected)!=len(paper_ids) or not selected<=set(snapshot.papers):
        raise ValueError('Unknown or duplicate paper selection')
    labels=[label for label in snapshot.verification.per_passage_labels.values() if label.paper_id in selected]
    return {name:sum(label.label==name and not label.truncated for label in labels) for name in ('SUPPORT','CONTRADICT','NEUTRAL')}
