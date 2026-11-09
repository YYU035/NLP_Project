"""
calibrate_dev.py
Downloads SciFact dev set via the official tarball, runs Oracle evaluation,
sweeps decision thresholds, and outputs Accuracy, Macro-F1, and Confusion Matrices.
"""

import os
import json
import tarfile
import urllib.request
from typing import List, Dict, Any, Tuple

from schemas import EvidencePassage, PassageLabel
from verifier import ClaimVerifier

DATA_TAR_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download_and_extract_scifact(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    claims_path = os.path.join(data_dir, "claims_dev.jsonl")

    # If already extracted, skip
    if os.path.exists(corpus_path) and os.path.exists(claims_path):
        print(f"Found local SciFact files in {data_dir}/")
        return corpus_path, claims_path

    tar_path = os.path.join(data_dir, "data.tar.gz")
    print(f"Downloading SciFact data archive from {DATA_TAR_URL}...")

    # Include a User-Agent to satisfy AWS S3
    req = urllib.request.Request(
        DATA_TAR_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
        out_file.write(response.read())

    print("Extracting dataset archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract files cleanly
        for member in tar.getmembers():
            # Check for corpus and dev files
            basename = os.path.basename(member.name)
            if basename in ["corpus.jsonl", "claims_dev.jsonl"]:
                member.name = basename  # Flatten into target dir
                tar.extract(member, path=data_dir)

    # Clean up the compressed archive
    if os.path.exists(tar_path):
        os.remove(tar_path)

    print("Data extraction complete.")
    return corpus_path, claims_path


def load_scifact_data(data_dir: str = "data") -> Tuple[Dict[int, Dict], List[Dict]]:
    corpus_path, claims_path = download_and_extract_scifact(data_dir)

    # 1. Load Corpus
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc

    # 2. Load Dev Claims
    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            claims.append(json.loads(line))

    return corpus, claims


def build_oracle_dataset(corpus: Dict[int, Dict], claims: List[Dict]) -> List[Dict[str, Any]]:
    """
    Constructs (claim, evidence_passage, gold_label) triples from SciFact dev.
    Gold labels: 'SUPPORT', 'CONTRADICT', 'NEUTRAL'
    """
    dataset = []

    for item in claims:
        claim_text = item["claim"]
        evidence_dict = item.get("evidence", {})
        cited_doc_ids = item.get("cited_doc_ids", [])

        # Add SUPPORT and CONTRADICT cases from annotated evidence
        for doc_id_str, ev_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            doc = corpus.get(doc_id)
            if not doc:
                continue

            abstract_sentences = doc.get("abstract", [])

            for ev in ev_list:
                label_str = ev.get("label")  # 'SUPPORT' or 'CONTRADICT'
                sentence_indices = ev.get("sentences", [])
                
                # Combine sentences if evidence spans multiple sentences
                passage_text = " ".join(
                    [abstract_sentences[i] for i in sentence_indices if i < len(abstract_sentences)]
                )

                if passage_text.strip():
                    dataset.append({
                        "claim": claim_text,
                        "paper_id": str(doc_id),
                        "passage_text": passage_text,
                        "gold_label": label_str,
                    })

        # Add NEUTRAL cases (cited papers where evidence was not found/neutral)
        for doc_id in cited_doc_ids:
            if str(doc_id) not in evidence_dict:
                doc = corpus.get(doc_id)
                if doc and doc.get("abstract"):
                    # Use the first 2 sentences of the cited paper as neutral text
                    neutral_text = " ".join(doc["abstract"][:2])
                    dataset.append({
                        "claim": claim_text,
                        "paper_id": str(doc_id),
                        "passage_text": neutral_text,
                        "gold_label": "NEUTRAL",
                    })

    return dataset


def compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    labels = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
    cm = {g: {p: 0 for p in labels} for g in labels}

    for g, p in zip(gold, pred):
        if g in cm and p in cm[g]:
            cm[g][p] += 1

    total = len(gold)
    correct = sum(cm[l][l] for l in labels)
    accuracy = correct / total if total > 0 else 0.0

    f1_scores = {}
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[other][l] for other in labels if other != l)
        fn = sum(cm[l][other] for other in labels if other != l)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[l] = f1

    macro_f1 = sum(f1_scores.values()) / len(labels)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": f1_scores,
        "confusion_matrix": cm,
    }


def main():
    print("=" * 60)
    print("SciFact Oracle Evaluation & Threshold Calibration")
    print("=" * 60)

    corpus, claims = load_scifact_data()
    oracle_dataset = build_oracle_dataset(corpus, claims)
    print(f"Total Oracle Claim-Evidence pairs: {len(oracle_dataset)}")

    # Initialize verifier
    verifier = ClaimVerifier(threshold=0.50)

    cached_predictions = []
    print("\nRunning inference over SciFact Dev Oracle samples...")

    for i, item in enumerate(oracle_dataset):
        passage = EvidencePassage(
            passage_id=f"p_{i}",
            paper_id=item["paper_id"],
            section="abstract",
            text=item["passage_text"]
        )
        res = verifier._classify_passage(item["claim"], passage)
        cached_predictions.append((item["gold_label"], res))

        if (i + 1) % 50 == 0 or (i + 1) == len(oracle_dataset):
            print(f"  Processed {i + 1}/{len(oracle_dataset)} pairs")

    # Sweep thresholds
    threshold_candidates = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    best_threshold = 0.65
    best_macro_f1 = -1.0
    best_results = None

    print("\n" + "=" * 60)
    print("Threshold Sensitivity Sweep")
    print("=" * 60)
    print(f"{'Threshold':<12} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Support F1':<12} | {'Contradict F1':<14} | {'Neutral F1':<10}")
    print("-" * 80)

    for thresh in threshold_candidates:
        golds = []
        preds = []

        for gold_label, verdict in cached_predictions:
            golds.append(gold_label)
            pred_label = verdict.label.value
            if pred_label in ["SUPPORT", "CONTRADICT"] and verdict.confidence < thresh:
                pred_label = "NEUTRAL"
            preds.append(pred_label)

        metrics = compute_metrics(golds, preds)
        acc = metrics["accuracy"]
        mf1 = metrics["macro_f1"]
        f1s = metrics["per_class_f1"]

        print(f"{thresh:<12.2f} | {acc:<10.4f} | {mf1:<10.4f} | {f1s['SUPPORT']:<12.4f} | {f1s['CONTRADICT']:<14.4f} | {f1s['NEUTRAL']:<10.4f}")

        if mf1 > best_macro_f1:
            best_macro_f1 = mf1
            best_threshold = thresh
            best_results = metrics

    print("\n" + "=" * 60)
    print(f"Optimal Calibration Threshold on SciFact Dev: {best_threshold:.2f} (Macro-F1: {best_macro_f1:.4f})")
    print("=" * 60)

    print("\nConfusion Matrix (Rows: Gold, Columns: Predicted):")
    labels = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
    header = f"{'Gold / Pred':<14} | " + " | ".join(f"{l:<12}" for l in labels)
    print(header)
    print("-" * len(header))
    for g in labels:
        row = f"{g:<14} | " + " | ".join(f"{best_results['confusion_matrix'][g][p]:<12}" for p in labels)
        print(row)


if __name__ == "__main__":
    main()