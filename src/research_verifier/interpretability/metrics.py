"""
eval_ig_dev.py
Evaluates Integrated Gradients explanations on SciFact dev.

For each (claim, cited abstract with annotated evidence):
  premise = the FULL abstract (all sentences), hypothesis = the claim,
  target  = the gold label (SUPPORT / CONTRADICT).

Two questions:
1. Plausibility: do IG's highest-scoring sentences match the sentences that
   human annotators marked as rationales?
     - Average precision (AP) of sentence scores against gold rationale sentences
     - Top-1 hit: is the highest-scoring sentence a gold rationale?
     - Both compared with a random sentence ranking
2. Faithfulness: does the model actually rely on the high-attribution tokens?
     - ERASER comprehensiveness and sufficiency (AOPC over 1/5/10/20/50%),
       each compared with removing/keeping the same number of random tokens

Usage (GPU recommended, e.g. Colab):
    python eval_ig_dev.py --limit 50          # quick check, ~minutes on a T4
    python eval_ig_dev.py                     # all dev pairs
Outputs in ig_outputs/: per_example.jsonl, summary.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from typing import Dict, List

from research_verifier.interpretability.gradients import IGExplainer
from research_verifier.verification.schemas import EvidencePassage, PassageLabel
from research_verifier.verification.engine import ClaimVerifier


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_examples(data_dir: str) -> List[Dict]:
    """One example per (claim, evidence document). Uses the files that
    calibrate_dev.py already downloads into data/."""
    corpus = {d["doc_id"]: d for d in load_jsonl(os.path.join(data_dir, "corpus.jsonl"))}
    examples = []
    for c in load_jsonl(os.path.join(data_dir, "claims_dev.jsonl")):
        for doc_id, ev_list in c.get("evidence", {}).items():
            doc = corpus.get(int(doc_id))
            if not doc or not ev_list:
                continue
            labels = {ev["label"] for ev in ev_list}
            if len(labels) != 1:
                continue  # mixed labels within one doc: ambiguous target, skip
            rationale = sorted({i for ev in ev_list for i in ev["sentences"]
                                if i < len(doc["abstract"])})
            examples.append({
                "claim_id": c["id"], "claim": c["claim"], "doc_id": str(doc_id),
                "sentences": doc["abstract"], "rationale": rationale, "gold_label": labels.pop(),
            })
    return examples


def average_precision(scores: List[float], relevant: set) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits, total = 0, 0.0
    for rank, i in enumerate(order, 1):
        if i in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else 0.0


def random_ap(n: int, relevant: set, rng: random.Random, trials: int = 50) -> float:
    return statistics.mean(average_precision([rng.random() for _ in range(n)], relevant)
                           for _ in range(trials))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None, verifier=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="ig_outputs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--internal-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    examples = build_examples(args.data_dir)
    if args.limit:
        examples = rng.sample(examples, min(args.limit, len(examples)))
    print(f"{len(examples)} claim-abstract pairs")

    verifier = verifier or ClaimVerifier()
    explainer = IGExplainer(verifier, n_steps=args.n_steps,
                            internal_batch_size=args.internal_batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    with open(os.path.join(args.out_dir, "per_example.jsonl"), "w", encoding="utf-8") as f:
        for n, ex in enumerate(examples, 1):
            passage = EvidencePassage(passage_id=f"{ex['claim_id']}_{ex['doc_id']}",
                                      paper_id=ex["doc_id"], section="abstract",
                                      text=" ".join(ex["sentences"]))
            exp, faith = explainer.explain_with_faithfulness(
                ex["claim"], passage, target=PassageLabel(ex["gold_label"]),
                sentences=ex["sentences"], seed=args.seed + n)
            rel = set(ex["rationale"])
            scores = exp.sentence_scores
            row = {
                "claim_id": ex["claim_id"], "doc_id": ex["doc_id"], "gold_label": ex["gold_label"],
                "predicted_label": exp.predicted_label.value,
                "correct": exp.predicted_label.value == ex["gold_label"],
                "target_probability": exp.target_probability,
                "convergence_delta": exp.convergence_delta,
                "relative_delta": abs(exp.convergence_delta) / max(
                    abs(exp.target_probability - exp.baseline_probability), 1e-6),
                "truncated": exp.truncated,
                "n_sentences": len(scores), "rationale": ex["rationale"],
                "sentence_scores": scores,
                "ap": average_precision(scores, rel),
                "ap_random": random_ap(len(scores), rel, rng),
                "top1_hit": max(range(len(scores)), key=lambda i: scores[i]) in rel,
                "top1_random": len(rel) / len(scores),
                "faithfulness": faith,
                "top_claim_words": [w.text for w in exp.top_words(5, "claim")],
                "top_evidence_words": [w.text for w in exp.top_words(8, "evidence")],
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if n % 10 == 0 or n == len(examples):
                print(f"  {n}/{len(examples)}  ({time.perf_counter() - t0:.0f}s)")

    def summarise(rs):
        rs = [r for r in rs if not r["truncated"]]
        if not rs:
            return {"n": 0}
        F = [r["faithfulness"] for r in rs]
        return {
            "n": len(rs),
            "verifier_accuracy_on_full_abstract": mean([r["correct"] for r in rs]),
            "plausibility": {
                "mean_ap": mean([r["ap"] for r in rs]),
                "mean_ap_random": mean([r["ap_random"] for r in rs]),
                "top1_hit_rate": mean([r["top1_hit"] for r in rs]),
                "top1_hit_random": mean([r["top1_random"] for r in rs]),
            },
            "faithfulness": {
                "aopc_comprehensiveness": mean([x["aopc_comprehensiveness"] for x in F]),
                "aopc_comprehensiveness_random": mean([x["aopc_comprehensiveness_random"] for x in F]),
                "aopc_sufficiency": mean([x["aopc_sufficiency"] for x in F]),
                "aopc_sufficiency_random": mean([x["aopc_sufficiency_random"] for x in F]),
            },
            "ig_reliability": {
                "median_abs_convergence_delta": statistics.median(abs(r["convergence_delta"]) for r in rs),
                "median_relative_delta": statistics.median(r["relative_delta"] for r in rs),
            },
        }

    summary = {
        "all": summarise(rows),
        "correct_predictions": summarise([r for r in rows if r["correct"]]),
        "incorrect_predictions": summarise([r for r in rows if not r["correct"]]),
        "by_label": {l: summarise([r for r in rows if r["gold_label"] == l])
                     for l in ("SUPPORT", "CONTRADICT")},
        "n_truncated_excluded": sum(r["truncated"] for r in rows),
        "settings": {**explainer.settings(), **vars(args)},
        "seconds": time.perf_counter() - t0,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    md = ["# Integrated Gradients on SciFact dev", "",
          "| Subset | n | AP | AP random | Top-1 hit | Top-1 random | Comp. AOPC | Comp. random | Suff. AOPC | Suff. random |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in [("All", summary["all"]), ("Correct", summary["correct_predictions"]),
                    ("Incorrect", summary["incorrect_predictions"]),
                    ("SUPPORT", summary["by_label"]["SUPPORT"]),
                    ("CONTRADICT", summary["by_label"]["CONTRADICT"])]:
        if not s.get("n"):
            continue
        p, fa = s["plausibility"], s["faithfulness"]
        md.append(f"| {name} | {s['n']} | {p['mean_ap']:.3f} | {p['mean_ap_random']:.3f} | "
                  f"{p['top1_hit_rate']:.3f} | {p['top1_hit_random']:.3f} | "
                  f"{fa['aopc_comprehensiveness']:.3f} | {fa['aopc_comprehensiveness_random']:.3f} | "
                  f"{fa['aopc_sufficiency']:.3f} | {fa['aopc_sufficiency_random']:.3f} |")
    s = summary["all"]
    if s.get("n"):
        md += ["", f"Truncated pairs excluded: {summary['n_truncated_excluded']}",
               f"Median |convergence delta| / |P(x) - P(baseline)|: "
               f"{s['ig_reliability']['median_relative_delta']:.3f} (below ~0.05 is good; "
               f"otherwise raise --n-steps)",
               "", "Reading the table: plausibility and comprehensiveness should be ABOVE random; "
               "sufficiency should be BELOW random."]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return summary


if __name__ == "__main__":
    main()
