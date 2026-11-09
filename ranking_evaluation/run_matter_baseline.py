"""Run a claim-only feasibility baseline, not an evidence-grounded system evaluation."""

import argparse
import json
import platform
import time
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ranking_evaluation.evaluate_matter_of_fact import evaluate
from shared.datasets.matter_of_fact import load
from shared.datasets.setup import file_hash


def run(train: dict, target: dict) -> dict:
    """Fit on training claims only; labels and source IDs never become features."""
    if train["protocol"]["split"] != "train" or target["protocol"]["split"] == "train":
        raise ValueError("Fit only on train and evaluate on a separate split")
    if train["protocol"]["scope"] != target["protocol"]["scope"]:
        raise ValueError("Training and evaluation scopes must match")
    def papers(dataset):
        return {row["source_paper_id"].split("v")[0] for row in dataset["gold"].values()}
    if papers(train) & papers(target):
        raise ValueError("Source paper overlap between train and evaluation")
    train_texts = {row["claim_text"].strip() for row in train["inputs"]}
    if any(row["claim_text"].strip() in train_texts for row in target["inputs"]):
        raise ValueError("Exact claim text overlap between train and evaluation")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                 sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=0)),
    ])
    started = time.perf_counter()
    model.fit([row["claim_text"] for row in train["inputs"]],
              [train["gold"][row["case_id"]]["label"] for row in train["inputs"]])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba([row["claim_text"] for row in target["inputs"]])
    predict_seconds = time.perf_counter() - started
    positive_column = list(model.classes_).index(True)
    export = {"status": "complete", "dataset_version": target["version"],
              "model": "claim-only TF-IDF word(1,2) + LogisticRegression C=1 random_state=0; no retrieval or evidence",
              "predictions": [{"case_id": row["case_id"],
                               "prediction": bool(prob[positive_column] > 0.5),
                               "feasible_probability": float(prob[positive_column]), "evidence": []}
                              for row, prob in zip(target["inputs"], probabilities)]}
    metrics = evaluate(target, export)
    return {"predictions": export, "evaluation": metrics,
            "protocol": {"role": "diagnostic claim-only baseline; NOT project system performance",
                         "input_features": "claim_text only; no IDs, dates, topics, evidence or generated explanations",
                         "training_version": train["version"], "evaluation_version": target["version"],
                         "training_cases": len(train["inputs"]), "evaluation_cases": len(target["inputs"]),
                         "source_paper_overlap": 0, "exact_claim_overlap": 0,
                         "threshold": 0.5, "hyperparameter_search": False,
                         "limitation": "No retrieval/evidence correctness measured; no feasibility reasoning guarantee",
                         "settings": model.get_params()["classifier"].get_params(),
                         "tfidf": {"ngram_range": [1, 2], "min_df": 2,
                                   "max_features": 50000, "sublinear_tf": True}},
            "timing": {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "platform": platform.platform()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/matter_of_fact"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--scope", choices=["eee", "all"], default="eee")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output exists; choose a new directory")
    result = run(load(args.data, split="train", scope=args.scope),
                 load(args.data, split=args.split, scope=args.scope))
    result["provenance"] = {"runner_sha256": file_hash(Path(__file__)),
                            "adapter_sha256": file_hash(Path("shared/datasets/matter_of_fact.py")),
                            "evaluator_sha256": file_hash(Path("ranking_evaluation/evaluate_matter_of_fact.py"))}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result["evaluation"][k] for k in ("cases", "accuracy", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
