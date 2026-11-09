"""Matter-of-Fact feasibility adapter with separated inference and gold data."""

import argparse
import calendar
import hashlib
import json
from datetime import date
from pathlib import Path

from shared.datasets.setup import verify_download

FILES = {"train": "matteroffact.train.2022.1376.json",
         "validation": "matteroffact.validation.2023.2538.json",
         "test": "matteroffact.test.20242025.4446.json"}
EEE_THEMES = {"theme_batteries", "theme_semi-conductors", "theme_superconductors"}


def cutoff_date(value: dict) -> str:
    """The release uses year/month; interpret cutoff as end of that month."""
    year, month = value["year"], value["month"]
    if type(year) is not int or type(month) is not int:
        raise ValueError("Cutoff year/month must be integers")
    return date(year, month, calendar.monthrange(year, month)[1]).isoformat()


def load(directory: Path, *, split: str = "validation", scope: str = "eee") -> dict:
    manifest = verify_download(directory, "matter_of_fact")
    if split not in FILES or scope not in ("eee", "all"):
        raise ValueError("Unknown Matter-of-Fact split or scope")
    rows = json.loads((directory / "benchmark" / FILES[split]).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of benchmark records")
    seen, inputs, gold = set(), [], {}
    for row in rows:
        identity, text, label = row["claim_id"], row["claim_text"], row["gold_label"]
        metadata = row["metadata"]
        if not isinstance(identity, str) or identity in seen:
            raise ValueError("Duplicate or invalid source claim ID")
        seen.add(identity)
        if not isinstance(text, str) or not text.strip() or type(label) is not bool:
            raise ValueError("Expected nonempty claim and strict boolean feasibility label")
        if not isinstance(metadata["paper_id"], str) or not metadata["paper_id"]:
            raise ValueError("Source paper reference is missing")
        topics = metadata["topics"]
        if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            raise ValueError("Invalid topic metadata")
        cutoff = cutoff_date(metadata["exclude_date"])
        if scope == "eee" and not EEE_THEMES.intersection(topics):
            continue
        # Original IDs expose the T/F answer. Opaque IDs are for joining only.
        case_id = "mof-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if case_id in gold:
            raise ValueError("Opaque case ID collision")
        inputs.append({"case_id": case_id, "claim_text": text,
                       "evidence_not_after": cutoff})
        gold[case_id] = {"label": label, "source_claim_id": identity,
                         "source_paper_id": metadata["paper_id"],
                         "topics": topics, "evidence_not_after": cutoff}
    if not inputs:
        raise ValueError("No cases match the selected scope")
    protocol = {"split": split, "scope": scope, "eee_themes": sorted(EEE_THEMES),
                "task": "binary literature-supported feasibility, not three-class NLI",
                "cutoff_policy": "inclusive end of metadata.exclude_date month"}
    version = hashlib.sha256(json.dumps({"manifest": manifest, "protocol": protocol}, sort_keys=True).encode()).hexdigest()
    return {"dataset": "Matter-of-Fact", "version": version, "manifest": manifest,
            "protocol": protocol, "source_split_count": len(rows), "inputs": inputs, "gold": gold}


def prepare(dataset: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "inference-input.json").write_text(json.dumps({
        "dataset": dataset["dataset"], "version": dataset["version"],
        "protocol": dataset["protocol"], "cases": dataset["inputs"],
        "input_policy": "Only claim_text and cutoff inform inference. case_id is an opaque join key, never a prompt feature.",
    }, indent=2) + "\n")
    (output / "evaluation-gold.json").write_text(json.dumps({
        "version": dataset["version"], "gold": dataset["gold"],
        "warning": "Evaluator-only; do not provide source IDs, labels or topics to inference.",
    }, indent=2) + "\n")
    (output / "predictions-template.json").write_text(json.dumps({
        "status": "pending", "dataset_version": dataset["version"],
        "model": "replace with actual model/revision/protocol",
        "predictions": [{"case_id": row["case_id"], "prediction": None, "evidence": []}
                        for row in dataset["inputs"]],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/matter_of_fact"))
    parser.add_argument("--split", choices=list(FILES), default="validation")
    parser.add_argument("--scope", choices=["eee", "all"], default="eee")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = load(args.data, split=args.split, scope=args.scope)
    prepare(dataset, args.output_dir)
    print(json.dumps({"source_split_count": dataset["source_split_count"],
                      "selected_cases": len(dataset["inputs"]),
                      "version": dataset["version"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
