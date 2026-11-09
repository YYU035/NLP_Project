"""Check training boundaries and evaluator coverage for the claim-only baseline."""

import pytest

pytest.importorskip("sklearn")

from ranking_evaluation.run_matter_baseline import run


def dataset(split):
    inputs, gold = [], {}
    for i in range(12):
        identity = f"{split}-{i}"
        inputs.append({"case_id": identity,
                       "claim_text": f"{split} example {i} " + ("stable efficient" if i % 2 else "unstable impossible")})
        gold[identity] = {"label": bool(i % 2), "source_paper_id": f"{split}-{i}v1",
                          "evidence_not_after": "2022-12-31"}
    return {"version": split, "protocol": {"split": split, "scope": "eee"},
            "inputs": inputs, "gold": gold}


def test_real_classifier_evaluator_and_paper_overlap():
    train, target = dataset("train"), dataset("validation")
    result = run(train, target)
    assert result["evaluation"]["cases"] == 12
    assert len(result["predictions"]["predictions"]) == 12
    assert all(row["evidence"] == [] for row in result["predictions"]["predictions"])
    target["gold"]["validation-0"]["source_paper_id"] = "train-0v2"
    with pytest.raises(ValueError, match="paper overlap"):
        run(train, target)


def test_reject_training_evaluation_and_claim_overlap():
    train, target = dataset("train"), dataset("validation")
    with pytest.raises(ValueError, match="separate split"):
        run(train, train)
    target["inputs"][0]["claim_text"] = train["inputs"][0]["claim_text"]
    with pytest.raises(ValueError, match="claim text overlap"):
        run(train, target)
