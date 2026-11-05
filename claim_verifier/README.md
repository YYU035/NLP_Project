Here is a professional, complete `README.md` for your module. It documents your design choices, interface contracts for Person 3 and Person 5, benchmark results, and instructions for running tests.

Create a file named **`README.md`** inside your `claim_verifier/` folder and paste the following:

---

````markdown
# Module 4: Claim Verification & Evidence Aggregation

**Owner:** Chong Yao (Person 4)  
**System Role:** Classify claim–evidence relationships using deep Natural Language Inference (NLI), enforce scientific safety guards, and aggregate passage-level findings into transparent paper-level and corpus-level verdicts.

---

## 1. Overview & Architectural Boundaries

This module acts as the scientific reasoning engine in the verification pipeline:

- **Upstream Interface:** Consumes ranked evidence passages (`EvidencePassage`) from **Person 3 (Yu Yue)**.
- **Downstream Interface:** Produces structured verdict payloads (`VerificationResponse` and `AggregationResponse`) consumed by **Person 5 (Kireeti)** for Streamlit rendering and explanation generation.

### Decoupled Two-Stage API Design

To ensure sub-second UI responsiveness in Streamlit without repeatedly running expensive neural networks:

1. `verify(claim, papers, passages_by_paper) -> VerificationResponse`
   - Heavy neural inference pass using DeBERTa-v3 on GPU.
   - Run **once** per user search query.
   - Caches raw probabilities, truncation statuses, and per-paper verdicts.
2. `aggregate(verification_response, subset_of_paper_ids) -> AggregationResponse`
   - Lightweight, deterministic re-aggregation in pure Python.
   - Executed **on-the-fly (< 50ms)** when a user checks/unchecks papers in the UI.
   - Zero GPU/neural forward passes.

---

## 2. Scientific & Safety Guards

1. **Strict Input Orientation:**
   - **Premise** = Evidence Passage Text.
   - **Hypothesis** = Original Claim (preserved verbatim without query modification).
2. **Truncation Guard:**
   - Passages exceeding the 512-token limit are flagged as `TRUNCATED` and automatically **excluded from confident aggregation** to prevent false deductions caused by missing conditional clauses or negations.
3. **Threshold Calibration & Uncertainty Discipline:**
   - A decision threshold tuned strictly on the SciFact dev benchmark (frozen at **`0.50`**).
   - Below-threshold classifications are forced to `NEUTRAL` and tagged with `SufficiencyCause.MODEL_UNCERTAINTY`.
4. **Per-Paper Voting & Internal Conflict Detection:**
   - Aggregations are capped at a maximum of 3 passages per paper so that verbose papers cannot dominate.
   - Papers showing simultaneously high Support and Contradiction evidence trigger `conflict_flag=True` and downgrade to `INSUFFICIENT`.
5. **Separation of Operational Failures vs. Scientific Verdicts:**
   - System/processing errors (`PROCESSING_FAILURE`, `NO_PASSAGE_AVAILABLE`, `TRUNCATED`) are tracked as operational metadata, distinct from scientific verdicts (`INSUFFICIENT`).

---

## 3. Directory Layout

```text
claim_verifier/
├── schemas.py           # Pydantic data schemas, enums & taxonomy definitions
├── verifier.py          # Core ClaimVerifier engine (verify and aggregate)
├── test_verifier.py     # Unit test suite (label mapping, truncation, conflicts)
├── calibrate_dev.py     # SciFact dev dataset loader & threshold calibration runner
├── demo_for_person5.py  # Quickstart integration script for UI handoff
├── data/                # SciFact dataset cache (corpus.jsonl, claims_dev.jsonl)
└── README.md            # Module documentation
```
````

---

## 4. Setup & Installation

### Requirements

- Python 3.10+ (tested on Python 3.13)
- PyTorch (CUDA GPU recommended; automatically falls back to CPU)
- Transformers, Pydantic, Pytest

Install dependencies:

```bash
pip install torch transformers pydantic pytest
```

---

## 5. Model Configuration & Dev Calibration

- **Base Checkpoint:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Precision:** `torch.float16` on CUDA (automatic fallback to `torch.float32` on CPU)
- **Token Max Length:** 512 tokens
- **Frozen Decision Threshold:** `0.50` (tuned on SciFact dev)

### SciFact Dev Benchmark Results (Oracle Setting)

Evaluated on **469** ground-truth claim–evidence pairs extracted from the SciFact dev split:

| Metric            | Score      |
| :---------------- | :--------- |
| **Accuracy**      | **64.61%** |
| **Macro-F1**      | **0.6637** |
| **Contradict F1** | **0.7814** |
| **Neutral F1**    | **0.6127** |
| **Support F1**    | **0.5968** |

#### Confusion Matrix

```text
Gold / Pred    | SUPPORT      | CONTRADICT   | NEUTRAL
-----------------------------------------------------------
SUPPORT        | 94           | 6            | 116
CONTRADICT     | 2            | 84           | 36
NEUTRAL        | 3            | 3            | 125
```

> **Key Finding:** Cross-talk between `SUPPORT` and `CONTRADICT` is exceptionally low (only 8 fatal confusions out of 469 cases, representing a **98.3% safety rate** against dangerous contradictions). Errors are predominantly conservative shifts into `NEUTRAL`.

To reproduce the benchmark sweep:

```bash
python calibrate_dev.py
```

---

## 6. Integration Contract for Teammates

### For Person 3 (Yu Yue) — Passage Inputs

Ensure candidate evidence conforms to the `EvidencePassage` schema:

```python
from schemas import EvidencePassage

passage = EvidencePassage(
    passage_id="doc123_s2",
    paper_id="doc123",
    section="abstract",
    text="Exact text of the evidence extracted from the abstract or PDF.",
    score=0.88
)
```

### For Person 5 (Kireeti) — Streamlit UI Integration

Run the demo to inspect the live handoff:

```bash
python demo_for_person5.py
```

Example integration inside Streamlit:

```python
from verifier import ClaimVerifier
from schemas import EvidencePassage

# 1. Initialize once
verifier = ClaimVerifier()

# 2. Heavy Neural Inference (run once per claim search)
verification_response = verifier.verify(claim, papers, passages_by_paper)

# Display per-paper verdicts
for paper in verification_response.per_paper_verdicts:
    st.write(f"Paper: {paper.paper_id} | Verdict: {paper.verdict.value} (Conf: {paper.confidence:.2f})")
    if paper.conflict_flag:
        st.warning("Conflicting evidence detected inside this paper!")

# 3. Fast UI Filter (run when user toggles checkboxes - < 50ms)
selected_ids = ["paper_1", "paper_2"]
ui_verdict = verifier.aggregate(verification_response, subset_of_paper_ids=selected_ids)

st.header(f"Overall Verdict: {ui_verdict.aggregated_verdict.value}")
```

---

## 7. Unit Tests

Run the test suite to validate label mappings, truncation guards, conflict flags, and sub-second aggregation:

```bash
python -m pytest -v -s test_verifier.py
```

```

```
