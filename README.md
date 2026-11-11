# Research Claim Verifier

Inspect scientific claims against traceable research evidence. The integrated application connects public paper discovery, BM25 and semantic retrieval, Reciprocal Rank Fusion, exact evidence selection, local DeBERTa inference and cited explanations in Streamlit.

This is the combined five-person project. Functional packages replace the original contribution folders; authorship and source pins remain in the [contribution record](docs/contributions/index.md).

## Install and run

Use **Python 3.12**. From a Windows checkout:

```powershell
.\setup.cmd
.\run.cmd
```

Setup creates `.venv`, installs pinned dependencies and bundled source packages, and explicitly provisions required public model files. See `scripts/setup.ps1` for available options. Initial setup needs internet and disk space; DeBERTa weights alone are about 870 MB. The app opens at `http://127.0.0.1:8501`.

Use `setup.cmd -SkipModels` for software-only setup, or `setup.cmd -WithInterpretability` to add Captum. Optional local generation is provisioned explicitly with `.venv\Scripts\python.exe -m research_verifier.cli setup-models --language-model`; template explanations remain the default.

The `.cmd` entry points invoke PowerShell with a process-scoped execution-policy setting, without changing permanent machine policy. Discovery is bundled in `src/paper_search/`; installation no longer fetches application code from an old private Git revision. `.env.example` documents settings and is not automatically loaded.

## Follow one claim

1. In **Find papers**, enter a claim, load a recorded search or import a search. Inspect candidate abstracts, provenance and advisory quality warnings, then select papers.
2. Prepare the selection and open **Verify evidence**. Retrieval ranks available selected records; exact passages feed the local verifier.
3. Inspect Supported, Contradicted, Conflicting or Insufficient with citations and source scope. Processing failures remain separate from insufficient evidence.
4. Change the paper selection to reuse stored predictions, then export JSON and reload through Cached replay.

The seven workspaces are **Project overview**, **Find papers**, **Verify evidence**, **Local PDF**, **Benchmark results**, **Model attribution** and **Evaluate explanations**. PDF upload preserves page references; saved runs support inspection without catalogue availability.

```text
Claim → discovery → BM25 + semantic retrieval → RRF → exact evidence
      → DeBERTa → scoped aggregation → cited explanation → export/replay
```

The primary abstract policy uses up to three semantic anchors, exact whole-abstract context when the claim pair fits 512 tokens, and a 0.65 paper relevance gate. The focused baseline remains an explicit comparison route; PDF selection is separate.

**No paid or hosted inference API is required.** Fresh Live search sends the claim to public OpenAlex, OpenAIRE and arXiv catalogues. Setup downloads public model files explicitly. Inference, ranking, templates, PDF extraction and replay run locally; missing weights do not trigger a hosted fallback.

See the [architecture](docs/architecture/overview.md), [interactive historical map](docs/architecture/project_map/project-summary.html) and [dependency/source inventory](docs/contributions/dependencies.md).

## Existing measured results

These frozen studies predate consolidation; cleanup is not a new accuracy experiment.

| Internal SciFact audit: 209 reserved claims | Original baseline | Primary policy |
|---|---:|---:|
| Accuracy | 56.5% | 67.9% |
| Macro F1 | 0.583 | 0.679 |
| Contradiction precision | 47.8% | 84.0% |
| Contradiction recall | 82.1% | 53.8% |
| Input evidence F1 | 0.210 | 0.165 |

The candidate fixes 45 errors and introduces 21 regressions. This is an internal audit of published training records, **not an official hidden test**. Overall accuracy improves while contradiction recall and rationale coverage fall.

Separate RRF Recall@10 results are **95.6% SciFact**, **39.2% CSFCube test** and **93.7% RECIPER**. Tasks, candidate pools and model/preprocessing protocols differ. CSFCube/RECIPER do not establish verifier accuracy. Read the [results and complete protocols](docs/results/index.md) before quoting scores.

## Validate

```powershell
.\test.cmd
.\validate.cmd
```

Software checks cover contracts, failures, source preservation, UI flows, replay and cached aggregation. Actual-model smoke and scientific evaluations are separate. The [migration validation report](docs/results/migration_validation.md) records checks rerun on the renamed release; historical test counts stay in their dated reports. Do not tune new policies on the inspected frozen audit.

After model setup, `validate.cmd -WithModels` adds the recorded-search real-model smoke. `validate.cmd -WithModels -Live` also makes an explicit public-catalogue search. `research-verifier check` runs only the local artifact/provenance checks. Default `test.cmd` and `validate.cmd` do not download or infer with models.

| Path | Purpose |
|---|---|
| `src/research_verifier/` | Functional pipeline, UI and runtime |
| `src/paper_search/` | Bundled discovery implementation |
| `tests/` | Software tests and controlled inputs |
| `examples/` | Recorded searches, scenarios, replays and benchmark snapshots |
| `evaluation/` | Metrics, protocols, frozen studies and review evidence |
| `config/` | Versioned policy settings |
| `scripts/` | Setup, launch and validation |
| `docs/` | Architecture, results, contributions and history |
| `data/`, `.cache/`, `outputs/` | Ignored downloads, models and run artifacts |

## Limits and review

A verdict describes selected evidence, not scientific truth; confidence is not calibrated correctness. Provider coverage, PDF reading order and population/generalisation errors remain limitations. OCR/scanned-PDF interpretation and arbitrary paper-URL fetching are unsupported.

Word-MASK sensitivity is available. IG remains experimental after failed real-case numerical gates. All 12 recorded raw local-Qwen generations failed the strict schema and used templates. The prepared human explanation-review study has no completed ratings.

Historical documents retain their original paths, terminology and counts. Start here for current navigation; see [history and provenance](docs/history/README.md).
