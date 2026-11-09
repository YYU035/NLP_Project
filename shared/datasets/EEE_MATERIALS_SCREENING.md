# EEE and Materials Dataset Screening

Date: 2026-11-09. Status: screening completed; no new primary dataset adopted.
The user declined the climate-science domain. ClimateCheck is no longer the
selected third dataset. Existing code/results remain as archived exploration.
SciFact and CSFCube remain the two confirmed evaluation datasets.

## Recommended candidate: RECIPER

Primary sources:

- Authors' code/data: https://github.com/ReaganWu/RECIPER
- Official release: https://huggingface.co/datasets/ReaganWZY/RECIPER
- Paper and construction method: https://arxiv.org/html/2604.11229v1

The released files were actually downloaded and inspected, not just described
from a search result. Pinned dataset revision:
`1e167badbf19d6fd78b2a49700028f4c68134219`.
Local files and SHA-256 manifest: `data/candidate_audits/reciper/` (Git-ignored).
The corpus is 46,475,801 bytes; QA file is 1,137,390 bytes.

Measured structural audit:

| Check | Observed |
|---|---:|
| Distinct source papers | 343 |
| Unique questions | 1,024 |
| QA source-paper references missing from corpus | 0 |
| Distinct papers referenced by QA | 343 |
| Sections | 2,763 |
| Original paragraph entries | 12,162 |
| Recipe entries | 544 |

QA records contain `question`, `answer`, `topic`, `paper_id`, `paper_title`,
`source`, `source_type`. Corpus records contain title, abstract, bibliographic
metadata, full-text sections and recipes. These provide a complete small fixed
corpus and known source-document targets without reconstructing a historical
literature index.

### Domain fit

This is materials research rather than a general climate/public-discourse task.
Inspection found battery, electronic-materials, dielectric and superconductivity
content, including a paper on lithium-ion cell state-of-power estimation and one
on graphene electron-transport layers in perovskite solar cells. It is adjacent
to EEE, not a complete electrical-engineering syllabus or a purely semiconductor
collection. Author affiliations/arXiv categories alone are not domain labels.

A mechanical title/abstract substring check found `battery` in 53 papers,
`electronic` in 75, `dielectric` in 13, `semiconductor` in 6 and `superconduct` in
15. These overlap and are not expert domain judgments. Do not silently use
these counts to declare an EEE subset; any subset needs a frozen selection rule
and review. Using the full materials collection avoids selecting easy examples.

### What it can evaluate honestly

The source `paper_id` supports source-paper Recall@K, MRR and nDCG, using all
343 papers as the candidate collection. Questions, rather than claims, are
legitimate input for this supplemental academic retrieval test; preserve them
unchanged instead of inventing scientific assertions.

The supplied evaluation code collapses retrieved chunks to distinct paper IDs
and tests against the single source paper. Therefore these are **source-paper
recovery metrics**, not exhaustive relevance judgments over every other paper.
A non-source paper may also answer a question. Preserve and report that limit.

No additional retrieval benchmark model was run during this screening. The
paper's published model scores are not project results.

### Important annotation boundary

The paper says questions/answers were generated with an LLM. The QA file has
no expert-labeled paragraph IDs, sentence rationales, answer character offsets,
or Supported/Contradicted/NEI verdicts. Recipe summaries are also generated.
The availability of paragraphs does not mean they are annotated gold evidence.
Do not treat all paragraphs of a correct paper as relevant evidence, or treat
the generated answer/recipe as human verification gold.

RECIPER is a practical candidate for Person 3 paper retrieval evaluation. It
will not independently establish sentence-evidence correctness or three-way
claim verification. SciFact remains the benchmark for those evidence metrics.
If the requirement is expert evidence gold in every dataset, RECIPER is not a
complete replacement and should not be adopted on that basis.

## Secondary candidate: solar-cell SCQA

Primary sources:

- https://huggingface.co/datasets/CambridgeMolecularEngineering/SCQA_FirstTurn
- https://huggingface.co/datasets/CambridgeMolecularEngineering/SCQA_SecondTurn
- Construction paper: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00307a

The FirstTurn release was actually downloaded and checked at revision
`2b439d1e6fdf94380d984b37cfc7ffe042e73fcc`.
Files/manifest: `data/candidate_audits/scqa_first_turn/` (Git-ignored).
SecondTurn was inspected through its metadata/card, not downloaded or evaluated.

| FirstTurn check | Train | Validation |
|---|---:|---:|
| Rows | 34,305 | 8,577 |
| Distinct question texts | 124 | 90 |
| Source DOI values | 6,250 | 4,025 |
| Answer spans failing exact substring check | 0 | 0 |

The files include source DOI (in `title`), context, question and answer offsets.
Source overlap: 3,814 DOI values and 3,968 exact context strings occur in both
train and validation. These are structural observations, not proof that an
untuned pretrained model leaked labels. They matter if training/tuning a model
or claiming source-independent generalization.

This domain is directly relevant to photovoltaics/EEE and the offsets support
extractive answer evaluation. However, many questions such as asking for a
voltage value make sense only with the supplied context. The same short query
can refer to different devices and papers. A valid offset also does not prove
that the selected answer semantically answers its question. The construction
paper describes automatic QA generation rather than full expert evidence gold.

Recommendation: consider SCQA for a separately scoped context-conditioned
property-extraction experiment, not as a direct open paper-retrieval benchmark.
Do not convert its random row split into a claim of document-independent testing.

## Other screened options

- EEE-Bench / MaScQA: engineering/materials problem solving, not a fixed
  literature-retrieval/evidence annotation task. Not preferred for this pipeline.
- The 2023 materials extractive-QA repository exposes a demo notebook and one
  sample PDF; a ready released labeled benchmark was not verified there.
  https://github.com/anaghasavit/material-science-extractive-QA
- A recent superconductivity RAG paper describes manually checked records and
  a 100-question experiment, but an accessible benchmark download was not
  verified in the checked article. Do not count it as integrated data.
  https://arxiv.org/html/2610.34391v1

## Proposed next implementation, if RECIPER is selected

1. Freeze the release and use original paragraph/abstract text. Keep gold answers,
   source labels and generated recipes out of the initial baseline features.
2. Retrieve against the complete 343-paper collection. For chunk retrieval,
   collapse chunk hits to unique paper IDs before source-paper metrics. Declare
   a fixed aggregation rule and avoid a gold-preselected per-query candidate pool.
3. Compare actual P2 BM25/SBERT and P3 RRF/RRF+CE on exactly the same corpus/queries.
   Tune only on a declared source-paper-disjoint development subset, or perform
   a clearly stated frozen zero-shot evaluation without tuning.
4. Report Recall@1/5/10, MRR/nDCG and measured latency. Label this as materials
   source-paper retrieval; retain SciFact's separate evidence P/R/F1.
5. If sentence/paragraph correctness is required for this domain, obtain
   independently validated evidence annotations before reporting evidence F1.

Screening audit: `outputs/datasets/eee-materials-candidate-screening.json`.
No models were trained/evaluated for these candidates and no Git push occurred.
