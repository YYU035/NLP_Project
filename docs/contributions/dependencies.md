# Dependencies, models and operating boundaries

This inventory does not replace original licenses or provider terms. Exact install versions live in `requirements/` and `pyproject.toml`; adaptation records preserve teammate source provenance.

## Software attribution

| Software | Role / handling |
|---|---|
| Python, Streamlit, Pydantic | Runtime, UI and contracts; retain distribution licenses |
| NumPy, Sentence-Transformers | Custom BM25 calculations and embeddings; BM25 is teammate code, not rank-bm25 |
| PyTorch, Transformers, Hugging Face Hub | Local inference and explicit setup downloads; hosted inference unused |
| spaCy, KeyBERT, httpx | Local discovery query generation and catalogue transport |
| PyMuPDF | Uploaded PDF extraction; retain its own distribution licensing terms |
| Captum | Optional experimental IG with numerical gates |
| pytest | Software testing, separate from scientific measurements |
| Archify 3.2.0 | Architecture snapshot; preserve bundled license/notice/third-party files together |

No general license for all teammate source was present at the audited pins. Preserve authorship, revisions and applicable notices; do not invent a project-wide license from model/dependency licenses. Public source availability is not a blanket redistribution license.

The lexical implementation's `porter` setting uses NLTK Porter when NLTK is installed and otherwise its bundled suffix-stripper fallback. NLTK is absent from the audited environment and pinned runtime lock. Adding it can change retrieval results and requires a separately declared comparison; the configured setting name alone does not establish which implementation ran.

## Model pins

| Model | Revision | Role / recorded license |
|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | `2a7cca6ad7605f3183bce07f677bda658059bbe9` | Production retrieval; pinned model card/license |
| sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Evidence selection; pinned model card/license |
| MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` | NLI; recorded MIT |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | Optional local generation; recorded Apache-2.0 |

Explicit setup retains public file revision/hash manifests. Runtime loads local files with remote code disabled. Models remain ignored under the configured data directory. Qwen is not the default explanation method and its recorded 12-case QC had no accepted raw generations.

## Datasets

| Dataset | Evaluation and provenance |
|---|---|
| SciFact | Corpus/claim/rationale files and declared splits/hashes in frozen protocols |
| CSFCube | Computer Science retrieval, pin `7ffe012c1d92c470f185fc086086caf5b8e05ef3`; retain original license/datasheet |
| RECIPER | Materials Science source-paper retrieval, pin `1e167badbf19d6fd78b2a49700028f4c68134219`; retain original license/readme and adaptation |

Download tools preserve official files and manifests. Compact result snapshots do not include tracked full model weights/corpora. CSFCube/RECIPER lack the NLI/rationale gold used by the verifier study.

## External calls

Live discovery permits public OpenAlex, OpenAIRE and arXiv clients without forwarding credentials. Availability and provider terms remain external dependencies. Inspectable paper links do not mean full text was read.

Setup may download public model, language-package and dataset files. Retrieval, selection, verification, aggregation, explanations, uploaded-PDF processing and replay are local. There is no paid inference dependency, hosted judge, explanation API-key requirement or automatic paid fallback. Unused hosted relevance-evaluation experiments are excluded from the bundled discovery runtime.
