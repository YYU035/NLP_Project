# Retrieval — BM25, SBERT & Hybrid (RRF) · Naila

Ranks papers for a claim with lexical (BM25), semantic (SBERT) and fused (Reciprocal Rank Fusion) retrieval. It runs on the fixed SciFact corpus for evaluation and on Person 1's live-search candidates for the demo.

## Setup

Run from the **repo root**:

```bash
pip install -r retrieval/requirements.txt
python -m retrieval.download_scifact        # -> data/scifact/*.jsonl
python -m pytest retrieval/tests -q         # 13 tests, no downloads needed
```

## Run the benchmark comparison

```bash
python -m retrieval.run_scifact --split dev
python -m retrieval.run_scifact --split dev --sbert-model sentence-transformers/allenai-specter
```

The first run encodes the 5,183 abstracts. On CPU, MiniLM takes a few minutes and SPECTER is slower. Embeddings are cached in `retrieval/cache/`, so later runs skip encoding.

Outputs go to `retrieval/outputs/<split>_<model>/`:

| File | What it is |
|---|---|
| `results.md` | Comparison table: MRR, Recall@1/5/10/20/100, nDCG@10, median and p95 query latency, plus index build time |
| `metrics.json` | Summary and per-claim metrics for each method |
| `failures.json` | Claims where only BM25, only SBERT, or neither found a relevant paper in the top 10 |
| `runs_rrf.jsonl` | **Handoff to Person 3**: top-10 RRF results per claim, with paper text attached |
| `runs_bm25.jsonl`, `runs_sbert.jsonl` | Top-100 per branch, for analysis |
| `config.json` | Every setting, model name, library version and git commit used |

**Splits.** SciFact test labels are not public, so report numbers on **dev**. If you tune anything (BM25 `k1`/`b`, `k_rrf`, model choice), tune on train and report dev once.

**Relevance.** The relevant papers for a claim are the doc IDs in its `evidence` field, the standard SciFact/BEIR qrels. NEI claims have no evidence documents, so they are excluded from retrieval metrics. They still matter for Person 4.

## Rank live-search candidates

```bash
python -m retrieval.search_live --papers path/to/person1_output.json \
    --claim "Vitamin D supplementation reduces fracture risk" --method rrf --k 10
```

```python
from retrieval.search_live import rank_candidates
ranked, papers = rank_candidates(claim, person1_records, method="rrf", k=10)
handoff = ranked.to_dict(papers)   # -> Person 3
```

Papers with no title and no abstract are left out of the index and listed in `settings.skipped_no_text`.

## Interface

All three methods share one interface:

```python
from retrieval.retriever import Retriever, build_all
r = Retriever("rrf").index(papers)          # or "bm25" / "sbert"
ranked = r.search(claim, k=10, claim_id="13")
```

`build_all(papers)` indexes BM25 and SBERT once and shares them across the three methods, so the comparison is like-for-like.

### Output schema (`RankedList`, one per claim per method)

```json
{
  "claim_id": "13", "claim": "<original claim, unchanged>", "method": "rrf",
  "corpus": "scifact", "latency_ms": 4.2, "settings": {...},
  "results": [{
    "paper_id": "4983", "rank": 1, "score": 0.0325, "method": "rrf",
    "component_ranks":  {"bm25": 1, "sbert": 2},
    "component_scores": {"bm25": 18.4, "sbert": 0.71},
    "paper": {"paper_id": "4983", "title": "...", "abstract": ["sent 0", "sent 1"],
              "full_text": null, "text_status": "abstract_only",
              "source": "scifact", "year": null, "url": null}
  }]
}
```

- **`abstract` is a list of sentences.** SciFact rationales are sentence indices into this list, so Person 3 should use these sentences as they are and not re-split them. Otherwise evidence F1 can't be computed.
- `component_ranks` is `null` for a branch that didn't return the paper in its top `fusion_depth`.

### Input from Person 1

`data.paper_from_live_record` accepts common field names: `paper_id`/`id`/`doi`/`paperId`, `title`, `abstract`/`summary` (a string or a list of sentences), `full_text`, `text_status`, `year`/`publication_date`, `url`, `source`. Once Person 1's format is final, map it there.

## Method details

- **BM25**: Okapi BM25 over title and abstract, with k1=0.9 and b=0.4 (Anserini/BEIR defaults, so results are comparable to published SciFact baselines). It uses Lucene-style IDF, lowercasing, stopword removal and the Porter stemmer.
- **SBERT**: L2-normalised `sentence-transformers` embeddings with cosine similarity. The default model is `all-MiniLM-L6-v2`. `allenai-specter` and `multi-qa-mpnet-base-cos-v1` are worth comparing.
- **RRF**: `score(d) = Σ 1/(k_rrf + rank_m(d))` over the top `fusion_depth`=100 from each branch, with `k_rrf`=60.
- **Latency** is per-query search time after a warm-up query, reported as median and p95. Index build time is reported separately.

## Limitations

- On live candidate sets of only tens of papers, BM25's IDF is estimated from very few documents, so it is noisier than on SciFact.
- Only titles and abstracts are indexed. Full text is passed through for Person 3 but not used for ranking.
