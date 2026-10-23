# Smart Document Search (v2 - hybrid retrieval)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # UI
python evaluate.py            # baseline vs. each module (P@3, R@10, MRR, nDCG@10)
```
Drop `.txt/.md/.pdf` files into `docs/` (or upload in the sidebar). Edit `thesaurus.json` for your domain.
Optional: `pip install sentence-transformers` and tick "Use neural embeddings" for real semantic search.

## Architecture
```
 INDEXING (once, cached)                       QUERY TIME
 ───────────────────────                       ─────────────────────────────────────────────
 .txt/.md/.pdf                                 raw query
   │ load_folder / upload                        │
   ▼                                             ▼
 passage chunking (~70 words,               [1] Typo layer: OOV tokens only, candidates from the
   sentence-aligned)                              CORPUS vocabulary (rapidfuzz Damerau-Levenshtein),
   │                                              guards: min length, same 1st letter, no morph. variants
   ▼                                             ▼
 CountVectorizer (stemmed vocab)            [2] Weighted expansion  (typed=1.0)
   ├─► BM25 matrix ─────────────┐                 • domain thesaurus      (0.4)
   ├─► TF-IDF ─► SVD/LSA (Z) ───┤                 • LSA term neighbours   (≤0.3, high threshold)
   └─► [optional] dense         │                 • pseudo-relevance fb.  (≤0.2, not highlighted)
       embeddings (E)           │                 ▼
                                │            [3] Two retrieval channels
 doc-level aggregates ──────────┤                 BM25 (lexical)  ║  LSA / embeddings (semantic)
 (for corpus topic map)         │                 ▼
                                └───────────► [4] Reciprocal Rank Fusion (k=60, semantic weight 0.7)
                                                  ▼
                                             [5] Passages → documents (best passage wins)
                                                  ▼
                                             [6] Snippet = best 2-sentence window by weighted
                                                 coverage+density; HTML-escaped highlighting
                                                 (yellow = typed, blue = expanded)
                                                  ▼
                                             [7] Cluster the RESULT SET: LSA vectors → KMeans,
                                                 k by silhouette, labels = distinctive TF-IDF terms
                                                 (query terms excluded) → topic filter in UI
```
Modules: `ir_engine.py` (engine, no UI) · `app.py` (Streamlit) · `evaluate.py` · `thesaurus.json` · `eval_queries.json`

## Setbacks of the original approach and how v2 addresses them
| # | Original design | Problem | v2 |
|---|---|---|---|
| 1 | `difflib` against a general dictionary / unrestricted vocab | Corrupts domain terms; O(V) pure-Python scan; corrects words that are already valid | rapidfuzz against corpus vocabulary, OOV-only, guards against false corrections (found and fixed `respond→responder`, `long→log` during testing) |
| 2 | WordNet synonym expansion, equal weights | Query drift (virus→biology, trojan→Troy); dilutes precision | Editable domain thesaurus, expansion weighted 0.3-0.4, capped per term, filtered to terms present in corpus |
| 3 | Claims to solve "semantic blindness" with synonyms | Synonym lists are not semantics | Separate semantic channel (LSA by default, neural embeddings optional) fused with BM25 via RRF; claim reduced to "mitigates" |
| 4 | TF-IDF cosine at document level | Long docs dominate; weak ranking; no answer location | BM25 on passages, doc score = best passage |
| 5 | Snippet = keyword-density paragraph, bold | Highlights only typed words; unescaped HTML risk | Sentence-window scoring, highlights typed vs. expanded differently, HTML-escaped |
| 6 | K-Means on full corpus, fixed k, raw sparse TF-IDF | Doesn't fix "broad query = jumble" (clusters corpus, not results); K-Means is poor on sparse high-dim data; unlabeled clusters | Cluster the *retrieved results* in LSA space, k by silhouette, auto-labels, falls back to one group if structure is weak |
| 7 | No evaluation | Can't prove any module helps | `evaluate.py` ablation across 6 configurations |
| 8 | Typo correction applied blindly | Hurts queries when it fires wrongly | Per-module toggles + "search exactly as typed" button |

## Limitations:
* LSA is *latent co-occurrence*, not true semantics; on small corpora it is noisy. Use embeddings for real conceptual matching.
* Thesaurus needs manual curation per domain; the bundled one is a starter.
* Clustering needs enough results (≥6) and enough corpus to be meaningful; labels are keyword-based.
* Typo layer cannot fix a typo whose correct form is absent from the corpus, or valid words missing from it.
* Bundled eval set: 13 docs / 12 queries written by the same author as the thesaurus, so scores are optimistic. Build judgments on your real corpus.
* Pure-Python stemmer is crude (e.g. `routing`/`route`); swap in NLTK Snowball if needed.
* Index is rebuilt in memory (fine up to ~10^4-10^5 passages); beyond that use FAISS/Elasticsearch.
* Scanned PDFs need OCR (not included).
