# Information Retrieval System for Medical Document Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Coursework for the *Information Retrieval* course at the University of Patras.
The project builds and evaluates several document-ranking models on the
**Cystic Fibrosis** medical literature collection (1,239 articles, 20 queries
with ground-truth relevance judgments), then explores document clustering with
both TF-IDF and semantic embeddings.

The code lives in Jupyter notebooks — there is no installable package or CLI.

## What's inside

| Notebook | What it does |
| --- | --- |
| `testing.ipynb` | Custom TF-IDF implementation (three TF weightings) + per-query evaluation (Precision@k, Recall@k, F1@k, MAP). |
| `vectorizer.ipynb` | scikit-learn `TfidfVectorizer` with a 270-combination grid search over the weighting/normalisation parameters. |
| `kmeans-vect.ipynb` | K-Means / agglomerative clustering on the TF-IDF vectors (elbow + silhouette analysis). |
| `embeddings.ipynb` | Semantic clustering with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim). |

Data files: `Queries.txt` (20 queries), `Relevant.txt` (relevance judgments),
and `docs/` (the document collection).

## Results

Retrieval (custom TF-IDF vs. grid-searched scikit-learn TF-IDF):

| Model | F1@k | Precision | Recall | MAP |
|-------|------|-----------|--------|-----|
| Raw TF-IDF | 0.2817 | 0.0392 | 0.9009 | 0.2620 |
| Log TF-IDF | 0.2817 | 0.0392 | 0.9009 | 0.2616 |
| Aug TF-IDF | 0.2753 | 0.0392 | 0.9009 | 0.2583 |
| **Optimised TF-IDF** | **0.3006** | **0.0420** | **0.9200** | **0.2750** |

*Best parameters: `ngram_range=(1,1)`, `sublinear_tf=True`, `min_df=1`, `max_df=0.7`, `norm='l2'`.*

Clustering:

| Method | Best k | Silhouette |
|--------|--------|-----------|
| K-Means (TF-IDF) | 17 | 0.18 |
| Agglomerative (TF-IDF) | 50 | 0.22 |
| K-Means (embeddings) | 12 | **0.31** |

**Takeaways.** On these short abstracts (~81 words avg, most terms appearing
≤2×) plain raw TF works as well as log/augmented weighting, and grid search
mainly helped by tuning `max_df` to drop over-common terms (F1@k 0.282 →
0.301). For clustering, sentence-transformer embeddings separated documents
noticeably better than TF-IDF vectors (silhouette 0.31 vs 0.18–0.22).

## Running it

```bash
git clone https://github.com/thonos-cpu/TFIDF-Search-Engine.git
cd TFIDF-Search-Engine

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy scikit-learn sentence-transformers yellowbrick matplotlib jupyter

jupyter notebook                # then open any of the .ipynb files
```

Run `testing.ipynb` for the custom TF-IDF + evaluation, `vectorizer.ipynb` for
the grid search (a few minutes), and `kmeans-vect.ipynb` / `embeddings.ipynb`
for clustering. The embedding notebook downloads the `all-MiniLM-L6-v2` model on
first run.

## Course

Department of Computer Engineering & Informatics, University of Patras —
Winter 2025–2026. Supervisor: Prof. C. Makris. TAs: N. Kalogeropoulos,
A. Bompotas.

## References

- Shaw, Wood, Wood & Tibbo (1991) — Cystic Fibrosis test collection.
- Salton & Buckley (1988) — *Term-weighting approaches in automatic text retrieval*.
- Reimers & Gurevych (2019) — *Sentence-BERT*.

## License

MIT — see [LICENSE](LICENSE).
