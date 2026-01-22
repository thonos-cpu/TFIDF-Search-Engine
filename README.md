# Information Retrieval System for Medical Document Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive information retrieval system developed for the University of Patras "Information Retrieval" course. This project implements and evaluates multiple document ranking models on the Cystic Fibrosis medical literature collection, featuring custom TF-IDF implementations, hyperparameter optimization, and advanced clustering techniques.

## 📋 Overview

This system processes 1,239 medical research articles from the Cystic Fibrosis database, implementing various information retrieval models to rank documents based on their relevance to user queries. The project compares custom-built algorithms against optimized scikit-learn implementations and explores document clustering using both traditional and modern embedding approaches.

**Key Highlights:**
- Custom TF-IDF implementation with multiple term weighting schemes
- Grid search optimization across 270+ hyperparameter combinations
- Document clustering using K-Means and Agglomerative methods
- Semantic analysis with sentence-transformers embeddings
- Comprehensive evaluation using Precision, Recall, F1-Score, and MAP metrics

## 🎯 Features

### Core Functionality

- **Document Preprocessing**: Stopword removal, stemming, and inverted index construction
- **Vector Space Model**: Custom implementation supporting three TF variants:
  - Raw TF (optimal for this collection)
  - Logarithmic TF
  - Augmented TF
- **Cosine Similarity**: Efficient document-query similarity computation
- **Hyperparameter Tuning**: Automated grid search for optimal TF-IDF configuration
- **Document Clustering**: 
  - TF-IDF based clustering (K-Means, Agglomerative)
  - Semantic embeddings clustering (sentence-transformers/all-MiniLM-L6-v2)
- **Performance Metrics**: Precision@k, Recall@k, F1@k, Average Precision, MAP

### Performance Results

| Model | F1@k | Precision | Recall | MAP |
|-------|------|-----------|--------|-----|
| Raw TF-IDF | 0.2817 | 0.0392 | 0.9009 | 0.2620 |
| Log TF-IDF | 0.2817 | 0.0392 | 0.9009 | 0.2616 |
| Aug TF-IDF | 0.2753 | 0.0392 | 0.9009 | 0.2583 |
| **Optimized TF-IDF** | **0.3006** | **0.0420** | **0.9200** | **0.2750** |

*Optimized parameters: `ngram_range=(1,1)`, `sublinear_tf=True`, `min_df=1`, `max_df=0.7`, `norm='l2'`*

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/information-retrieval-system.git
   cd information-retrieval-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

```
information_retrieval/
├── src/
│   ├── testing.py              # Custom TF-IDF implementation & evaluation
│   ├── vectorizer.py           # scikit-learn grid search optimization
│   ├── kmeans-vect.py          # K-Means clustering with TF-IDF
│   ├── agglomerative.py        # Agglomerative clustering with TF-IDF
│   └── embeddings.py           # Semantic embeddings & clustering
├── data/
│   ├── docs/                   # Cystic Fibrosis document collection
│   ├── Queries.txt             # 20 medical research queries
│   └── Relevant.txt            # Ground truth relevance judgments
├── requirements.txt
├── README.md
└── ir2025_1093503_1097449_technical_report.pdf
```

## 💻 Usage

### 1. Basic Document Retrieval

Run the custom TF-IDF implementation and evaluate all queries:

```bash
python src/testing.py
```

This will:
- Load and preprocess 1,239 documents
- Build an inverted index
- Evaluate all 20 queries
- Display precision, recall, F1-score, and MAP metrics

### 2. Hyperparameter Optimization

Find optimal TF-IDF parameters using grid search:

```bash
python src/vectorizer.py
```

Expected runtime: ~18 minutes for 270 parameter combinations.

### 3. Document Clustering (TF-IDF)

**K-Means Clustering:**
```bash
python src/kmeans-vect.py
```

**Agglomerative Clustering:**
```bash
python src/agglomerative.py
```

Both scripts generate:
- Elbow method plots
- Silhouette score analysis
- 2D PCA visualizations

### 4. Semantic Embeddings Clustering

Use pre-trained sentence transformers for semantic document clustering:

```bash
python src/embeddings.py
```

This approach:
- Encodes documents using `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Determines optimal cluster count via silhouette analysis
- Provides superior semantic grouping compared to TF-IDF

## 📊 Methodology

### Preprocessing Pipeline

1. **Tokenization**: Split documents into individual terms
2. **Stopword Removal**: Filter ~100 common English stopwords
3. **Normalization**: Convert to lowercase, remove non-alphabetic characters
4. **Stemming**: Reduce terms to root forms (optional, not used in final implementation)

### TF-IDF Weighting Schemes

We implemented and compared three term frequency variants:

```python
# Raw TF (best performer for short medical documents)
TF(t, d) = count(t, d)

# Logarithmic TF
TF_log(t, d) = 1 + log(count(t, d))

# Augmented TF
TF_aug(t, d) = 0.5 + 0.5 * (count(t, d) / max_count(d))
```

**Why Raw TF Won**: The Cystic Fibrosis collection contains very short documents (average 81 words) with minimal term repetition (89.8% of terms appear ≤2 times). Logarithmic and augmented schemes compress the already-sparse signal, losing discriminative power.

### Clustering Analysis

| Method | Best k | Silhouette Score | Notes |
|--------|--------|------------------|-------|
| K-Means (TF-IDF) | 17 | 0.18 | Elbow method indication |
| Agglomerative (TF-IDF) | 50 | 0.22 | Increasing trend observed |
| K-Means (Embeddings) | 12 | 0.31 | **Best semantic grouping** |

The embedding-based approach significantly outperforms TF-IDF clustering, capturing semantic relationships beyond simple term overlap.

## 🔬 Evaluation Metrics

```python
# Precision@k: Ratio of relevant docs in top-k results
Precision@k = relevant_retrieved[:k] / k

# Recall@k: Fraction of all relevant docs found in top-k
Recall@k = relevant_retrieved[:k] / total_relevant

# F1@k: Harmonic mean of Precision@k and Recall@k
F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)

# Average Precision: Position-aware precision
AP = Σ(Precision@i * is_relevant(i)) / total_relevant

# Mean Average Precision: Average AP across all queries
MAP = Σ(AP_q) / num_queries
```

## 📈 Key Findings

1. **Raw TF-IDF is optimal for short documents**: In collections with low term frequency (medical abstracts), sophisticated normalization schemes can be counterproductive.

2. **Hyperparameter tuning matters**: Optimized `max_df=0.7` improved F1@k by 6.7% by filtering overly common terms while preserving medical terminology.

3. **Semantic embeddings excel at clustering**: The 384-dimensional sentence-transformer embeddings captured medical concept relationships that term-based methods missed, achieving 72% higher silhouette scores.

4. **Collection statistics matter**: Understanding document length distribution (81 words avg) and term frequency patterns (0.1% of terms with TF>10) guided successful model selection.

## 🛠️ Technical Details

### Dependencies

```txt
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
yellowbrick>=1.5
```

### System Requirements

- **RAM**: 4GB minimum (8GB recommended for embeddings)
- **Storage**: 500MB for documents + 2GB for pre-trained models
- **CPU**: Multi-core recommended for grid search

### Performance Benchmarks

| Operation | Time |
|-----------|------|
| Document preprocessing | ~2s |
| Inverted index construction | ~1s |
| Single query ranking | ~50ms |
| Grid search (270 configs) | ~18min |
| Embedding generation (1,239 docs) | ~3min |


Department of Computer Engineering & Informatics  
University of Patras  
Winter Semester 2025-2026

**Supervisor**: Prof. C. Makris  
**Teaching Assistants**: N. Kalogeropoulos, A. Bompotas

## 📄 License

This project was developed as coursework for the Information Retrieval course at the University of Patras. Feel free to use it as a reference, but please cite appropriately if you build upon this work.

## 🙏 Acknowledgments

- **Cystic Fibrosis Database**: Shaw, W.M., Wood, J.B., Wood, R.E., & Tibbo, H.R. (1991)
- **TF-IDF Theory**: Salton & Buckley (1988), "Term-weighting approaches in automatic text retrieval"
- **Vector Space Model**: Salton, Wong & Yang (1975)
- **Sentence Transformers**: Reimers & Gurevych (2019)

---

**Questions or suggestions?** Feel free to open an issue or submit a pull request. We welcome contributions that improve retrieval performance or extend the system to new medical corpora!
