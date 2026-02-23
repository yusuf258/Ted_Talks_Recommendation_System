# TED Talks Recommendation System | Sentence-BERT

Semantic recommendation engine for TED Talks using Sentence-BERT (SBERT) embeddings and cosine similarity to suggest thematically related talks.

## Problem Statement
Given a TED Talk, recommend the **10 most semantically similar talks** from a corpus of 2,467 talks. Unlike keyword-matching approaches, this system understands meaning — "climate change" and "global warming" are correctly treated as related topics.

## Dataset
| Attribute | Detail |
|---|---|
| Files | `ted_main.csv` + `transcripts.csv` |
| Records | 2,467 TED Talks |
| Join Key | `url` column |
| Features Used | `title`, `transcript`, `main_speaker` |

## Methodology
1. **Data Loading** — Merge ted_main (metadata) and transcripts on `url`
2. **Preprocessing** — Fill missing transcripts, select relevant columns
3. **SBERT Encoding** — Encode all transcripts using `all-MiniLM-L6-v2` model → 384-dim embeddings
4. **Cosine Similarity Matrix** — Compute 2,467 × 2,467 pairwise similarity matrix
5. **Recommendation Function** — Return Top-10 most similar talks (excluding self)
6. **Model Persistence** — Save similarity matrix, talk metadata, and title index for Streamlit deployment

## Results
| Component | Detail |
|---|---|
| Model | `all-MiniLM-L6-v2` — ~80MB, 384-dim output |
| Similarity Matrix | 2,467 × 2,467 cosine similarity |
| Output | Top-10 semantically closest talks |

> Semantic embeddings capture conceptual similarity beyond keyword overlap — significantly outperforms TF-IDF for recommendation quality.

## Technologies
`Python` · `sentence-transformers` · `scikit-learn` · `Pandas` · `NumPy` · `joblib`

## File Structure
```
21_Ted_Talks_Recommendation_System/
├── project_notebook.ipynb    # Main notebook
├── ted_main.csv              # Talk metadata
├── transcripts.csv           # Full talk transcripts
└── models/
    ├── cosine_sim_dl.pkl     # Precomputed similarity matrix
    ├── ted_data.pkl          # Talk metadata for display
    └── indices.pkl           # Title-to-index mapping
```

## How to Run
```bash
cd 21_Ted_Talks_Recommendation_System
jupyter notebook project_notebook.ipynb
```

### Requirements
```bash
pip install sentence-transformers
```
