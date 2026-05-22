# Fake News Detection

Binary classification of news articles as real (0) or fake (1) using the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) (~72,000 articles).

The project explores four classical NLP feature representations as baselines, then fine-tunes BERT for a stronger result.

---

## Pipeline Overview

```
WELFake_Dataset.csv
        │
        ▼
01_preprocessing.ipynb
  remove_punctuation → remove_stopwords → lemmatize
        │
        ▼
  data/lemmatized.csv
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
02_feature_engineering.ipynb          03_bert_finetune.ipynb
  TF-IDF (sklearn)                      bert-base-uncased
  Word2Vec (gensim, mean-pool)          fine-tuned 3 epochs
  Sentence2Vec (TF-IDF weighted)        AdamW lr=2e-5
  Doc2Vec (gensim PV-DM)               ──────────────────
  ── Logistic Regression baseline ──    bert-finetuned/
```

---

## Results

| Model | Representation | Accuracy |
|---|---|---|
| Logistic Regression | TF-IDF | ~0.94 |
| Logistic Regression | Word2Vec | ~0.87 |
| Logistic Regression | Sentence2Vec | ~0.88 |
| Logistic Regression | Doc2Vec | ~0.89 |
| **BERT fine-tuned** | Contextual embeddings | **~0.97+** |

> Results are approximate; exact figures depend on random seeds and training duration.

---

## Setup

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

Then run the notebooks in order:
1. `01_preprocessing.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_bert_finetune.ipynb` 

---

## Data

**WELFake** (Verma et al., 2021) — 72,134 news articles scraped from four sources (Kaggle, McIntire, Reuters, BuzzFeed Political), balanced between real and fake.

> Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021).  
> *WELFake: Word embedding over linguistic features for fake news detection.*  
> IEEE Transactions on Computational Social Systems.

---

## Key Design Decisions

- **One preprocessing module** (`src/preprocessing.py`) replaces three separate fragmented scripts/notebooks. All cleaning logic is in one place and importable.
- **One vectorizer module** (`src/vectorizers.py`) replaces four separate files. Each representation is a single function call.
- **Consistent data paths** — all notebooks read from and write to `data/` relative to the repo root.
- **Baseline comparison in Notebook 2** — logistic regression on each representation gives an apples-to-apples view before committing to BERT.
