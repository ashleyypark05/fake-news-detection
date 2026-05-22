"""
vectorizers.py
--------------
Feature-engineering helpers for the Fake News Detection pipeline.

Provides:
    build_tfidf          – TF-IDF matrix from a text column
    build_word2vec       – train / load a Word2Vec model; average-pool to doc vectors
    build_sentence2vec   – TF-IDF-weighted Word2Vec sentence vectors
    build_doc2vec        – train / load a Doc2Vec model; infer doc vectors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf(
    df: pd.DataFrame,
    text_col: str = "text",
    max_features: int = 2000,
    ngram_range: tuple = (1, 2),
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer and return (tfidf_features_df, fitted_vectorizer).

    The returned DataFrame contains only the feature columns (no metadata).
    """
    texts = df[text_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    tfidf_df = pd.DataFrame(
        matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return tfidf_df, vectorizer


# ---------------------------------------------------------------------------
# Word2Vec (mean-pooled)
# ---------------------------------------------------------------------------

def _tokenize_series(series: pd.Series) -> list[list[str]]:
    return [word_tokenize(str(t).lower()) for t in series]


def _mean_vector(words: list[str], model: Word2Vec, dim: int) -> np.ndarray:
    valid = [w for w in words if w in model.wv.key_to_index]
    if not valid:
        return np.zeros(dim)
    return np.mean(model.wv[valid], axis=0)


def build_word2vec(
    df: pd.DataFrame,
    text_col: str = "text",
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    save_path: str | None = "word2vec.model",
) -> tuple[pd.DataFrame, Word2Vec]:
    """
    Train a skip-gram Word2Vec model and return (w2v_features_df, model).

    Columns are named w2v_0 .. w2v_{vector_size-1}.
    Pass save_path=None to skip saving.
    """
    tokenized = _tokenize_series(df[text_col])
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=epochs,
    )
    if save_path:
        model.save(save_path)

    dim = model.vector_size
    vectors = np.vstack([_mean_vector(words, model, dim) for words in tokenized])
    cols = [f"w2v_{i}" for i in range(dim)]
    return pd.DataFrame(vectors, columns=cols), model


# ---------------------------------------------------------------------------
# Sentence2Vec (TF-IDF-weighted Word2Vec)
# ---------------------------------------------------------------------------

def _weighted_vector(
    words: list[str],
    model: Word2Vec,
    tfidf_row: pd.Series,
    dim: int,
) -> np.ndarray:
    vecs, weights = [], []
    for word in words:
        if word in model.wv.key_to_index and word in tfidf_row.index:
            vecs.append(model.wv[word] * tfidf_row[word])
            weights.append(tfidf_row[word])
    if not vecs:
        return np.zeros(dim)
    return np.average(vecs, axis=0, weights=weights)


def build_sentence2vec(
    df: pd.DataFrame,
    w2v_model: Word2Vec,
    tfidf_df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Build TF-IDF-weighted sentence vectors from a pre-trained Word2Vec model.

    Parameters
    ----------
    df        : DataFrame with the text column
    w2v_model : fitted Word2Vec model (from build_word2vec)
    tfidf_df  : per-document TF-IDF feature DataFrame (rows aligned to df)
    """
    tokenized = _tokenize_series(df[text_col])
    dim = w2v_model.vector_size
    vectors = np.vstack(
        [
            _weighted_vector(words, w2v_model, tfidf_df.iloc[i], dim)
            for i, words in enumerate(tokenized)
        ]
    )
    cols = [f"s2v_{i}" for i in range(dim)]
    return pd.DataFrame(vectors, columns=cols)


# ---------------------------------------------------------------------------
# Doc2Vec
# ---------------------------------------------------------------------------

def build_doc2vec(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: str = "title",
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 40,
    save_path: str | None = "doc2vec.model",
) -> tuple[pd.DataFrame, Doc2Vec]:
    """
    Train a Doc2Vec model on combined title+text and return (d2v_features_df, model).

    Columns are named d2v_0 .. d2v_{vector_size-1}.
    """
    combined = (
        df[title_col].fillna("").astype(str)
        + " "
        + df[text_col].fillna("").astype(str)
    )
    tagged = [
        TaggedDocument(words=row.split(), tags=[i])
        for i, row in enumerate(combined)
    ]
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    if save_path:
        model.save(save_path)

    vectors = np.vstack([model.dv[i] for i in range(len(tagged))])
    cols = [f"d2v_{i}" for i in range(vector_size)]
    return pd.DataFrame(vectors, columns=cols), model
