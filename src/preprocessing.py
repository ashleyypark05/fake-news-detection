"""
preprocessing.py
----------------
Reusable text-cleaning functions for the Fake News Detection pipeline.

Steps (in order):
    1. remove_punctuation  – strip unicode punctuation + standard ASCII punctuation
    2. remove_stopwords    – drop NLTK English stopwords
    3. lemmatize           – WordNet lemmatisation via NLTK

Each function accepts a single string and returns a cleaned string,
making them safe to use with df[col].apply(...).
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent if already present)
for pkg in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(pkg, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Step 1 – Punctuation removal
# ---------------------------------------------------------------------------

def remove_punctuation(text: str) -> str:
    """Strip unicode curly quotes/dashes and standard ASCII punctuation."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\u201c\u201d\u2018\u2019\u2013\u2014\u2026]", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 2 – Stopword removal
# ---------------------------------------------------------------------------

def remove_stopwords(text: str) -> str:
    """Tokenise and drop English stopwords."""
    if not isinstance(text, str):
        return text
    tokens = word_tokenize(text)
    return " ".join(w for w in tokens if w.lower() not in _STOP_WORDS)


# ---------------------------------------------------------------------------
# Step 3 – Lemmatisation
# ---------------------------------------------------------------------------

def lemmatize(text: str) -> str:
    """WordNet-lemmatise every token in the text."""
    if not isinstance(text, str):
        return text
    tokens = word_tokenize(text)
    return " ".join(_LEMMATIZER.lemmatize(w) for w in tokens)


# ---------------------------------------------------------------------------
# Convenience: full pipeline on a DataFrame
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: str = "title",
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline (punctuation -> stopwords -> lemmatise)
    to `text_col` and `title_col` in-place and return the cleaned DataFrame.

    Parameters
    ----------
    df        : source DataFrame (not modified; a copy is returned)
    text_col  : name of the article-body column
    title_col : name of the headline column
    """
    df = df.copy()

    for col in (text_col, title_col):
        if col not in df.columns:
            continue
        df[col] = df[col].apply(remove_punctuation)
        df[col] = df[col].apply(remove_stopwords)
        df[col] = df[col].apply(lemmatize)
        # Remove stray commas left by earlier pipeline versions
        df[col] = df[col].str.replace(",", " ", regex=False)

    return df
