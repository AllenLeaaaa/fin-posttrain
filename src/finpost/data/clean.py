"""Text normalization and dataset-level cleaning for ECTSum."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SPACES_RE = re.compile(r"[ \t ]+")


def normalize_text(text: str) -> str:
    """NFKC-normalize, collapse runs of spaces, and drop blank lines.

    Line breaks are kept: ECTSum stores one sentence per line, and ROUGE-Lsum
    uses newlines as sentence boundaries.
    """
    text = unicodedata.normalize("NFKC", text)
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize text, drop empty rows and leaked duplicates, add length columns.

    Duplicate handling: if the same transcript shows up in both train and
    val/test, the train copy is dropped so the evaluation splits stay
    identical to the official ones. Duplicates within a single split are
    reduced to one copy.

    Returns the cleaned frame and a report of what was removed.
    """
    report: dict = {"n_raw": {s: int(n) for s, n in df["split"].value_counts().items()}}
    df = df.copy()
    df["transcript"] = df["transcript"].map(normalize_text)
    df["summary"] = df["summary"].map(normalize_text)

    empty = (df["transcript"] == "") | (df["summary"] == "")
    report["dropped_empty"] = int(empty.sum())
    df = df[~empty]

    # Evaluation splits win: sort so val/test rows come before train rows.
    priority = df["split"].map({"test": 0, "val": 1, "train": 2})
    df = df.assign(_priority=priority).sort_values(["_priority", "doc_id"])
    dup = df.duplicated(subset="transcript", keep="first")
    report["dropped_duplicate"] = {s: int(n) for s, n in df.loc[dup, "split"].value_counts().items()}
    df = df[~dup].drop(columns="_priority")

    df["n_words_transcript"] = df["transcript"].str.split().str.len()
    df["n_words_summary"] = df["summary"].str.split().str.len()
    df["n_summary_lines"] = df["summary"].str.count("\n") + 1

    df = df.sort_values(["split", "doc_id"]).reset_index(drop=True)
    report["n_clean"] = {s: int(n) for s, n in df["split"].value_counts().items()}
    return df, report
