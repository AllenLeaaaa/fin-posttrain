"""ROUGE F-scores via Google's ``rouge-score`` package."""

from __future__ import annotations

from functools import lru_cache

from rouge_score import rouge_scorer

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


@lru_cache(maxsize=1)
def _scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """F-measure for each ROUGE type.

    rougeLsum treats each newline-separated line as a sentence, which fits
    bullet-point summaries, so both strings should keep one bullet per line.
    """
    scores = _scorer().score(reference, prediction)
    return {name: scores[name].fmeasure for name in ROUGE_TYPES}
