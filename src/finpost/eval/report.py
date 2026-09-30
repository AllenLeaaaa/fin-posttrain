"""Per-example scoring and corpus-level aggregation with bootstrap CIs."""

from __future__ import annotations

import numpy as np

from .numeric import number_set, numeric_prf, unsupported_numbers
from .rouge import rouge_scores

# Metrics where higher is better, except num_halluc_rate.
METRICS = (
    "rouge1", "rouge2", "rougeL", "rougeLsum",
    "num_precision", "num_recall", "num_f1",
    "num_halluc_rate",
)


def score_example(prediction: str, reference: str, source: str, allow_rounding: bool = False) -> dict:
    """All metrics for one summary. Numeric fields are None when they don't apply."""
    row: dict = dict(rouge_scores(prediction, reference))
    prf = numeric_prf(prediction, reference)
    row["num_precision"], row["num_recall"], row["num_f1"] = prf["precision"], prf["recall"], prf["f1"]

    n_pred = len(number_set(prediction))
    n_unsupported = len(unsupported_numbers(prediction, source, allow_rounding))
    row["n_pred_numbers"] = n_pred
    row["n_unsupported_numbers"] = n_unsupported
    row["num_halluc_rate"] = n_unsupported / n_pred if n_pred else None
    return row


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def aggregate(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean and 95% bootstrap CI per metric, skipping None values.

    ``n`` is how many examples the metric applied to. The hallucination rate
    is also reported micro-averaged (total unsupported / total numbers),
    which is less noisy than averaging per-summary rates when summaries
    contain only one or two numbers.
    """
    out: dict = {}
    for name in METRICS:
        values = [r[name] for r in rows if r.get(name) is not None]
        if not values:
            out[name] = {"mean": None, "ci95": None, "n": 0}
            continue
        out[name] = {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values, n_boot, seed=seed), "n": len(values)}

    total_numbers = sum(r["n_pred_numbers"] for r in rows)
    total_unsupported = sum(r["n_unsupported_numbers"] for r in rows)
    out["num_halluc_rate_micro"] = {
        "mean": total_unsupported / total_numbers if total_numbers else None,
        "n_numbers": total_numbers,
    }
    out["share_without_numbers"] = sum(r["n_pred_numbers"] == 0 for r in rows) / len(rows) if rows else None
    return out
