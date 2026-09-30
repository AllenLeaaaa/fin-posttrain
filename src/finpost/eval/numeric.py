"""Number-based faithfulness metrics.

Earnings summaries mostly consist of figures (EPS, revenue, guidance), so the
two questions that matter most are:

* Did the summary capture the reference's numbers?  -> ``numeric_prf``
  (precision/recall/F1 of the summary's numbers against the reference's).
* Did the summary make numbers up?  -> ``numeric_support``
  (the share of the summary's numbers that appear in the source transcript).
  ``1 - support`` is the numeric hallucination rate. This mirrors
  "Num-Prec" from the ECTSum paper.

Numbers are compared by value, so "0.80" == "0.8" and "1,234" == "1234".
Units and scale words are ignored ("5%", "5 percent", and "$5" are all 5).
Numbers glued to letters, such as "q3", "fy2021", or "3x", are skipped
because they are labels, not reported figures. Written-out numbers ("two")
are not handled.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

# Integer part with optional thousands separators, optional decimals. An
# optional k/m/b suffix ("$1.8B") is allowed but not captured.
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?=[kKmMbB]?(?!\w))")


def extract_numbers(text: str) -> list[tuple[Decimal, int]]:
    """Return (value, number_of_decimals) for each figure in ``text``, in order."""
    out = []
    for int_part, frac in _NUM_RE.findall(text):
        literal = int_part.replace(",", "") + frac
        out.append((Decimal(literal), len(frac) - 1 if frac else 0))
    return out


def number_set(text: str) -> set[Decimal]:
    return {value for value, _ in extract_numbers(text)}


def unsupported_numbers(summary: str, source: str, allow_rounding: bool = False) -> set[Decimal]:
    """Numbers in ``summary`` that can't be found in ``source``.

    With ``allow_rounding``, a summary figure also counts as supported when a
    source figure rounds to it at the summary's precision (source 2.47 -> 2.5).
    Off by default so results are comparable to exact-match Num-Prec.
    """
    source_values = number_set(source)
    missing = set()
    for value, decimals in extract_numbers(summary):
        if value in source_values:
            continue
        if allow_rounding:
            quantum = Decimal(1).scaleb(-decimals)
            if any(s.quantize(quantum, rounding=ROUND_HALF_UP) == value for s in source_values):
                continue
        missing.add(value)
    return missing


def numeric_support(summary: str, source: str, allow_rounding: bool = False) -> float | None:
    """Share of the summary's distinct numbers found in the source; None if it has none."""
    values = number_set(summary)
    if not values:
        return None
    missing = unsupported_numbers(summary, source, allow_rounding)
    return 1.0 - len(missing) / len(values)


def numeric_prf(prediction: str, reference: str) -> dict[str, float | None]:
    """Set precision/recall/F1 of the prediction's numbers against the reference's.

    If the reference has no numbers the metric doesn't apply and every field
    is None. If the prediction has no numbers but the reference does,
    precision is 0: omitting every figure counts as a miss, not as undefined.
    """
    ref = number_set(reference)
    if not ref:
        return {"precision": None, "recall": None, "f1": None}
    pred = number_set(prediction)
    hit = len(pred & ref)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
