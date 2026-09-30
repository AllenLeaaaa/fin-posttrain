from decimal import Decimal

import pytest

from finpost.eval import numeric, report
from finpost.eval.rouge import rouge_scores


@pytest.mark.parametrize(
    "text, expected",
    [
        ("eps $0.83.", [Decimal("0.83")]),
        ("revenue rose 2.5 percent to $452.2 million.", [Decimal("2.5"), Decimal("452.2")]),
        ("sales of $1,234,567 in 2021", [Decimal("1234567"), Decimal("2021")]),
        ("guidance $1.8B, margin 5%", [Decimal("1.8"), Decimal("5")]),
        # labels, not figures
        ("q3 and fy2021 results, 3x leverage, covid19", []),
    ],
)
def test_extract_numbers(text, expected):
    assert [v for v, _ in numeric.extract_numbers(text)] == expected


def test_extract_numbers_records_decimals():
    assert numeric.extract_numbers("0.80 and 12") == [(Decimal("0.80"), 2), (Decimal("12"), 0)]


def test_values_compare_numerically():
    assert numeric.number_set("0.80") == numeric.number_set("0.8")


def test_unsupported_numbers_exact_and_rounding():
    source = "revenue grew 2.47% to $452.2 million"
    summary = "revenue up 2.5 percent to $452.2 million; eps $1.10"
    assert numeric.unsupported_numbers(summary, source) == {Decimal("2.5"), Decimal("1.10")}
    assert numeric.unsupported_numbers(summary, source, allow_rounding=True) == {Decimal("1.10")}


def test_numeric_support():
    assert numeric.numeric_support("eps $0.83", "eps was $0.83") == 1.0
    assert numeric.numeric_support("eps $0.83, rev $5", "eps was $0.83") == 0.5
    assert numeric.numeric_support("no figures here", "eps was $0.83") is None


def test_numeric_prf():
    ref = "eps $0.83. revenue $452.2 million."
    assert numeric.numeric_prf("eps $0.83 and 7 stores", ref) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert numeric.numeric_prf("strong quarter", ref) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert numeric.numeric_prf("eps $0.83", "no numbers") == {"precision": None, "recall": None, "f1": None}


def test_rouge_identical_and_disjoint():
    assert rouge_scores("revenue rose\neps fell", "revenue rose\neps fell")["rougeLsum"] == pytest.approx(1.0)
    assert rouge_scores("apples", "oranges")["rouge1"] == 0.0


def test_score_example_and_aggregate():
    source = "eps was $0.83. revenue was $452.2 million."
    ref = "eps $0.83.\nrevenue $452.2 million."
    rows = [
        report.score_example("eps $0.83.\nrevenue $452.2 million.", ref, source),
        report.score_example("eps $0.99.", ref, source),
        report.score_example("a good quarter.", ref, source),
    ]
    assert rows[0]["num_halluc_rate"] == 0.0
    assert rows[1]["num_halluc_rate"] == 1.0
    assert rows[2]["num_halluc_rate"] is None

    agg = report.aggregate(rows, n_boot=200)
    assert agg["num_halluc_rate"]["n"] == 2
    assert agg["num_halluc_rate"]["mean"] == pytest.approx(0.5)
    # 1 unsupported out of 3 numbers overall
    assert agg["num_halluc_rate_micro"]["mean"] == pytest.approx(1 / 3)
    assert agg["share_without_numbers"] == pytest.approx(1 / 3)
    lo, hi = agg["rouge1"]["ci95"]
    assert lo <= agg["rouge1"]["mean"] <= hi


def test_bootstrap_ci_is_deterministic():
    values = [0.1, 0.5, 0.9, 0.3]
    assert report.bootstrap_ci(values, seed=1) == report.bootstrap_ci(values, seed=1)
