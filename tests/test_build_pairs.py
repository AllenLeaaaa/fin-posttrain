"""Spec for finpost.dpo.build_pairs.

The tests only pin down relational behavior (ordering, constraints, edge
cases), not exact reward values, so the weighting is up to the implementer.
They are skipped while reward/select_pair raise NotImplementedError.
"""

import pytest

from finpost.dpo.build_pairs import PairConfig, ScoredCandidate, build_pairs, reward, select_pair


def _call(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"TODO(Allen): implement {fn.__name__}")


def _scores(rouge=0.3, num_f1=0.5, halluc=0.0):
    return {"rougeLsum": rouge, "num_f1": num_f1, "num_halluc_rate": halluc}


def _cand(text, **kw):
    return ScoredCandidate(text=text, scores=_scores(**kw))


CFG = PairConfig()


# ---------------------------------------------------------------- reward


def test_reward_increases_with_rouge():
    assert _call(reward, _scores(rouge=0.5), CFG) > _call(reward, _scores(rouge=0.2), CFG)


def test_reward_increases_with_numeric_f1():
    assert _call(reward, _scores(num_f1=0.8), CFG) > _call(reward, _scores(num_f1=0.1), CFG)


def test_reward_decreases_with_hallucination():
    assert _call(reward, _scores(halluc=0.0), CFG) > _call(reward, _scores(halluc=0.5), CFG)


def test_reward_handles_missing_metrics():
    # num_f1 is None when the reference has no numbers; num_halluc_rate is None
    # when the candidate has no numbers. Neither should crash or return NaN.
    value = _call(reward, {"rougeLsum": 0.3, "num_f1": None, "num_halluc_rate": None}, CFG)
    assert value == value


# ---------------------------------------------------------------- select_pair


def test_select_pair_prefers_higher_reward():
    good, mid, bad = _cand("good", rouge=0.6, num_f1=0.9), _cand("mid"), _cand("bad", rouge=0.05, num_f1=0.0)
    chosen, rejected = _call(select_pair, [mid, bad, good], CFG)
    assert _call(reward, chosen.scores, CFG) > _call(reward, rejected.scores, CFG)
    assert chosen.text == "good"


def test_select_pair_none_when_margin_too_small():
    cands = [_cand("a", rouge=0.30), _cand("b", rouge=0.301), _cand("c", rouge=0.302)]
    assert _call(select_pair, cands, PairConfig(min_margin=0.05)) is None


def test_select_pair_none_with_fewer_than_two_candidates():
    assert _call(select_pair, [_cand("only")], CFG) is None
    assert _call(select_pair, [], CFG) is None


def test_select_pair_never_chooses_hallucinating_candidate():
    # Highest ROUGE but it invents numbers; with max_halluc_for_chosen=0 it
    # must not be chosen.
    liar = _cand("liar", rouge=0.9, num_f1=0.9, halluc=0.4)
    honest = _cand("honest", rouge=0.5, num_f1=0.6, halluc=0.0)
    weak = _cand("weak", rouge=0.05, num_f1=0.0, halluc=0.0)
    pair = _call(select_pair, [liar, honest, weak], PairConfig(max_halluc_for_chosen=0.0))
    assert pair is not None and pair[0].text != "liar"


def test_select_pair_ignores_duplicate_texts():
    # Two samples with identical text carry no preference signal.
    same = [_cand("same", rouge=0.9), _cand("same", rouge=0.1)]
    assert _call(select_pair, same, CFG) is None


# ---------------------------------------------------------------- build_pairs


def test_build_pairs_emits_trl_conversational_format():
    items = [
        ("DOC1", "transcript one", [_cand("good", rouge=0.6, num_f1=0.9), _cand("bad", rouge=0.05, num_f1=0.0)]),
        ("DOC2", "transcript two", [_cand("x", rouge=0.3), _cand("y", rouge=0.3)]),
    ]
    records, stats = _call(build_pairs, items, CFG)
    assert stats == {"prompts": 2, "pairs": 1, "no_pair": 1}
    rec = records[0]
    assert rec["doc_id"] == "DOC1"
    assert rec["prompt"][-1]["role"] == "user" and "transcript one" in rec["prompt"][-1]["content"]
    assert rec["chosen"] == [{"role": "assistant", "content": "good"}]
    assert rec["rejected"] == [{"role": "assistant", "content": "bad"}]
    assert rec["reward_chosen"] > rec["reward_rejected"]
