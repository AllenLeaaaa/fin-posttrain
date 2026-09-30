"""Turn scored candidate summaries into DPO preference pairs.

Where this sits in Phase A:
  1. For each training transcript, sample K summaries from the SFT model
     (temperature > 0).
  2. Score every candidate with ``finpost.eval.report.score_example``
     (an LLM-judge score can be added to ``scores`` later).
  3. ``build_pairs`` picks one (chosen, rejected) pair per transcript and
     emits records in TRL's conversational preference format for DPOTrainer.

TODO(Allen): implement ``reward`` and ``select_pair``. tests/test_build_pairs.py
describes the required behavior; those tests are skipped until the functions
stop raising NotImplementedError. Decisions to make (and be able to defend):
  - How to weigh ROUGE vs. numeric F1 vs. hallucination in one scalar, and how
    to treat metrics that are None (e.g. num_f1 when the reference has no
    numbers).
  - Whether a candidate with unsupported numbers may ever be "chosen". Keep in
    mind that ~13% of the numbers in the human references are not in the
    transcript (results/data/ectsum_report.json).
  - What reward margin makes a pair informative rather than noise.
  - Best-vs-worst, or best-vs-random-worse? (Best-vs-worst gives large margins
    but the rejected answer can be trivially bad.)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from finpost.prompts import build_messages


@dataclass
class ScoredCandidate:
    text: str
    # Output of score_example: rouge1/2/L/Lsum, num_precision/recall/f1,
    # num_halluc_rate, ... Numeric fields may be None.
    scores: dict[str, float | None]


@dataclass
class PairConfig:
    # Starting point only; tune once SFT samples exist.
    weights: dict[str, float] = field(
        default_factory=lambda: {"rougeLsum": 1.0, "num_f1": 1.0, "num_halluc_rate": -2.0}
    )
    # Pairs whose reward gap is smaller than this are dropped.
    min_margin: float = 0.05
    # A candidate whose num_halluc_rate exceeds this can't be chosen.
    # None disables the constraint.
    max_halluc_for_chosen: float | None = 0.0


def reward(scores: dict[str, float | None], cfg: PairConfig) -> float:
    """Collapse one candidate's metrics into a scalar; higher is better."""
    raise NotImplementedError("TODO(Allen)")


def select_pair(
    candidates: list[ScoredCandidate], cfg: PairConfig
) -> tuple[ScoredCandidate, ScoredCandidate] | None:
    """Pick (chosen, rejected) for one prompt, or None if no informative pair exists."""
    raise NotImplementedError("TODO(Allen)")


def build_pairs(
    items: Iterable[tuple[str, str, list[ScoredCandidate]]], cfg: PairConfig
) -> tuple[list[dict], dict]:
    """Build DPO records from (doc_id, transcript, candidates) triples.

    ``transcript`` must already be truncated with the same token budget used
    when the candidates were sampled, so the DPO prompt matches what the model
    actually saw.

    Returns (records, stats). Each record uses TRL's conversational preference
    format, plus doc_id and the two rewards for later analysis.
    """
    records, stats = [], Counter()
    for doc_id, transcript, candidates in items:
        stats["prompts"] += 1
        pair = select_pair(candidates, cfg)
        if pair is None:
            stats["no_pair"] += 1
            continue
        chosen, rejected = pair
        records.append(
            {
                "doc_id": doc_id,
                "prompt": build_messages(transcript),
                "chosen": [{"role": "assistant", "content": chosen.text}],
                "rejected": [{"role": "assistant", "content": rejected.text}],
                "reward_chosen": reward(chosen.scores, cfg),
                "reward_rejected": reward(rejected.scores, cfg),
            }
        )
        stats["pairs"] += 1
    return records, dict(stats)
