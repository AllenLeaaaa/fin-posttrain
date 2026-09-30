"""Download ECTSum, clean it, and write data/ectsum.sqlite plus a data report.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-0.5B-Instruct   # also report token lengths

The report goes to results/data/ectsum_report.json and is meant to be committed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from finpost.data import clean as clean_mod
from finpost.data import ectsum, store
from finpost.eval.numeric import number_set, unsupported_numbers

ROOT = Path(__file__).resolve().parent.parent


def _percentiles(values) -> dict:
    qs = [50, 90, 95, 99, 100]
    return {f"p{q}": int(np.percentile(values, q)) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "ectsum.sqlite")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "data" / "ectsum_report.json")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer used to report token-length percentiles")
    args = parser.parse_args()

    print(f"Downloading ECTSum @ {ectsum.ECTSUM_COMMIT[:12]} ...")
    ectsum_dir = ectsum.download(args.raw_dir)
    raw = ectsum.load_raw(ectsum_dir)
    df, report = clean_mod.clean(raw)

    report["source"] = {"repo": ectsum.ECTSUM_REPO, "commit": ectsum.ECTSUM_COMMIT, "license": "GPL-3.0"}
    report["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["words"] = {}
    for split, part in df.groupby("split"):
        report["words"][split] = {
            "transcript": _percentiles(part["n_words_transcript"]),
            "summary": _percentiles(part["n_words_summary"]),
            "summary_lines": _percentiles(part["n_summary_lines"]),
        }

    # How often do the human-written references cite numbers that aren't in the
    # transcript? This is the floor for the numeric hallucination metric.
    report["reference_unsupported_numbers"] = {}
    for split, part in df.groupby("split"):
        total = sum(len(number_set(s)) for s in part["summary"])
        missing = sum(len(unsupported_numbers(s, t)) for s, t in zip(part["summary"], part["transcript"]))
        report["reference_unsupported_numbers"][split] = {
            "numbers": total,
            "unsupported": missing,
            "rate": round(missing / total, 4),
        }

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        n_tokens = [len(ids) for ids in tok(df["transcript"].tolist(), add_special_tokens=False)["input_ids"]]
        df["n_tokens_transcript"] = n_tokens
        report["tokens"] = {"tokenizer": args.tokenizer}
        for split, part in df.groupby("split"):
            lengths = part["n_tokens_transcript"]
            report["tokens"][split] = {
                "transcript": _percentiles(lengths),
                "share_over_2048": round(float((lengths > 2048).mean()), 3),
                "share_over_3072": round(float((lengths > 3072).mean()), 3),
                "share_over_4096": round(float((lengths > 4096).mean()), 3),
            }

    store.write_db(df, args.db, meta={k: report[k] for k in ("source", "created_at", "n_raw", "n_clean")})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {len(df)} documents to {args.db}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
