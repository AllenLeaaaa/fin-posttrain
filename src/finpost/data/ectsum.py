"""Download and load ECTSum (Mukherjee et al., EMNLP 2022).

ECTSum pairs earnings-call transcripts with short bullet-point summaries that
were written by Reuters journalists. The official split is train/val/test =
1681/249/495. The dataset is GPL-3.0, so this repo only ships the download
code, never the data itself.
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

ECTSUM_REPO = "rajdeep345/ECTSum"
# Pinned so the dataset can't silently change under us.
ECTSUM_COMMIT = "6909f1fc543104c1c60cf9de63e799f6620d1b0a"
SPLITS = ("train", "val", "test")

_DOC_ID_RE = re.compile(r"^(?P<ticker>.+)_q(?P<quarter>[1-4])_(?P<year>\d{4})$")


def download(raw_dir: Path, commit: str = ECTSUM_COMMIT) -> Path:
    """Fetch the pinned ECTSum commit and extract ``data/final`` into ``raw_dir/ectsum``.

    Returns the extracted directory. Skips the download if it already exists.
    """
    out_dir = Path(raw_dir) / "ectsum"
    if all((out_dir / split / "ects").is_dir() for split in SPLITS):
        return out_dir

    url = f"https://codeload.github.com/{ECTSUM_REPO}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Member paths look like "ECTSum-<sha>/data/final/<split>/<kind>/<file>.txt"
            parts = member.name.split("/")
            if not member.isfile() or len(parts) != 6 or parts[1:3] != ["data", "final"]:
                continue
            split, kind, filename = parts[3], parts[4], parts[5]
            if split not in SPLITS or kind not in ("ects", "gt_summaries"):
                continue
            dest = out_dir / split / kind / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())

    return out_dir


def parse_doc_id(doc_id: str) -> tuple[str | None, int | None, int | None]:
    """Split a file stem like ``AAN_q3_2021`` into (ticker, quarter, year)."""
    match = _DOC_ID_RE.match(doc_id)
    if match is None:
        return None, None, None
    return match["ticker"], int(match["quarter"]), int(match["year"])


def load_raw(ectsum_dir: Path) -> pd.DataFrame:
    """Read every (transcript, summary) pair into one DataFrame.

    Raises if a transcript has no matching summary or vice versa, since that
    would mean the download is incomplete.
    """
    rows = []
    for split in SPLITS:
        ect_files = {p.stem: p for p in (Path(ectsum_dir) / split / "ects").glob("*.txt")}
        sum_files = {p.stem: p for p in (Path(ectsum_dir) / split / "gt_summaries").glob("*.txt")}
        if ect_files.keys() != sum_files.keys():
            missing = sorted(ect_files.keys() ^ sum_files.keys())
            raise ValueError(f"{split}: unpaired files, e.g. {missing[:5]}")
        for doc_id in sorted(ect_files):
            ticker, quarter, year = parse_doc_id(doc_id)
            rows.append(
                {
                    "doc_id": doc_id,
                    "split": split,
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "transcript": ect_files[doc_id].read_text(encoding="utf-8"),
                    "summary": sum_files[doc_id].read_text(encoding="utf-8"),
                }
            )
    return pd.DataFrame(rows)
