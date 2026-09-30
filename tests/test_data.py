import pandas as pd
import pytest

from finpost.data import clean as clean_mod
from finpost.data import ectsum, store


def _write_pair(root, split, doc_id, transcript, summary):
    for kind, text in (("ects", transcript), ("gt_summaries", summary)):
        path = root / split / kind / f"{doc_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.parametrize(
    "doc_id, expected",
    [
        ("AAN_q3_2021", ("AAN", 3, 2021)),
        ("BRK_B_q1_2020", ("BRK_B", 1, 2020)),
        ("weird-name", (None, None, None)),
    ],
)
def test_parse_doc_id(doc_id, expected):
    assert ectsum.parse_doc_id(doc_id) == expected


def test_load_raw_pairs_files(tmp_path):
    for split in ectsum.SPLITS:
        _write_pair(tmp_path, split, "X_q1_2021", f"{split} transcript", "summary")
    df = ectsum.load_raw(tmp_path)
    assert sorted(df["split"]) == sorted(ectsum.SPLITS)
    assert set(df["ticker"]) == {"X"}


def test_load_raw_rejects_unpaired(tmp_path):
    for split in ectsum.SPLITS:
        _write_pair(tmp_path, split, "X_q1_2021", "t", "s")
    (tmp_path / "test" / "ects" / "Y_q2_2021.txt").write_text("orphan")
    with pytest.raises(ValueError, match="unpaired"):
        ectsum.load_raw(tmp_path)


def test_normalize_text_keeps_lines_and_collapses_spaces():
    raw = "  Revenue  rose  5%. \n\n\tEPS was $1.20.  \n"
    assert clean_mod.normalize_text(raw) == "Revenue rose 5%.\nEPS was $1.20."


def _frame(rows):
    return pd.DataFrame(
        [{"doc_id": d, "split": s, "ticker": None, "quarter": None, "year": None, "transcript": t, "summary": m}
         for d, s, t, m in rows]
    )


def test_clean_drops_train_copy_of_test_transcript():
    df, report = clean_mod.clean(
        _frame(
            [
                ("A", "train", "same text", "s1"),
                ("B", "test", "same text", "s2"),
                ("C", "train", "other", "s3"),
            ]
        )
    )
    assert set(zip(df["doc_id"], df["split"])) == {("B", "test"), ("C", "train")}
    assert report["dropped_duplicate"] == {"train": 1}


def test_clean_drops_empty_and_counts_lengths():
    df, report = clean_mod.clean(
        _frame([("A", "train", "   ", "s"), ("B", "train", "one two three", "line one\nline two")])
    )
    assert report["dropped_empty"] == 1
    row = df.iloc[0]
    assert (row["n_words_transcript"], row["n_words_summary"], row["n_summary_lines"]) == (3, 4, 2)


def test_store_roundtrip(tmp_path):
    df, _ = clean_mod.clean(
        _frame([(f"D{i}", "test", f"transcript {i}", f"summary {i}") for i in range(5)])
    )
    db = tmp_path / "x.sqlite"
    store.write_db(df, db, meta={"source": {"commit": "abc"}})
    loaded = store.load_split(db, "test", limit=3)
    assert list(loaded["doc_id"]) == ["D0", "D1", "D2"]
    assert store.load_split(db, "train").empty
    assert store.load_meta(db) == {"source": {"commit": "abc"}}
