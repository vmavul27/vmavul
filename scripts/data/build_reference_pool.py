#!/usr/bin/env python3
"""Build the synthesis reference pool of one training dataset (Step 2(ii)).

Only vulnerable functions of the TRAINING split are used (Section 4.1: "construct the
synthesis reference pool exclusively from the training splits").  Functions whose CWE
falls outside the ten weakness families are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalized_hash, read_jsonl, write_jsonl  # noqa: E402
from vmavul.space import family_of_cwe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    ap.add_argument("--train-set", required=True, choices=["megavul", "primevul", "icvul", "titanvul"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    train = read_jsonl(a.datasets / a.train_set / "train.jsonl")
    test_hashes = {normalized_hash(r["func"]) for name in ("megavul", "primevul", "icvul", "titanvul")
                   if (a.datasets / name / "test.jsonl").exists()
                   for r in read_jsonl(a.datasets / name / "test.jsonl")}
    pool = []
    for r in train:
        assert r.get("split") == "train"
        if int(r["target"]) != 1:
            continue
        fam = family_of_cwe(r.get("cwe"))
        if fam is None or normalized_hash(r["func"]) in test_hashes:
            continue
        pool.append({"id": r["id"], "dataset": a.train_set, "split": "train", "cwe": r["cwe"],
                     "family": fam, "func": r["func"]})
    out = a.out or Path("data/reference_pool") / f"{a.train_set}.jsonl"
    n = write_jsonl(out, pool)
    print(json.dumps({"train_set": a.train_set, "reference_functions": n,
                      "per_family": Counter(p["family"] for p in pool)}, indent=1))


if __name__ == "__main__":
    main()
