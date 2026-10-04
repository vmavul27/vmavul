#!/usr/bin/env python3
"""Draw a seeded uniform-random augmentation set from synthesized vulnerabilities.

Section 4.3 samples 15,000 functions for each approach. No sample is selected or ranked
by a quality score, similarity to seeds, or detector/test-set performance.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from common import clang_format_all, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="JSONL with a `code` field (all synthesized samples)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-format", action="store_true", help="skip clang-format normalization")
    a = ap.parse_args()
    rows = [r for r in read_jsonl(a.input) if (r.get("code") or "").strip()]
    if len(rows) < a.n:
        print(f"WARNING: only {len(rows)} samples available (< {a.n}); using all of them", file=sys.stderr)
    picked = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    if not a.no_format:
        for r, code in zip(picked, clang_format_all([r["code"] for r in picked])):
            r["code"] = code
    out = [{"id": r.get("id"), "code": r["code"], "label": 1, "source": r.get("source", "")}
           for r in picked]
    n = write_jsonl(a.out, out)
    meta = {"input": str(a.input), "available": len(rows), "sampled": n, "seed": a.seed, "method": "uniform_random"}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
