#!/usr/bin/env python3
"""Aggregate downstream detector results (mean over seeds) into Tables 2 and 3.

RQ2 (Tables 2-3): F1 per detector x training set x test set x approach; the number of
settings where VMAVul is best; the settings where an approach falls below Orig.; and the
mean relative F1 gain of Eq. (3):
    Imp(VMAVul, b) = 1/N * sum_i (F1_i^VMAVul - F1_i^b) / F1_i^b * 100%
over all paired (detector, training set, test set) settings, plus its in-dataset /
cross-dataset / per-detector breakdowns.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

TRAIN = ["megavul", "primevul", "icvul", "titanvul"]
SHORT = {"megavul": "M", "primevul": "P", "icvul": "I", "titanvul": "T"}


def load(results: Path, metric: str) -> dict:
    """(detector, train, config, test) -> mean metric over seeds (in %), plus seed counts."""
    acc = defaultdict(list)
    for p in sorted(results.glob("*.json")):
        r = json.loads(p.read_text())
        for test, m in r["test"].items():
            acc[(r["detector"], r["train_set"], r["config"], test)].append(100.0 * m[metric])
    return {k: (statistics.mean(v), len(v)) for k, v in acc.items()}


def imp(table: dict, keys: list, ours: str, base: str) -> float | None:
    vals = []
    for (det, tr, te) in keys:
        a, b = table.get((det, tr, ours, te)), table.get((det, tr, base, te))
        if a and b and b[0] > 0:
            vals.append((a[0] - b[0]) / b[0])
    return 100.0 * sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/rq2"))
    ap.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"])
    ap.add_argument("--ours", default=None)
    a = ap.parse_args()
    t = load(a.results, a.metric)
    dets = sorted({k[0] for k in t})
    configs = sorted({k[2] for k in t})
    report: dict = {"seeds_per_cell": sorted({n for _, n in t.values()})}
    ours = a.ours or "vmavul"
    baselines = [c for c in ["vulgen", "vgx", "gvi", "vulscriber"] if c in configs]
    keys = [(d, tr, te) for d in dets for tr in TRAIN for te in TRAIN]
    rows = []
    for d, tr, te in keys:
        row = {c: round(t[(d, tr, c, te)][0], 2) for c in configs if (d, tr, c, te) in t}
        rows.append({"detector": d, "train": tr, "test": SHORT[te], **row})
    report["table"] = rows
    synth = [c for c in baselines + [ours] if c in configs]
    best = [r for r in rows if all(c in r for c in synth) and max(synth, key=lambda c: r[c]) == ours]
    report["vmavul_best_settings"] = f"{len(best)}/{len(rows)}"
    report["below_orig"] = {c: sum(1 for r in rows if c in r and "orig" in r and r[c] < r["orig"])
                            for c in synth}
    groups = {"all": keys,
              "in_dataset": [k for k in keys if k[1] == k[2]],
              "cross_dataset": [k for k in keys if k[1] != k[2]]}
    groups |= {f"detector={d}": [k for k in keys if k[0] == d] for d in dets}
    report["improvement_eq3"] = {g: {b: imp(t, ks, ours, b) for b in baselines} for g, ks in groups.items()}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
