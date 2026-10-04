"""Shared helpers for downstream detector training and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

DATASETS = ("megavul", "primevul", "icvul", "titanvul")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_split(datasets: Path, name: str, split: str) -> list[dict]:
    return [{"code": r["func"], "label": int(r["target"])} for r in read_jsonl(datasets / name / f"{split}.jsonl")]


def load_training(datasets: Path, train_set: str, aug: str | None, seed: int) -> list[dict]:
    rows = load_split(datasets, train_set, "train")
    if aug and aug.lower() not in ("none", "orig"):
        rows += [{"code": r["code"], "label": 1} for r in read_jsonl(Path(aug))]
    random.Random(seed).shuffle(rows)
    return rows


def prf(labels: list[int], preds: list[int]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(labels)}


def write_result(out_dir: Path, record: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{record['detector']}__{record['train_set']}__{record['config']}__seed{record['seed']}.json"
    p.write_text(json.dumps(record, indent=1))
    return p
