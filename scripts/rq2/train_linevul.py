#!/usr/bin/env python3
"""Run the official LineVul implementation for function-level classification.

This adapter only prepares the repository's JSONL data as LineVul CSV files, invokes a
pinned, clean checkout of the public LineVul code, and converts its logged metrics to the
common RQ2 result format. Model construction, tokenization, training, checkpoint
selection, and inference are performed by LineVul itself.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_split, load_training, write_result  # noqa: E402


LINEVUL_COMMIT = "9401ec6e60b762a9308645961268203244bee048"
MODEL_ID = "microsoft/codebert-base"


def _write_linevul_csv(path: Path, rows: list[dict]) -> None:
    """Write the columns required by the official LineVul train and test loaders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["processed_func", "target", "flaw_line_index", "flaw_line"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "processed_func": row["code"],
                "target": int(row["label"]),
                "flaw_line_index": "",
                "flaw_line": "",
            })


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_linevul_checkout(repo: Path) -> Path:
    script = repo / "linevul" / "linevul_main.py"
    if not script.is_file():
        raise RuntimeError(
            f"official LineVul checkout not found at {repo}; run scripts/rq2/setup_linevul.sh"
        )
    try:
        commit = _git_output(repo, "rev-parse", "HEAD")
        modified = _git_output(repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{repo} is not a usable LineVul git checkout") from exc
    if commit != LINEVUL_COMMIT:
        raise RuntimeError(f"LineVul must be at {LINEVUL_COMMIT}, found {commit}")
    if modified:
        raise RuntimeError("tracked files in the LineVul checkout are modified")
    return script.resolve()


def _run_logged(cmd: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)
    return "".join(lines)


def _metric(log: str, name: str) -> float:
    matches = re.findall(rf"\b{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)", log)
    if not matches:
        raise RuntimeError(f"official LineVul log did not contain {name}")
    return float(matches[-1])


def _best_validation_f1(log: str) -> float:
    values = re.findall(r"Best f1:\s*([0-9]+(?:\.[0-9]+)?)", log)
    if not values:
        raise RuntimeError("official LineVul did not produce a validation-selected checkpoint")
    return max(map(float, values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-set", required=True, choices=DATASETS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--aug", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    ap.add_argument("--out", type=Path, default=Path("results/rq2"))
    ap.add_argument("--work", type=Path, default=Path("work/linevul"))
    ap.add_argument("--linevul-dir", type=Path,
                    default=Path(os.environ.get("VMAVUL_LINEVUL_DIR", "third_party/LineVul")))
    ap.add_argument("--linevul-python", type=Path, default=None,
                    help="Python from the official LineVul environment (default: <linevul-dir>/.venv/bin/python)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-train-samples", type=int, default=0, help="debug only")
    a = ap.parse_args()

    linevul_dir = a.linevul_dir.resolve()
    script = _verify_linevul_checkout(linevul_dir)
    python = (a.linevul_python or linevul_dir / ".venv" / "bin" / "python").resolve()
    if not python.is_file():
        raise RuntimeError(f"LineVul Python not found at {python}; run scripts/rq2/setup_linevul.sh")

    run_name = f"linevul__{a.train_set}__{a.config}__seed{a.seed}"
    run_dir = (a.work / run_name).resolve()
    inputs = run_dir / "inputs"
    checkpoint_dir = run_dir / "official_output"
    run_dir.mkdir(parents=True, exist_ok=True)

    train = load_training(a.datasets, a.train_set, a.aug, a.seed)
    if a.max_train_samples:
        train = train[: a.max_train_samples]
    valid = load_split(a.datasets, a.train_set, "valid")
    train_csv, valid_csv = inputs / "train.csv", inputs / "valid.csv"
    _write_linevul_csv(train_csv, train)
    _write_linevul_csv(valid_csv, valid)

    common = [
        str(python), str(script),
        "--output_dir", str(checkpoint_dir),
        "--model_name", "model.bin",
        "--model_type", "roberta",
        "--tokenizer_name", MODEL_ID,
        "--model_name_or_path", MODEL_ID,
        "--block_size", str(a.max_len),
        "--eval_batch_size", str(a.eval_batch_size),
        "--seed", str(a.seed),
    ]
    train_cmd = common + [
        "--do_train", "--evaluate_during_training",
        "--train_data_file", str(train_csv),
        "--eval_data_file", str(valid_csv),
        "--epochs", str(a.epochs),
        "--train_batch_size", str(a.batch_size),
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(a.lr),
        "--max_grad_norm", "1.0",
    ]
    t0 = time.time()
    train_log = _run_logged(train_cmd, run_dir, run_dir / "train.log")
    best_valid = _best_validation_f1(train_log)

    tests = {}
    for name in DATASETS:
        rows = load_split(a.datasets, name, "test")
        test_csv = inputs / f"test_{name}.csv"
        _write_linevul_csv(test_csv, rows)
        log = _run_logged(
            common + ["--do_test", "--test_data_file", str(test_csv)],
            run_dir, run_dir / f"test_{name}.log",
        )
        tests[name] = {
            "precision": _metric(log, "test_precision"),
            "recall": _metric(log, "test_recall"),
            "f1": _metric(log, "test_f1"),
            "n": len(rows),
        }

    write_result(a.out, {
        "detector": "linevul", "train_set": a.train_set, "config": a.config, "seed": a.seed,
        "aug": a.aug, "n_train": len(train), "best_valid_f1": best_valid,
        "best_checkpoint": str(checkpoint_dir / "checkpoint-best-f1" / "model.bin"),
        "selection": "validation_f1", "implementation": "official-linevul",
        "linevul_commit": LINEVUL_COMMIT, "test": tests,
        "hours": (time.time() - t0) / 3600,
        "hparams": {k: str(v) for k, v in vars(a).items()},
    })


if __name__ == "__main__":
    main()
