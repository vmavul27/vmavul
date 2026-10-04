#!/usr/bin/env python3
"""Emit (and optionally run) the downstream detector-training job matrix.

The matrix is {LineVul, Qwen3.5-4B, Llama3.1-8B} x four training datasets x
{Orig., VulGen, VGX, GVI, VulScribeR, VMAVul} x seeds 0..4.

Augmentation files are expected at ``<aug_dir>/<train_set>/<config>.jsonl`` or
``<aug_dir>/<config>.jsonl``.
Without ``--run`` the commands are only printed (one per line).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_SETS = ["megavul", "primevul", "icvul", "titanvul"]
CONFIGS = ["orig", "vulgen", "vgx", "gvi", "vulscriber", "vmavul"]


def aug_path(aug_dir: Path, train_set: str, config: str) -> str | None:
    if config == "orig":
        return None
    for p in (aug_dir / train_set / f"{config}.jsonl", aug_dir / f"{config}.jsonl"):
        if p.exists():
            return str(p)
    return str(aug_dir / train_set / f"{config}.jsonl")


def jobs(aug_dir: Path, seeds: list[int], out: Path) -> list[list[str]]:
    detectors = ["linevul", "qwen3.5-4b", "llama3.1-8b"]
    cmds = []
    for det in detectors:
        for ts in TRAIN_SETS:
            for cfg in CONFIGS:
                for seed in seeds:
                    script = "train_linevul.py" if det == "linevul" else "train_llm_lora.py"
                    cmd = [sys.executable, str(HERE / script), "--train-set", ts, "--config", cfg,
                           "--seed", str(seed), "--out", str(out)]
                    if det != "linevul":
                        cmd += ["--backbone", det]
                    a = aug_path(aug_dir, ts, cfg)
                    if a:
                        cmd += ["--aug", a]
                    cmds.append(cmd)
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug-dir", type=Path, default=Path("data/augmentation"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--run", action="store_true", help="execute the jobs on the GPUs in --gpus")
    ap.add_argument("--gpus", default="0")
    a = ap.parse_args()
    out = a.out or Path("results/rq2")
    cmds = jobs(a.aug_dir, [int(s) for s in a.seeds.split(",")], out)
    if not a.run:
        for c in cmds:
            print(shlex.join(c))
        print(f"# {len(cmds)} jobs", file=sys.stderr)
        return
    gpus = a.gpus.split(",")
    running: dict[str, subprocess.Popen] = {}
    queue = list(cmds)
    while queue or running:
        for g, p in list(running.items()):
            if p.poll() is not None:
                del running[g]
        for g in gpus:
            if g not in running and queue:
                cmd = queue.pop(0)
                running[g] = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
        time.sleep(10)


if __name__ == "__main__":
    main()
