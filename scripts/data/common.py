"""Shared JSONL, de-duplication, and formatting helpers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def normalized_hash(code: str) -> str:
    """Hash of the code with comments and all whitespace removed (duplicate detection)."""
    text = _WS_RE.sub("", _COMMENT_RE.sub("", code or ""))
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _fmt_one(code: str) -> str:
    try:
        p = subprocess.run(["clang-format", "--style=LLVM", "--assume-filename=input.c"], input=code,
                           capture_output=True, text=True, timeout=10)
        return p.stdout if p.returncode == 0 and p.stdout.strip() else code
    except (OSError, subprocess.TimeoutExpired):
        return code


def clang_format_all(codes: list[str], workers: int = 8) -> list[str]:
    """Format an augmentation set with clang-format (LLVM style) when available."""
    if not shutil.which("clang-format"):
        return codes
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_fmt_one, codes, chunksize=64))
