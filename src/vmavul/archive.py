"""The archive A: accepted samples stored by (measured) cell, persisted as JSONL."""

from __future__ import annotations

import base64
import json
import random
import threading
from pathlib import Path
from typing import Optional

import numpy as np


def _enc(v: np.ndarray) -> str:
    return base64.b64encode(v.astype(np.float16).tobytes()).decode("ascii")


def _dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


class Archive:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.cells: dict[tuple, list[dict]] = {}
        self.vectors: dict[tuple, list[np.ndarray]] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._insert(rec, _dec(rec["embedding"]) if rec.get("embedding") else None)

    def _insert(self, rec: dict, vec: Optional[np.ndarray]) -> None:
        cell = tuple(rec["cell"])
        self.cells.setdefault(cell, []).append(rec)
        if vec is not None:
            self.vectors.setdefault(cell, []).append(vec)

    def add(self, rec: dict, vec: Optional[np.ndarray]) -> None:
        with self._lock:
            rec = dict(rec)
            if vec is not None:
                rec["embedding"] = _enc(vec)
            self._insert(rec, vec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def min_distance(self, cell, vec: np.ndarray) -> Optional[float]:
        from .tools.embedding import min_cosine_distance
        with self._lock:
            vecs = self.vectors.get(tuple(cell)) if cell is not None else None
            if not vecs:
                return None
            return min_cosine_distance(vec, np.stack(vecs))

    def count(self, cell) -> int:
        with self._lock:
            return len(self.cells.get(tuple(cell), []))

    def covered(self) -> set[tuple]:
        with self._lock:
            return {c for c, v in self.cells.items() if v}

    def size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self.cells.values())

    def random_sample(self, cell, rng: random.Random) -> Optional[dict]:
        with self._lock:
            items = self.cells.get(tuple(cell))
            return rng.choice(items) if items else None

    def all_samples(self) -> list[dict]:
        with self._lock:
            return [r for v in self.cells.values() for r in v]
