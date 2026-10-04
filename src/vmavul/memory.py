"""Refinement memory: lessons from successful refinements, retrieved during generation
(Step 2(iii)).  An entry is stored whenever a refined candidate passes the check that
the previous version failed; retrieval returns the most recent entries of the target
weakness family, preferring entries recorded for the same target cell."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MemoryEntry:
    family: Optional[str]
    cell: list
    failed_check: str
    diagnostic: str
    lesson: str


class RefinementMemory:
    def __init__(self, path: Optional[Path] = None, enabled: bool = True) -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled
        self.entries: list[MemoryEntry] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.entries.append(MemoryEntry(**json.loads(line)))

    def add(self, entry: MemoryEntry) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.entries.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def retrieve(self, family: Optional[str], cell: Optional[tuple], k: int) -> list[MemoryEntry]:
        if not self.enabled or k <= 0:
            return []
        with self._lock:
            pool = [e for e in self.entries if family is None or e.family == family]
        same = [e for e in pool if cell is not None and tuple(e.cell) == tuple(cell)]
        rest = [e for e in pool if e not in same]
        ordered = list(reversed(same)) + list(reversed(rest))
        return ordered[:k]
