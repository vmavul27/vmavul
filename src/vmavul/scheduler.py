"""Step 1: coverage-guided scheduling (Section 3.3).

Stages
------
(1) Bootstrapping -- for each weakness family in turn, target uniformly random
    admissible cells of that family until accepted samples cover ``bootstrap_cells``
    (3) distinct cells of the family, or no new cell is covered in ``patience`` (3)
    consecutive iterations.
(2) Coverage Expansion -- target the uncovered, unexhausted admissible cell with the
    most covered neighbours (cells differing in exactly one dimension), ties broken
    uniformly at random.  A cell still uncovered after ``k`` attempts is exhausted.  The
    stage ends when no uncovered and unexhausted cell remains.
(3) Densification -- until the synthesis budget is reached: first target covered cells
    with |A(b)| < tau (varying one of their samples); once every covered cell holds tau
    samples, vary an archived sample toward an uncovered neighbouring cell of the same
    weakness family if one exists, otherwise diversify it within its own cell.

Accepted samples are archived by their *measured* descriptors (see ``agent.py``).
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .archive import Archive
from .config import HyperParams
from .space import ManifestationSpace, VALUES


@dataclass
class Task:
    stage: str                       # bootstrapping | coverage_expansion | densification
    target: Optional[tuple]          # target VMA cell
    family: Optional[str]            # target weakness family
    vary: Optional[dict] = None      # archived sample to vary (Densification)
    mode: str = "generate"           # generate | densify | toward_neighbor | diversify


class Scheduler:
    def __init__(self, space: ManifestationSpace, archive: Archive, hyper: HyperParams, budget: int,
                 rng: random.Random, state_path: Optional[Path] = None,
                 densify_skip_exhausted: bool = True) -> None:
        self.space = space
        self.archive = archive
        self.h = hyper
        self.budget = budget
        self.rng = rng
        self.state_path = Path(state_path) if state_path else None
        self.densify_skip_exhausted = densify_skip_exhausted
        self._lock = threading.RLock()
        self.families: list[Optional[str]] = list(VALUES["D1"]) if "D1" in space.dims else [None]
        self.stage = "bootstrapping"
        self.family_index = 0
        self.no_new_streak = 0
        self.attempts: dict[str, int] = {}
        self.exhausted: set[str] = set()
        self.iterations = 0
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.state_path and self.state_path.exists():
            s = json.loads(self.state_path.read_text())
            self.stage, self.family_index = s["stage"], s["family_index"]
            self.no_new_streak, self.attempts = s["no_new_streak"], s["attempts"]
            self.exhausted, self.iterations = set(s["exhausted"]), s["iterations"]

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "stage": self.stage, "family_index": self.family_index, "no_new_streak": self.no_new_streak,
            "attempts": self.attempts, "exhausted": sorted(self.exhausted), "iterations": self.iterations,
        }, indent=1))
        tmp.replace(self.state_path)

    # ------------------------------------------------------------------ helpers
    def done(self) -> bool:
        return self.archive.size() >= self.budget

    def _family_covered(self, family: Optional[str]) -> int:
        cells = set(self.space.cells_of_family(family))
        return len(cells & self.archive.covered())

    def _uncovered_candidates(self) -> list[tuple]:
        covered = self.archive.covered()
        return [c for c in self.space.admissible_cells
                if c not in covered and self.space.key(c) not in self.exhausted]

    def _advance_bootstrap(self) -> None:
        while self.stage == "bootstrapping":
            if self.family_index >= len(self.families):
                self.stage = "coverage_expansion"
                break
            fam = self.families[self.family_index]
            if self._family_covered(fam) >= self.h.bootstrap_cells_per_family \
                    or self.no_new_streak >= self.h.bootstrap_patience:
                self.family_index += 1
                self.no_new_streak = 0
                continue
            break
        if self.stage == "coverage_expansion" and not self._uncovered_candidates():
            self.stage = "densification"

    # ------------------------------------------------------------------ API
    def next(self) -> Optional[Task]:
        with self._lock:
            if self.done():
                return None
            self._advance_bootstrap()
            if self.stage == "bootstrapping":
                fam = self.families[self.family_index]
                return Task("bootstrapping", self.rng.choice(self.space.cells_of_family(fam)), fam)
            if self.stage == "coverage_expansion":
                cands = self._uncovered_candidates()
                if cands:
                    covered = self.archive.covered()
                    scores = [sum(1 for n in self.space.neighbors(c) if n in covered) for c in cands]
                    best = max(scores)
                    cell = self.rng.choice([c for c, s in zip(cands, scores) if s == best])
                    return Task("coverage_expansion", cell, self.space.family_of(cell))
                self.stage = "densification"
            return self._densify()

    def _densify(self) -> Optional[Task]:
        covered = sorted(self.archive.covered())
        if not covered:
            return None
        low = [c for c in covered if self.archive.count(c) < self.h.tau_density]
        if low:
            least = min(self.archive.count(c) for c in low)
            cell = self.rng.choice([c for c in low if self.archive.count(c) == least])
            return Task("densification", cell, self.space.family_of(cell),
                        self.archive.random_sample(cell, self.rng), "densify")
        cell = self.rng.choice(covered)
        sample = self.archive.random_sample(cell, self.rng)
        fam = self.space.family_of(cell)
        cov = set(covered)
        nbrs = [n for n in self.space.neighbors(cell) if n not in cov
                and self.space.family_of(n) == fam
                and not (self.densify_skip_exhausted and self.space.key(n) in self.exhausted)]
        if nbrs:
            target = self.rng.choice(nbrs)
            return Task("densification", target, fam, sample, "toward_neighbor")
        return Task("densification", cell, fam, sample, "diversify")

    def observe(self, task: Task, accepted_cell: Optional[tuple], was_new_cell: bool) -> None:
        with self._lock:
            self.iterations += 1
            if task.stage == "bootstrapping":
                self.no_new_streak = 0 if was_new_cell else self.no_new_streak + 1
            if task.target is not None:
                key = self.space.key(task.target)
                if task.target not in self.archive.covered():
                    self.attempts[key] = self.attempts.get(key, 0) + 1
                    if self.attempts[key] >= self.h.k_exhausted_attempts:
                        self.exhausted.add(key)
            self.save()
