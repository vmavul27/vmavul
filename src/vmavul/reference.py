"""Reference pool of real-world vulnerable functions (Step 2(ii)).

Built by ``scripts/data/build_reference_pool.py`` from the TRAINING split only; the
loader refuses any record whose split is not ``train``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from .space import family_of_cwe


class ReferencePool:
    def __init__(self, records: list[dict]) -> None:
        bad = [r.get("id") for r in records if r.get("split") != "train"]
        if bad:
            raise ValueError(f"reference pool must contain training-split functions only; "
                             f"{len(bad)} non-train records (e.g. {bad[:3]})")
        self.records = records
        self.by_family: dict[str, list[dict]] = {}
        for r in records:
            fam = r.get("family") or family_of_cwe(r.get("cwe"))
            if fam:
                self.by_family.setdefault(fam, []).append(r)

    @classmethod
    def load(cls, path: str) -> "ReferencePool":
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return cls(recs)

    def sample(self, family: Optional[str], n: int, rng: random.Random) -> list[dict]:
        """``n`` functions of ``family`` drawn uniformly at random (any family if None)."""
        pool = self.by_family.get(family, []) if family else self.records
        if not pool:
            return []
        picks = rng.sample(pool, min(n, len(pool)))
        return [{"code": p["func"], "cwe": p.get("cwe", ""), "id": p.get("id", "")} for p in picks]
