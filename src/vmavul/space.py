"""The Vulnerability Manifestation Atlas (VMA): dimensions, values and admissible cells.

Implements Section 3.2 of the paper:

* D1 Weakness Family      -- 10 families (Table 1)
* D2 Intra-procedural Taint Length -- zero / short / medium / long
* D3 Control-flow Form    -- linear / conditional / loop / exception / nested
* D4 Source-referencing Guard -- unguarded / guarded
* D5 Size-complexity Tier -- small / medium / large

10 x 4 x 5 x 2 x 3 = 1,200 cells; three definitional constraints exclude 264 cells,
leaving 936 admissible cells.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

DIMENSIONS: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5")
DIMENSION_NAMES = {
    "D1": "weakness_family",
    "D2": "taint_length",
    "D3": "control_flow_form",
    "D4": "source_referencing_guard",
    "D5": "size_complexity_tier",
}

# D1: family -> (display name, CWE ids).
FAMILIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "buffer_overflow": ("Buffer Overflow", (119, 125, 787)),
    "null_dereference": ("NULL Dereference", (476,)),
    "use_after_free": ("Use After Free", (415, 416)),
    "integer_overflow": ("Integer Overflow", (190, 191)),
    "command_injection": ("Command Injection", (77, 78)),
    "sql_injection": ("SQL Injection", (89,)),
    "path_traversal": ("Path Traversal", (22,)),
    "resource_leak": ("Resource Leak", (401, 404)),
    "division_by_zero": ("Division by Zero", (369,)),
    "format_string": ("Format String", (134,)),
}

VALUES: dict[str, tuple[str, ...]] = {
    "D1": tuple(FAMILIES),
    "D2": ("zero", "short", "medium", "long"),
    "D3": ("linear", "conditional", "loop", "exception", "nested"),
    "D4": ("unguarded", "guarded"),
    "D5": ("small", "medium", "large"),
}

CWE_TO_FAMILY: dict[int, str] = {cwe: fam for fam, (_, cwes) in FAMILIES.items() for cwe in cwes}


def family_of_cwe(cwe: str | int | None) -> Optional[str]:
    """Map ``"CWE-787"`` / ``787`` to its family (None if outside the 10 families)."""
    if cwe is None:
        return None
    text = str(cwe).strip().upper().replace("CWE-", "").replace("CWE", "")
    try:
        return CWE_TO_FAMILY.get(int(text))
    except ValueError:
        return None


def bin_taint_length(edges: int) -> str:
    """D2 binning: 0 -> zero, 1-2 -> short, 3-5 -> medium, >=6 -> long (Table 1)."""
    if edges < 0:
        raise ValueError("taint length must be non-negative")
    if edges == 0:
        return "zero"
    if edges <= 2:
        return "short"
    if edges <= 5:
        return "medium"
    return "long"


def classify_size(nloc: int, ccn: int) -> str:
    """D5: Small (NLOC<30 and CCN<5), Large (NLOC>=100 or CCN>=12), else Medium."""
    if nloc >= 100 or ccn >= 12:
        return "large"
    if nloc < 30 and ccn < 5:
        return "small"
    return "medium"


def classify_control_flow(structure_kinds: Sequence[str]) -> str:
    """D3 from the kinds of the distinct controlling structures of the sink.

    ``structure_kinds`` holds one entry per control statement on which the sink is
    transitively control dependent (``conditional`` / ``loop`` / ``exception``).
    0 -> linear, 1 -> its kind, >=2 -> nested.
    """
    if not structure_kinds:
        return "linear"
    if len(structure_kinds) >= 2:
        return "nested"
    kind = structure_kinds[0]
    if kind not in ("conditional", "loop", "exception"):
        raise ValueError(f"unknown control structure kind {kind!r}")
    return kind


Cell = tuple  # tuple of dimension values in the order of ``ManifestationSpace.dims``


@dataclass(frozen=True)
class Constraint:
    """Excludes cells where every (dimension, value) pair in ``when`` holds."""

    name: str
    when: tuple[tuple[str, str], ...]

    def dims(self) -> set[str]:
        return {d for d, _ in self.when}

    def violated(self, assignment: dict[str, str]) -> bool:
        return all(assignment.get(d) == v for d, v in self.when)


# The three constraints of Section 3.2.
CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint("uaf_cannot_be_zero_taint", (("D1", "use_after_free"), ("D2", "zero"))),
    Constraint("linear_cannot_be_guarded", (("D3", "linear"), ("D4", "guarded"))),
    Constraint("exception_cannot_be_guarded", (("D3", "exception"), ("D4", "guarded"))),
)


class ManifestationSpace:
    """The five-dimensional manifestation space and its admissible cells."""

    def __init__(self, dims: Sequence[str] = DIMENSIONS) -> None:
        unknown = [d for d in dims if d not in DIMENSIONS]
        if unknown:
            raise ValueError(f"unknown dimensions {unknown}")
        self.dims: tuple[str, ...] = tuple(d for d in DIMENSIONS if d in dims)
        self.constraints = tuple(c for c in CONSTRAINTS if c.dims() <= set(self.dims))
        self._all = [tuple(c) for c in itertools.product(*(VALUES[d] for d in self.dims))]
        self._admissible = [c for c in self._all if self._is_admissible(c)]
        self._admissible_set = set(self._admissible)

    # ------------------------------------------------------------------ cells
    def as_dict(self, cell: Cell) -> dict[str, str]:
        return dict(zip(self.dims, cell))

    def cell_from(self, descriptor: dict[str, Optional[str]]) -> Optional[Cell]:
        """Project a full descriptor onto this space (None if a needed value is missing)."""
        values = []
        for d in self.dims:
            v = descriptor.get(d)
            if v is None:
                return None
            values.append(v)
        return tuple(values)

    def _is_admissible(self, cell: Cell) -> bool:
        assignment = self.as_dict(cell)
        return not any(c.violated(assignment) for c in self.constraints)

    def is_admissible(self, cell: Optional[Cell]) -> bool:
        return cell is not None and cell in self._admissible_set

    @property
    def total_cells(self) -> int:
        return len(self._all)

    @property
    def admissible_cells(self) -> list[Cell]:
        return list(self._admissible)

    def cells_of_family(self, family: Optional[str]) -> list[Cell]:
        if family is None or "D1" not in self.dims:
            return list(self._admissible)
        i = self.dims.index("D1")
        return [c for c in self._admissible if c[i] == family]

    def family_of(self, cell: Cell) -> Optional[str]:
        if "D1" not in self.dims:
            return None
        return cell[self.dims.index("D1")]

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        """Admissible cells that differ from ``cell`` in exactly one dimension."""
        for i, d in enumerate(self.dims):
            for v in VALUES[d]:
                if v == cell[i]:
                    continue
                other = cell[:i] + (v,) + cell[i + 1:]
                if other in self._admissible_set:
                    yield other

    @staticmethod
    def key(cell: Cell) -> str:
        return "|".join(cell)

    def parse_key(self, key: str) -> Cell:
        cell = tuple(key.split("|"))
        if len(cell) != len(self.dims):
            raise ValueError(f"cell key {key!r} does not match dims {self.dims}")
        return cell

    def summary(self) -> dict:
        return {
            "dims": list(self.dims),
            "total_cells": self.total_cells,
            "excluded_cells": self.total_cells - len(self._admissible),
            "admissible_cells": len(self._admissible),
            "constraints": [c.name for c in self.constraints],
        }


def iter_descriptor_values(descriptor: dict[str, Optional[str]], dims: Iterable[str]) -> list[str]:
    return [str(descriptor.get(d)) for d in dims]
