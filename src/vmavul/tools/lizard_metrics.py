"""Lizard 1.23.0 metrics of the target function alone (D5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import lizard


@dataclass
class SizeMetrics:
    nloc: int
    ccn: int


def measure(function_code: str, language: str = "c", name: Optional[str] = None) -> Optional[SizeMetrics]:
    """NLOC and CCN of the function in ``function_code`` (the calling context is excluded
    because only the target function is passed in)."""
    filename = "target.cpp" if language == "c++" else "target.c"
    info = lizard.analyze_file.analyze_source_code(filename, function_code)
    funcs = info.function_list
    if not funcs:
        return None
    chosen = next((f for f in funcs if name and f.name.split("::")[-1] == name), None)
    if chosen is None:
        chosen = max(funcs, key=lambda f: f.nloc)
    return SizeMetrics(int(chosen.nloc), int(chosen.cyclomatic_complexity))
