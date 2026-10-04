"""Render a (possibly partial) manifestation descriptor as natural-language requirements
derived from the Table 1 characterization rules (Step 2(i))."""

from __future__ import annotations

from typing import Optional

from .space import FAMILIES

_FAMILY_SINK = {
    "buffer_overflow": "a buffer access / memory-copy whose size or index can exceed the buffer bounds",
    "null_dereference": "a dereference of a pointer that can be NULL",
    "use_after_free": "a use of a pointer after it has been freed",
    "integer_overflow": "an arithmetic operation that can overflow and feed a size/index/allocation",
    "command_injection": "a shell/exec call whose command string can be influenced by input",
    "sql_injection": "a database query built from input without parameterization",
    "path_traversal": "a filesystem call whose path can be influenced by input",
    "resource_leak": "an acquired resource (memory/file/socket) that can leak on some path",
    "division_by_zero": "a division or modulo whose divisor can be zero",
    "format_string": "a printf-family call whose format string can be influenced by input",
}

_D2 = {"zero": "0 def-use edges (the source value reaches the sink directly, within the sink statement)",
       "short": "1-2 def-use edges on the shortest source-to-sink path",
       "medium": "3-5 def-use edges on the shortest source-to-sink path",
       "long": "at least 6 def-use edges on the shortest source-to-sink path"}
_D3 = {"linear": "no control structure governs the sink (it executes unconditionally)",
       "conditional": "exactly one conditional (if/switch) governs whether the sink executes",
       "loop": "exactly one loop governs whether/how often the sink executes",
       "exception": "exactly one catch handler governs the sink",
       "nested": "at least two control structures jointly govern the sink"}
_D4 = {"unguarded": "no predicate governing the sink checks the vulnerability-relevant (source) value",
       "guarded": "at least one predicate governing the sink references the source value or a value "
                  "derived from it (the guard may be incomplete or incorrect)"}
_D5 = {"small": "Small: NLOC < 30 and cyclomatic complexity < 5",
       "medium": "Medium: between the Small and Large tiers",
       "large": "Large: NLOC >= 100 or cyclomatic complexity >= 12"}

_LABELS = {"D1": "Weakness Family", "D2": "Intra-procedural Taint Length", "D3": "Control-flow Form",
           "D4": "Source-referencing Guard", "D5": "Size-complexity Tier"}
_MAPS = {"D2": _D2, "D3": _D3, "D4": _D4, "D5": _D5}


def requirements(descriptor: dict[str, Optional[str]], dims) -> list[str]:
    out: list[str] = []
    for d in dims:
        val = descriptor.get(d)
        if val is None:
            continue
        if d == "D1":
            name, cwes = FAMILIES[val]
            out.append(f"{_LABELS[d]}: {name} (CWE {', '.join(str(c) for c in cwes)}). The vulnerability must "
                       f"be realized through {_FAMILY_SINK[val]}.")
        else:
            out.append(f"{_LABELS[d]}: {val} -- {_MAPS[d][val]}.")
    return out
