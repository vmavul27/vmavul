"""Candidate artifacts: vulnerable function, patched function and shared calling context.

The generation prompt asks for three tagged blocks::

    <vulnerable_function> ... </vulnerable_function>
    <patched_function> ... </patched_function>
    <calling_context> ... </calling_context>

The calling context supplies types, auxiliary APIs and a ``main`` driver; it contains
the line ``/* @TARGET_FUNCTION@ */`` where the (vulnerable or patched) target function
is inserted.  If the marker is missing, the function is inserted before ``main``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

MARKER = "/* @TARGET_FUNCTION@ */"

_TAG_RE = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in ("vulnerable_function", "patched_function", "calling_context", "fix_summary")
}
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9+]*\s*\n(.*?)\n```\s*$", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9+]*\s*\n(.*?)\n```", re.DOTALL)
_CPP_HINTS = re.compile(r"(#include\s*<(iostream|string|vector|map|memory|cstdio|cstring|cstdlib)>"
                        r"|\bstd::|\bnamespace\b|\bclass\s+\w+|\btemplate\s*<|\bnew\s+\w|\bdelete\b"
                        r"|\btry\s*\{|\bcatch\s*\()")
_FUNC_DEF_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z_][\w:<>\*\s&,]*?[\s\*&])?([A-Za-z_]\w*(?:::[A-Za-z_~]\w*)*)\s*\(([^;{}]*)\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?\{", re.MULTILINE)
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "do", "else", "catch"}


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = _TAG_RE[tag].search(text or "")
    return _strip_fence(m.group(1)) if m else None


def first_code_block(text: str) -> Optional[str]:
    m = _ANY_FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return None


def function_name(code: str) -> Optional[str]:
    """Name of the first function *definition* in ``code``."""
    for m in _FUNC_DEF_RE.finditer(code or ""):
        name = m.group(1).split("::")[-1]
        if name not in _KEYWORDS:
            return name
    return None


def detect_language(*codes: str) -> str:
    return "c++" if any(_CPP_HINTS.search(c or "") for c in codes) else "c"


@dataclass
class Candidate:
    vulnerable: str
    patched: str
    context: str
    target_name: str
    language: str = "c"
    fix_summary: str = ""
    generator: str = ""
    refiners: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "Candidate":
        """Parse an LLM response; raises ``ValueError`` with a readable reason."""
        vuln = extract_tag(text, "vulnerable_function")
        patched = extract_tag(text, "patched_function")
        ctx = extract_tag(text, "calling_context")
        missing = [n for n, v in (("vulnerable_function", vuln), ("patched_function", patched),
                                  ("calling_context", ctx)) if not v]
        if missing:
            raise ValueError("response is missing the tagged block(s): " + ", ".join(missing))
        name = function_name(vuln or "")
        if not name:
            raise ValueError("could not find a function definition in <vulnerable_function>")
        pname = function_name(patched or "")
        if pname != name:
            raise ValueError(f"patched function is named {pname!r}, expected {name!r}")
        return cls(vulnerable=vuln, patched=patched, context=ctx, target_name=name,
                   language=detect_language(vuln, patched, ctx),
                   fix_summary=extract_tag(text, "fix_summary") or "")

    # ------------------------------------------------------------------ units
    def unit(self, version: str) -> tuple[str, tuple[int, int]]:
        """Full compilation unit for ``version`` ('vulnerable'|'patched') and the
        1-based (first, last) line range of the target function inside it."""
        fn = self.vulnerable if version == "vulnerable" else self.patched
        ctx = self.context
        if MARKER in ctx:
            before, after = ctx.split(MARKER, 1)
        else:
            m = re.search(r"^\s*int\s+main\s*\(", ctx, re.MULTILINE)
            if m:
                before, after = ctx[: m.start()], ctx[m.start():]
            else:
                before, after = ctx, ""
        before = before.rstrip("\n") + "\n"
        start = before.count("\n") + 1
        end = start + fn.count("\n")
        return before + fn + "\n" + after.lstrip("\n"), (start, end)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def render_artifacts(c: Candidate) -> str:
    """Render a candidate back into the tagged format (used in refinement prompts)."""
    return (f"<vulnerable_function>\n{c.vulnerable}\n</vulnerable_function>\n"
            f"<patched_function>\n{c.patched}\n</patched_function>\n"
            f"<calling_context>\n{c.context}\n</calling_context>")
