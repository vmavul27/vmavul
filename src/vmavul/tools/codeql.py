"""CodeQL static checking (Step 3(2)): run the family's predefined queries on a version.

A database is created by compiling the unit with Clang under ``codeql database create``;
the selected queries are run with ``codeql database analyze`` and the SARIF results are
mapped back to the query selectors (and hence families) that produced them.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .clang import base_flags, source_name
from .process import run


@dataclass
class CodeQLFinding:
    query_id: str
    query_path: str
    message: str
    line: int


@dataclass
class CodeQLResult:
    ok: bool
    findings: list[CodeQLFinding] = field(default_factory=list)
    error: str = ""


def _resolve_selector(pack_root: Path, selector: str) -> list[Path]:
    p = pack_root / selector
    if p.is_dir():
        return sorted(q for q in p.rglob("*.ql"))
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"CodeQL query selector not found in pack: {selector}")


_ID_RE = re.compile(r"@id\s+(\S+)")


def query_id(path: Path) -> str:
    m = _ID_RE.search(path.read_text(errors="replace")[:4000])
    return m.group(1) if m else path.stem


def queries_for(pack_root: str, selectors: Iterable[str]) -> dict[str, str]:
    """Map absolute .ql path -> selector (only queries that declare an ``@id``)."""
    root = Path(pack_root)
    if not root.is_dir():
        raise FileNotFoundError(f"codeql/cpp-queries pack not found: {pack_root!r} "
                                "(set VMAVUL_CODEQL_CPP_QUERIES)")
    out: dict[str, str] = {}
    for sel in selectors:
        for q in _resolve_selector(root, sel):
            out[str(q.resolve())] = sel
    return out


def analyze(codeql: str, pack_root: str, unit_code: str, language: str, selectors: list[str],
            clang: str, work: Path, *, timeout: float = 300.0, threads: int = 1, tolerant: bool = False) -> tuple[CodeQLResult, dict[str, str]]:
    """Run ``selectors`` on ``unit_code``.  Returns the result and the query->selector map."""
    qmap = queries_for(pack_root, selectors)
    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / source_name(language)
    src.write_text(unit_code)
    db = work / "db"
    # The build is a small shell script; tolerant mode lets the extractor inspect
    # incomplete units when explicitly requested.
    build = " ".join(shlex.quote(x) for x in [clang] + base_flags(language) + ["-w", "-c", src.name, "-o", "unit.o"])
    script = src_dir / "build.sh"
    script.write_text("#!/bin/sh\n" + build + (" || true\nexit 0\n" if tolerant else "\n"))
    res = run([codeql, "database", "create", str(db), "--language=cpp", f"--source-root={src_dir}",
               f"--command=sh {script.name}", "--overwrite", f"--threads={threads}"], timeout=timeout)
    if res.returncode != 0:
        return CodeQLResult(False, error=(res.stderr or res.stdout)[-4000:]), qmap
    sarif = work / "results.sarif"
    res = run([codeql, "database", "analyze", str(db), *qmap.keys(), "--format=sarif-latest",
               f"--output={sarif}", f"--threads={threads}", "--no-print-diagnostics-summary",
               "--no-sarif-add-snippets"], timeout=timeout)
    if res.returncode != 0 or not sarif.exists():
        return CodeQLResult(False, error=(res.stderr or res.stdout)[-4000:]), qmap
    return CodeQLResult(True, parse_sarif(json.loads(sarif.read_text()))), qmap


def parse_sarif(sarif: dict) -> list[CodeQLFinding]:
    findings: list[CodeQLFinding] = []
    for run_ in sarif.get("runs", []):
        for r in run_.get("results", []):
            rid = r.get("ruleId", "") or (r.get("rule") or {}).get("id", "")
            line = 0
            locs = r.get("locations") or []
            if locs:
                line = int(locs[0].get("physicalLocation", {}).get("region", {}).get("startLine", 0) or 0)
            findings.append(CodeQLFinding(rid, "", (r.get("message") or {}).get("text", ""), line))
    return findings


def selector_ids(qmap: dict[str, str]) -> dict[str, set[str]]:
    """selector -> set of query ``@id`` values it contains."""
    out: dict[str, set[str]] = {}
    for qpath, sel in qmap.items():
        out.setdefault(sel, set()).add(query_id(Path(qpath)))
    return out
