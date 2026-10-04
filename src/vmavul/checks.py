"""Step 3: tool-supported quality checks, in order of increasing cost.

(1) syntax checking  ->  (2) weakness-specific checking  ->  (3) manifestation
characterization  ->  (4) within-cell diversity checking.  A failure at any check
returns the diagnostics of that check (consumed by refinement, Step 4).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import yaml

from .artifacts import Candidate
from .config import HyperParams, ToolConfig, resolve_tools
from .space import FAMILIES, ManifestationSpace, bin_taint_length, classify_control_flow, classify_size

STAGES = ("syntax", "weakness", "characterization", "diversity")


# --------------------------------------------------------------------------- results
@dataclass
class FamilyEvidence:
    family: str
    static_applicable: bool = True
    static_vuln_hit: bool = False
    static_patched_hit: bool = False
    runtime_applicable: bool = False
    runtime_vuln_hit: bool = False
    runtime_patched_hit: bool = False
    runtime_vuln_done: bool = False
    runtime_patched_done: bool = False
    runtime_require_both: bool = True
    runtime_unrelated_error: str = ""
    static_error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def static_pass(self) -> bool:
        if self.static_error:
            return False
        return self.static_vuln_hit and not self.static_patched_hit

    @property
    def runtime_pass(self) -> bool:
        if not self.runtime_vuln_done:
            return False
        if self.runtime_require_both and not self.runtime_patched_done:
            return False
        return self.runtime_vuln_hit and not self.runtime_patched_hit

    @property
    def contradiction(self) -> Optional[str]:
        if self.static_patched_hit and not self.static_vuln_hit:
            return "the patched version triggers the CodeQL finding but the vulnerable version does not"
        if self.static_vuln_hit and self.static_patched_hit:
            return "the patched version still triggers the CodeQL finding"
        if self.runtime_patched_hit:
            return "the patched version still triggers the sanitizer finding"
        if self.runtime_unrelated_error:
            return f"runtime failure caused by an unrelated error: {self.runtime_unrelated_error}"
        return None

    @property
    def passed(self) -> bool:
        return (self.static_pass or self.runtime_pass) and self.contradiction is None


@dataclass
class CheckOutcome:
    accepted: bool
    stage: str                                  # failed stage, or "accepted"
    diagnostics: str = ""
    family: Optional[str] = None                # D1 (family whose checks passed)
    descriptor: dict = field(default_factory=dict)
    cell: Optional[tuple] = None
    evidence: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- toolchain
class Toolchain(Protocol):
    def syntax(self, cand: Candidate) -> tuple[bool, str]: ...
    def weakness(self, cand: Candidate, families: list[str]) -> dict[str, FamilyEvidence]: ...
    def characterize(self, cand: Candidate, family: str) -> tuple[dict, dict]: ...
    def embed(self, code: str) -> np.ndarray: ...


def load_sinks(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class RealToolchain:
    """Clang 17 / CodeQL 2.25.4 / ASan-UBSan-LSan / Joern 4.0.547 / Lizard / UniXcoder."""

    def __init__(self, tools: ToolConfig, sinks: dict) -> None:
        from .tools import joern as joern_mod
        from .tools.embedding import Embedder
        self.t = resolve_tools(tools)
        self.sinks = sinks
        self.joern = joern_mod.JoernPool(self.t.joern_home, self.t.joern_servers, self.t.joern_base_port)
        self.embedder = Embedder(self.t.embed_model, self.t.embed_device)

    def _work(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="vmavul_chk_", dir=self.t.work_dir or None))

    def syntax(self, cand: Candidate) -> tuple[bool, str]:
        from .tools.clang import compile_unit, source_name
        from .tools.sandbox import scan_destructive
        bad = scan_destructive(cand.vulnerable, cand.patched, cand.context)
        if bad:
            return False, ("The candidate embeds destructive shell commands (" + ", ".join(bad) +
                           "); remove them -- the program must be self-contained and harmless.")
        work = self._work()
        try:
            msgs = []
            for version in ("vulnerable", "patched"):
                code, _ = cand.unit(version)
                d = work / version
                d.mkdir()
                src = d / source_name(cand.language)
                src.write_text(code)
                res = compile_unit(self.t.clang, src, d / "prog", cand.language,
                                   timeout=self.t.compile_timeout_s)
                if not res.ok:
                    msgs.append(f"[{version} version + calling context]\n{res.diagnostics[-3000:]}")
            return (not msgs), "\n\n".join(msgs)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def weakness(self, cand: Candidate, families: list[str], *, differential: bool = True,
                 tolerant: bool = False, runtime: bool = True) -> dict[str, FamilyEvidence]:
        """Static (CodeQL) and runtime (sanitizer) evidence per family."""
        from .tools import codeql, sanitizer
        fams = self.sinks["families"]
        ev = {f: FamilyEvidence(f) for f in families}
        work = self._work()
        try:
            # ---- static: one database per version, the union of the families' queries
            selectors = sorted({q for f in families for q in fams[f]["codeql_queries"]})
            hits: dict[str, set[str]] = {}
            versions = ("vulnerable", "patched") if differential else ("vulnerable",)
            for version in versions:
                code, (lo, hi) = cand.unit(version)
                res, qmap = codeql.analyze(self.t.codeql, self.t.codeql_cpp_queries, code, cand.language,
                                           selectors, self.t.clang, work / f"codeql_{version}",
                                           timeout=self.t.codeql_timeout_s, threads=self.t.codeql_threads,
                                           tolerant=tolerant)
                if not res.ok:
                    msg = f"CodeQL failed on the {version} version: {(res.error or '')[-800:]}"
                    for f in families:
                        ev[f].notes.append(msg)
                        ev[f].static_error = f"{ev[f].static_error}; {msg}" if ev[f].static_error else msg
                    hits[version] = set()
                    continue
                in_target = [x for x in res.findings if lo <= x.line <= hi]
                hits[version] = {x.query_id for x in in_target}
                ids = codeql.selector_ids(qmap)
                for f in families:
                    fam_ids = set().union(*(ids.get(s, set()) for s in fams[f]["codeql_queries"]))
                    found = [x for x in in_target if x.query_id in fam_ids]
                    if version == "vulnerable":
                        ev[f].static_vuln_hit = bool(found)
                    else:
                        ev[f].static_patched_hit = bool(found)
                    for x in found:
                        ev[f].notes.append(f"CodeQL [{version}] {x.query_id} line {x.line - lo + 1} of the "
                                           f"target function: {x.message[:300]}")
            # ---- runtime: one build per sanitizer kind and version
            reports: dict[tuple[str, str], sanitizer.SanitizerReport] = {}
            for f in families:
                kind = fams[f].get("sanitizer")
                if not kind or not runtime:
                    continue
                ev[f].runtime_applicable = True
                ev[f].runtime_require_both = "patched" in versions
                patterns = [p.lower() for p in fams[f].get("sanitizer_errors", [])]
                for version in versions:
                    key = (kind, version)
                    if key not in reports:
                        code, rng = cand.unit(version)
                        reports[key] = sanitizer.run_with_sanitizer(
                            self.t.clang, kind, code, cand.language, cand.target_name, rng,
                            work / f"{kind}_{version}", compile_timeout=self.t.compile_timeout_s,
                            run_timeout=self.t.run_timeout_s, memory_mb=self.t.run_memory_mb,
                            sandbox=self.t.sandbox_command, sandbox_mode=self.t.sandbox)
                    rep = reports[key]
                    if not rep.built or not rep.ran:
                        detail = rep.build_error[-500:] if not rep.built else (rep.build_error or "did not run")
                        ev[f].notes.append(f"{kind} did not complete ({version}): {detail}")
                        continue
                    matches = rep.triggered and any(p in rep.raw.lower() for p in patterns)
                    if version == "vulnerable":
                        ev[f].runtime_vuln_done = True
                        ev[f].runtime_vuln_hit = matches and rep.in_target
                        if rep.triggered and not (matches and rep.in_target):
                            ev[f].runtime_unrelated_error = rep.error_text or "abnormal termination"
                    else:
                        ev[f].runtime_patched_done = True
                        ev[f].runtime_patched_hit = matches
                        if rep.triggered and not matches:
                            ev[f].runtime_unrelated_error = (
                                "patched version: " + (rep.error_text or "abnormal termination")
                            )
                    if rep.triggered:
                        ev[f].notes.append(f"{kind} [{version}]: {rep.error_text} "
                                           f"(in target function: {rep.in_target})")
                    else:
                        ev[f].notes.append(f"{kind} [{version}]: no error reported (exit {rep.exit_code})")
            return ev
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def characterize(self, cand: Candidate, family: str) -> tuple[dict, dict]:
        from .tools import joern, lizard_metrics
        code, _ = cand.unit("vulnerable")
        analysis = joern.analyze(self.joern, code, cand.language, cand.target_name,
                                 self.sinks["families"][family], self.sinks.get("common_sources", {}),
                                 timeout=self.t.joern_timeout_s)
        size = lizard_metrics.measure(cand.vulnerable, cand.language, cand.target_name)
        return descriptor_from_analysis(family, analysis, size)

    def embed(self, code: str) -> np.ndarray:
        return self.embedder.embed([code])[0]


def descriptor_from_analysis(family: Optional[str], analysis, size) -> tuple[dict, dict]:
    """Apply the Table 1 rules to the raw Joern/Lizard outputs."""
    desc: dict = {"D1": family, "D2": None, "D3": None, "D4": None, "D5": None}
    info: dict = {}
    if size is not None:
        desc["D5"] = classify_size(size.nloc, size.ccn)
        info["nloc"], info["ccn"] = size.nloc, size.ccn
    if analysis is not None and analysis.target_found:
        info.update(taint_edges=analysis.taint_edges, sink=analysis.sink, source=analysis.source,
                    flow=analysis.flow, structures=analysis.structures, predicates=analysis.predicates,
                    n_sinks=len(analysis.sinks), n_sources=len(analysis.sources))
        if analysis.taint_edges is not None:
            desc["D2"] = bin_taint_length(int(analysis.taint_edges))
        if analysis.sink is not None:
            desc["D3"] = classify_control_flow([s["kind"] for s in analysis.structures])
            desc["D4"] = "guarded" if analysis.guarded else "unguarded"
        info["sink_candidates"] = analysis.sinks[:10]
        info["source_candidates"] = analysis.sources[:10]
    else:
        info["target_found"] = False
    return desc, info


# --------------------------------------------------------------------------- checker
class QualityChecker:
    def __init__(self, toolchain: Toolchain, space: ManifestationSpace, hyper: HyperParams) -> None:
        self.tc = toolchain
        self.space = space
        self.hyper = hyper

    def check(self, cand: Candidate, target_family: Optional[str], archive) -> CheckOutcome:
        evidence: dict = {}
        # (1) syntax
        ok, diag = self.tc.syntax(cand)
        if not ok:
            return CheckOutcome(False, "syntax", diag)
        # (2) weakness-specific
        if target_family is None:
            return CheckOutcome(False, "weakness", "no target weakness family was specified")
        family = target_family
        families = [target_family]
        ev = self.tc.weakness(cand, families)
        evidence["weakness"] = {f: _ev_dict(e) for f, e in ev.items()}
        if not ev[target_family].passed:
            return CheckOutcome(False, "weakness", weakness_diagnostics(ev, families), evidence=evidence)
        evidence["family"] = family
        # (3) characterization
        descriptor, info = self.tc.characterize(cand, family)
        evidence["characterization"] = info
        missing = [d for d in self.space.dims if descriptor.get(d) is None]
        if missing:
            return CheckOutcome(False, "characterization",
                                characterization_diagnostics(missing, info), family, descriptor, evidence=evidence)
        cell = self.space.cell_from(descriptor)
        if not self.space.is_admissible(cell):
            return CheckOutcome(False, "characterization",
                                f"measured descriptor {descriptor} is not an admissible cell",
                                family, descriptor, evidence=evidence)
        # (4) within-cell diversity (minimum cosine distance must not fall below delta)
        emb = self.tc.embed(cand.vulnerable)
        dist = archive.min_distance(cell, emb)
        evidence["min_cosine_distance"] = dist
        if dist is not None and dist < self.hyper.delta_min_distance:
            return CheckOutcome(False, "diversity",
                                f"within-cell similarity failure: minimum cosine distance {dist:.3f} to the "
                                f"samples of cell {cell} is below delta={self.hyper.delta_min_distance}",
                                family, descriptor, cell, evidence, emb)
        return CheckOutcome(True, "accepted", "", family, descriptor, cell, evidence, emb)


def _ev_dict(e: FamilyEvidence) -> dict:
    return {"static_vuln_hit": e.static_vuln_hit, "static_patched_hit": e.static_patched_hit,
            "static_error": e.static_error,
            "runtime_applicable": e.runtime_applicable, "runtime_vuln_hit": e.runtime_vuln_hit,
            "runtime_patched_hit": e.runtime_patched_hit,
            "runtime_vuln_done": e.runtime_vuln_done, "runtime_patched_done": e.runtime_patched_done,
            "unrelated_error": e.runtime_unrelated_error,
            "passed": e.passed, "notes": e.notes[:12]}


def weakness_diagnostics(ev: dict[str, FamilyEvidence], families: list[str]) -> str:
    lines = []
    for f in families:
        e = ev[f]
        if len(families) > 1 and not (e.static_vuln_hit or e.runtime_vuln_hit or e.contradiction):
            continue
        lines.append(f"Weakness family {FAMILIES[f][0]}:")
        if not (e.static_vuln_hit or e.runtime_vuln_hit):
            lines.append("  - neither CodeQL nor the sanitizer reports the weakness inside the target function "
                         "of the vulnerable version")
        if e.contradiction:
            lines.append(f"  - contradicting evidence: {e.contradiction}")
        lines += [f"  - {n}" for n in e.notes[:10]]
    if not lines:
        lines.append("No weakness family's CodeQL queries or sanitizers reported a finding inside the target "
                     "function of the vulnerable version.")
    return "\n".join(lines)


def characterization_diagnostics(missing: list[str], info: dict) -> str:
    out = [f"Could not determine dimension(s) {', '.join(missing)} of the manifestation descriptor."]
    if info.get("target_found") is False:
        out.append("Joern could not find the target function definition.")
    else:
        if info.get("n_sinks", 0) == 0:
            out.append("No sink of the weakness family (vulnerable operation / relevant API argument) was "
                       "identified in the target function.")
        elif info.get("taint_edges") is None:
            out.append("No intra-procedural data flow from a source (parameter, external input call, global "
                       "variable, locally read value; the freed pointer for Use After Free) to a sink was "
                       "recovered.")
        out.append(f"Sinks found: {[s.get('code') for s in info.get('sink_candidates', [])]}")
        out.append(f"Sources found: {[s.get('code') for s in info.get('source_candidates', [])]}")
    return "\n".join(out)
