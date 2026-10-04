"""Prompt construction for generation (Step 2) and refinement (Step 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import jinja2

from .descriptors_nl import requirements

REFINE_INSTRUCTIONS = {
    # (1) syntax failure: repair from the Clang diagnostics, keep the pair intact
    "syntax": "Repair the code so that both versions and the calling context compile with Clang, based on the "
              "diagnostics above, while preserving the intended vulnerability-patch pair. The "
              "/* @TARGET_FUNCTION@ */ marker must be at file scope immediately before main, never inside a "
              "function.",
    # (2) weakness-specific failure: revise the vulnerable operation or the patch
    "weakness": "Using the CodeQL / sanitizer diagnostics above, revise the vulnerable operation or the patch so "
                "that the target weakness holds in the vulnerable version (inside the target function, exercised "
                "by the driver) and is eliminated in the patched version, without introducing unrelated errors. "
                "The driver and helpers must remain sanitizer-clean; do not overflow a driver buffer merely to "
                "prepare the target's input. A no-argument execution of main must always reach the target with an "
                "in-source concrete input that triggers the vulnerable path.",
    # (3) incomplete characterization: make the sink and its sources identifiable
    "characterization": "Using the analysis results above, make the program elements required for "
                        "characterization identifiable: the vulnerable operation (sink) must be an explicit "
                        "operation or API call in the target function, and the vulnerability-relevant value must "
                        "flow to it intra-procedurally from a source (parameter, external input call, global "
                        "variable or locally read value; for Use After Free, the freed pointer).",
    # (4) within-cell similarity failure: vary the implementation, keep the vulnerability
    "diversity": "The candidate is too similar to samples already archived for this manifestation. Vary its "
                 "implementation (scenario, data structures, naming, statement structure) while preserving the "
                 "intended vulnerability and the target manifestation.",
}

VARY_INSTRUCTIONS = {
    "densify": "Vary the scenario and implementation while keeping the same manifestation descriptor.",
    "toward_neighbor": "Change the sample so that it satisfies the target manifestation descriptor above "
                       "(which differs from the sample's current manifestation in one dimension).",
    "diversify": "Diversify the implementation within the same manifestation descriptor.",
}


class PromptBuilder:
    def __init__(self, prompts_dir: str, max_reference_chars: int = 6000) -> None:
        self.dir = Path(prompts_dir)
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(self.dir)),
                                      undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
        self.max_ref = max_reference_chars
        self.system = (self.dir / "system.txt").read_text()

    def _refs(self, refs: list[dict]) -> list[dict]:
        out = []
        for r in refs:
            code = r["code"]
            if len(code) > self.max_ref:
                code = code[: self.max_ref] + "\n/* ... truncated ... */"
            out.append({"code": code, "cwe": r.get("cwe", "")})
        return out

    def generation(self, descriptor: dict, dims, references: list[dict], memory: list,
                   vary: Optional[str] = None, vary_mode: str = "densify") -> tuple[str, str]:
        user = self.env.get_template("generate.j2").render(
            requirements=requirements(descriptor, dims), references=self._refs(references),
            memory=memory, vary=vary, vary_instruction=VARY_INSTRUCTIONS.get(vary_mode, ""))
        return self.system, user

    def refinement(self, failed_check: str, descriptor: dict, dims, artifacts: str,
                   diagnostics: str) -> tuple[str, str]:
        user = self.env.get_template("refine.j2").render(
            failed_check=failed_check, requirements=requirements(descriptor, dims), artifacts=artifacts,
            diagnostics=diagnostics[-6000:], instruction=REFINE_INSTRUCTIONS[failed_check])
        return self.system, user
