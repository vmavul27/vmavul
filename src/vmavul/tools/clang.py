"""Clang 17 compilation (syntax check, Step 3(1)) and sanitizer builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .process import ProcResult, run

SANITIZER_FLAGS = {
    "asan": ["-fsanitize=address", "-fno-omit-frame-pointer"],
    "lsan": ["-fsanitize=address", "-fno-omit-frame-pointer"],   # LSan via ASan's leak checker
    "ubsan": ["-fsanitize=signed-integer-overflow,unsigned-integer-overflow,shift,"
              "integer-divide-by-zero,float-divide-by-zero,null,implicit-integer-truncation",
              "-fno-sanitize-recover=all"],
}


def source_name(language: str) -> str:
    return "unit.cpp" if language == "c++" else "unit.c"


def base_flags(language: str) -> list[str]:
    if language == "c++":
        return ["-x", "c++", "-std=gnu++17"]
    return ["-x", "c", "-std=gnu11"]


@dataclass
class CompileResult:
    ok: bool
    diagnostics: str
    binary: str = ""


def compile_unit(clang: str, source: Path, out: Path, language: str, *, extra: Sequence[str] = (),
                 timeout: float = 60.0, link: bool = True) -> CompileResult:
    cmd = [clang] + base_flags(language) + ["-g", "-O0", "-w", "-fno-color-diagnostics"] + list(extra)
    cmd += [str(source)]
    if link:
        cmd += ["-o", str(out), "-lm"] + (["-lstdc++"] if language == "c++" else [])
    else:
        cmd += ["-c", "-o", str(out)]
    res: ProcResult = run(cmd, cwd=str(source.parent), timeout=timeout)
    ok = res.returncode == 0 and not res.timed_out
    diag = res.stderr.strip() if not res.timed_out else "clang timed out"
    return CompileResult(ok, diag, str(out) if ok else "")
