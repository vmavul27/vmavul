"""Runtime checking (Step 3(2)): run the driver under ASan / UBSan / LSan.

A finding counts for the target function only if the report locates the error inside
the target function: the top in-program stack frame (ASan), the allocation stack
(LSan) or the reported source line (UBSan) must fall into the target function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .clang import SANITIZER_FLAGS, compile_unit, source_name
from .process import run

_FRAME_RE = re.compile(r"#\d+\s+0x[0-9a-f]+\s+in\s+([^\s(]+)(?:\(.*?\))?\s+(\S+?):(\d+)", re.I)
_UBSAN_RE = re.compile(r"(\S+?):(\d+):\d+: runtime error: (.*)")


@dataclass
class SanitizerReport:
    built: bool
    ran: bool
    triggered: bool = False          # any sanitizer error / abnormal termination
    error_text: str = ""             # first sanitizer error line
    in_target: bool = False          # error located inside the target function
    timed_out: bool = False
    exit_code: int = 0
    raw: str = ""
    build_error: str = ""


def _in_range(line: int, rng: tuple[int, int]) -> bool:
    return rng[0] <= line <= rng[1]


def parse_report(text: str, unit_file: str, target_name: str, target_lines: tuple[int, int],
                 kind: str) -> tuple[str, bool]:
    """Return (error headline, located-in-target-function)."""
    headline = ""
    for line in text.splitlines():
        if "ERROR: AddressSanitizer" in line or "ERROR: LeakSanitizer" in line \
                or "runtime error:" in line:
            headline = line.strip()
            break
    if not headline:
        return "", False
    if "runtime error:" in headline:
        m = _UBSAN_RE.search(headline)
        if m and Path(m.group(1)).name == unit_file:
            return headline, _in_range(int(m.group(2)), target_lines)
        return headline, False
    if kind == "lsan" or "LeakSanitizer" in headline:
        # any allocation stack that passes through the target function
        frames = [(f, Path(fl).name, int(ln)) for f, fl, ln in _FRAME_RE.findall(text)]
        return headline, any(fl == unit_file and (f == target_name or _in_range(ln, target_lines))
                             for f, fl, ln in frames)
    # ASan: the first frame located in the unit file (skip interceptors / libc)
    after = text[text.find(headline):]
    for f, fl, ln in _FRAME_RE.findall(after):
        if Path(fl).name == unit_file:
            return headline, (f == target_name or _in_range(int(ln), target_lines))
    return headline, False


def _symbolizer_env(clang: str) -> dict:
    """Point the sanitizers at the llvm-symbolizer of the same LLVM installation, so that
    reports carry file:line frames (needed to locate the error in the target function)."""
    import shutil
    cands = []
    exe = shutil.which(clang) or clang
    d = Path(exe).resolve().parent
    m = re.search(r"-(\d+)$", Path(exe).name)
    cands += [d / "llvm-symbolizer", d / f"llvm-symbolizer-{m.group(1)}" if m else d / "llvm-symbolizer"]
    found = next((str(c) for c in cands if c.exists()), None) or shutil.which("llvm-symbolizer")
    return {"ASAN_SYMBOLIZER_PATH": found, "LLVM_SYMBOLIZER_PATH": found} if found else {}


def run_with_sanitizer(clang: str, kind: str, unit_code: str, language: str, target_name: str,
                       target_lines: tuple[int, int], work: Path, *, compile_timeout: float = 60,
                       run_timeout: float = 10, memory_mb: int = 0,
                       sandbox: Sequence[str] = (), sandbox_mode: str = "bwrap",
                       args: Optional[list[str]] = None) -> SanitizerReport:
    work.mkdir(parents=True, exist_ok=True)
    src = work / source_name(language)
    src.write_text(unit_code)
    exe = work / f"prog_{kind}"
    comp = compile_unit(clang, src, exe, language, extra=SANITIZER_FLAGS[kind], timeout=compile_timeout)
    if not comp.ok:
        return SanitizerReport(False, False, build_error=comp.diagnostics[-4000:])
    env = {"PATH": "/usr/bin:/bin", **_symbolizer_env(clang),
           "ASAN_OPTIONS": "detect_leaks=%d:abort_on_error=0:symbolize=1:allocator_may_return_null=1"
                           % (1 if kind == "lsan" else 0),
           "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"}
    if sandbox_mode == "bwrap":
        from .sandbox import SandboxUnavailableError, prefix
        try:
            sym = env.get("ASAN_SYMBOLIZER_PATH")
            sandbox = prefix(work, readonly=[sym] if sym else [])
        except SandboxUnavailableError as exc:
            return SanitizerReport(True, False, build_error=f"runtime check skipped: {exc}")
    elif sandbox_mode == "none":
        sandbox = ()
    # ASan's shadow memory needs a large virtual address space: no RLIMIT_AS for asan/lsan.
    res = run([str(exe)] + list(args or []), cwd=str(work), timeout=run_timeout, env=env,
              memory_mb=0 if kind in ("asan", "lsan") else memory_mb, sandbox=sandbox)
    text = (res.stderr or "") + "\n" + (res.stdout or "")
    headline, in_target = parse_report(text, src.name, target_name, target_lines, kind)
    abnormal = res.timed_out or res.returncode < 0 or bool(headline)
    return SanitizerReport(True, True, triggered=abnormal, error_text=headline or
                           ("timeout" if res.timed_out else (f"signal {-res.returncode}" if res.returncode < 0 else "")),
                           in_target=in_target, timed_out=res.timed_out, exit_code=res.returncode,
                           raw=text[-6000:])
