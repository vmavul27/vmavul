"""Subprocess helpers with timeouts, resource limits and an optional sandbox prefix."""

from __future__ import annotations

import os
import resource
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run(cmd: Sequence[str], *, cwd: Optional[str] = None, timeout: float = 60.0,
        env: Optional[dict] = None, memory_mb: int = 0, sandbox: Sequence[str] = (),
        stdin: Optional[str] = None) -> ProcResult:
    def _limits() -> None:  # pragma: no cover - runs in the child
        os.setsid()
        if memory_mb:
            lim = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    full = list(sandbox) + list(cmd)
    try:
        p = subprocess.run(full, cwd=cwd, env=env, input=stdin, capture_output=True, text=True,
                           errors="replace", timeout=timeout, preexec_fn=_limits)
        return ProcResult(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ProcResult(-1, out, err, timed_out=True)
    except FileNotFoundError as exc:
        return ProcResult(127, "", f"command not found: {exc}")


def scratch_dir(base: str = "", prefix: str = "vmavul_") -> tempfile.TemporaryDirectory:
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=base or None)
