"""Fail-closed execution sandbox (bubblewrap) for the runtime checks.

Generated programs are compiled and executed to observe whether the vulnerability
manifests.  Running model-authored code directly on the host is a destructive-command
risk, so every execution is wrapped in a ``bwrap`` jail:

* read-only host root (``--ro-bind / /``): compiler runtime and ``llvm-symbolizer``
  stay reachable, but writes to the real filesystem fail;
* the common user/data trees are replaced by empty tmpfs mounts;
* only the candidate's own temporary work directory is writable;
* no network (``--unshare-all``), fresh PID/IPC/UTS/user/cgroup namespaces,
  ``--die-with-parent`` and ``--new-session``.

Fail-closed: if ``bwrap`` is unavailable the runtime check is NOT executed (the caller
records it as not applicable) unless the user explicitly opts out with
``tools.sandbox: none``.  A static pre-filter additionally rejects candidates that embed
unambiguous destructive shell commands before anything is compiled.

"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path


class SandboxUnavailableError(RuntimeError):
    """Raised when the sandbox runtime (bwrap) is missing -- never run unsandboxed."""


_MASK_TMPFS = ("/tmp", "/run", "/var/tmp", "/data", "/home", "/root", "/mnt", "/srv")


@lru_cache(maxsize=1)
def _locate_bwrap() -> str | None:
    return shutil.which("bwrap")


def bwrap_available() -> bool:
    return _locate_bwrap() is not None


def prefix(workdir: str | Path, *, readonly: Sequence[str | Path] = ()) -> list[str]:
    """Return the ``bwrap ... --`` prefix that confines a command to ``workdir``.

    ``readonly`` re-exposes extra paths read-only (e.g. a compiler located under a
    masked tree)."""
    bwrap = _locate_bwrap()
    if bwrap is None:
        raise SandboxUnavailableError("bwrap not found; refusing to execute generated code")
    wd = str(Path(workdir).resolve())
    args = [bwrap, "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
    for mount in _MASK_TMPFS:              # masked first; the work dir is re-bound below
        if Path(mount).exists():
            args += ["--tmpfs", mount]
    for ro in readonly:
        rp = str(Path(ro).resolve())
        args += ["--ro-bind", rp, rp]
    args += ["--bind", wd, wd, "--chdir", wd, "--unshare-all", "--die-with-parent", "--new-session", "--"]
    return args


_DESTRUCTIVE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brm\s+-\S*[rf]\S*(?:\s+-\S+)*\s+(?:/(?=[\s'\";)|&]|$|\*|etc|usr|bin|boot|lib|var|"
                r"data|home|root|dev|sys|proc|opt|mnt)|~|\$\{?HOME|\*)", re.I), "rm_rf_system_path"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b", re.I), "mkfs"),
    (re.compile(r"\bwipefs\b|\bmkswap\b", re.I), "wipefs_mkswap"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/", re.I), "dd_to_device"),
    (re.compile(r">\s*/dev/(?:sd|nvme|vd|hd|disk|mapper)", re.I), "redirect_block_device"),
    (re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:?\s*&\s*\}\s*;\s*:", re.I), "fork_bomb"),
    (re.compile(r"(?:curl|wget|fetch)\b[^\n|]*\|\s*(?:sh|bash|zsh|python\d?)\b", re.I), "download_pipe_shell"),
    (re.compile(r"\.ssh/authorized_keys", re.I), "authorized_keys_write"),
    (re.compile(r"\bcrontab\b\s+-", re.I), "crontab_mutation"),
    (re.compile(r"\b(?:shutdown|reboot|poweroff)\b|\bhalt\s+-|\binit\s+0\b", re.I), "system_power_control"),
)


def scan_destructive(*texts: str) -> list[str]:
    """Labels of destructive shell patterns found in ``texts`` (empty = clean)."""
    blob = "\n".join(t for t in texts if t)
    return sorted({label for pattern, label in _DESTRUCTIVE_PATTERNS if pattern.search(blob)})
