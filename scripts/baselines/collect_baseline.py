#!/usr/bin/env python3
"""Normalize the synthesized vulnerable samples of a compared approach into JSONL.

Inputs are the outputs of the approaches' publicly released implementations (run with
the settings recommended by their authors).  ALL
synthesized vulnerable samples are collected (e.g. for VulScribeR all three strategies:
mutation, injection and extension); the 15,000-sample augmentation sets are then drawn
uniformly at random by ``scripts/sample_augmentation.py``.

Output records: ``{id, code, label: 1, source}``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common import write_jsonl  # noqa: E402

csv.field_size_limit(10 ** 9)


def from_vulgen(paths):
    """VulGen/VGX CSV outputs: rows with target==1, code in processed_func (or func)."""
    for p in paths:
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("target", "1")).strip() == "1":
                    yield r.get("processed_func") or r.get("func") or ""


def from_c_files(paths):
    """Directories / zips of generated C files (VGX releases *_1.c = vulnerable)."""
    for p in paths:
        if p.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for n in sorted(z.namelist()):
                    if n.endswith("_1.c"):
                        yield z.read(n).decode("utf-8", "replace")
        else:
            for f in sorted(glob.glob(str(Path(p) / "**" / "*_1.c"), recursive=True)):
                yield Path(f).read_text(errors="replace")


def from_jsonl(paths, field):
    """JSON / JSONL outputs (VulScribeR ``generated``, GVI ``code``); zips are scanned."""
    def _lines(fh):
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    for p in paths:
        if p.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                for n in sorted(z.namelist()):
                    if n.endswith(".jsonl"):
                        yield from (r.get(field, "") for r in _lines(io.TextIOWrapper(z.open(n), encoding="utf-8", errors="replace")))
                    elif n.endswith(".zip"):
                        inner = io.BytesIO(z.read(n))
                        with zipfile.ZipFile(inner) as z2:
                            for m in sorted(z2.namelist()):
                                if m.endswith(".jsonl"):
                                    yield from (r.get(field, "") for r in _lines(io.TextIOWrapper(z2.open(m), encoding="utf-8", errors="replace")))
        elif p.endswith(".json"):
            data = json.loads(Path(p).read_text())
            for r in (data if isinstance(data, list) else data.get("data", [])):
                yield r.get(field, "")
        else:
            with open(p, encoding="utf-8", errors="replace") as fh:
                yield from (r.get(field, "") for r in _lines(fh))


READERS = {
    "vulgen": lambda ps: from_vulgen(ps),
    "vgx": lambda ps: from_c_files(ps) if any(p.endswith(".zip") or Path(p).is_dir() for p in ps) else from_vulgen(ps),
    "vulscriber": lambda ps: from_jsonl(ps, "generated"),
    "gvi": lambda ps: from_jsonl(ps, "code"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", required=True, choices=sorted(READERS))
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seen, rows = set(), []
    for code in READERS[a.approach](a.inputs):
        code = (code or "").strip()
        if code and code not in seen:
            seen.add(code)
            rows.append({"id": f"{a.approach}::{len(rows)}", "code": code, "label": 1, "source": a.approach})
    print(f"{a.approach}: {write_jsonl(a.out, rows)} distinct non-empty vulnerable samples -> {a.out}")


if __name__ == "__main__":
    main()
