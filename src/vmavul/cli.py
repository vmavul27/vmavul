"""Command-line interface.

  vmavul space                         print the admissible-cell count
  vmavul synthesize --config CFG       run the synthesis agent (resumable)
  vmavul export --run DIR --out F      export accepted samples as detector-training JSONL
  vmavul check-tools --config CFG      verify the external toolchain versions
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def cmd_space(_args) -> int:
    from .space import ManifestationSpace
    print(json.dumps(ManifestationSpace().summary()))
    return 0


def cmd_synthesize(args) -> int:
    from .agent import Agent
    from .checks import RealToolchain, load_sinks
    from .config import load_run_config
    from .llm import ModelPool
    from .reference import ReferencePool

    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.reference_pool:
        overrides["reference_pool"] = args.reference_pool
    if args.budget:
        overrides["budget"] = args.budget
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_run_config(args.config, overrides)
    pool = ModelPool.from_yaml(cfg.models, random.Random(cfg.seed + 1))
    refs = ReferencePool.load(cfg.reference_pool)
    toolchain = RealToolchain(cfg.tools, load_sinks(cfg.sinks))
    try:
        summary = Agent(cfg, pool, toolchain, refs).run()
    finally:
        toolchain.joern.stop()
    print(json.dumps(summary, indent=1))
    return 0


def cmd_export(args) -> int:
    from .archive import Archive
    arch = Archive(Path(args.run) / "archive.jsonl")
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in arch.all_samples():
            c = rec["candidate"]
            row = {"id": rec["id"], "code": c["vulnerable"], "label": 1, "patched": c.get("patched", ""),
                   "context": c.get("context", ""), "target_name": c.get("target_name", ""),
                   "family": rec.get("family"), "descriptor": rec.get("descriptor", {}),
                   "cell": rec.get("cell"), "source": rec.get("variant", "vmavul")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n} samples to {args.out}")
    return 0


def cmd_check_tools(args) -> int:
    import subprocess
    from .config import load_run_config, resolve_tools
    t = resolve_tools(load_run_config(args.config).tools)
    rows = []
    jars = sorted(Path(t.joern_home).glob("lib/io.joern.joern-cli-*.jar")) if t.joern_home else []
    if jars:      # the launcher starts a REPL; read the version from the CLI jar instead
        rows.append(("joern", jars[-1].name[len("io.joern.joern-cli-"):-len(".jar")]))
    else:
        rows.append(("joern", "NOT AVAILABLE (set VMAVUL_JOERN_HOME to the joern-cli directory)"))
    for name, cmd in (("clang", [t.clang, "--version"]),
                      ("codeql", [t.codeql, "version", "--format=terse"]),
                      ("java", ["java", "-version"])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            rows.append((name, (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "?"))
        except (OSError, subprocess.TimeoutExpired) as exc:
            rows.append((name, f"NOT AVAILABLE ({exc})"))
    import lizard
    rows.append(("lizard", getattr(lizard, "version", "?")))
    rows.append(("codeql/cpp-queries", t.codeql_cpp_queries or "NOT SET"))
    for name, val in rows:
        print(f"{name:20s} {val}")
    import json
    import shutil
    rows_ok = True
    manifest = json.loads((Path(__file__).resolve().parents[2] / "configs" / "toolchain.json").read_text()) \
        if (Path(__file__).resolve().parents[2] / "configs" / "toolchain.json").exists() else None
    print(f"{'bubblewrap':20s} {shutil.which('bwrap') or 'NOT AVAILABLE (runtime checks will be skipped)'}")
    if manifest:
        ext = manifest["external"]
        for name, val in rows:
            key = {"codeql/cpp-queries": "codeql/cpp-queries"}.get(name, name)
            want = ext.get(key, {}).get("version")
            if want and want not in str(val):
                rows_ok = False
                print(f"  ! {name}: expected version {want}")
    print("expected: Clang 17.0.6, CodeQL 2.25.4, cpp-queries 1.6.2, Joern 4.0.547, "
          "Lizard 1.23.0, Java 17")
    return 0 if rows_ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vmavul", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("space").set_defaults(fn=cmd_space)
    s = sub.add_parser("synthesize")
    s.add_argument("--config", required=True)
    s.add_argument("--output-dir")
    s.add_argument("--reference-pool")
    s.add_argument("--budget", type=int)
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=cmd_synthesize)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)
    c = sub.add_parser("check-tools")
    c.add_argument("--config", required=True)
    c.set_defaults(fn=cmd_check_tools)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
