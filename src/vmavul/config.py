"""Configuration objects (loaded from YAML).  All paths/endpoints come from the
config file or environment variables -- nothing machine-specific is hard-coded."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in strings (recursively)."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


@dataclass
class HyperParams:
    """Section 3.4 hyper-parameters."""

    k_exhausted_attempts: int = 10      # k: attempts before an uncovered cell is exhausted
    tau_density: int = 50               # tau: density threshold
    delta_min_distance: float = 0.2     # delta: within-cell minimum cosine distance
    r_refinement_rounds: int = 3        # r: refinement limit
    bootstrap_cells_per_family: int = 3     # "until accepted samples cover three distinct cells"
    bootstrap_patience: int = 3             # "... or no new cell in three consecutive iterations"


@dataclass
class GenerationConfig:
    n_references: int = 2
    n_memory_entries: int = 3
    max_reference_chars: int = 6000     # truncate very long reference functions
    temperature: float = 0.8
    max_tokens: int = 4096
    request_timeout_s: float = 600.0


@dataclass
class ToolConfig:
    clang: str = "${VMAVUL_CLANG:-clang-17}"
    codeql: str = "${VMAVUL_CODEQL:-codeql}"
    codeql_cpp_queries: str = "${VMAVUL_CODEQL_CPP_QUERIES:-}"   # path of codeql/cpp-queries 1.6.2
    joern_home: str = "${VMAVUL_JOERN_HOME:-}"                   # contains the `joern` launcher
    joern_servers: int = 2
    joern_base_port: int = 0            # 0 = pick free ports automatically
    joern_timeout_s: float = 180.0
    codeql_timeout_s: float = 300.0
    codeql_threads: int = 1
    compile_timeout_s: float = 60.0
    run_timeout_s: float = 10.0
    run_memory_mb: int = 2048
    sandbox: str = "bwrap"              # "bwrap" (fail-closed, default) | "none" | "custom"
    sandbox_command: list[str] = field(default_factory=list)  # prefix used when sandbox == "custom"
    embed_model: str = "${VMAVUL_EMBED_MODEL:-microsoft/unixcoder-base}"
    embed_device: str = "${VMAVUL_EMBED_DEVICE:-cpu}"
    work_dir: str = "${VMAVUL_TMPDIR:-}"   # scratch for builds; default = system temp dir


@dataclass
class RunConfig:
    output_dir: str = "runs/vmavul"
    reference_pool: str = ""            # JSONL produced by scripts/data/build_reference_pool.py
    budget: int = 15000                 # accepted samples to synthesize
    max_iterations: int = 0             # safety cap (0 = 20 x budget)
    workers: int = 1                    # parallel iterations (1 = exact sequential semantics)
    seed: int = 0
    models: str = "configs/models.yaml"
    sinks: str = "configs/sinks.yaml"
    prompts_dir: str = "prompts"
    hyper: HyperParams = field(default_factory=HyperParams)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)


def _build(cls, data: dict[str, Any]):
    if data is None:
        return cls()
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {cls.__name__}.{key}")
        ftype = known[key].type
        default = known[key].default_factory() if callable(known[key].default_factory) else None  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
        del ftype
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("extends", None)
    if base:
        parent = load_yaml(Path(path).parent / base)
        data = _deep_merge(parent, data)
    return data


def load_run_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> RunConfig:
    data = load_yaml(path)
    if overrides:
        data = _deep_merge(data, overrides)
    return _build(RunConfig, expand_env(data))


def resolve_tools(tools: ToolConfig) -> ToolConfig:
    """Return a copy of ``tools`` with environment variables expanded."""
    values = {f.name: expand_env(getattr(tools, f.name)) for f in fields(tools)}
    return ToolConfig(**values)
