"""The VMA-guided vulnerability synthesis agent: the four-step loop of Section 3.3."""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from .archive import Archive
from .artifacts import Candidate, render_artifacts
from .checks import STAGES, CheckOutcome, QualityChecker
from .config import RunConfig
from .llm import LLMError, ModelPool
from .memory import MemoryEntry, RefinementMemory
from .prompts import PromptBuilder
from .reference import ReferencePool
from .scheduler import Scheduler, Task
from .space import ManifestationSpace


class Agent:
    def __init__(self, cfg: RunConfig, pool: ModelPool, toolchain, references: Optional[ReferencePool],
                 out_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.h = cfg.hyper
        self.out = Path(out_dir or cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(cfg.seed)
        self.pool = pool
        self.refs = references
        if references is None:
            raise ValueError("a (training-split) reference pool is required")
        self.space = ManifestationSpace()
        self.archive = Archive(self.out / "archive.jsonl")
        self.memory = RefinementMemory(self.out / "refinement_memory.jsonl", enabled=True)
        self.scheduler = Scheduler(self.space, self.archive, self.h, cfg.budget, self.rng,
                                   self.out / "scheduler_state.json")
        self.checker = QualityChecker(toolchain, self.space, self.h)
        self.prompts = PromptBuilder(cfg.prompts_dir, cfg.generation.max_reference_chars)
        self._log_lock = threading.Lock()
        (self.out / "run_config.json").write_text(json.dumps(_cfg_dict(cfg), indent=1))
        (self.out / "space.json").write_text(json.dumps(self.space.summary(), indent=1))

    # ------------------------------------------------------------------ helpers
    def _log(self, event: dict) -> None:
        with self._log_lock, (self.out / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _complete(self, client, system: str, user: str, calls: list) -> str:
        g = self.cfg.generation
        calls.append(client.spec.name)
        return client.complete(system, user, temperature=g.temperature, max_tokens=g.max_tokens,
                               timeout=g.request_timeout_s)

    # ------------------------------------------------------------------ one iteration
    def iteration(self, task: Task, rng: random.Random) -> dict:
        calls: list[str] = []
        event = {"time": time.time(), "stage": task.stage, "mode": task.mode,
                 "target": list(task.target) if task.target else None, "family": task.family}
        try:
            descriptor = self.space.as_dict(task.target) if task.target else {}
            refs = self.refs.sample(task.family, self.cfg.generation.n_references, rng)
            mem = self.memory.retrieve(task.family, task.target, self.cfg.generation.n_memory_entries)
            vary = render_artifacts(Candidate.from_dict(task.vary["candidate"])) if task.vary else None
            system, user = self.prompts.generation(descriptor, self.space.dims, refs, mem, vary, task.mode)
            event["references"] = [r.get("id") for r in refs]
            generator = self.pool.select_generator(task.stage)
            text = self._complete(generator, system, user, calls)
            cand, outcome = self._parse(text)
            history = []
            prev: Optional[CheckOutcome] = None
            for rnd in range(self.h.r_refinement_rounds + 1):
                if cand is not None:
                    cand.generator = generator.spec.name
                    outcome = self.checker.check(cand, task.family, self.archive)
                history.append(outcome.stage)
                if prev is not None and cand is not None and _progressed(prev.stage, outcome.stage):
                    self.memory.add(MemoryEntry(task.family, list(task.target or []), prev.stage,
                                                prev.diagnostics[:600], cand.fix_summary or ""))
                if outcome.accepted:
                    break
                if rnd == self.h.r_refinement_rounds:
                    break
                refiner = self.pool.select_refiner(task.stage, generator)
                artifacts = render_artifacts(cand) if cand is not None else text
                stage = outcome.stage if outcome.stage in STAGES else "syntax"
                system, user = self.prompts.refinement(stage, descriptor, self.space.dims, artifacts,
                                                       outcome.diagnostics)
                text = self._complete(refiner, system, user, calls)
                prev = outcome
                new_cand, parse_outcome = self._parse(text)
                if new_cand is not None:
                    new_cand.refiners = (cand.refiners if cand else []) + [refiner.spec.name]
                    cand = new_cand
                else:
                    cand, outcome = None, parse_outcome
            event.update(history=history, llm_calls=len(calls), models=calls, accepted=outcome.accepted,
                         failed_stage=None if outcome.accepted else outcome.stage)
            if outcome.accepted and cand is not None:
                with self.archive._lock:
                    new_cell = self.archive.count(outcome.cell) == 0
                    rec = {"id": uuid.uuid4().hex, "cell": list(outcome.cell), "descriptor": outcome.descriptor,
                           "family": outcome.family, "target": event["target"], "stage": task.stage,
                           "mode": task.mode, "rounds": len(history) - 1, "candidate": cand.to_dict(),
                           "evidence": _trim(outcome.evidence), "variant": "vmavul"}
                    self.archive.add(rec, outcome.embedding)
                event.update(cell=list(outcome.cell), new_cell=new_cell, sample_id=rec["id"])
        except LLMError as exc:
            event.update(accepted=False, failed_stage="llm_error", error=str(exc)[:500], llm_calls=len(calls))
        return event

    def _parse(self, text: str):
        try:
            return Candidate.parse(text), None
        except ValueError as exc:
            return None, CheckOutcome(False, "syntax", f"Malformed response: {exc}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cap = self.cfg.max_iterations or 20 * self.cfg.budget
        counter = {"n": self.scheduler.iterations}
        lock = threading.Lock()

        def worker(wid: int) -> None:
            while True:
                with lock:
                    if counter["n"] >= cap:
                        return
                    counter["n"] += 1
                    it = counter["n"]
                task = self.scheduler.next()
                if task is None:
                    return
                ev = self.iteration(task, random.Random(self.cfg.seed * 1_000_003 + it))
                ev["iteration"] = it
                self._log(ev)
                self.scheduler.observe(task, tuple(ev["cell"]) if ev.get("cell") else None,
                                       bool(ev.get("new_cell")))

        n = max(1, self.cfg.workers)
        if n == 1:
            worker(0)
        else:
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(worker, range(n)))
        summary = {"accepted": self.archive.size(), "covered_cells": len(self.archive.covered()),
                   "admissible_cells": len(self.space.admissible_cells), "iterations": self.scheduler.iterations,
                   "exhausted_cells": len(self.scheduler.exhausted), "stage": self.scheduler.stage}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary


def _progressed(prev_stage: str, stage: str) -> bool:
    order = list(STAGES) + ["accepted"]
    return prev_stage in order and stage in order and order.index(stage) > order.index(prev_stage)


def _trim(ev: dict) -> dict:
    s = json.dumps(ev, default=str)
    return json.loads(s) if len(s) < 20000 else {"truncated": True, "family": ev.get("family")}


def _cfg_dict(cfg: RunConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)
