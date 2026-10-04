"""LLM clients and the generation/refinement model pool (Section 3.4).

* ``OpenAIChatClient`` speaks the OpenAI ``/chat/completions`` protocol (OpenAI API,
  vLLM, SGLang, TGI, ... all expose it).
* ``AnthropicClient`` speaks the Anthropic ``/v1/messages`` protocol.
* ``ModelPool`` picks the generator uniformly at random from the pool of the current
  stage and the refiner uniformly at random among models of a *different family*.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .config import expand_env, load_yaml


class LLMError(RuntimeError):
    pass


@dataclass
class ModelSpec:
    name: str
    family: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str = ""
    token_param: str = "max_tokens"     # some APIs expect "max_completion_tokens"
    send_temperature: bool = True       # some reasoning APIs only accept their default
    extra: dict = field(default_factory=dict)


class ChatClient(Protocol):
    spec: ModelSpec

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 timeout: float) -> str: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:          # unterminated opening tag stripped by the server
        text = text.split("</think>", 1)[1]
    return text.strip()


def _post_json(url: str, payload: dict, headers: dict, timeout: float, retries: int = 3) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                break
            time.sleep(2 ** attempt)
    raise LLMError(f"request to {url} failed: {last}")


class OpenAIChatClient:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 timeout: float) -> str:
        if not self.spec.model_id:
            raise LLMError(f"model_id of {self.spec.name} is not configured")
        key = os.environ.get(self.spec.api_key_env, "") if self.spec.api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        payload = {
            "model": self.spec.model_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.spec.token_param: max_tokens,
            **({"temperature": temperature} if self.spec.send_temperature else {}),
            **self.spec.extra,
        }
        out = _post_json(self.spec.base_url.rstrip("/") + "/chat/completions", payload, headers, timeout)
        try:
            return strip_reasoning(out["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed response from {self.spec.name}: {out}") from exc


class AnthropicClient:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 timeout: float) -> str:
        if not self.spec.model_id:
            raise LLMError(f"model_id of {self.spec.name} is not configured")
        key = os.environ.get(self.spec.api_key_env, "")
        if not key:
            raise LLMError(f"environment variable {self.spec.api_key_env} is not set")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        payload = {
            "model": self.spec.model_id,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            **({"temperature": temperature} if self.spec.send_temperature else {}),
            **self.spec.extra,
        }
        out = _post_json(self.spec.base_url.rstrip("/") + "/v1/messages", payload, headers, timeout)
        try:
            return strip_reasoning("".join(b.get("text", "") for b in out["content"] if b.get("type") == "text"))
        except (KeyError, TypeError) as exc:
            raise LLMError(f"malformed response from {self.spec.name}: {out}") from exc


def make_client(spec: ModelSpec) -> ChatClient:
    if spec.provider == "openai":
        return OpenAIChatClient(spec)
    if spec.provider == "anthropic":
        return AnthropicClient(spec)
    raise ValueError(f"unknown provider {spec.provider!r} for model {spec.name}")


class ModelPool:
    """Stage-specific generator pools with cross-family refinement."""

    def __init__(self, pools: dict[str, list[ChatClient]], stage_pools: dict[str, str],
                 rng: Optional[random.Random] = None) -> None:
        self.pools = pools
        self.stage_pools = stage_pools
        self.rng = rng or random.Random(0)
        self._lock = threading.Lock()
        for stage, pool in stage_pools.items():
            if pool not in pools or not pools[pool]:
                raise ValueError(f"stage {stage!r} refers to empty/unknown pool {pool!r}")
            families = {c.spec.family for c in pools[pool]}
            if len(families) < 2:
                raise ValueError(f"pool {pool!r} needs >=2 model families for cross-family refinement")

    @classmethod
    def from_yaml(cls, path: str, rng: Optional[random.Random] = None,
                  client_factory=make_client) -> "ModelPool":
        data = expand_env(load_yaml(path))
        pools = {name: [client_factory(ModelSpec(**m)) for m in models]
                 for name, models in data["pools"].items()}
        return cls(pools, data["stage_pools"], rng)

    def _pool(self, stage: str) -> list[ChatClient]:
        return self.pools[self.stage_pools.get(stage, self.stage_pools.get("unscheduled", ""))]

    def select_generator(self, stage: str) -> ChatClient:
        with self._lock:
            return self.rng.choice(self._pool(stage))

    def select_refiner(self, stage: str, generator: ChatClient) -> ChatClient:
        candidates = [c for c in self._pool(stage) if c.spec.family != generator.spec.family]
        if not candidates:
            raise LLMError(f"no model of a different family than {generator.spec.family}")
        with self._lock:
            return self.rng.choice(candidates)
