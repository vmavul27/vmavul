"""UniXcoder embeddings for the within-cell diversity check (Step 3(4))."""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np


class Embedder:
    def __init__(self, model: str = "microsoft/unixcoder-base", device: str = "cpu",
                 max_tokens: int = 512) -> None:
        self.model_name = model
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def embed(self, codes: Sequence[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised last hidden states (768-d)."""
        import torch
        with self._lock:
            self._load()
            out = []
            with torch.no_grad():
                for i in range(0, len(codes), 32):
                    enc = self._tok(list(codes[i:i + 32]), padding=True, truncation=True,
                                    max_length=self.max_tokens, return_tensors="pt").to(self.device)
                    hidden = self._model(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                    vec = torch.nn.functional.normalize(vec, dim=-1)
                    out.append(vec.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), dtype=np.float32)


def min_cosine_distance(vec: np.ndarray, others: Optional[np.ndarray]) -> Optional[float]:
    """Minimum cosine distance (1 - cos) between ``vec`` and the rows of ``others``."""
    if others is None or len(others) == 0:
        return None
    sims = others @ vec / (np.linalg.norm(others, axis=1) * np.linalg.norm(vec) + 1e-12)
    return float(1.0 - sims.max())
