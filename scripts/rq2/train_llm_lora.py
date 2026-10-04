#!/usr/bin/env python3
"""Fine-tune an open-weight LLM (Qwen3.5-4B or Llama3.1-8B) with LoRA for function-level
binary vulnerability classification and evaluate on the four official test sets.

The model answers "Yes"/"No"; the training loss is on the answer token only and the
prediction is P(Yes) > P(No) at the answer position.  Model selection uses ONLY the
validation split of the training dataset (best validation F1 over epochs); the test sets
are evaluated once with the selected adapter.  Hyper-parameters are fixed across all
augmentation settings.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, load_split, load_training, prf, write_result  # noqa: E402

BACKBONES = {
    "qwen3.5-4b": {"model_id": "Qwen/Qwen3.5-4B", "epochs": 3,
                   "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                               "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]},
    "llama3.1-8b": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "epochs": 2,
                    "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
}
SYSTEM = "You are a software vulnerability classifier. Reply with exactly Yes or No."
USER = "Is the following C or C++ function vulnerable?\n<FUNCTION>\n{code}\n</FUNCTION>\nReply with exactly Yes or No."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--train-set", required=True, choices=DATASETS)
    ap.add_argument("--config", required=True)
    ap.add_argument("--aug", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    ap.add_argument("--out", type=Path, default=Path("results/rq2"))
    ap.add_argument("--work", type=Path, default=Path("work/llm"))
    ap.add_argument("--epochs", type=int, default=0, help="0 = backbone default (Qwen 3, Llama 2)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-train-samples", type=int, default=0, help="debug only")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

    bb = BACKBONES[a.backbone]
    epochs = a.epochs or bb["epochs"]
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(bb["model_id"])
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    model = AutoModelForCausalLM.from_pretrained(bb["model_id"], dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             bias="none", task_type=TaskType.CAUSAL_LM,
                                             target_modules=[t for t in bb["targets"]]))

    prefix_suffix = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": USER.format(code="\x00")}],
                                            tokenize=False, add_generation_prompt=True).split("\x00")
    pre_ids = tok.encode(prefix_suffix[0], add_special_tokens=False)
    post_ids = tok.encode(prefix_suffix[1], add_special_tokens=False)
    budget = a.max_len - len(pre_ids) - len(post_ids) - 1

    def encode(code: str) -> list[int]:
        ids = tok.encode(code, add_special_tokens=False)
        if len(ids) > budget:                       # keep head 75% / tail 25%
            head = math.ceil(budget * 0.75)
            ids = ids[:head] + ids[len(ids) - (budget - head):]
        return pre_ids + ids + post_ids

    @torch.no_grad()
    def predict(rows: list[dict]) -> list[int]:
        model.eval()
        out = []
        for r in rows:
            ids = torch.tensor([encode(r["code"])], device="cuda")
            logits = model(input_ids=ids).logits[0, -1]
            out.append(int(logits[yes_id] > logits[no_id]))
        model.train()
        return out

    train = load_training(a.datasets, a.train_set, a.aug, a.seed)
    if a.max_train_samples:
        train = train[: a.max_train_samples]
    valid = load_split(a.datasets, a.train_set, "valid")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(train) / a.grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * steps), steps)
    run = f"{a.backbone}__{a.train_set}__{a.config}__seed{a.seed}"
    best_f1, best_epoch, best_dir = -1.0, 0, a.work / run / "best_adapter"
    t0 = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        random.Random(a.seed * 1000 + epoch).shuffle(order)
        for step, i in enumerate(order, 1):
            ids = encode(train[i]["code"]) + [yes_id if train[i]["label"] else no_id]
            x = torch.tensor([ids], device="cuda")
            labels = torch.full_like(x, -100)
            labels[0, -1] = x[0, -1]
            loss = model(input_ids=x, labels=labels).loss / a.grad_accum
            loss.backward()
            if step % a.grad_accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        f1 = prf([r["label"] for r in valid], predict(valid))["f1"]
        print(f"[{run}] epoch {epoch} valid_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            model.save_pretrained(str(best_dir))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(best_dir / "adapter_model.safetensors")))
    tests = {}
    for name in DATASETS:
        rows = load_split(a.datasets, name, "test")
        tests[name] = prf([r["label"] for r in rows], predict(rows))
    write_result(a.out, {"detector": a.backbone, "train_set": a.train_set, "config": a.config, "seed": a.seed,
                         "aug": a.aug, "n_train": len(train), "best_valid_f1": best_f1, "best_epoch": best_epoch,
                         "selection": "validation_f1", "test": tests, "hours": (time.time() - t0) / 3600,
                         "hparams": {k: str(v) for k, v in vars(a).items()}})


if __name__ == "__main__":
    main()
