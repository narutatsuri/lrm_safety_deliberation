"""
Shared helpers for the refactored STAIR training pipeline.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

STAIR_SYSTEM_PROMPT = (
    "You are a helpful assistant capable of multi-step reasoning. "
    "If you are provided with a task that can benefit from in-depth thinking, "
    "e.g., math, coding, and logical reasoning, you should solve the problem "
    "step by step and eventually give your answer.\n"
    "Use <|Reasoning_step|> and <|/Reasoning_step|> to mark the start and end "
    "of one step of reasoning, and wrap your final answer with <|Output|> and "
    "<|/Output|> after sufficient reasoning steps."
)

STAIR_SPECIAL_TOKENS = [
    "<|Reasoning_step|>",
    "<|/Reasoning_step|>",
    "<|Output|>",
    "<|/Output|>",
]

LORA_TARGET_MODULES = {
    "Llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "Qwen": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "Olmo": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "Phi4": [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
    "GPTOSS": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
}

THINKING_MODELS = {"Qwen", "Olmo"}


def add_stair_special_tokens(tokenizer) -> list[str]:
    tokens_to_add = [t for t in STAIR_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    return tokens_to_add


def patch_qwen_chat_template(tokenizer) -> bool:
    if not tokenizer.chat_template:
        return False
    remove_text = (
        "{% if '</think>' in content %}"
        "{% set content = content.split('</think>')[-1] %}"
        "{% endif %}"
    )
    patched = tokenizer.chat_template.replace(remove_text.strip(), "")
    if patched != tokenizer.chat_template:
        tokenizer.chat_template = patched
        return True
    return False


def build_stair_response(answer: str, base_model: str) -> str:
    answer = answer.strip()
    if base_model in THINKING_MODELS:
        return f"<think>\n\n</think>\n\n{answer}"
    return answer


def build_stair_messages(prompt: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": STAIR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def render_stair_example(tokenizer, prompt: str, answer: str, base_model: str) -> tuple[str, str]:
    del base_model  # kept for call-site symmetry
    full = tokenizer.apply_chat_template(
        build_stair_messages(prompt, answer),
        tokenize=False,
        add_generation_prompt=False,
    )
    query = tokenizer.apply_chat_template(
        build_stair_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    return full, query


def load_json_records(path: str | Path, max_samples: int = 0, seed: int = 42) -> list[dict[str, Any]]:
    with open(path) as f:
        records = json.load(f)
    if max_samples > 0 and len(records) > max_samples:
        rng = random.Random(seed)
        if records and all(isinstance(record, dict) and "type" in record for record in records):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                grouped.setdefault(str(record["type"]), []).append(record)
            for group in grouped.values():
                rng.shuffle(group)
            ordered_keys = sorted(grouped)
            sampled: list[dict[str, Any]] = []
            while len(sampled) < max_samples and any(grouped.values()):
                for key in ordered_keys:
                    if grouped[key]:
                        sampled.append(grouped[key].pop())
                        if len(sampled) == max_samples:
                            break
            records = sampled
        else:
            indices = list(range(len(records)))
            rng.shuffle(indices)
            records = [records[i] for i in indices[:max_samples]]
    return records


def materialize_json_subset(
    src_path: str | Path,
    dst_path: str | Path,
    max_samples: int,
    seed: int = 42,
) -> Path:
    records = load_json_records(src_path, max_samples=max_samples, seed=seed)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return dst_path


def stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def latest_state_checkpoint(root: str | Path) -> str:
    root = Path(root)
    best = ""
    best_step = -1
    for path in sorted(root.rglob("checkpoint-*")):
        if not (path / "training_state.pt").exists():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if step > best_step:
            best = str(path)
            best_step = step
    return best


def latest_trainer_checkpoint(root: str | Path) -> str:
    root = Path(root)
    best = ""
    best_step = -1
    for path in sorted(root.rglob("checkpoint-*")):
        if not (path / "trainer_state.json").exists():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if step > best_step:
            best = str(path)
            best_step = step
    return best


def find_merged_dir(root: str | Path) -> str:
    root = Path(root)
    if not root.exists():
        return ""
    merged = sorted(root.rglob("merged-final"))
    return str(merged[-1]) if merged else ""


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    out: list[list[Any]] = []
    bucket: list[Any] = []
    for item in items:
        bucket.append(item)
        if len(bucket) == size:
            out.append(bucket)
            bucket = []
    if bucket:
        out.append(bucket)
    return out
