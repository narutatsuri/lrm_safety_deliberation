"""Shared helper for the M=4 inference-defense oscillation audit.

Provides:
  - SAMPLING_PRESETS: per-model recommended sampling settings (from each model's
    generation_config.json, except GPT-OSS which uses the harmony default
    T=1.0, top_p=1.0 with no top_k).
  - apply_sampling_preset(args, model_key): override args.temperature, .top_p,
    .top_k in place when --sampling-preset is set.
  - load_gid_filter(path): read a JSON file containing a list of int global_ids.
  - filter_dataset_by_gids(dataset, gid_set): retain only rows whose global_id
    is in gid_set; preserve original index as `__orig_idx__` so sample_id (which
    callers compute as args.start + i) still matches the source asr/orr row index.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# --- GPT-OSS harmony channel-aware (thinking, response) splitter ---
# Mirrors scripts/defenses_eval/reparse_gptoss_cells.py:split_gptoss (validated).
# The base split_thinking_and_response (</think>-based) does NOT understand the
# harmony channel format, so for GPT-OSS the final response gets glued onto the
# thinking trace. Use this whenever is_gpt_oss.
_CHANNEL_TOKEN_RE = re.compile(r"<\|channel\|>final\|?<?\|?message\|?>?(.*)", re.DOTALL)
_ASSISTANT_FINAL_RE = re.compile(r"assistantfinal(.*)", re.DOTALL)
_ASSISTANT_COMMENTARY_RE = re.compile(r"assistantcommentary[\w\s=]*?(?=[A-Z])(.*)", re.DOTALL)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def split_gptoss(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    m = _CHANNEL_TOKEN_RE.search(text)
    if m:
        thinking = text[: m.start()].strip()
        response = _HARMONY_TOKEN_RE.sub("", m.group(1)).strip()
        return thinking, response
    m = _ASSISTANT_FINAL_RE.search(text)
    if m:
        return text[: m.start()].rstrip(".").strip(), m.group(1).strip()
    m = _ASSISTANT_COMMENTARY_RE.search(text)
    if m:
        return text[: m.start()].rstrip(".").strip(), m.group(1).strip()
    return text.strip(), ""


SAMPLING_PRESETS: dict[str, dict] = {
    "Qwen3-8B":         {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "Olmo-3-7B-Think":  {"temperature": 0.6, "top_p": 0.95, "top_k": 50},
    "Phi-4-reasoning":  {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    "GPT-OSS-20B":      {"temperature": 1.0, "top_p": 1.0,  "top_k": 0},
}


def apply_sampling_preset(args, model_key: str | None) -> None:
    if not model_key:
        return
    if model_key not in SAMPLING_PRESETS:
        raise ValueError(
            f"unknown --sampling-preset {model_key!r}; "
            f"valid: {sorted(SAMPLING_PRESETS)}"
        )
    preset = SAMPLING_PRESETS[model_key]
    args.temperature = preset["temperature"]
    args.top_p = preset["top_p"]
    args.top_k = preset["top_k"]


def load_gid_filter(path: str | None) -> set[int] | None:
    if not path:
        return None
    p = Path(path)
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        gids = data.get("gids") or data.get("global_ids") or []
    else:
        gids = data
    return {int(g) for g in gids}


def filter_dataset_by_gids(
    dataset: list[dict], gid_set: set[int] | None
) -> list[dict]:
    """Filter rows whose `global_id` is in gid_set.

    Tags each kept row with `__orig_idx__` (original asr/orr row index) so
    callers can use it as the canonical sample_id rather than the post-filter
    enumerate index.
    """
    if gid_set is None:
        return [{**row, "__orig_idx__": i} for i, row in enumerate(dataset)]
    out = []
    for i, row in enumerate(dataset):
        gid = row.get("global_id")
        if gid is None:
            continue
        if int(gid) in gid_set:
            out.append({**row, "__orig_idx__": i})
    return out
