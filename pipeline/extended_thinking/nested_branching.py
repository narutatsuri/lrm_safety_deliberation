#!/usr/bin/env python3
"""Full-pool nested-branching variance + force-extension generator.

For each prompt in the K=32 think pool (READ-ONLY source):
  1. Deterministically subsample M=8 of K=32 existing think rollouts → M prefixes.
  2. For each prefix, for B in {20, 40, 60, 80} (cut < 100):
        * truncate the prefix's *thinking-token* portion at B% (relative to its
          natural `</think>` position),
        * append closure marker,
        * sample N=8 final responses.
  3. For each prefix, for B in {120, 140, 160, 180, 200} (cut > 100):
        * derive r = (cut - 100) / 100.0  (so B=120 → r=0.2; B=200 → r=1.0).
        * Phase 1a: continue from the full natural thinking (no close), with
          `bad_words=['</think>']` (`<|end|>` for GPT-OSS), for
          `extension_budget = int(r * natural_thinking_length)` tokens, n=1.
        * Phase 1b: append closure marker, sample N=8 final responses.

OUTPUT (NEW path; NEVER touches eval_results/k32_full_pool/):
  eval_results/k32_nested_branching_full/<model>/<split>/cut{B}/shard_*.jsonl

Per row:
  {
    "global_pool_idx": int,
    "prefix_idx": 0..M-1,
    "source_rollout_idx": int,
    "cut_pct": int,
    "prompt": str,            (carried through for cls)
    "split": ..., "benchmark": ..., "benchmark_sample_id": ..., "category": ...,
    "natural_thinking_length": int,
    "truncated_thinking_length": int,
    "extension_length": int | null,
    "rollouts": [N dicts with raw_response/response/n_tokens/finish_reason]
  }
"""
from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/nested_branching_variance/k32_full/nested_branching.py
# The model registry (scripts.neurips_final.common.MODELS) and the 4-guardrail
# majority vote now live in the `lrm_safety_deliberation` library (lrm_safety_deliberation.models /
# lrm_safety_deliberation.guardrails); this driver still imports the original modules via ROOT.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))

from scripts.neurips_final.common import MODELS
from evaluation.full_eval_pipeline.datasets import read_jsonl
from evaluation.full_eval_pipeline.pipeline import (
    _build_chat_prompts,
    _byte_level_decode_if_needed,
    _strip_response_channel,
)
from evaluation.base import strip_thinking

K32_SOURCE_ROOT = ROOT / "eval_results" / "k32_full_pool"
OUT_ROOT_DEFAULT = ROOT / "eval_results" / "k32_nested_branching_full"

# Closure markers (match apply_16k_rule.py / pipeline.py).
CLOSE_NONCHANNEL = "\n</think>\n"
CLOSE_CHANNEL = "<|end|><|start|>assistant<|channel|>final<|message|>"

# Close-marker substring used to locate the natural close in raw_response.
# Channel marker INCLUDES the channel-switch lead-up so natural_thinking ends
# strictly before "<|end|>" — required to avoid duplicating the channel-switch
# sequence when we append `closure` after cut_ids in the r=0.2 extension path.
CLOSE_MARKER_SUBSTR = {
    "nonchannel": "</think>",
    "channel": "<|end|><|start|>assistant<|channel|>final<|message|>",
}

# Token strings to ban during Phase-1a force-extension.
BAD_WORDS = {
    "nonchannel": ["</think>"],
    "channel": ["<|end|>", "<|return|>"],
}

ANSWER_BUDGET = 1024            # Phase 2 max_tokens for final answer
PHASE2_TEMP_DEFAULT = None      # None → use model's sampling.temperature
CHUNK_PROMPTS = int(os.environ.get("NB_CHUNK_PROMPTS", "16"))

# Override max_model_len for nested-branching (extension path, cut > 100).
# K=32 source rollouts have up to 16K thinking tokens; at r=1.0 (B=200) the
# extension adds another ~16K, plus prompt + closure + 1K answer headroom →
# need ~34K. We use 36K. Per-model cap below honors each checkpoint's native
# max_position_embeddings (Phi-4 = 32768; long-nat rows on Phi-4 skip via the
# existing `no_room_for_extension` branch).
MAX_MODEL_LEN_OVERRIDE = 36864


def get_model_position_cap(model_path: str) -> int:
    """Return model's native max_position_embeddings from config.json."""
    cfg = Path(model_path) / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("max_position_embeddings", 65536)
    return 65536


def shard_slice(n_total: int, shard_idx: int, n_shards: int) -> tuple[int, int]:
    chunk = (n_total + n_shards - 1) // n_shards
    start = shard_idx * chunk
    end = min(n_total, start + chunk)
    return start, end


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stable_seed(model: str, split: str, gid: int, base: int) -> int:
    s = f"{model}|{split}|{gid}|{base}".encode()
    return int(hashlib.sha256(s).hexdigest(), 16) & 0xFFFFFFFF


def load_k32_shards(model: str, split: str) -> dict[int, dict]:
    """Read all K=32 think shards for (model, split). Returns {global_pool_idx: row}."""
    cell_dir = K32_SOURCE_ROOT / model / "think" / split
    shard_files = sorted(cell_dir.glob("shard_*.jsonl"))
    if not shard_files:
        raise SystemExit(f"FAIL_NO_K32_SHARDS in {cell_dir}")
    by_gid: dict[int, dict] = {}
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                r = json.loads(line)
                gid = r.get("global_pool_idx")
                if gid is None:
                    continue
                by_gid[gid] = r
    return by_gid


def find_close_pos(raw: str, is_channel: bool) -> int:
    """Byte-position of close marker in raw_response. -1 if not found."""
    needle = CLOSE_MARKER_SUBSTR["channel" if is_channel else "nonchannel"]
    return raw.find(needle)


def thinking_token_ids(raw: str, close_pos: int, tokenizer) -> list[int]:
    """Tokenize raw[:close_pos] (the thinking portion) and return token ids."""
    if close_pos <= 0:
        return []
    thinking_text = raw[:close_pos]
    return tokenizer(thinking_text, add_special_tokens=False).input_ids


def build_decoded(raw: str, is_channel: bool) -> str:
    return _strip_response_channel(raw) if is_channel else strip_thinking(raw)


def existing_rows_by_key(out_path: Path) -> set[tuple[int, int]]:
    """Return set of (gid, prefix_idx) already present in out_path."""
    if not out_path.exists():
        return set()
    keys: set[tuple[int, int]] = set()
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r.get("global_pool_idx"), r.get("prefix_idx")))
    return keys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True, choices=["asr", "orr"])
    p.add_argument("--shard_idx", type=int, required=True)
    p.add_argument("--n_shards", type=int, required=True)
    p.add_argument("--K_existing", type=int, default=32)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--cuts", type=str, default="20,40,60,80,120")
    p.add_argument("--output_root", default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--subsample_seed_base", type=int, default=20260517)
    p.add_argument("--phase1a_seed_base", type=int, default=70260517)
    p.add_argument("--phase1b_seed_base", type=int, default=80260517)
    p.add_argument("--phase2_seed_base", type=int, default=90260517)
    p.add_argument("--ext_ratio", type=float, default=None,
                   help="Override extension ratio for cut>100 cells. "
                        "Default (None) derives r=(cut-100)/100 per-cut, so "
                        "cut=120→r=0.2, cut=140→r=0.4, ..., cut=200→r=1.0.")
    args = p.parse_args()

    assert args.model in MODELS, f"Unknown model {args.model}"
    assert 0 <= args.shard_idx < args.n_shards
    cuts = [int(x) for x in args.cuts.split(",") if x.strip()]
    assert args.M <= args.K_existing

    model_cfg = MODELS[args.model]
    is_channel = bool(model_cfg.get("is_channel", False))
    use_plain_prompt = bool(model_cfg.get("is_pretrained", False))
    closure = CLOSE_CHANNEL if is_channel else CLOSE_NONCHANNEL
    bad_words = BAD_WORDS["channel" if is_channel else "nonchannel"]

    out_root = Path(args.output_root)
    out_dirs = {b: out_root / args.model / args.split / f"cut{b}" for b in cuts}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{args.shard_idx:03d}_of_{args.n_shards:03d}"
    out_paths = {b: out_dirs[b] / f"{tag}.jsonl" for b in cuts}
    audit_path = out_root / args.model / args.split / f"{tag}.audit.json"

    # ---------- Load source K=32 rollouts ----------
    print(f"[nb] loading K=32 source for {args.model}/{args.split} ...", flush=True)
    by_gid = load_k32_shards(args.model, args.split)

    # ---------- Slice prompts for this shard ----------
    data_path = ROOT / "evaluation" / "full_eval_pipeline" / "data" / f"{args.split}.jsonl"
    records = read_jsonl(data_path)
    n_total = len(records)
    start, end = shard_slice(n_total, args.shard_idx, args.n_shards)
    my_records = records[start:end]
    my_gids = list(range(start, end))
    ext_cuts_preview = ", ".join(
        f"B={c}→r={(args.ext_ratio if args.ext_ratio is not None else (c - 100) / 100.0):.2f}"
        for c in cuts if c > 100
    ) or "(no extension cuts)"
    print(
        f"[shard] {args.model} {args.split} {tag}: prompts {start}..{end} "
        f"(n={len(my_records)}); cuts={cuts} M={args.M} N={args.N} "
        f"is_channel={is_channel}; extension: {ext_cuts_preview}",
        flush=True,
    )

    # Resume: per-cut, what (gid, prefix_idx) keys are already done?
    done_keys: dict[int, set[tuple[int, int]]] = {
        b: existing_rows_by_key(out_paths[b]) for b in cuts
    }
    for b in cuts:
        print(f"[resume] cut{b}: {len(done_keys[b])} (gid,prefix) rows already present", flush=True)

    # ---------- vLLM + tokenizer ----------
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    model_path = str(ROOT / model_cfg["model_path"])
    # Use the override (36864) so r=1.0 (B=200) fits: prompt + 16K nat + 16K ext +
    # closure + 1K answer ≈ 34K. Cap at the model's native max_position_embeddings
    # (Phi-4 = 32768) — long-nat rows that overflow this cap skip via the existing
    # `no_room_for_extension` branch.
    position_cap = get_model_position_cap(model_path)
    desired = max(model_cfg.get("max_model_len", 16384), MAX_MODEL_LEN_OVERRIDE)
    max_model_len = min(desired, position_cap)
    print(
        f"[vllm] Loading {model_path}  max_model_len={max_model_len} "
        f"(override={MAX_MODEL_LEN_OVERRIDE}, position_cap={position_cap})",
        flush=True,
    )
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-build chat-template prompts for my prompts (one per gid).
    chat_prompts = _build_chat_prompts(
        tokenizer, my_records, no_think=False, use_plain_prompt=use_plain_prompt,
    )
    chat_prompt_ids = [tokenizer(t, add_special_tokens=False).input_ids for t in chat_prompts]

    # Sampling base.
    samp = model_cfg.get("sampling", {})
    base_sampling: dict = {}
    for k in ("temperature", "top_p", "top_k", "min_p"):
        if k in samp:
            base_sampling[k] = samp[k]
    base_sampling.setdefault("temperature", 0.7)
    print(f"[sampling] base={base_sampling}", flush=True)

    def make_sp_final(n: int, seed: int) -> SamplingParams:
        kw = dict(base_sampling)
        kw.update(dict(n=n, seed=seed, max_tokens=ANSWER_BUDGET,
                       skip_special_tokens=not is_channel))
        return SamplingParams(**kw)

    def make_sp_extend(max_tokens: int, seed: int) -> SamplingParams:
        kw = dict(base_sampling)
        kw.update(dict(n=1, seed=seed, max_tokens=max_tokens,
                       skip_special_tokens=False, bad_words=bad_words))
        return SamplingParams(**kw)

    # Headroom: we need prompt_ids + cut_thinking_ids + closure_ids + ANSWER_BUDGET <= max_model_len.
    # The cut_thinking can be up to natural_thinking_length (~16K). Need at least 1K headroom.
    # If prompt_token_len + nat_think_len + closure + 1024 > max_model_len, mark as skipped.

    # ---------- Main loop: chunked over prompts ----------
    t0 = time.time()
    n_processed = 0
    n_skipped = 0
    n_skip_reasons: dict[str, int] = {}
    skipped_gid_set: set[int] = set()

    # Per-cut row buffers (to write at chunk end via atomic full-rewrite).
    rows_buffer: dict[int, list[dict]] = {b: [] for b in cuts}
    # Pre-load existing rows for full-rewrite (atomic) on chunk boundary.
    for b in cuts:
        if out_paths[b].exists():
            with open(out_paths[b]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows_buffer[b].append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def record_skip(gid: int, prefix_idx: int, cut: int, reason: str, **extra):
        nonlocal n_skipped
        n_skipped += 1
        n_skip_reasons[reason] = n_skip_reasons.get(reason, 0) + 1
        rec = {
            "global_pool_idx": gid,
            "prefix_idx": prefix_idx,
            "cut_pct": cut,
            "skipped": True,
            "skip_reason": reason,
            "rollouts": [],
        }
        rec.update(extra)
        rows_buffer[cut].append(rec)
        done_keys[cut].add((gid, prefix_idx))

    skip_phase2_target = ANSWER_BUDGET + 64  # headroom past prompt+thinking+closure

    for chunk_start in range(0, len(my_records), CHUNK_PROMPTS):
        chunk_end = min(chunk_start + CHUNK_PROMPTS, len(my_records))
        chunk_local = range(chunk_start, chunk_end)

        # ===== Build batches for this chunk =====
        # final_batch: list of (TokensPrompt, SamplingParams, meta) where meta
        #   identifies (gid, prefix_idx, cut_pct, full_thinking_text|None, cut_text,
        #   nat_len, cut_len, source_rollout_idx, extension_text|None, extension_len|None)
        # ext_batch: list for Phase 1a (extension) with n=1, custom max_tokens.

        ext_requests: list[tuple[TokensPrompt, SamplingParams]] = []
        ext_meta: list[dict] = []  # one per ext request

        final_requests: list[tuple[TokensPrompt, SamplingParams]] = []
        final_meta: list[dict] = []

        for li in chunk_local:
            gid = my_gids[li]
            record = my_records[li]
            prompt_ids = chat_prompt_ids[li]

            src_row = by_gid.get(gid)
            if src_row is None or src_row.get("over_budget"):
                for prefix_idx in range(args.M):
                    for cut in cuts:
                        if (gid, prefix_idx) in done_keys[cut]:
                            continue
                        record_skip(
                            gid, prefix_idx, cut,
                            "no_source_or_over_budget",
                            prompt=record.get("prompt"),
                            split=record.get("split"),
                            benchmark=record.get("benchmark"),
                            benchmark_sample_id=record.get("benchmark_sample_id"),
                            category=record.get("category"),
                        )
                continue

            # Subsample M prefixes among K_existing valid rollouts.
            valid_indices = [
                i for i, ro in enumerate(src_row.get("rollouts", []))
                if ro is not None and ro.get("raw_response")
            ]
            if len(valid_indices) < args.M:
                for prefix_idx in range(args.M):
                    for cut in cuts:
                        if (gid, prefix_idx) in done_keys[cut]:
                            continue
                        record_skip(
                            gid, prefix_idx, cut,
                            "too_few_valid_rollouts",
                            prompt=record.get("prompt"),
                            split=record.get("split"),
                            benchmark=record.get("benchmark"),
                            benchmark_sample_id=record.get("benchmark_sample_id"),
                            category=record.get("category"),
                            n_valid=len(valid_indices),
                        )
                continue

            seed = stable_seed(args.model, args.split, gid, args.subsample_seed_base)
            chosen = sorted(random.Random(seed).sample(valid_indices, args.M))

            for prefix_idx, src_idx in enumerate(chosen):
                # Skip if all cuts already done for this (gid, prefix_idx).
                pending_cuts = [c for c in cuts if (gid, prefix_idx) not in done_keys[c]]
                if not pending_cuts:
                    continue

                ro = src_row["rollouts"][src_idx]
                raw = ro["raw_response"]
                close_pos = find_close_pos(raw, is_channel)
                if close_pos < 0:
                    # No natural close in this rollout (rare; under 16K rule, force-close stitched it).
                    for cut in pending_cuts:
                        record_skip(
                            gid, prefix_idx, cut,
                            "no_close_in_prefix",
                            prompt=record.get("prompt"),
                            split=record.get("split"),
                            benchmark=record.get("benchmark"),
                            benchmark_sample_id=record.get("benchmark_sample_id"),
                            category=record.get("category"),
                            source_rollout_idx=src_idx,
                        )
                    continue

                think_ids = thinking_token_ids(raw, close_pos, tokenizer)
                nat_len = len(think_ids)
                if nat_len == 0:
                    for cut in pending_cuts:
                        record_skip(
                            gid, prefix_idx, cut,
                            "empty_thinking",
                            prompt=record.get("prompt"),
                            split=record.get("split"),
                            benchmark=record.get("benchmark"),
                            benchmark_sample_id=record.get("benchmark_sample_id"),
                            category=record.get("category"),
                            source_rollout_idx=src_idx,
                        )
                    continue

                closure_ids = tokenizer(closure, add_special_tokens=False).input_ids

                for cut in pending_cuts:
                    if cut > 100:
                        # Phase 1a (extension): n=1, max_tokens=int(r*nat_len), bad_words.
                        # r is derived per-cut: cut=120→0.2, cut=140→0.4, ..., cut=200→1.0.
                        # `--ext_ratio` overrides this if explicitly set (legacy).
                        r = args.ext_ratio if args.ext_ratio is not None else (cut - 100) / 100.0
                        ext_budget = max(1, int(r * nat_len))
                        # Cap ext_budget so phase1b prompt fits with headroom.
                        max_ext = max_model_len - (len(prompt_ids) + nat_len + len(closure_ids) + skip_phase2_target)
                        if max_ext < 1:
                            record_skip(
                                gid, prefix_idx, cut,
                                "no_room_for_extension",
                                prompt=record.get("prompt"),
                                split=record.get("split"),
                                benchmark=record.get("benchmark"),
                                benchmark_sample_id=record.get("benchmark_sample_id"),
                                category=record.get("category"),
                                source_rollout_idx=src_idx,
                                natural_thinking_length=nat_len,
                            )
                            continue
                        ext_budget_eff = min(ext_budget, max_ext)
                        phase1a_input_ids = prompt_ids + think_ids
                        # Seed includes `cut` so each B value gets a distinct
                        # extension trace (don't reuse the r=0.2 extension at r=1.0).
                        seed_a = stable_seed(args.model, args.split,
                                             gid * 10000 + prefix_idx * 100 + cut,
                                             args.phase1a_seed_base)
                        ext_requests.append((
                            TokensPrompt(prompt_token_ids=phase1a_input_ids),
                            make_sp_extend(ext_budget_eff, seed_a),
                        ))
                        ext_meta.append(dict(
                            gid=gid, prefix_idx=prefix_idx, src_idx=src_idx,
                            record=record, cut=cut, r=r,
                            nat_len=nat_len, think_ids=think_ids,
                            closure_ids=closure_ids, prompt_ids=prompt_ids,
                            ext_budget=ext_budget_eff,
                        ))
                    else:
                        # Cut at B% (B in 20/40/60/80).
                        cut_len = max(1, int(cut * nat_len / 100.0))
                        cut_ids = think_ids[:cut_len]
                        # Build phase2 input: prompt + cut + closure
                        phase2_input_ids = prompt_ids + cut_ids + closure_ids
                        if len(phase2_input_ids) + ANSWER_BUDGET > max_model_len:
                            record_skip(
                                gid, prefix_idx, cut,
                                "no_room_for_answer",
                                prompt=record.get("prompt"),
                                split=record.get("split"),
                                benchmark=record.get("benchmark"),
                                benchmark_sample_id=record.get("benchmark_sample_id"),
                                category=record.get("category"),
                                source_rollout_idx=src_idx,
                                natural_thinking_length=nat_len,
                                truncated_thinking_length=cut_len,
                            )
                            continue
                        seed_p2 = stable_seed(args.model, args.split,
                                              gid * 10000 + prefix_idx * 100 + cut,
                                              args.phase2_seed_base)
                        final_requests.append((
                            TokensPrompt(prompt_token_ids=phase2_input_ids),
                            make_sp_final(args.N, seed_p2),
                        ))
                        final_meta.append(dict(
                            gid=gid, prefix_idx=prefix_idx, src_idx=src_idx,
                            record=record, cut=cut, nat_len=nat_len, cut_len=cut_len,
                            cut_ids=cut_ids, prompt_ids=prompt_ids,
                            closure_ids=closure_ids,
                            extension_length=None,
                        ))

        # ===== Phase 1a (extension) batched run =====
        if ext_requests:
            prompts = [p for p, _ in ext_requests]
            sps = [sp for _, sp in ext_requests]
            outs = llm.generate(prompts, sps, use_tqdm=False)
            # For each extension, enqueue Phase 1b (final answer, N=args.N).
            for meta, out in zip(ext_meta, outs):
                comp = out.outputs[0]
                ext_text = _byte_level_decode_if_needed(comp.text or "")
                ext_n = len(comp.token_ids)
                # Re-tokenize ext_text under our tokenizer for stable ids.
                ext_ids = tokenizer(ext_text, add_special_tokens=False).input_ids
                # Safety: confirm no </think> / channel-switch leaked through (post-hoc).
                # If leaked, truncate text at first occurrence.
                needle = CLOSE_MARKER_SUBSTR["channel" if is_channel else "nonchannel"]
                leak_pos = ext_text.find(needle)
                if leak_pos >= 0:
                    ext_text = ext_text[:leak_pos]
                    ext_ids = tokenizer(ext_text, add_special_tokens=False).input_ids
                phase1b_input_ids = (
                    meta["prompt_ids"] + meta["think_ids"] + ext_ids + meta["closure_ids"]
                )
                if len(phase1b_input_ids) + ANSWER_BUDGET > max_model_len:
                    record_skip(
                        meta["gid"], meta["prefix_idx"], meta["cut"],
                        "phase1b_overlong",
                        prompt=meta["record"].get("prompt"),
                        split=meta["record"].get("split"),
                        benchmark=meta["record"].get("benchmark"),
                        benchmark_sample_id=meta["record"].get("benchmark_sample_id"),
                        category=meta["record"].get("category"),
                        source_rollout_idx=meta["src_idx"],
                        natural_thinking_length=meta["nat_len"],
                        extension_length=len(ext_ids),
                    )
                    continue
                seed_p1b = stable_seed(args.model, args.split,
                                       meta["gid"] * 10000 + meta["prefix_idx"] * 100 + meta["cut"],
                                       args.phase1b_seed_base)
                final_requests.append((
                    TokensPrompt(prompt_token_ids=phase1b_input_ids),
                    make_sp_final(args.N, seed_p1b),
                ))
                final_meta.append(dict(
                    gid=meta["gid"], prefix_idx=meta["prefix_idx"],
                    src_idx=meta["src_idx"],
                    record=meta["record"], cut=meta["cut"],
                    nat_len=meta["nat_len"],
                    cut_len=meta["nat_len"] + len(ext_ids),
                    cut_ids=meta["think_ids"] + ext_ids,
                    prompt_ids=meta["prompt_ids"],
                    closure_ids=meta["closure_ids"],
                    extension_length=len(ext_ids),
                    extension_text=ext_text,
                ))

        # ===== Phase 2 / Phase 1b (final answer) batched run =====
        if final_requests:
            prompts = [p for p, _ in final_requests]
            sps = [sp for _, sp in final_requests]
            outs = llm.generate(prompts, sps, use_tqdm=False)
            for meta, out in zip(final_meta, outs):
                rollouts = []
                # cut_text preserves the prefix's natural opening (e.g. GPT-OSS
                # already includes "<|channel|>analysis<|message|>"; Qwen3/Phi-4
                # already include "<think>"; Olmo's chat template inserted the
                # "<think>" so cut_text omits it). We append the model's
                # natural closure marker + the final response — no extra wrap.
                cut_text = tokenizer.decode(meta["cut_ids"], skip_special_tokens=False)
                for comp in out.outputs:
                    fin_raw = _byte_level_decode_if_needed(comp.text or "")
                    fin_n = len(comp.token_ids)
                    fin_fr = comp.finish_reason
                    full_raw = cut_text + closure + fin_raw
                    resp = build_decoded(full_raw, is_channel)
                    rollouts.append({
                        "raw_response": full_raw,
                        "response": resp,
                        "n_tokens": fin_n,
                        "finish_reason": fin_fr,
                    })
                row = {
                    "global_pool_idx": meta["gid"],
                    "prefix_idx": meta["prefix_idx"],
                    "source_rollout_idx": meta["src_idx"],
                    "cut_pct": meta["cut"],
                    "prompt": meta["record"].get("prompt"),
                    "split": meta["record"].get("split"),
                    "benchmark": meta["record"].get("benchmark"),
                    "benchmark_sample_id": meta["record"].get("benchmark_sample_id"),
                    "category": meta["record"].get("category"),
                    "natural_thinking_length": meta["nat_len"],
                    "truncated_thinking_length": meta["cut_len"],
                    "extension_length": meta.get("extension_length"),
                    "rollouts": rollouts,
                }
                if meta["cut"] > 100 and "extension_text" in meta:
                    row["extension_text"] = meta["extension_text"]
                rows_buffer[meta["cut"]].append(row)
                done_keys[meta["cut"]].add((meta["gid"], meta["prefix_idx"]))

        # ===== Atomic chunk write per cut =====
        for b in cuts:
            atomic_write_jsonl(out_paths[b], rows_buffer[b])

        n_processed += len(list(chunk_local))
        dt = time.time() - t0
        print(
            f"[chunk {chunk_start}-{chunk_end}] ext={len(ext_requests)} "
            f"final={len(final_requests)} skipped_acc={n_skipped} "
            f"elapsed={dt:.0f}s",
            flush=True,
        )

        # Chunk-level audit update (so partial progress is visible).
        # r_by_cut: realized extension ratio per cut > 100.
        # If --ext_ratio is set explicitly it overrides for all such cuts; otherwise
        # we derive r = (cut - 100) / 100 per cut.
        r_by_cut = {
            c: (args.ext_ratio if args.ext_ratio is not None else (c - 100) / 100.0)
            for c in cuts if c > 100
        }
        audit = {
            "model": args.model, "split": args.split,
            "shard_idx": args.shard_idx, "n_shards": args.n_shards,
            "K_existing": args.K_existing, "M": args.M, "N": args.N,
            "cuts": cuts, "ext_ratio": args.ext_ratio, "r_by_cut": r_by_cut,
            "n_prompts_in_shard": len(my_records),
            "n_prompts_processed": n_processed,
            "n_skipped": n_skipped,
            "skip_reasons": n_skip_reasons,
            "elapsed_seconds": dt,
            "chunk_prompts": CHUNK_PROMPTS,
            "sampling_base": base_sampling,
            "model_path": model_path,
            "out_root": str(out_root),
        }
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

    dt = time.time() - t0
    print(f"[summary] processed={n_processed} skipped={n_skipped} wall={dt:.0f}s", flush=True)

    del llm, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
