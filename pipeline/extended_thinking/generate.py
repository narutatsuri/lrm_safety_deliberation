#!/usr/bin/env python3
"""K=32 rollout generation, sharded with chunked incremental writes + malformed resample.

Invariants:
- Output JSONL is atomically rewritten after every chunk of CHUNK_PROMPTS prompts,
  so SLURM timeouts don't lose progress.
- Resumable: re-runs load existing rows and re-sample only positions whose
  finish_reason != 'stop'. Designed to handle "mode-collapse" truncations
  (degenerate repetition loops) by drawing fresh samples with different seeds.
- Schema unchanged from v1: each row has `rollouts: [{raw_response, response,
  n_tokens, finish_reason}]` and the existing `over_budget`/`truncated_any` flags.
- Audit gains `n_resampled` and `chunk_prompts`, but `n_truncated` /
  `n_rollouts_total` / `n_rollouts_expected` keep their original meaning for
  final_audit.py compatibility.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/nested_branching_variance/k32_full/generate.py
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

# Per-model max_new_tokens. Calibration carried from v1; truncations are mostly
# mode collapse (degenerate loops), not legitimate long thinking, so the primary
# mitigation is re-sampling malformed positions with fresh seeds.
PER_MODEL_MAXNEW = {
    "Qwen3-8B":        {"think": 14000, "nothink": 8192},
    "Olmo-3-7B-Think": {"think": 14000, "nothink": 8192},
    "Phi-4-reasoning": {"think": 28000, "nothink": 16384},
    "GPT-OSS-20B":     {"think": 24000, "nothink": 24000},
}

PER_MODEL_MAXLEN = {
    "Qwen3-8B":        32768,
    "Olmo-3-7B-Think": 32768,
    "Phi-4-reasoning": 32768,
    "GPT-OSS-20B":     32768,
}

CHUNK_PROMPTS = int(os.environ.get("K32_CHUNK_PROMPTS", "32"))
MAX_RESAMPLE_ATTEMPTS = int(os.environ.get("K32_MAX_RESAMPLE", "2"))


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


def init_row(rec: dict, global_id: int, shard_idx: int, prompt_token_len: int, K: int) -> dict:
    row = dict(rec)
    row["global_pool_idx"] = global_id
    row["shard_idx"] = shard_idx
    row["prompt_token_len"] = prompt_token_len
    row["rollouts"] = [None] * K
    row["finish_reasons"] = [None] * K
    row["truncated_any"] = False
    row["over_budget"] = False
    return row


def mark_over_budget(rec: dict, global_id: int, shard_idx: int, prompt_token_len: int) -> dict:
    row = dict(rec)
    row["global_pool_idx"] = global_id
    row["shard_idx"] = shard_idx
    row["prompt_token_len"] = prompt_token_len
    row["rollouts"] = []
    row["finish_reasons"] = []
    row["truncated_any"] = False
    row["over_budget"] = True
    return row


def count_existing_state(existing: dict, K: int) -> tuple[int, int, int]:
    """Return (n_clean, n_malformed, n_unfilled) across all existing rows' K slots."""
    clean = malformed = unfilled = 0
    for row in existing.values():
        if row.get("over_budget"):
            continue
        for fr in row["finish_reasons"]:
            if fr is None:
                unfilled += 1
            elif fr == "stop":
                clean += 1
            else:
                malformed += 1
    return clean, malformed, unfilled


def write_audit(audit_path: Path, args, max_new: int, eff_max_model_len: int,
                my_records: list[dict], existing: dict, sp_kwargs: dict,
                model_path: str, data_path: Path, wall_seconds: float,
                n_resampled: int, n_over_budget: int) -> None:
    n_truncated_final = 0
    n_rollouts_total = 0
    for row in existing.values():
        for fr in row.get("finish_reasons", []):
            if fr is None:
                continue  # unfilled slot - not counted as a rollout
            n_rollouts_total += 1
            if fr != "stop":
                n_truncated_final += 1
    n_expected = (len(my_records) - n_over_budget) * args.K
    audit = {
        "model": args.model,
        "mode": args.mode,
        "split": args.split,
        "shard_idx": args.shard_idx,
        "n_shards": args.n_shards,
        "K": args.K,
        "max_new_tokens": max_new,
        "max_model_len": eff_max_model_len,
        "n_prompts_in_shard": len(my_records),
        "n_prompts_over_budget": n_over_budget,
        "n_rollouts_total": n_rollouts_total,
        "n_rollouts_expected": n_expected,
        "n_truncated": n_truncated_final,
        "n_resampled": n_resampled,
        "trunc_rate": n_truncated_final / max(1, n_rollouts_total),
        "wall_seconds": wall_seconds,
        "tps_observed": (n_rollouts_total * 1000) / max(1.0, wall_seconds) if wall_seconds else 0.0,
        "chunk_prompts": CHUNK_PROMPTS,
        "max_resample_attempts": MAX_RESAMPLE_ATTEMPTS,
        "sampling": sp_kwargs,
        "prompt_data_path": str(data_path),
        "model_path": model_path,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True, choices=["asr", "orr"])
    p.add_argument("--mode", required=True, choices=["think", "nothink"])
    p.add_argument("--shard_idx", type=int, required=True)
    p.add_argument("--n_shards", type=int, required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--output_root", default=str(ROOT / "eval_results" / "k32_full_pool"))
    args = p.parse_args()

    assert args.model in MODELS, f"Unknown model {args.model}"
    assert args.model in PER_MODEL_MAXNEW, f"No max_new_tokens config for {args.model}"
    assert 0 <= args.shard_idx < args.n_shards, "shard_idx out of range"

    model_cfg = MODELS[args.model]
    no_think = (args.mode == "nothink")
    max_new = PER_MODEL_MAXNEW[args.model]["nothink" if no_think else "think"]
    is_channel = bool(model_cfg.get("is_channel", False))
    use_plain_prompt = bool(model_cfg.get("is_pretrained", False))

    out_dir = Path(args.output_root) / args.model / args.mode / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{args.shard_idx:03d}_of_{args.n_shards:03d}"
    gen_path = out_dir / f"{tag}.jsonl"
    audit_path = out_dir / f"{tag}.audit.json"

    data_path = ROOT / "evaluation" / "full_eval_pipeline" / "data" / f"{args.split}.jsonl"
    records = read_jsonl(data_path)
    n_total = len(records)
    start, end = shard_slice(n_total, args.shard_idx, args.n_shards)
    my_records = records[start:end]
    print(
        f"[shard] {args.model} {args.mode} {args.split} {tag}: "
        f"prompts {start}..{end} (n={len(my_records)}/{n_total}); "
        f"max_new_tokens={max_new}; CHUNK={CHUNK_PROMPTS} MAX_RESAMPLE={MAX_RESAMPLE_ATTEMPTS}",
        flush=True,
    )

    # Load existing rows (resume / re-sample mode)
    existing: dict[int, dict] = {}
    if gen_path.exists():
        with open(gen_path) as f:
            for line in f:
                row = json.loads(line)
                local_idx = row["global_pool_idx"] - start
                # Normalize: make sure finish_reasons is K-length and rollouts mirrors it
                K = args.K
                if not row.get("over_budget"):
                    while len(row.setdefault("finish_reasons", [])) < K:
                        row["finish_reasons"].append(None)
                    while len(row.setdefault("rollouts", [])) < K:
                        row["rollouts"].append(None)
                existing[local_idx] = row
        clean, malformed, unfilled = count_existing_state(existing, args.K)
        print(
            f"[resume] Loaded {len(existing)} rows; clean={clean} malformed={malformed} unfilled={unfilled}",
            flush=True,
        )

    # Quick exit if everything (across all expected prompts) is clean
    all_present = len(existing) == len(my_records)
    if all_present:
        _, malformed, unfilled = count_existing_state(existing, args.K)
        if malformed == 0 and unfilled == 0:
            print(f"[skip] All {len(my_records)} prompts already have K={args.K} clean rollouts.", flush=True)
            # Leave the existing audit untouched; it is authoritative for this state.
            return 0

    # Bring up vLLM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = str(ROOT / model_cfg["model_path"])
    eff_max_model_len = PER_MODEL_MAXLEN.get(args.model, model_cfg["max_model_len"])
    print(f"[vllm] Loading {model_path}  max_model_len={eff_max_model_len}", flush=True)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
        max_model_len=eff_max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = _build_chat_prompts(
        tokenizer, my_records, no_think=no_think, use_plain_prompt=use_plain_prompt,
    )
    budget = eff_max_model_len - max_new - 64
    prompt_lens: list[int] = []
    for t in texts:
        prompt_lens.append(len(tokenizer(t, add_special_tokens=False).input_ids))
    over_set = set(i for i, n in enumerate(prompt_lens) if n > budget)
    if over_set:
        print(
            f"[over-budget] {len(over_set)}/{len(texts)} prompts exceed budget={budget}; "
            f"will be marked over_budget with empty rollouts",
            flush=True,
        )

    # Initialize/normalize rows that don't yet exist
    for local_i in range(len(my_records)):
        if local_i in over_set:
            existing[local_i] = mark_over_budget(
                my_records[local_i], start + local_i, args.shard_idx, prompt_lens[local_i],
            )
            continue
        if local_i not in existing:
            existing[local_i] = init_row(
                my_records[local_i], start + local_i, args.shard_idx,
                prompt_lens[local_i], args.K,
            )
        else:
            # An existing row might have been written before over_set was known.
            # If it was previously over_budget but no longer is (config change),
            # we trust the new computation and reset its rollouts. Conservative:
            # only reset if the existing row is over_budget mismatch.
            row = existing[local_i]
            if row.get("over_budget") and local_i not in over_set:
                existing[local_i] = init_row(
                    my_records[local_i], start + local_i, args.shard_idx,
                    prompt_lens[local_i], args.K,
                )

    # Build base sampling kwargs (n + seed per request)
    samp = model_cfg.get("sampling", {})
    sp_base: dict = dict(max_tokens=max_new, skip_special_tokens=not is_channel)
    for k in ("temperature", "top_p", "top_k", "min_p"):
        if k in samp:
            sp_base[k] = samp[k]
    sp_base.setdefault("temperature", 0.7)
    print(f"[sampling] base={sp_base}", flush=True)

    def make_sp(n: int, seed: int | None) -> SamplingParams:
        kw = dict(sp_base)
        kw["n"] = n
        if seed is not None:
            kw["seed"] = seed
        return SamplingParams(**kw)

    def rollout_from(out_text: str, finish_reason: str, n_tokens: int) -> dict:
        raw = _byte_level_decode_if_needed(out_text)
        if no_think:
            resp = _strip_response_channel(raw) if is_channel else raw.strip()
        else:
            resp = _strip_response_channel(raw) if is_channel else strip_thinking(raw)
        return {
            "raw_response": raw,
            "response": resp,
            "n_tokens": n_tokens,
            "finish_reason": finish_reason,
        }

    t0 = time.time()
    total_resampled = 0  # rollouts produced via re-sample (attempt > 0)

    for chunk_start in range(0, len(my_records), CHUNK_PROMPTS):
        chunk_end = min(chunk_start + CHUNK_PROMPTS, len(my_records))
        chunk_t0 = time.time()

        # Gather work: (local_i, positions_to_fill)
        work: list[tuple[int, list[int]]] = []
        for local_i in range(chunk_start, chunk_end):
            if local_i in over_set:
                continue
            row = existing[local_i]
            bad = [pi for pi, fr in enumerate(row["finish_reasons"]) if fr != "stop"]
            if bad:
                work.append((local_i, bad))

        if not work:
            # Chunk already clean (resumed run hitting an already-complete slice)
            continue

        base_seed = args.shard_idx * 10**7 + chunk_start * 1000
        for attempt in range(MAX_RESAMPLE_ATTEMPTS + 1):
            if not work:
                break
            prompts_to_run = [texts[li] for li, _ in work]
            sps = []
            for (li, positions), _ in zip(work, prompts_to_run):
                seed = (base_seed + attempt * 17 + li) if attempt > 0 else None
                sps.append(make_sp(n=len(positions), seed=seed))
            outputs = llm.generate(prompts_to_run, sps, use_tqdm=False)

            new_work: list[tuple[int, list[int]]] = []
            n_filled = 0
            n_resampled_chunk = 0  # fills that replaced a previously-malformed slot
            for (li, positions), out in zip(work, outputs):
                row = existing[li]
                still_bad: list[int] = []
                for pos_idx, pos in enumerate(positions):
                    prev_fr = row["finish_reasons"][pos]
                    r = out.outputs[pos_idx]
                    fr = r.finish_reason
                    new_roll = rollout_from(r.text, fr, len(r.token_ids))
                    row["rollouts"][pos] = new_roll
                    row["finish_reasons"][pos] = fr
                    if fr == "stop":
                        n_filled += 1
                        if prev_fr is not None and prev_fr != "stop":
                            n_resampled_chunk += 1
                    else:
                        still_bad.append(pos)
                row["truncated_any"] = any(fr != "stop" for fr in row["finish_reasons"])
                if still_bad:
                    new_work.append((li, still_bad))

            total_resampled += n_resampled_chunk
            still_bad_total = sum(len(p) for _, p in new_work)
            print(
                f"[chunk {chunk_start}-{chunk_end} attempt {attempt}] "
                f"filled={n_filled} resampled={n_resampled_chunk} still_bad={still_bad_total}",
                flush=True,
            )
            work = new_work

        # Atomic rewrite of full file after this chunk
        rows_to_write = [existing[i] for i in range(len(my_records))]
        atomic_write_jsonl(gen_path, rows_to_write)
        # Update audit incrementally (so external monitoring sees progress)
        write_audit(
            audit_path, args, max_new, eff_max_model_len, my_records, existing,
            sp_kwargs=sp_base, model_path=model_path, data_path=data_path,
            wall_seconds=time.time() - t0, n_resampled=total_resampled,
            n_over_budget=len(over_set),
        )
        print(
            f"[chunk {chunk_start}-{chunk_end}] wrote {len(rows_to_write)} rows in "
            f"{time.time() - chunk_t0:.1f}s; total_elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    dt = time.time() - t0
    write_audit(
        audit_path, args, max_new, eff_max_model_len, my_records, existing,
        sp_kwargs=sp_base, model_path=model_path, data_path=data_path,
        wall_seconds=dt, n_resampled=total_resampled, n_over_budget=len(over_set),
    )

    # Final summary
    n_clean, n_malformed, n_unfilled = count_existing_state(existing, args.K)
    n_rollouts_total = n_clean + n_malformed
    n_expected = (len(my_records) - len(over_set)) * args.K
    print(
        f"[summary] clean={n_clean} truncated_final={n_malformed} unfilled={n_unfilled} "
        f"(/{n_expected} expected); resampled={total_resampled} over_budget={len(over_set)} "
        f"wall={dt:.1f}s",
        flush=True,
    )

    del llm, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if n_unfilled > 0:
        print(f"FAIL_UNFILLED: {n_unfilled} rollouts never sampled (likely OOM or early exit).", flush=True)
        return 5
    if n_malformed > 0:
        print(
            f"FAIL_TRUNCATION_RESIDUAL: {n_malformed} rollouts still malformed after "
            f"{MAX_RESAMPLE_ATTEMPTS} retries.",
            flush=True,
        )
        return 2
    if n_rollouts_total != n_expected:
        print(f"FAIL_ROLLOUT_COUNT: got {n_rollouts_total}, expected {n_expected}.", flush=True)
        return 3
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
