#!/usr/bin/env python3
"""
Causal cuts generator. Two modes:

- `flip` (Phase 1): cut the thinking trace at every per-sentence sign flip
  (+1 <-> -1), append force-close `</think>`, generate K rollouts. Preserved
  for backward compatibility with existing rollouts.jsonl.

- `segment` (Phase 2): collapse per-sentence labels into stance segments under
  carry_forward zero handling, find salient transitions where segment S_pre
  (sign A, length >= L_pre) is followed by S_post (sign B = -A, length >= L_min),
  and emit a matched (pre, post) cut pair per transition. K rollouts each at
  pre-cut and post-cut prefixes. (L_pre, L_min) thresholds are recorded as
  metainfo so a single (1,1) generation pass supports the entire sweep at
  analysis time.

Output:
  flip mode    -> rollouts.jsonl
  segment mode -> rollouts_segment.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/causal_cuts/generate_cuts.py
# The model registry (scripts.neurips_final.common.MODELS) and the 4-guardrail
# majority vote now also live in the `lrm_safety_deliberation` library (lrm_safety_deliberation.models /
# lrm_safety_deliberation.guardrails); this driver still imports the original repo modules via
# ROOT. Set LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
PROJECT_ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from scripts.neurips_final.common import MODELS, ROOT

DEFAULT_AUDIT_PATH = ROOT / "experiments" / "causal_cuts" / "eval_results" / "stance_audit_full" / "labeled_traces.jsonl"
# NOTE: BASE_ROOT and SUPP_ROOT point at the canonical top-level NeurIPS-final
# generation files (created 2026-04-17). Earlier code expected them to live
# under experiments/causal_cuts/eval_results/ — that path now only contains
# classifications/representations/prefill_analysis subdirs, no `generations/`.
BASE_ROOT = ROOT / "eval_results" / "neurips_final"
SUPP_ROOT = ROOT / "eval_results" / "neurips_final_supplementary"
OUT_ROOT = ROOT / "experiments" / "causal_cuts" / "eval_results" / "causal_cuts"

CHUNK_LEN = 200
BASE_SEED = 1234

AUDIT_NAME_TO_MODEL_KEY = {
    "GPT-OSS-20B": "GPT-OSS-20B",
    "Qwen3-8B": "Qwen3-8B",
    "Olmo3-7B-Think": "Olmo-3-7B-Think",
    "Phi-4-reasoning": "Phi-4-reasoning",
}


# ── chunking (must match run_full_audit.py exactly) ──────────────

def chunk_trace(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= CHUNK_LEN:
            cur += s + " "
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 20]


# ── Phase 1: per-sentence sign flips ─────────────────────────────

def sign_flip_points(labels: list[int]) -> list[tuple[int, str]]:
    flips = []
    for i in range(1, len(labels)):
        if labels[i - 1] * labels[i] < 0:
            direction = f"{labels[i-1]:+d}->{labels[i]:+d}"
            flips.append((i, direction))
    return flips


# ── Phase 2: stance segments under carry_forward ─────────────────

def segments_carry_forward(labels: list[int]) -> list[tuple[int, int, int, int]]:
    """Return list of (sign, start_chunk_idx, end_chunk_idx, length_chunks)
    after forward-filling 0-labeled chunks into the surrounding sign.
    Leading 0s become a 0-segment that is skipped for transitions.
    """
    n = len(labels)
    if n == 0:
        return []
    filled = list(labels)
    last = 0
    for i in range(n):
        if filled[i] == 0:
            filled[i] = last
        else:
            last = filled[i]
    segs = []
    cur_sign = filled[0]
    cur_start = 0
    cur_len = 1
    for i in range(1, n):
        if filled[i] == cur_sign:
            cur_len += 1
        else:
            segs.append((cur_sign, cur_start, i - 1, cur_len))
            cur_sign = filled[i]
            cur_start = i
            cur_len = 1
    segs.append((cur_sign, cur_start, n - 1, cur_len))
    return segs


def salient_transitions(
    labels: list[int], L_pre: int, L_min: int
) -> list[tuple[int, int, int, int, int, int]]:
    """For each adjacent (S_pre, S_post) pair with opposite nonzero signs and
    length thresholds satisfied, return:
      (transition_idx, S_pre.sign, S_pre.end_chunk_idx, S_pre.length,
                       S_post.sign, S_post.end_chunk_idx, S_post.length)

    cut points used downstream:
      pre_cut_chunk_idx  = S_pre.end_chunk_idx + 1   (prefix length in chunks)
      post_cut_chunk_idx = S_post.end_chunk_idx + 1
    """
    segs = segments_carry_forward(labels)
    out = []
    t_idx = 0
    for i in range(len(segs) - 1):
        a = segs[i]
        b = segs[i + 1]
        if a[0] == 0 or b[0] == 0:
            continue
        if a[0] * b[0] >= 0:
            continue
        if a[3] < L_pre or b[3] < L_min:
            continue
        out.append(
            (
                t_idx,
                a[0],  # S_pre sign
                a[2],  # S_pre end_chunk_idx
                a[3],  # S_pre length
                b[0],  # S_post sign
                b[2],  # S_post end_chunk_idx
                b[3],  # S_post length
            )
        )
        t_idx += 1
    return out


# ── tokeniser helpers ────────────────────────────────────────────

def tok_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )


def open_think_ids(tokenizer) -> list[int]:
    return tok_ids(tokenizer, "<think>\n")


def close_think_ids(tokenizer) -> list[int]:
    return tok_ids(tokenizer, "\n</think>\n\n")


def gptoss_analysis_ids(tokenizer) -> list[int]:
    return tok_ids(tokenizer, "<|channel|>analysis<|message|>")


def gptoss_final_ids(tokenizer) -> list[int]:
    return tok_ids(tokenizer, "<|end|><|start|>assistant<|channel|>final<|message|>")


def prompt_has_open_think(text: str) -> bool:
    return bool(re.search(r"<think>\s*$", text))


# ── base trace loaders ───────────────────────────────────────────

def load_base_traces(model_key: str, split: str) -> dict[int, dict]:
    """Load canonical base traces. For Qwen3-8B benign, also load ORFuzz
    supplementary traces (gid > 2884, benchmark==orfuzzset)."""
    base_path = BASE_ROOT / model_key / "generations" / "base" / f"{split}.jsonl"
    base_by_gid: dict[int, dict] = {}
    with open(base_path) as f:
        for line in f:
            d = json.loads(line)
            base_by_gid[d["global_id"]] = d

    if model_key == "Qwen3-8B" and split == "benign":
        supp_path = SUPP_ROOT / model_key / "generations" / "base" / "orfuzz.jsonl"
        if supp_path.exists():
            with open(supp_path) as f:
                for line in f:
                    d = json.loads(line)
                    gid = d["global_id"]
                    if gid not in base_by_gid:
                        base_by_gid[gid] = d
    return base_by_gid


# ── main generation ──────────────────────────────────────────────

def run(args) -> None:
    model_key = args.model
    split = args.split
    audit_name_map = {v: k for k, v in AUDIT_NAME_TO_MODEL_KEY.items()}
    audit_name = audit_name_map.get(model_key, model_key)

    spec = MODELS[model_key]
    model_path = str(ROOT / spec["model_path"])
    max_model_len = args.max_model_len or spec.get("max_model_len", 8192)
    gpu_util = spec.get("gpu_memory_utilization", 0.90)
    sampling_kw = spec.get("sampling", {"temperature": 0.6, "top_p": 0.95})
    is_channel = bool(spec.get("is_channel", False))
    max_new = args.max_new
    prompt_budget = max_model_len - max_new - 64

    out_dir = OUT_ROOT / model_key / split
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.out_suffix or ""
    if args.transition_mode == "flip":
        out_path = out_dir / f"rollouts{suffix}.jsonl"
    else:
        if args.n_shards > 1:
            out_path = (
                out_dir / f"rollouts_segment{suffix}__shard{args.prompt_shard}of{args.n_shards}.jsonl"
            )
        else:
            out_path = out_dir / f"rollouts_segment{suffix}.jsonl"

    # ── resume map ───────────────────────────────────────────────
    # Per-rollout dedup: track (gid, t_idx, kind, rollout_idx) so calling with
    # a larger --K appends only the missing rollout indices to existing cells.
    # For flip mode the dedup key is (gid, cut_idx, rollout_idx).
    done_rollouts = set()  # full keys including rollout_idx
    rollouts_per_cut = {}  # cut_key -> set of completed rollout_idx values
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("skipped"):
                        continue
                    r_idx = int(d.get("rollout_idx", 0))
                    if args.transition_mode == "flip":
                        cut_key = (d["global_id"], d["cut_idx"])
                    else:
                        cut_key = (d["global_id"], d["transition_idx"], d["cut_kind"])
                    done_rollouts.add(cut_key + (r_idx,))
                    rollouts_per_cut.setdefault(cut_key, set()).add(r_idx)
                except Exception:
                    pass
    n_full_done = sum(1 for s in rollouts_per_cut.values() if len(s) >= args.K)
    print(f"[{model_key}/{split}] already done: {len(done_rollouts)} rollouts across "
          f"{len(rollouts_per_cut)} cuts ({n_full_done} cuts already at K>={args.K})")

    # ── audit rows ───────────────────────────────────────────────
    audit_path = Path(args.audit_path)
    print(f"[{model_key}/{split}] audit-path: {audit_path}")
    # Accept either the audit-name convention (e.g. "Olmo3-7B-Think") or the
    # canonical model_key (e.g. "Olmo-3-7B-Think"). The majority-labels file
    # uses the canonical name; the legacy GPT-5.4-only file uses the audit name.
    accepted_names = {audit_name, model_key}
    audit_rows: dict[int, dict] = {}
    n_dropped_none = 0
    with open(audit_path) as f:
        for line in f:
            d = json.loads(line)
            if "labels" not in d:
                continue
            if d.get("model") in accepted_names and d["split"] == split:
                # Drop traces with any None labels (2-annotator unresolved chunks).
                # The cut-detection logic requires int labels per chunk.
                if any(x is None for x in d["labels"]):
                    n_dropped_none += 1
                    continue
                audit_rows[d["global_id"]] = d
    if n_dropped_none > 0:
        print(f"[{model_key}/{split}] dropped {n_dropped_none} traces with None labels")

    # ── base traces ──────────────────────────────────────────────
    base_by_gid = load_base_traces(model_key, split)

    # ── prompt-shard filter (modulo on global_id sorted order) ──
    sorted_gids = sorted(audit_rows.keys())
    if args.n_shards > 1:
        sorted_gids = [
            g for i, g in enumerate(sorted_gids) if i % args.n_shards == args.prompt_shard
        ]
    audit_rows = {g: audit_rows[g] for g in sorted_gids}
    print(
        f"[{model_key}/{split}] audit rows after shard {args.prompt_shard}/{args.n_shards}: {len(audit_rows)}"
    )

    # ── collect tasks ────────────────────────────────────────────
    # task fields differ per mode; common: (gid, prefix_text, prefix_tokens, full_thinking, prompt_text, meta_extra)
    tasks: list[dict] = []
    for gid, audit in audit_rows.items():
        labels = audit["labels"]
        base = base_by_gid.get(gid)
        if base is None or not base.get("thinking_trace"):
            continue
        chunks = chunk_trace(base["thinking_trace"])
        if not chunks:
            continue
        full_thinking = base["thinking_trace"]
        prompt_text = base["prompt"]
        benchmark = audit.get("benchmark", base.get("benchmark", ""))

        if args.transition_mode == "flip":
            for cut_idx, flip_dir in sign_flip_points(labels):
                cut_key = (gid, cut_idx)
                if len(rollouts_per_cut.get(cut_key, set())) >= args.K:
                    continue
                tasks.append(
                    dict(
                        gid=gid,
                        cut_idx=cut_idx,
                        flip_dir=flip_dir,
                        chunks=chunks,
                        full_thinking=full_thinking,
                        prompt_text=prompt_text,
                        n_chunks_total=len(chunks),
                        benchmark=benchmark,
                        existing_r_idx=rollouts_per_cut.get(cut_key, set()).copy(),
                    )
                )
        else:
            transitions = salient_transitions(labels, args.L_pre, args.L_min)
            for (
                t_idx,
                s_pre_sign,
                s_pre_end,
                s_pre_len,
                s_post_sign,
                s_post_end,
                s_post_len,
            ) in transitions:
                pre_cut = s_pre_end + 1
                post_cut = s_post_end + 1
                # safety: pre_cut may equal 0 if S_pre starts at index 0 with length covering only chunk 0
                if pre_cut <= 0 or post_cut <= pre_cut:
                    continue
                if pre_cut > len(chunks) or post_cut > len(chunks):
                    continue
                for kind, cut_chunk_idx in (("pre", pre_cut), ("post", post_cut)):
                    cut_key = (gid, t_idx, kind)
                    if len(rollouts_per_cut.get(cut_key, set())) >= args.K:
                        continue
                    tasks.append(
                        dict(
                            gid=gid,
                            transition_idx=t_idx,
                            cut_kind=kind,
                            cut_chunk_idx=cut_chunk_idx,
                            existing_r_idx=rollouts_per_cut.get(cut_key, set()).copy(),
                            s_pre_sign=int(s_pre_sign),
                            s_pre_end=s_pre_end,
                            s_pre_len=s_pre_len,
                            s_post_sign=int(s_post_sign),
                            s_post_end=s_post_end,
                            s_post_len=s_post_len,
                            chunks=chunks,
                            full_thinking=full_thinking,
                            prompt_text=prompt_text,
                            n_chunks_total=len(chunks),
                            benchmark=benchmark,
                        )
                    )

    print(f"[{model_key}/{split}] tasks to generate: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        return

    if args.dry_run:
        # Dump per-cut summary so caller can plan SLURM sizing without loading the model.
        n_cuts = len(tasks)
        n_traces_with_cuts = len({t["gid"] for t in tasks})
        n_rollouts_planned = n_cuts * args.K
        print(f"[{model_key}/{split}] DRY RUN: {n_cuts} cuts across "
              f"{n_traces_with_cuts} traces; would generate {n_rollouts_planned} rollouts at K={args.K}")
        return

    # ── load model ───────────────────────────────────────────────
    print(f"Loading {model_key}...")
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if is_channel:
        open_ids = gptoss_analysis_ids(tokenizer)
        close_ids = gptoss_final_ids(tokenizer)
    else:
        open_ids = open_think_ids(tokenizer)
        close_ids = close_think_ids(tokenizer)

    sampling = SamplingParams(
        n=args.K, seed=BASE_SEED, max_tokens=max_new, **sampling_kw
    )

    # ── build prompts ────────────────────────────────────────────
    prompts: list[TokensPrompt] = []
    metas: list[dict] = []
    skipped_rows: list[dict] = []

    for task in tasks:
        chunks = task["chunks"]
        if args.transition_mode == "flip":
            cut_chunk_idx = task["cut_idx"]
        else:
            cut_chunk_idx = task["cut_chunk_idx"]

        prefix_chunks_text = " ".join(chunks[:cut_chunk_idx])
        prefix_ids = tok_ids(tokenizer, prefix_chunks_text)
        full_ids_think = tok_ids(tokenizer, task["full_thinking"])

        prompt_rendered = apply_chat_template(tokenizer, task["prompt_text"])
        prompt_ids = tok_ids(tokenizer, prompt_rendered)

        if is_channel:
            # GPT-OSS Harmony: rendered prompt ends just before channel marker;
            # we must always prepend analysis_ids (no stray "<think>" exists).
            full_ids = list(prompt_ids) + open_ids + prefix_ids + close_ids
        elif prompt_has_open_think(prompt_rendered):
            full_ids = list(prompt_ids) + prefix_ids + close_ids
        else:
            full_ids = list(prompt_ids) + open_ids + prefix_ids + close_ids

        # common metainfo
        common = dict(
            global_id=task["gid"],
            split=split,
            n_chunks_total=task["n_chunks_total"],
            benchmark=task["benchmark"],
            prefix_n_chunks=cut_chunk_idx,
            prefix_n_tokens=len(prefix_ids),
            full_thinking_tokens=len(full_ids_think),
            prefix_frac_of_full_trace=(
                len(prefix_ids) / max(1, len(full_ids_think))
            ),
            prompt_token_len=len(full_ids),
        )
        if args.transition_mode == "flip":
            common.update(
                cut_idx=task["cut_idx"], flip_dir=task["flip_dir"], cut_kind="post"
            )
        else:
            common.update(
                transition_idx=task["transition_idx"],
                cut_kind=task["cut_kind"],
                cut_chunk_idx=task["cut_chunk_idx"],
                s_pre_sign=task["s_pre_sign"],
                s_pre_len_chunks=task["s_pre_len"],
                s_pre_end_chunk=task["s_pre_end"],
                s_post_sign=task["s_post_sign"],
                s_post_len_chunks=task["s_post_len"],
                s_post_end_chunk=task["s_post_end"],
                L_pre_used=args.L_pre,
                L_min_used=args.L_min,
            )

        if len(full_ids) > prompt_budget:
            for r in range(args.K):
                row = dict(common)
                row.update(
                    rollout_idx=r,
                    response="",
                    raw_response="",
                    thinking_prefix=prefix_chunks_text,
                    seed=BASE_SEED,
                    skipped=True,
                    skip_reason="overlong_prefix",
                )
                skipped_rows.append(row)
            continue

        prompts.append(TokensPrompt(prompt_token_ids=full_ids))
        metas.append({
            "common": common,
            "prefix_text": prefix_chunks_text,
            "existing_r_idx": task.get("existing_r_idx", set()),
        })

    print(f"[{model_key}/{split}] prompts={len(prompts)} skipped={len(skipped_rows)//max(1,args.K)}")

    # Write skipped rows up front so they're durable.
    if skipped_rows:
        with open(out_path, "a") as fout:
            for row in skipped_rows:
                fout.write(json.dumps(row) + "\n")

    # Batched generation — flush every BATCH prompts so a TLE doesn't lose
    # the whole run. vLLM continuous batching is preserved within each batch.
    BATCH = max(1, args.batch_size)
    n_done = 0
    for start in range(0, len(prompts), BATCH):
        end = min(start + BATCH, len(prompts))
        batch_prompts = prompts[start:end]
        batch_metas = metas[start:end]
        if not batch_prompts:
            continue
        outputs = llm.generate(batch_prompts, sampling)
        with open(out_path, "a") as fout:
            for m, out in zip(batch_metas, outputs):
                common = m["common"]
                prefix_text = m["prefix_text"]
                existing_r_idx = m.get("existing_r_idx", set())
                for r_idx, comp in enumerate(out.outputs):
                    if r_idx in existing_r_idx:
                        # Already in file from a prior K=32 (or smaller) run; preserve as-is.
                        continue
                    if is_channel:
                        raw = (
                            f"<|channel|>analysis<|message|>{prefix_text}"
                            f"<|end|><|start|>assistant<|channel|>final<|message|>"
                            f"{comp.text}"
                        )
                    else:
                        raw = f"<think>\n{prefix_text}\n</think>\n\n{comp.text}"
                    row = dict(common)
                    row.update(
                        rollout_idx=r_idx,
                        response=comp.text.strip(),
                        raw_response=raw,
                        thinking_prefix=prefix_text,
                        seed=BASE_SEED,
                        skipped=False,
                        skip_reason="",
                    )
                    fout.write(json.dumps(row) + "\n")
        n_done += len(batch_prompts)
        print(f"[{model_key}/{split}] flushed batch {start}-{end} ({n_done}/{len(prompts)} prompts)", flush=True)

    total = len(prompts) * args.K + len(skipped_rows)
    print(f"[{model_key}/{split}] wrote {total} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["Qwen3-8B", "Olmo-3-7B-Think",
                            "Phi-4-reasoning", "GPT-OSS-20B"])
    p.add_argument("--split", required=True, choices=["harmful", "benign"])
    p.add_argument(
        "--transition-mode", default="segment", choices=["flip", "segment"]
    )
    p.add_argument("--L-pre", type=int, default=1)
    p.add_argument("--L-min", type=int, default=1)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--prompt-shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Flush rollouts every N prompts (resume-safe under TLE)")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Override model max_model_len (smaller = more KV concurrency)")
    p.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH),
                   help="Audit JSONL with per-trace 'labels' (per-chunk stance). "
                        "Default = GPT-5.4 single-annotator file. Override to "
                        "use majority-of-3 labels for the rebuttal/paper analysis.")
    p.add_argument("--out-suffix", default="",
                   help="Append a suffix to the output filename "
                        "(e.g. '_majority' -> rollouts_segment_majority.jsonl). "
                        "Use this when running with --audit-path != default to "
                        "avoid overwriting the GPT-5.4-only baseline rollouts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect cuts and print the task count per cell, but skip "
                        "loading the LLM and skip rollout generation. "
                        "Use to plan SLURM job sizing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
