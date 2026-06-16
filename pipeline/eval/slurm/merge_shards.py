#!/usr/bin/env python3
"""Merge row-slice shard outputs for one (model, split) cell into a canonical
non-sharded summary, recomputing the 4-guardrail majority-vote metrics.

Shard dirs are written by the pipeline as:
    eval_results/<tag>/<split>/<MODEL>__shard_<start>_<end>__/{generations,results}.jsonl

Output (canonical cell):
    eval_results/<tag>/<split>/<MODEL>/{generations,results,summary}.jsonl

Verifies that the shards tile [0, expected_total) exactly with no gap/overlap.

Usage: python merge_shards.py <MODEL> <split> [--tag asr_orr_16k]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.full_eval_pipeline.pipeline import FullEvalPipeline
from evaluation.full_eval_pipeline.datasets import build_split_records

SHARD_RE = re.compile(r"__shard_(\d+)_(\d+)__$")


def read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("split")
    ap.add_argument("--tag", default="asr_orr_16k")
    ap.add_argument("--num_rollouts", type=int, default=1)
    args = ap.parse_args()

    K = max(1, args.num_rollouts)
    split_root = ROOT / "eval_results" / args.tag / args.split
    n_prompts = len(build_split_records(args.split))
    expected_total = n_prompts * K  # K rows per prompt

    shard_dirs = []
    for d in split_root.glob(f"{args.model}__shard_*__"):
        m = SHARD_RE.search(d.name)
        if m:
            shard_dirs.append((int(m.group(1)), int(m.group(2)), d))
    shard_dirs.sort()
    if not shard_dirs:
        print(f"ERROR: no shard dirs for {args.model}/{args.split} under {split_root}")
        return 2

    # Verify exact tiling of the prompt range [0, n_prompts) (shard bounds are prompt indices).
    cursor = 0
    for start, end, d in shard_dirs:
        if start != cursor:
            print(f"ERROR: gap/overlap — expected start {cursor}, got {start} ({d.name})")
            return 3
        cursor = end
    if cursor != n_prompts:
        print(f"ERROR: shards cover prompts [0,{cursor}) but expected {n_prompts}")
        return 3

    merged_gen: list[dict] = []
    merged_res: list[dict] = []
    for start, end, d in shard_dirs:
        res_p = d / "results.jsonl"
        gen_p = d / "generations.jsonl"
        if not res_p.exists():
            print(f"ERROR: missing {res_p} (shard not classified yet?)")
            return 4
        res = read_jsonl(res_p)
        if len(res) != (end - start) * K:
            print(f"ERROR: {d.name} has {len(res)} rows, expected {(end - start) * K}")
            return 4
        merged_res.extend(res)
        if gen_p.exists():
            merged_gen.extend(read_jsonl(gen_p))

    if len(merged_res) != expected_total:
        print(f"ERROR: merged {len(merged_res)} rows != expected {expected_total}")
        return 5

    out_dir = split_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in merged_res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if merged_gen:
        with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
            for r in merged_gen:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Recompute summary (no GPU; _summarize only needs split + rows + num_rollouts).
    pipe = FullEvalPipeline(split=args.split, model_path=str(ROOT / "models" / args.model), num_rollouts=K)
    summary = pipe._summarize(merged_res)
    summary["merged_from_shards"] = [d.name for _, _, d in shard_dirs]
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    mk = "majority_asr" if args.split.startswith("asr") else "majority_orr"
    print(f"OK {args.model}/{args.split}: n={summary['n_total']} ({summary['n_prompts']} prompts x{K}) "
          f"{mk}={summary[mk]:.4f} ±{summary[mk+'_std']:.4f} "
          f"({len(shard_dirs)} shards) -> {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
