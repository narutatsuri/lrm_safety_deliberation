#!/usr/bin/env python3
"""
Label per-chunk stance (-1 refuse / 0 neutral / +1 comply) on defense-trained
model thinking traces using GPT-5.4. Mirrors run_full_audit.py exactly:
same chunker, same prompt template, same model.

Per cell × split, samples N traces matched to the global_ids that already exist
in experiments/cot_audit/eval_results/stance_audit_full/labeled_traces.jsonl for the corresponding
base model. Resume-safe: appends to a single jsonl, skips (cell, split, gid)
already present.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

random.seed(0)

PRICE_IN = 1.25 / 1e6
PRICE_OUT = 10.0 / 1e6

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/cot_audit/label_defense_stances.py
# Bundled here because sample_traces_n500.py reuses chunk_trace / extract_thinking.
# The model registry + 4-guardrail vote now also live in the shared `lrm_safety_deliberation`
# library (lrm_safety_deliberation.models / lrm_safety_deliberation.guardrails). Set LRM_SAFETY_ARTIFACTS to
# point at the artifact tree (which carries experiments/cot_audit/prompts.py).
# Canonical template + system prompt — single source of truth.
# Re-exported for backwards-compat with scripts that did:
#     from experiments.cot_audit.label_defense_stances import PROMPT_TMPL
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.prompts import PROMPT_TMPL, SYSTEM_PROMPT  # noqa: E402,F401

# Map cell name -> (display, base-model display in stance_audit_full, defense gen path).
CELLS = {
    "STAR1-Qwen3-8B":       ("STAR1-Qwen3-8B",       "Qwen3-8B"),
    "SafeKey-Qwen3-8B":     ("SafeKey-Qwen3-8B",     "Qwen3-8B"),
    "R1ACT-Qwen3-8B":       ("R1ACT-Qwen3-8B",       "Qwen3-8B"),
    "ThinkSafe-Qwen3-8B":   ("ThinkSafe-Qwen3-8B",   "Qwen3-8B"),
    "RAPO-Qwen3-8B":        ("RAPO-Qwen3-8B",        "Qwen3-8B"),
    "STAR1-Olmo-3-7B-Think":     ("STAR1-Olmo-3-7B-Think",     "Olmo3-7B-Think"),
    "SafeKey-Olmo-3-7B-Think":   ("SafeKey-Olmo-3-7B-Think",   "Olmo3-7B-Think"),
    "R1ACT-Olmo-3-7B-Think":     ("R1ACT-Olmo-3-7B-Think",     "Olmo3-7B-Think"),
    "ThinkSafe-Olmo-3-7B-Think": ("ThinkSafe-Olmo-3-7B-Think", "Olmo3-7B-Think"),
    "RAPO-Olmo-3-7B-Think":      ("RAPO-Olmo-3-7B-Think",      "Olmo3-7B-Think"),
    "R1ACT-Phi-4-reasoning":     ("R1ACT-Phi-4-reasoning",     "Phi-4-reasoning"),
    # Cells added 2026-05-04: new defenses + GPT-OSS-20B base
    "STAR1-Phi-4-reasoning":     ("STAR1-Phi-4-reasoning",     "Phi-4-reasoning"),
    "STAIR-Qwen3-8B":            ("STAIR-Qwen3-8B",            "Qwen3-8B"),
    "STAIR-Phi-4-reasoning":     ("STAIR-Phi-4-reasoning",     "Phi-4-reasoning"),
    "ThinkSafe-Phi-4-reasoning": ("ThinkSafe-Phi-4-reasoning", "Phi-4-reasoning"),
    "ThinkSafe-GPT-OSS-20B":     ("ThinkSafe-GPT-OSS-20B",     "GPT-OSS-20B"),
}

# Map from script-level split name to defenses_eval __fullpipe__ split dir name.
SPLIT_TO_FULLPIPE = {"harmful": "asr", "benign": "orr"}

_OSS_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)",
    re.DOTALL,
)


def extract_thinking(raw: str, is_channel: bool = False, response: str = "") -> str:
    """Extract the thinking-trace span from a raw model output.

    Handles three formats:
      - GPT-OSS Harmony: `<|channel|>analysis<|message|>...<|end|>`
      - Qwen3 / Phi-4: `<think>...</think>` then response.
      - Olmo / implicit: no tags. The thinking is the prefix of raw_response
        and the assistant final answer is the `response` field; recover by
        stripping the response suffix.
    """
    if not raw:
        return ""
    if is_channel or "<|channel|>analysis<|message|>" in raw:
        m = _OSS_ANALYSIS_RE.search(raw)
        return m.group(1).strip() if m else ""
    if "<think>" in raw:
        if "</think>" in raw:
            return raw.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        return raw.split("<think>", 1)[1].strip()
    if response:
        if raw.endswith(response):
            return raw[:-len(response)].strip()
        if response in raw:
            return raw.split(response, 1)[0].strip()
    # Olmo prefills <think> in the PROMPT, so generated text can carry the closing
    # </think> with no opening tag. When the final answer is empty (refuse-in-thinking,
    # common for R1-ACT-Olmo), the response-fallback above yields nothing — recover the
    # thinking as everything before </think>. Fires only on the previously-"" case, so
    # extraction is unchanged wherever the branches above already succeeded.
    if "</think>" in raw:
        return raw.split("</think>", 1)[0].strip()
    return ""


def chunk_trace(text, max_len=200):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= max_len:
            cur += s + " "
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 20]


def load_base_gids(stance_path, base_display):
    """Return {(split, gid): True} for traces already labeled on this base model."""
    out = {}
    with open(stance_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("model") != base_display or "labels" not in d:
                continue
            out[(d["split"], int(d["global_id"]))] = True
    return out


def load_cell_gens(cell_root, split):
    """Return list of dicts (one per generation row) keyed by global_id.

    Tries the legacy path first
        <cell_root>/generations/base/<split>.jsonl
    then falls back to the canonical __fullpipe__ layout
        <cell_root>/__fullpipe__/<asr|orr>/<basename(cell)>/generations.jsonl
    extracting `thinking_trace` from `raw_response` when absent.
    """
    rows = {}
    legacy = Path(cell_root) / "generations" / "base" / f"{split}.jsonl"
    candidates = [legacy]
    fullpipe_split = SPLIT_TO_FULLPIPE.get(split, split)
    cell_basename = Path(cell_root).name
    candidates.append(
        Path(cell_root) / "__fullpipe__" / fullpipe_split / cell_basename / "generations.jsonl"
    )
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return rows
    is_channel_cell = "GPT-OSS" in cell_basename
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            gid = d.get("global_id")
            if gid is None:
                continue
            tt = (d.get("thinking_trace") or "").strip()
            if not tt:
                tt = extract_thinking(
                    d.get("raw_response", "") or "",
                    is_channel=is_channel_cell,
                    response=d.get("response", "") or "",
                )
            if len(tt) < 50:
                continue
            d["thinking_trace"] = tt
            rows[int(gid)] = d
    return rows


def call_gpt(cell_display, base_display, trace_data, chunks, retries=3):
    prompt_text = trace_data.get("prompt", "")[:200]
    formatted = "\n".join(f"  [{i}] {c}" for i, c in enumerate(chunks))
    user_msg = PROMPT_TMPL.format(
        model=cell_display,
        prompt=prompt_text,
        n_chunks=len(chunks),
        trace=formatted,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "user", "content": user_msg}],
                temperature=0,
                max_completion_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)
            return {
                "cell": cell_display,
                "base_model": base_display,
                "split": trace_data["__split__"],
                "global_id": int(trace_data["global_id"]),
                "benchmark": trace_data.get("benchmark", ""),
                "prompt": prompt_text,
                "n_chunks": len(chunks),
                "labels": result["labels"],
                "summary": result.get("summary", ""),
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "error": str(e),
                    "cell": cell_display,
                    "base_model": base_display,
                    "split": trace_data["__split__"],
                    "global_id": int(trace_data["global_id"]),
                }


def load_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "error" not in d:
                        done.add((d["cell"], d["split"], int(d["global_id"])))
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS.keys()))
    ap.add_argument("--n-per-cell", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument(
        "--defenses-root",
        default="experiments/cot_audit/eval_results/generations/defense",
    )
    ap.add_argument(
        "--stance-base",
        default="experiments/cot_audit/eval_results/stance_audit_full/labeled_traces.jsonl",
    )
    ap.add_argument("--out-dir", default="experiments/cot_audit/eval_results/defenses_stance")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labeled_traces.jsonl"
    cost_path = out_dir / "cost_log.json"

    # Build work list, matched to existing base gids.
    all_work = []
    base_gid_cache = {}
    for cell_key in args.cells:
        if cell_key not in CELLS:
            print(f"  ! unknown cell {cell_key}, skipping")
            continue
        cell_display, base_display = CELLS[cell_key]
        cell_root = Path(args.defenses_root) / cell_key
        if base_display not in base_gid_cache:
            base_gid_cache[base_display] = load_base_gids(args.stance_base, base_display)
        base_gids = base_gid_cache[base_display]
        for split in ("harmful", "benign"):
            cand_keys = sorted(g for (s, g) in base_gids if s == split)
            gens = load_cell_gens(cell_root, split)
            kept = []
            for gid in cand_keys:
                if gid not in gens:
                    continue
                d = gens[gid]
                ch = chunk_trace(d["thinking_trace"])
                if len(ch) >= 3:
                    kept.append((d, ch))
                if len(kept) == args.n_per_cell:
                    break
            if len(kept) < args.n_per_cell:
                for gid in cand_keys:
                    if gid not in gens:
                        continue
                    d = gens[gid]
                    if any(d is k[0] for k in kept):
                        continue
                    ch = chunk_trace(d["thinking_trace"])
                    if len(ch) >= 2:
                        kept.append((d, ch))
                    if len(kept) == args.n_per_cell:
                        break
            print(
                f"  cell={cell_key:32s} split={split:7s} "
                f"matched={len(kept)} (base pool={len(cand_keys)})"
            )
            for td, chunks in kept[: args.n_per_cell]:
                td["__split__"] = split
                all_work.append((cell_display, base_display, td, chunks))

    total = len(all_work)
    print(f"\nTotal work items: {total}")

    done = load_done(out_path)
    work = [
        (c, b, td, ch)
        for c, b, td, ch in all_work
        if (c, td["__split__"], int(td["global_id"])) not in done
    ]
    print(f"Already done: {len(done)}  Remaining: {len(work)}\n")
    sys.stdout.flush()

    if not work:
        print("Nothing to do.")
        return

    completed = len(done)
    errors = 0
    total_in = 0
    total_out = 0
    t0 = time.time()
    write_lock = Lock()

    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(call_gpt, c, b, td, ch): (c, td) for c, b, td, ch in work
            }
            for fut in as_completed(futures):
                result = fut.result()
                with write_lock:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    completed += 1
                    if "error" in result:
                        errors += 1
                        print(
                            f"  ✗ {result.get('cell','?')} {result.get('split','?')} "
                            f"gid={result.get('global_id','?')}: {result['error'][:60]}"
                        )
                    else:
                        total_in += result["input_tokens"]
                        total_out += result["output_tokens"]
                    if completed % 50 == 0 or completed == total + len(done):
                        elapsed = time.time() - t0
                        cost = total_in * PRICE_IN + total_out * PRICE_OUT
                        rate = (completed - len(done)) / max(elapsed, 1e-3)
                        remaining = (total + len(done)) - completed
                        eta = remaining / rate if rate > 0 else 0
                        print(
                            f"  [{completed:4d}/{total + len(done)}]  "
                            f"cost=${cost:.3f}  errors={errors}  "
                            f"{rate:.1f} req/s  ETA {eta/60:.1f}m"
                        )
                        sys.stdout.flush()

    elapsed = time.time() - t0
    cost = total_in * PRICE_IN + total_out * PRICE_OUT
    print(f"\nDone in {elapsed/60:.1f}m  |  cost=${cost:.4f}  |  errors={errors}")

    with open(cost_path, "w") as f:
        json.dump(
            {
                "total_cost_usd": cost,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "n_labeled": completed - len(done),
                "n_errors": errors,
                "elapsed_s": elapsed,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
