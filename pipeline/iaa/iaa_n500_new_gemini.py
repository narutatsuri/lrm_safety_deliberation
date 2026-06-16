#!/usr/bin/env python3
"""IAA n=500 re-run with NEW protocol (PROMPT_TMPL + SYSTEM_PROMPT) — Gemini-3-Pro.

Mirrors annotate_gemini_batched.py but imports the shared template from
experiments.cot_audit.prompts, strips the user-Q line, and applies the system
prompt (was previously Opus-only).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

# Provenance: copied verbatim (logic unchanged) from the original research repo
#   experiments/cot_audit/iaa_n500_new_gemini.py
# Prompt template comes from the original cot_audit.prompts via ROOT; the model
# registry / 4-guardrail vote live in `lrm_safety_deliberation`, and the final κ table is
# produced by camera_ready/reproduce/tab_iaa.py (lrm_safety_deliberation.stats κ). Set
# LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.prompts import PROMPT_TMPL_NEW as PROMPT_TMPL, SYSTEM_PROMPT  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

PRICE_IN = 1.25 / 1e6
PRICE_OUT = 10.0 / 1e6
MODEL = "gemini-3-pro-preview"


def call_gemini(client, row, retries=3):
    chunks = row["chunks"]
    formatted = "\n".join(f"  [{i}] {c}" for i, c in enumerate(chunks))
    user_msg = PROMPT_TMPL.format(
        model=row["cell"],
        n_chunks=len(chunks),
        trace=formatted,
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    raise
                parsed = json.loads(m.group(0))
            labels = [int(x) for x in parsed["labels"]]
            summary = str(parsed.get("summary", ""))[:500]
            usage = getattr(resp, "usage_metadata", None)
            return {
                "cell": row["cell"], "base_model": row["base_model"],
                "split": row["split"], "global_id": int(row["global_id"]),
                "benchmark": row.get("benchmark", ""),
                "n_chunks": len(chunks),
                "labels": labels,
                "summary": summary,
                "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "error": str(e)[:300],
                    "cell": row["cell"], "base_model": row["base_model"],
                    "split": row["split"], "global_id": int(row["global_id"]),
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT / "experiments/cot_audit/data/sample_n500_traces.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "experiments/cot_audit/outputs/iaa_n500_new/gemini.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    rows = [json.loads(l) for l in open(args.sample)]
    print(f"[load] sample: {len(rows)} traces", flush=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                d = json.loads(ln)
                if "labels" in d:
                    done.add((d["cell"], d["split"], int(d["global_id"])))
            except Exception:
                pass
    todo = [r for r in rows if (r["cell"], r["split"], int(r["global_id"])) not in done]
    print(f"[resume] done: {len(done)}  todo: {len(todo)}", flush=True)
    if not todo:
        return

    lock = Lock()
    t0 = time.time()
    n_ok = 0; n_err = 0; in_toks = 0; out_toks = 0

    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(call_gemini, client, r): r for r in todo}
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                fout.write(json.dumps(res) + "\n"); fout.flush()
                if "error" in res:
                    n_err += 1
                else:
                    n_ok += 1
                    in_toks += res.get("input_tokens", 0)
                    out_toks += res.get("output_tokens", 0)
                total = n_ok + n_err
                if total % 25 == 0 or total == len(todo):
                    dt = time.time() - t0
                    cost = in_toks * PRICE_IN + out_toks * PRICE_OUT
                    print(f"  [{total}/{len(todo)}]  ok={n_ok} err={n_err}  "
                          f"rate={total/max(dt,1e-3):.2f}/s  cost=${cost:.4f}", flush=True)

    dt = time.time() - t0
    cost = in_toks * PRICE_IN + out_toks * PRICE_OUT
    print(f"\n[done] {n_ok} ok, {n_err} err in {dt:.1f}s  cost=${cost:.4f}", flush=True)

    (out_path.parent / "gemini_cost.json").write_text(json.dumps({
        "model": MODEL, "n_ok": n_ok, "n_err": n_err,
        "input_tokens": in_toks, "output_tokens": out_toks, "cost_usd": cost,
    }, indent=2))


if __name__ == "__main__":
    main()
