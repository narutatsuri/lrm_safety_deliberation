#!/usr/bin/env python3
"""IAA n=500 re-run with NEW protocol — Opus 4.7 (Anthropic direct)."""
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
#   experiments/cot_audit/iaa_n500_new_opus.py
# Prompt template comes from the original cot_audit.prompts via ROOT; the model
# registry / 4-guardrail vote live in `lrm_safety_deliberation`, and the final κ table is
# produced by camera_ready/reproduce/tab_iaa.py (lrm_safety_deliberation.stats κ). Set
# LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from experiments.cot_audit.prompts import PROMPT_TMPL_NEW as PROMPT_TMPL, SYSTEM_PROMPT  # noqa: E402

import anthropic  # noqa: E402

PRICE_IN = 15.0 / 1e6
PRICE_OUT = 75.0 / 1e6
MODEL = "claude-opus-4-7"


def call_opus(client, row, retries=3):
    chunks = row["chunks"]
    formatted = "\n".join(f"  [{i}] {c}" for i, c in enumerate(chunks))
    user_msg = PROMPT_TMPL.format(
        model=row["cell"],
        n_chunks=len(chunks),
        trace=formatted,
    )
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = (resp.content[0].text if resp.content else "").strip()
            if not raw:
                raise ValueError("empty response (likely safety refusal)")
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            raw = re.sub(r"(?<=[,\[\s])\+(\d)", r"\1", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    raise
                parsed = json.loads(m.group(0))
            return {
                "cell": row["cell"], "base_model": row["base_model"],
                "split": row["split"], "global_id": int(row["global_id"]),
                "benchmark": row.get("benchmark", ""),
                "n_chunks": len(chunks),
                "labels": [int(x) for x in parsed["labels"]],
                "summary": str(parsed.get("summary", ""))[:500],
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
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
    ap.add_argument("--out", default=str(ROOT / "experiments/cot_audit/outputs/iaa_n500_new/opus47.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cost-cap-usd", type=float, default=40.0)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY not set (check .env)")
    client = anthropic.Anthropic(api_key=api_key)

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
    cap = args.cost_cap_usd
    cap_hit = False
    print(f"[cap] hard spend cap: ${cap:.2f}", flush=True)

    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(call_opus, client, r): r for r in todo}
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
                cost_so_far = in_toks * PRICE_IN + out_toks * PRICE_OUT
                if total % 20 == 0 or total == len(todo):
                    dt = time.time() - t0
                    print(f"  [{total}/{len(todo)}]  ok={n_ok} err={n_err}  "
                          f"rate={total/max(dt,1e-3):.2f}/s  cost=${cost_so_far:.4f}", flush=True)
                if cost_so_far > cap and not cap_hit:
                    cap_hit = True
                    print(f"  [CAP HIT] ${cost_so_far:.2f} > ${cap:.2f} — letting in-flight finish, no new submits", flush=True)
                    for f in futures:
                        if not f.done():
                            f.cancel()

    dt = time.time() - t0
    cost = in_toks * PRICE_IN + out_toks * PRICE_OUT
    print(f"\n[done] {n_ok} ok, {n_err} err in {dt:.1f}s  cost=${cost:.4f}  cap_hit={cap_hit}", flush=True)

    (out_path.parent / "opus47_cost.json").write_text(json.dumps({
        "model": MODEL, "n_ok": n_ok, "n_err": n_err,
        "input_tokens": in_toks, "output_tokens": out_toks, "cost_usd": cost,
        "cap_hit": cap_hit,
    }, indent=2))


if __name__ == "__main__":
    main()
