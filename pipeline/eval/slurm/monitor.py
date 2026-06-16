#!/usr/bin/env python3
"""Self-contained monitor for the sharded ASR/ORR fleet (K=1 or K=4). Runs detached;
loops until all cells merge (or timeout/fatal). Handles: classification tripwire,
resubmit of failed shards (once), per-cell merge, final report.

Config via env:
  MON_TAG       output tag (default asr_orr_16k)
  MON_MANIFEST  dispatched manifest filename (default dispatched_sharded.tsv)
  MON_K         rollouts per prompt (default 1)
"""
from __future__ import annotations
import json, os, subprocess, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", "/scratch/gpfs/ARORA/nr3764/inference_skill_composition"))
TAG = os.environ.get("MON_TAG", "asr_orr_16k")
MANIFEST = ROOT / "scripts/slurm/asr_orr_16k" / os.environ.get("MON_MANIFEST", "dispatched_sharded.tsv")
K = int(os.environ.get("MON_K", "1"))
NAME_PREFIX = MANIFEST.read_text().split("\t")[1].split("_")[0] if MANIFEST.exists() else "asr16k"
REPORT = ROOT / "scripts/slurm/asr_orr_16k" / f"FINAL_REPORT_{TAG}.md"
VENV = ROOT / ".venv/bin/python"
POLL, TIMEOUT = 120, 4 * 3600
ASR_BENCH = ["advbench", "harmbench", "strongreject", "sorrybench", "jailbreakbench", "wildjailbreak", "fortress", "hexphi"]
ORR_BENCH = ["or_bench", "falsereject", "coconot", "xstest_safe"]


def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def load_manifest():
    out = []
    for line in MANIFEST.read_text().splitlines():
        jid, name, model, split, start, end = line.split("\t")
        out.append(dict(name=name, model=model, split=split, start=int(start), end=int(end)))
    return out


def shard_dir(s):
    return ROOT / "eval_results" / TAG / s["split"] / f'{s["model"]}__shard_{s["start"]}_{s["end"]}__'


def nlines(p):
    return sum(1 for ln in p.open() if ln.strip()) if p.exists() else 0


def shard_done(s):
    return nlines(shard_dir(s) / "results.jsonl") == (s["end"] - s["start"]) * K


def active_names():
    return {n for n in sh("squeue --me -h -o '%j'").splitlines() if n.startswith(NAME_PREFIX)}


def resubmit(s):
    mem = "80G" if s["model"] == "GPT-OSS-20B" else "64G"
    seed_env = "NUM_ROLLOUTS=%d SEED=0 " % K if K > 1 else ""
    cmd = (f'sbatch --parsable --job-name="{s["name"]}" --output=".slurm/{s["name"]}_%j.out" '
           f'--partition=pli-c --account=arora --qos=pli-c --nodes=1 --ntasks=1 --gres=gpu:1 '
           f'--cpus-per-gpu=4 --mem={mem} --time=03:30:00 '
           f'--wrap="cd {ROOT} && ROW_SLICE={s["start"]}:{s["end"]} {seed_env}OUT_TAG={TAG} '
           f'bash scripts/slurm/asr_orr_16k/run_one.sh {s["model"]} {s["split"]}"')
    return sh(f"cd {ROOT} && {cmd}")


def main():
    shards = load_manifest()
    cells = defaultdict(list)
    for s in shards:
        cells[(s["model"], s["split"])].append(s)
    log, tripwire, resub, missing, merged, t0 = [], None, {}, defaultdict(int), {}, time.time()

    while True:
        names = active_names()
        if tripwire is None:
            for s in shards:
                if shard_done(s):
                    try:
                        summ = json.loads((shard_dir(s) / "summary.json").read_text())
                        need = {"wildguard", "qwen3guard", "granite_guardian", "oss_safeguard"}
                        ok = need.issubset(set(summ.get("guardrails", {}))) and summ.get("per_benchmark")
                        if K > 1:
                            ok = ok and summ.get("num_rollouts") == K
                        tripwire = True if ok else f"FAIL on {s['name']}: guardrails/per_bench/K bad"
                    except Exception as e:
                        tripwire = f"FAIL reading {s['name']}: {e}"
                    break
            if isinstance(tripwire, str):
                sh(f"squeue --me -h -o '%j %i' | awk '/{NAME_PREFIX}/{{print $2}}' | xargs -r scancel")
                log.append(f"FATAL tripwire {tripwire}; cancelled fleet."); break
            elif tripwire is True:
                log.append("cls tripwire PASS")

        for s in shards:
            if shard_done(s):
                missing[s["name"]] = 0
                continue
            if s["name"] not in names:
                missing[s["name"]] += 1
                if missing[s["name"]] >= 2 and resub.get(s["name"], 0) < 1:
                    jid = resubmit(s); resub[s["name"]] = 1; missing[s["name"]] = 0
                    log.append(f"resubmitted {s['name']} -> {jid}")
            else:
                missing[s["name"]] = 0

        for key, cs in cells.items():
            if key in merged or not all(shard_done(s) for s in cs):
                continue
            model, split = key
            r = subprocess.run(
                f"cd {ROOT} && {VENV} scripts/slurm/asr_orr_16k/merge_shards.py {model} {split} --tag {TAG} --num_rollouts {K}",
                shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                merged[key] = r.stdout.strip(); log.append(f"merged {model}/{split}: {r.stdout.strip()}")
            else:
                log.append(f"MERGE ERR {model}/{split}: {r.stdout.strip()} {r.stderr.strip()[:200]}")

        if len(merged) == len(cells):
            log.append("all cells merged."); break
        if time.time() - t0 > TIMEOUT:
            log.append(f"TIMEOUT; merged {len(merged)}/{len(cells)}"); break
        time.sleep(POLL)

    write_report(cells, merged, log, time.time() - t0)


def write_report(cells, merged, log, elapsed):
    models = ["Qwen3-8B", "Olmo-3-7B-Think", "Phi-4-reasoning", "GPT-OSS-20B"]
    summ = {}
    for split in ("asr_full", "orr_full"):
        for m in models:
            p = ROOT / "eval_results" / TAG / split / m / "summary.json"
            if p.exists():
                summ[(m, split)] = json.loads(p.read_text())
    L = [f"# {TAG} final report (K={K})", "", f"Monitor wall-clock: {elapsed/60:.1f} min", "", "## Event log"]
    L += [f"- {x}" for x in log] + ["", "## Headline (majority 4-guardrail vote; mean +/- std over K rollouts)"]
    L += ["| Model | ASR n | majority ASR | ORR n | majority ORR |", "|---|---|---|---|---|"]
    def cell(d, key):
        if not d: return "—"
        return f'{d[key]*100:.1f} ± {d.get(key+"_std",0)*100:.1f}'
    for m in models:
        a, o = summ.get((m, "asr_full")), summ.get((m, "orr_full"))
        L.append(f"| {m} | {a['n_total'] if a else '—'} | {cell(a,'majority_asr')} | {o['n_total'] if o else '—'} | {cell(o,'majority_orr')} |")
    for title, split, key, benches in [("ASR", "asr_full", "majority_asr", ASR_BENCH), ("ORR", "orr_full", "majority_orr", ORR_BENCH)]:
        L += ["", f"## Per-benchmark {title} (mean ± std %)", "| Model | " + " | ".join(benches) + " |", "|---|" + "---|" * len(benches)]
        for m in models:
            d = summ.get((m, split))
            cells_pb = []
            for b in benches:
                pb = d["per_benchmark"].get(b) if d else None
                cells_pb.append(f'{pb[key]*100:.1f}±{pb.get(key+"_std",0)*100:.1f}' if pb else "—")
            L.append(f"| {m} | " + " | ".join(cells_pb) + " |")
    L += ["", f"Canonical: eval_results/{TAG}/{{asr_full,orr_full}}/<MODEL>/summary.json"]
    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
