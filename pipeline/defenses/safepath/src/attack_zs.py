"""
Zero-shot SafePath inference wrapper.

Implements the inference-time variant from SAFEPATH by prefixing:
  <think>\nLet's think about safety first.
immediately after the assistant generation prompt, while leaving <think> open.

This is an inference-only intervention (no finetuning) and is implemented
against an OpenAI-compatible local endpoint (vLLM).

This is a DEFENSE runner (decode-time wrapper); the `attack`-prefixed filename
is historical only. PROVENANCE: original path defenses/safepath/src/attack_zs.py.
The base-model registry now lives in lrm_safety_deliberation.models; eval is pipeline/eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from transformers import AutoTokenizer

import sys as _sys
_here = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..", "_common")))
from m4_sampling import (  # noqa: E402
    apply_sampling_preset,
    load_gid_filter,
    filter_dataset_by_gids,
    split_gptoss,
)


SAFETY_PRIMER = "Let's think about safety first."


def project_root_from_here() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # defenses/safepath/src -> project root
    return os.path.abspath(os.path.join(this_dir, "..", "..", ".."))


def load_dataset(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def apply_chat_template_safe(tokenizer, messages, *, enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    name_or_path = getattr(tokenizer, "name_or_path", "") or ""
    if "qwen" in name_or_path.lower():
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        # Fallback for tokenizers without a chat template
        user_text = messages[0]["content"]
        return f"User: {user_text}\nAssistant:"


def split_thinking_and_response(text: str) -> tuple[str, str]:
    if text is None:
        return "", ""
    if not isinstance(text, str):
        text = str(text)

    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()

    if "</think>" in text:
        left, right = text.split("</think>", 1)
        return left.strip(), right.strip()

    # No </think> in the model output. The prompt prefills `<think>\n...`,
    # so the generation IS the thinking continuation. Treat the whole text
    # as truncated thinking and leave the response empty so guardrails see
    # a clean signal instead of a noisy analysis dump.
    return text.strip(), ""


def maybe_force_close(
    raw_text: str,
    base_prompt: str,
    *,
    client,
    model_name: str,
    tokenizer,
    is_gpt_oss: bool,
    think_token_cap: int = 16384,
    response_max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = None,
    seed: int | None = None,
) -> tuple[str, str, bool]:
    """Apply the max-16K-thinking + force-close rule.

    If raw_text already closes thinking (</think> for `<think>`-format models,
    a final-channel marker for GPT-OSS), parse normally.

    Otherwise truncate raw_text to think_token_cap tokens, append the closure
    marker, and request a short response generation from the model.

    Returns (thinking, response, was_force_closed).
    """
    if is_gpt_oss:
        has_closure = (
            "assistantfinal" in raw_text
            or "<|channel|>final|>" in raw_text
            or "<|channel|>final<|message|>" in raw_text
        )
    else:
        has_closure = "</think>" in raw_text

    if has_closure:
        if is_gpt_oss:
            thinking, response = split_gptoss(raw_text)
        else:
            thinking, response = split_thinking_and_response(raw_text)
        return thinking, response, False

    thinking = raw_text or ""
    if tokenizer is not None and thinking:
        toks = tokenizer.encode(thinking, add_special_tokens=False)
        if len(toks) > think_token_cap:
            toks = toks[:think_token_cap]
            thinking = tokenizer.decode(toks, skip_special_tokens=False)

    close_str = (
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        if is_gpt_oss
        else "</think>\n"
    )
    force_prompt = base_prompt + thinking + close_str

    extra_body: dict = {}
    if top_k is not None and top_k > 0:
        extra_body["top_k"] = int(top_k)
    if seed is not None:
        extra_body["seed"] = int(seed)

    response = ""
    for attempt in range(3):
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=force_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=response_max_tokens,
                extra_body=extra_body if extra_body else None,
            )
            response = (resp.choices[0].text or "").strip()
            if is_gpt_oss:
                response = re.sub(r"<\|[^|]+\|>", "", response).strip()
            break
        except Exception as e:
            print(f"  [force-close] attempt {attempt + 1}/3 failed: {e}")
            time.sleep(2)

    return thinking, response, True


def resolve_tokenizer_path(target_model: str, tokenizer_path: str | None) -> str:
    if tokenizer_path:
        return tokenizer_path

    project_root = project_root_from_here()
    candidate = os.path.join(project_root, "models", target_model)
    if os.path.exists(candidate):
        return candidate
    return target_model


def normalize_to_open_think(base_prompt: str, think_open: str) -> str:
    """
    Normalize prompt so generation starts at one open think block.

    Some chat templates already emit `<think> ... </think>` placeholders. For
    SafePath zero-shot prefill we need to continue from an *open* `<think>`.
    """
    if "<think>" in base_prompt:
        prefix = base_prompt.split("<think>", 1)[0]
        return f"{prefix}{think_open}"
    return f"{base_prompt}{think_open}"


def build_prompt(
    *,
    tokenizer,
    goal: str,
    think_open: str,
    primer_text: str,
    enable_thinking_template: bool,
) -> str:
    base_prompt = apply_chat_template_safe(
        tokenizer,
        [{"role": "user", "content": goal}],
        enable_thinking=enable_thinking_template,
    )
    open_think_prompt = normalize_to_open_think(base_prompt, think_open)
    return f"{open_think_prompt}{primer_text}"


def make_target_fn(
    *,
    base_url: str,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_tokens: int,
    tokenizer,
    is_gpt_oss: bool,
    think_token_cap: int = 16384,
    force_close_max_tokens: int = 2048,
    seed: int | None = None,
) -> callable:
    client = OpenAI(api_key="dummy", base_url=base_url)
    extra_body = {}
    if top_k is not None and top_k > 0:
        extra_body["top_k"] = top_k
    if seed is not None:
        extra_body["seed"] = int(seed)

    def call_target(prompt: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                resp = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                )
                raw_text = (resp.choices[0].text or "").strip()
                thinking, response, was_forced = maybe_force_close(
                    raw_text,
                    prompt,
                    client=client,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    is_gpt_oss=is_gpt_oss,
                    think_token_cap=think_token_cap,
                    response_max_tokens=force_close_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
                return {
                    "raw_generation": raw_text,
                    "thinking_trace": thinking,
                    "response": response,
                    "force_closed": was_forced,
                }
            except Exception as e:
                print(f"  [target] attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        return {
            "raw_generation": "",
            "thinking_trace": "",
            "response": "",
            "force_closed": False,
            "error": "target_call_failed",
        }

    return call_target


def load_completed_sample_ids(path: str) -> set[int]:
    completed: set[int] = set()
    if not os.path.exists(path):
        return completed

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            has_error = bool(rec.get("error"))
            response = rec.get("target_response", "")
            has_response = isinstance(response, str) and bool(response.strip())
            if not has_error and has_response and "sample_id" in rec:
                completed.add(rec["sample_id"])
    return completed


def run(
    *,
    tasks: list[dict],
    target_fn: callable,
    output_path: str,
    max_workers: int,
    save_raw_generation: bool,
    rollout_idx: int = 0,
    seed: int | None = None,
) -> int:
    total = len(tasks)
    done = 0
    lock = threading.Lock()

    with open(output_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(target_fn, task["attack_prompt"]): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "raw_generation": "",
                        "thinking_trace": "",
                        "response": "",
                        "error": f"task_exception: {e}",
                    }

                record = {
                    "sample_id": task["sample_id"],
                    "global_id": task.get("global_id", task["sample_id"]),
                    "benchmark": task.get("benchmark", ""),
                    "rollout_idx": int(rollout_idx),
                    "seed": int(seed) if seed is not None else None,
                    "goal": task["goal"],
                    "category": task["category"],
                    "attack_prompt": task["attack_prompt"],
                    "safety_primer": task["safety_primer"],
                    "target_response": result.get("response", ""),
                    "target_thinking_trace": result.get("thinking_trace", ""),
                    "force_closed": bool(result.get("force_closed", False)),
                }
                if save_raw_generation:
                    record["raw_target_generation"] = result.get("raw_generation", "")
                if result.get("error"):
                    record["error"] = result["error"]

                with lock:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    done += 1

                if done % 10 == 0 or done == total:
                    print(f"  Progress: {done}/{total}")

    return done


def main():
    parser = argparse.ArgumentParser(description="Zero-shot SafePath prefix-injection wrapper")
    parser.add_argument("--dataset", required=True, help="JSONL with at least 'goal' field")
    parser.add_argument("--target-url", default="http://localhost:8001/v1", help="OpenAI-compatible target URL")
    parser.add_argument("--target-model", default="Qwen3-8B", help="Target model name as served by endpoint")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path/ID; default models/<target-model>")
    parser.add_argument("--think-open", default="<think>\n", help="Thinking block opener")
    parser.add_argument("--primer-text", default=SAFETY_PRIMER, help="SafePath safety primer text")
    parser.add_argument(
        "--enable-thinking-template",
        action="store_true",
        default=False,
        help="Pass enable_thinking=True into tokenizer chat template for Qwen tokenizers",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50, help="<=0 disables top-k")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Phase-1 max generation tokens (also the thinking cap for force-close).")
    parser.add_argument("--force-close-max-tokens", type=int, default=2048, help="Max tokens for force-close response generation.")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent API requests")
    parser.add_argument("--save-raw-generation", action="store_true", default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--gid-list", default=None, help="JSON file with list of int global_ids to keep")
    parser.add_argument("--rollout-idx", type=int, default=0, help="Rollout index (0..M-1); used as seed offset and naming tag")
    parser.add_argument("--sampling-preset", default=None,
                        choices=["Qwen3-8B","Olmo-3-7B-Think","Phi-4-reasoning","GPT-OSS-20B"],
                        help="Override --temperature/--top-p/--top-k with each model's recommended defaults")
    parser.add_argument("--seed-base", type=int, default=10000, help="Base seed; rollout uses seed_base + rollout_idx")
    args = parser.parse_args()
    apply_sampling_preset(args, args.sampling_preset)

    project_root = project_root_from_here()
    if args.output is None:
        args.output = os.path.join(project_root, "results", "safepath_zs", f"{args.target_model}.jsonl")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = load_dataset(args.dataset)
    if args.end is not None:
        dataset = dataset[args.start:args.end]
    elif args.start > 0:
        dataset = dataset[args.start:]
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    gid_set = load_gid_filter(args.gid_list)
    dataset = filter_dataset_by_gids(dataset, gid_set)
    if gid_set is not None:
        print(f"Filtered to {len(dataset)} rows matching --gid-list ({len(gid_set)} gids requested)")

    tokenizer_path = resolve_tokenizer_path(args.target_model, args.tokenizer_path)
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    tasks: list[dict] = []
    for i, row in enumerate(dataset):
        goal = row.get("goal", row.get("prompt", row.get("ptq", "")))
        prompt = build_prompt(
            tokenizer=tokenizer,
            goal=goal,
            think_open=args.think_open,
            primer_text=args.primer_text,
            enable_thinking_template=args.enable_thinking_template,
        )
        sid = int(row.get("__orig_idx__", args.start + i))
        tasks.append(
            {
                "sample_id": sid,
                "global_id": int(row.get("global_id", sid)),
                "benchmark": row.get("benchmark", ""),
                "goal": goal,
                "category": row.get("category", ""),
                "attack_prompt": prompt,
                "safety_primer": args.primer_text,
            }
        )
    print(f"Prepared {len(tasks)} prompts")

    completed_ids = load_completed_sample_ids(args.output)
    if completed_ids:
        before = len(tasks)
        tasks = [t for t in tasks if t["sample_id"] not in completed_ids]
        print(f"Resuming: {len(completed_ids)} completed, {len(tasks)} remaining (from {before})")

    if not tasks:
        print("All tasks already completed.")
        return

    is_gpt_oss = "gpt-oss" in args.target_model.lower()
    rollout_seed = args.seed_base + int(args.rollout_idx)
    target_fn = make_target_fn(
        base_url=args.target_url,
        model_name=args.target_model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
        is_gpt_oss=is_gpt_oss,
        think_token_cap=args.max_tokens,
        force_close_max_tokens=args.force_close_max_tokens,
        seed=rollout_seed,
    )

    print(
        f"Running SafePath-ZS: target={args.target_model} @ {args.target_url}, "
        f"workers={args.max_workers}, tasks={len(tasks)}"
    )
    wrote = run(
        tasks=tasks,
        target_fn=target_fn,
        output_path=args.output,
        max_workers=args.max_workers,
        save_raw_generation=args.save_raw_generation,
        rollout_idx=args.rollout_idx,
        seed=rollout_seed,
    )
    print(f"\nWrote {wrote} new records to {args.output}")


if __name__ == "__main__":
    main()
