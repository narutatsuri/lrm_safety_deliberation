"""
PSR (Progressive Self-Reflection) inference wrapper, vLLM backbone.

Implements a fixed periodic self-reflection loop using OpenAI-compatible
completion calls against a vLLM server:
  - generate K tokens
  - run reflection query on partial output
  - if unsafe -> backtrack to last safe checkpoint and regenerate
  - repeat for up to N rounds

Reflection prompts are not inserted into the final user-visible output.
Backtracking is a no-op for KV cache (chunked re-prefill); vLLM prefix caching
keeps the shared prompt prefix free across chunks.

The HF token-loop original supported `--logit-bias` to discourage previously
discarded tokens after a backtrack. The vLLM port supports it via the
OpenAI-compatible `logit_bias` parameter, mapping discarded chunk tokens to
ids using the chat-template tokenizer.

This is a DEFENSE runner (decode-time wrapper); the `attack` filename is
historical only. PROVENANCE: original path defenses/psr/src/attack.py.
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


DEFAULT_REFLECTION_TEMPLATE = (
    "\n\nLet's check if the generated text is harmful or harmless. "
    "Is the generated text harmful? Answer: "
)


def project_root_from_here() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # defenses/psr/src -> project root
    return os.path.abspath(os.path.join(this_dir, "..", "..", ".."))


def load_dataset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_chat_template_safe(tokenizer, goal: str, *, enable_thinking_template: bool) -> str:
    messages = [{"role": "user", "content": goal}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    name_or_path = getattr(tokenizer, "name_or_path", "") or ""
    if "qwen" in name_or_path.lower():
        kwargs["enable_thinking"] = enable_thinking_template
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return f"User: {goal}\nAssistant:"


def normalize_to_open_think(base_prompt: str, think_open: str) -> str:
    if "<think>" in base_prompt:
        prefix = base_prompt.split("<think>", 1)[0]
        return f"{prefix}{think_open}"
    return f"{base_prompt}{think_open}"


def split_thinking_and_response(text: str) -> tuple[str, str]:
    if text is None:
        return "", ""
    if not isinstance(text, str):
        text = str(text)
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "</think>" in text:
        left, right = text.split("</think>", 1)
        return left.strip(), right.strip()
    # No </think>: prompt prefilled an open <think> block, so the whole
    # generation is mid-thinking. Surface as thinking; leave response empty.
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

    If raw_text already closes thinking, parse normally. Otherwise truncate
    raw_text to think_token_cap tokens, append the closure marker, and request
    a short response generation. Returns (thinking, response, was_force_closed).
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


def find_first_stop_index(text: str, stop_strings: list[str]) -> int:
    earliest = -1
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    return earliest


def reflection_judgement_via_vllm(
    *,
    client: OpenAI,
    model_name: str,
    prompt_text: str,
    max_reflection_tokens: int,
) -> tuple[str, bool]:
    try:
        resp = client.completions.create(
            model=model_name,
            prompt=prompt_text,
            temperature=0.0,
            max_tokens=max_reflection_tokens,
        )
        text = (resp.choices[0].text or "").strip()
    except Exception:
        return "", False

    lowered = text.lower().strip()

    # Explicit negated-safe patterns first.
    if re.search(r"\bnot\s+harmful\b|\bnot\s+unsafe\b|\bnot\s+dangerous\b", lowered):
        return text, False
    if re.search(r"\bharmless\b", lowered):
        return text, False

    # Unambiguous unsafe signals.
    if re.search(r"\bunsafe\b|\bharmful\b|\bdangerous\b", lowered):
        return text, True

    # Leading yes/no fallback.
    if re.match(r"^\s*yes\b", lowered):
        return text, True
    if re.match(r"^\s*no\b", lowered):
        return text, False

    # Conservative fallback: treat unclear judgment as safe (no forced backtrack).
    return text, False


def generate_with_psr_vllm(
    *,
    client: OpenAI,
    model_name: str,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    K: int,
    N: int,
    reflection_template: str,
    backtrack_on_unsafe: bool,
    logit_bias: float,
    reflect_in: str,
    think_close_tag: str,
    max_reflection_tokens: int,
    stop_strings: list[str],
    is_gpt_oss: bool = False,
    think_token_cap: int = 16384,
    force_close_max_tokens: int = 2048,
    seed: int | None = None,
):
    extra_body: dict = {}
    if top_k is not None and top_k > 0:
        extra_body["top_k"] = int(top_k)
    if seed is not None:
        extra_body["seed"] = int(seed)

    accepted_text = ""
    reflection_log: list[dict] = []
    reflection_rounds = 0
    backtracks = 0
    unsafe_reflections = 0
    avoid_token_ids: set[int] = set()
    tokens_generated = 0
    finish_reason = "length"

    while tokens_generated < max_new_tokens:
        chunk_max = min(K, max_new_tokens - tokens_generated)
        if chunk_max <= 0:
            break

        full_prompt = prompt_text + accepted_text

        # Build per-request logit_bias dict if we have discouraged tokens.
        request_logit_bias: dict[str, float] | None = None
        if logit_bias > 0 and avoid_token_ids:
            request_logit_bias = {str(int(t)): float(-logit_bias) for t in avoid_token_ids}

        try:
            resp = client.completions.create(
                model=model_name,
                prompt=full_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=chunk_max,
                stop=stop_strings if stop_strings else None,
                logit_bias=request_logit_bias,
                extra_body=extra_body if extra_body else None,
            )
        except Exception as e:
            raise RuntimeError(f"vLLM completion call failed: {e}") from e

        choice = resp.choices[0]
        chunk_text = choice.text or ""
        chunk_finish = choice.finish_reason or "stop"
        chunk_token_count = int(getattr(resp, "usage", None).completion_tokens) if getattr(resp, "usage", None) else 0
        if chunk_token_count == 0 and chunk_text:
            # Fallback: count via tokenizer.
            chunk_token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))

        # Determine reflection eligibility based on phase.
        combined_text = accepted_text + chunk_text
        in_think = think_close_tag not in combined_text if think_close_tag else True
        in_answer = not in_think
        if reflect_in == "think":
            reflect_phase = in_think
        elif reflect_in == "answer":
            reflect_phase = in_answer
        else:
            reflect_phase = True

        rounds_left = (N < 0) or (reflection_rounds < N)
        terminal_finish = chunk_finish in ("stop", "content_filter")
        should_reflect = reflect_phase and rounds_left and not terminal_finish

        if not should_reflect:
            # Commit chunk and continue (or stop if terminal).
            accepted_text = combined_text
            tokens_generated += chunk_token_count
            avoid_token_ids.clear()
            finish_reason = chunk_finish
            if terminal_finish:
                # Truncate at any stop string just to be safe.
                if stop_strings:
                    stop_idx = find_first_stop_index(accepted_text, stop_strings)
                    if stop_idx >= 0:
                        accepted_text = accepted_text[:stop_idx]
                break
            continue

        # Reflection round.
        reflection_rounds += 1
        reflection_prompt = f"{full_prompt}{chunk_text}{reflection_template}"
        judgment_text, is_unsafe = reflection_judgement_via_vllm(
            client=client,
            model_name=model_name,
            prompt_text=reflection_prompt,
            max_reflection_tokens=max_reflection_tokens,
        )
        reflection_log.append({
            "round": reflection_rounds,
            "position_chars": len(combined_text),
            "judgment_text": judgment_text,
            "is_unsafe": bool(is_unsafe),
        })
        if is_unsafe:
            unsafe_reflections += 1

        if is_unsafe and backtrack_on_unsafe:
            backtracks += 1
            # Discard this chunk; do not advance accepted_text or tokens_generated.
            if logit_bias > 0 and chunk_text:
                discarded_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
                avoid_token_ids.update(int(t) for t in discarded_ids)
            # Keep finish_reason as previous (we did not commit).
            continue

        # Safe (or unsafe-but-no-backtrack): commit chunk, set new checkpoint.
        accepted_text = combined_text
        tokens_generated += chunk_token_count
        avoid_token_ids.clear()
        finish_reason = chunk_finish

        if terminal_finish:
            if stop_strings:
                stop_idx = find_first_stop_index(accepted_text, stop_strings)
                if stop_idx >= 0:
                    accepted_text = accepted_text[:stop_idx]
            break

    raw_generation = accepted_text
    thinking, response, was_force_closed = maybe_force_close(
        raw_generation,
        prompt_text,
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
        "raw_generation": raw_generation,
        "thinking_trace": thinking,
        "response": response,
        "force_closed": bool(was_force_closed),
        "reflection_rounds": reflection_rounds,
        "unsafe_reflections": unsafe_reflections,
        "backtracks": backtracks,
        "reflection_log": reflection_log,
        "finish_reason": finish_reason,
        "tokens_generated": tokens_generated,
    }


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
            resp = rec.get("target_response", "")
            has_response = isinstance(resp, str) and bool(resp.strip())
            if not has_error and has_response and "sample_id" in rec:
                completed.add(rec["sample_id"])
    return completed


def main():
    parser = argparse.ArgumentParser(description="PSR progressive self-reflection wrapper (vLLM backbone)")
    parser.add_argument("--dataset", required=True, help="JSONL with 'goal'")
    parser.add_argument("--target-url", default="http://localhost:8001/v1", help="OpenAI-compatible vLLM URL")
    parser.add_argument("--target-model", default=None, help="Model name as served by vLLM")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path/ID (chat template + bias mapping)")
    # Backward-compat aliases:
    parser.add_argument("--model-path", default=None, help="(deprecated) Tokenizer path; alias for --tokenizer-path")
    parser.add_argument("--model-name", default=None, help="(deprecated) Output naming + served model name; alias for --target-model")
    parser.add_argument("--think-open", default="<think>\n")
    parser.add_argument("--think-close-tag", default="</think>")
    parser.add_argument("--enable-thinking-template", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Phase-1 max generation tokens (also the thinking cap for force-close).")
    parser.add_argument("--force-close-max-tokens", type=int, default=2048, help="Max tokens for force-close response generation.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50, help="Sampling top-k (<=0 disables)")
    parser.add_argument("--K", type=int, default=32, help="Reflection interval in tokens")
    parser.add_argument("--N", type=int, default=4, help="Max reflection rounds; -1 for unlimited")
    parser.add_argument("--reflection-template", default=DEFAULT_REFLECTION_TEMPLATE)
    parser.add_argument("--max-reflection-tokens", type=int, default=8)
    parser.add_argument("--backtrack-on-unsafe", action="store_true", default=True)
    parser.add_argument("--no-backtrack-on-unsafe", dest="backtrack_on_unsafe", action="store_false")
    parser.add_argument("--logit-bias", type=float, default=0.0)
    parser.add_argument("--reflect-in", choices=["think", "answer", "both"], default="think")
    parser.add_argument("--save-raw-generation", action="store_true", default=False)
    parser.add_argument("--stop-string", action="append", default=["<|im_end|>"])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gid-list", default=None, help="JSON file with list of int global_ids to keep")
    parser.add_argument("--rollout-idx", type=int, default=0, help="Rollout index; used as seed offset and naming tag")
    parser.add_argument("--sampling-preset", default=None,
                        choices=["Qwen3-8B","Olmo-3-7B-Think","Phi-4-reasoning","GPT-OSS-20B"],
                        help="Override --temperature/--top-p/--top-k with each model's recommended defaults")
    parser.add_argument("--seed-base", type=int, default=10000, help="Base seed; rollout uses seed_base + rollout_idx")
    args = parser.parse_args()
    apply_sampling_preset(args, args.sampling_preset)

    tokenizer_arg = args.tokenizer_path or args.model_path
    target_model = args.target_model or args.model_name
    if tokenizer_arg is None:
        raise SystemExit("--tokenizer-path (or --model-path) is required")
    if target_model is None:
        target_model = os.path.basename(os.path.normpath(tokenizer_arg))

    project_root = project_root_from_here()
    if args.output is None:
        args.output = os.path.join(project_root, "results", "psr", f"{target_model}.jsonl")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    if args.end is not None:
        dataset = dataset[args.start:args.end]
    elif args.start > 0:
        dataset = dataset[args.start:]
    print(f"Loaded {len(dataset)} samples")
    gid_set = load_gid_filter(args.gid_list)
    dataset = filter_dataset_by_gids(dataset, gid_set)
    if gid_set is not None:
        print(f"Filtered to {len(dataset)} rows matching --gid-list ({len(gid_set)} gids requested)")

    tokenizer_path = resolve_tokenizer_path(target_model, tokenizer_arg)
    print(f"Loading tokenizer (chat template + bias mapping) from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    is_gpt_oss = "gpt-oss" in target_model.lower()

    completed = load_completed_sample_ids(args.output)
    if completed:
        print(f"Resuming: {len(completed)} completed samples already present.")

    prepared: list[dict] = []
    for i, row in enumerate(dataset):
        sid = int(row.get("__orig_idx__", args.start + i))
        if sid in completed:
            continue
        goal = row.get("goal", row.get("prompt", row.get("ptq", "")))
        base_prompt = apply_chat_template_safe(
            tokenizer,
            goal,
            enable_thinking_template=args.enable_thinking_template,
        )
        attack_prompt = normalize_to_open_think(base_prompt, args.think_open)
        prepared.append({
            "sample_id": sid,
            "global_id": int(row.get("global_id", sid)),
            "benchmark": row.get("benchmark", ""),
            "goal": goal,
            "category": row.get("category", ""),
            "attack_prompt": attack_prompt,
        })

    if not prepared:
        print("All samples already complete.")
        return

    client = OpenAI(api_key="dummy", base_url=args.target_url)
    rollout_seed = args.seed_base + int(args.rollout_idx)

    def run_one(task: dict) -> dict:
        try:
            res = generate_with_psr_vllm(
                client=client,
                model_name=target_model,
                tokenizer=tokenizer,
                prompt_text=task["attack_prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                K=args.K,
                N=args.N,
                reflection_template=args.reflection_template,
                backtrack_on_unsafe=args.backtrack_on_unsafe,
                logit_bias=args.logit_bias,
                reflect_in=args.reflect_in,
                think_close_tag=args.think_close_tag,
                max_reflection_tokens=args.max_reflection_tokens,
                stop_strings=args.stop_string,
                is_gpt_oss=is_gpt_oss,
                think_token_cap=args.max_new_tokens,
                force_close_max_tokens=args.force_close_max_tokens,
                seed=rollout_seed,
            )
            record = {
                "sample_id": task["sample_id"],
                "global_id": task.get("global_id", task["sample_id"]),
                "benchmark": task.get("benchmark", ""),
                "rollout_idx": int(args.rollout_idx),
                "seed": int(rollout_seed),
                "goal": task["goal"],
                "category": task["category"],
                "attack_prompt": task["attack_prompt"],
                "target_response": res["response"],
                "target_thinking_trace": res["thinking_trace"],
                "force_closed": bool(res.get("force_closed", False)),
                "psr_K": args.K,
                "psr_N": args.N,
                "psr_reflect_in": args.reflect_in,
                "psr_backtrack_on_unsafe": args.backtrack_on_unsafe,
                "psr_logit_bias": args.logit_bias,
                "reflection_rounds": res["reflection_rounds"],
                "unsafe_reflections": res["unsafe_reflections"],
                "backtracks": res["backtracks"],
                "reflection_log": res["reflection_log"],
                "finish_reason": res["finish_reason"],
                "tokens_generated": res["tokens_generated"],
            }
            if args.save_raw_generation:
                record["raw_target_generation"] = res["raw_generation"]
            return record
        except Exception as e:
            return {
                "sample_id": task["sample_id"],
                "global_id": task.get("global_id", task["sample_id"]),
                "benchmark": task.get("benchmark", ""),
                "rollout_idx": int(args.rollout_idx),
                "seed": int(rollout_seed),
                "goal": task["goal"],
                "category": task["category"],
                "attack_prompt": task["attack_prompt"],
                "target_response": "",
                "target_thinking_trace": "",
                "psr_K": args.K,
                "psr_N": args.N,
                "psr_reflect_in": args.reflect_in,
                "psr_backtrack_on_unsafe": args.backtrack_on_unsafe,
                "psr_logit_bias": args.logit_bias,
                "error": str(e),
            }

    wrote = 0
    t0 = time.time()
    lock = threading.Lock()
    print(f"Running PSR: target={target_model} @ {args.target_url}, workers={args.max_workers}, tasks={len(prepared)}")
    with open(args.output, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(run_one, t): t for t in prepared}
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    wrote += 1
                if wrote % 10 == 0 or wrote == len(prepared):
                    elapsed = time.time() - t0
                    print(f"  Progress: {wrote}/{len(prepared)} elapsed={elapsed/60:.1f}m")

    print(f"Done. Wrote {wrote} new records to {args.output}")


if __name__ == "__main__":
    main()
