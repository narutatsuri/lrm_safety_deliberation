# PROVENANCE: original path defenses/rapo/rapo_rl.py (RAPO stage 2: GRPO RL with
# an LLM reward judge). The base-model registry now lives in lrm_safety_deliberation.models;
# eval is pipeline/eval/.
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from load_dataset import load_data
from utils import (
    BENIGN_JUDGE_SYSTEM_PROMPT,
    REWARD_JUDGE_SYSTEM_PROMPT,
    SAFETY_JUDGE_SYSTEM_PROMPT,
    str_judge,
)

_reward_client = None
_reward_model_name = None
_log_path = None
_save_json = []
_reward_concurrency = 1
# In-process reward path: when set, reward model runs on the same GPU as the
# training policy to keep single-GPU util high and avoid low-util cancellation.
_reward_model_local = None
_reward_tokenizer_local = None
# OpenRouter detection: when the reward server URL points at OpenRouter, the
# upstream provider does not honor `chat_template_kwargs={"enable_thinking": False}`
# silently passed via extra_body. Use the documented OpenRouter recipe instead:
#   (a) prepend `/no_think\n` to every system AND user message (Qwen3 honors
#       per-turn `/no_think` more reliably than persistent system instructions);
#   (b) pass `reasoning={"effort": "none", "exclude": True}` in extra_body.
# Verified 2026-05-04 against `qwen/qwen3-8b` paid: returns clean content with
# reasoning_tokens=0, matching the paper-faithful enable_thinking=False behavior.
_reward_is_openrouter = False


def _parse_dataset_recipe(s: str):
    if not s.strip():
        return []
    recipe = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        name, count = part.split(":")
        recipe.append((name.strip(), int(count.strip())))
    return recipe


def _extract_tag_number(text: str, tag: str):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    v = m.group(1).strip()
    try:
        return float(v)
    except ValueError:
        digits = re.findall(r"[-+]?\d*\.?\d+", v)
        return float(digits[0]) if digits else None


def _get_general_reward(text: str):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    score = _extract_tag_number(text, "score")
    if score is None:
        digits = re.findall(r"\d", text)
        score = float(digits[0]) if digits else 0.0
    return score * 2.0 - 1.0


def _get_risk_reward(text: str):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    reward = _extract_tag_number(text, "reward")
    if reward is None:
        return 0.0
    return reward if -2.0 < reward < 2.0 else 0.0


def _extract_safety_reasoning(completion_obj):
    raw = completion_obj[0]["content"]
    try:
        thinking, response = raw.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        response = response.strip()
        safety_reasoning = thinking.split("\n\n", 1)[0].strip()
        return safety_reasoning, thinking, response
    except ValueError:
        head = raw.split("\n\n", 1)[0].strip()
        return head, "", ""


def _call_reward_model(messages_list: list[list[dict]], max_tokens: int = 1024) -> list[str]:
    """Call the reward model. Prefers in-process transformers inference on the
    same GPU as the training policy when a local reward model was loaded; falls
    back to the OpenAI-compatible HTTP client otherwise."""

    if _reward_model_local is not None and _reward_tokenizer_local is not None:
        import torch
        tokenizer = _reward_tokenizer_local
        model = _reward_model_local
        prompts = []
        for msgs in messages_list:
            try:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            prompts.append(text)

        results: list[str] = []
        # Conservative batch size to keep activation memory modest alongside the
        # training model on the same GPU.
        batch_size = 4
        device = next(model.parameters()).device
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=3072,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            in_len = enc["input_ids"].shape[1]
            for j in range(out.shape[0]):
                gen_ids = out[j, in_len:]
                results.append(
                    tokenizer.decode(gen_ids, skip_special_tokens=True)
                )
        return results

    # Fallback: HTTP OpenAI-compatible reward server
    client = _reward_client
    concurrency = max(1, min(int(_reward_concurrency), len(messages_list)))

    if _reward_is_openrouter:
        # OpenRouter path: inject /no_think and reasoning suppression. Mutates a
        # copy so the caller's message dicts are not modified.
        def _prep(msgs):
            out = []
            for m in msgs:
                content = m.get("content", "") or ""
                if not content.startswith("/no_think"):
                    content = "/no_think\n" + content
                out.append({**m, "content": content})
            return out
        extra_body = {"reasoning": {"effort": "none", "exclude": True}}
    else:
        def _prep(msgs):
            return msgs
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def _one_call(msgs):
        try:
            resp = client.chat.completions.create(
                model=_reward_model_name,
                messages=_prep(msgs),
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body=extra_body,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"Reward model call failed: {e}")
            return ""

    if concurrency == 1:
        return [_one_call(msgs) for msgs in messages_list]

    results = [""] * len(messages_list)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_index = {
            pool.submit(_one_call, msgs): idx for idx, msgs in enumerate(messages_list)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return results


def reward_func(prompts, completions, harmful_label, **kwargs):
    global _save_json
    full_completions = [c[0]["content"].strip() for c in completions]
    extracted = [_extract_safety_reasoning(c) for c in completions]
    safety_reasonings = [x[0] for x in extracted]
    plain_completions = [x[2] for x in extracted]

    # Risk-aware reward
    risk_messages = []
    for i, sr in enumerate(safety_reasonings):
        ori_prompt = prompts[i][0]["content"]
        length_hint = (
            "## Hint: The length of the **Original Prompt is "
            f"{ori_prompt.count('. ') + 1} sentences**; The length of the **Safety Reasoning Trace is {sr.count('. ') + 1} sentences**. "
            "Please follow the **sentence length criteria** in the system prompt to judge the level, case and reward."
        )
        risk_user_prompt = f"## [Original Prompt] {ori_prompt}\n\n## [Safety Reasoning Trace] {sr}\n\n{length_hint}"
        risk_messages.append([
            {"role": "system", "content": REWARD_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": risk_user_prompt},
        ])

    risk_texts = _call_reward_model(risk_messages)
    risk_rewards = [_get_risk_reward(t) for t in risk_texts]

    # General reward (safety for harmful, helpfulness for benign)
    general_messages = []
    for i, completion in enumerate(plain_completions):
        ori_prompt = prompts[i][0]["content"]
        if int(harmful_label[i]) == 1:
            user_prompt = SAFETY_JUDGE_SYSTEM_PROMPT.format(prompt=ori_prompt, response=completion)
        else:
            user_prompt = BENIGN_JUDGE_SYSTEM_PROMPT.format(prompt=ori_prompt, response=completion)
        general_messages.append([{"role": "user", "content": user_prompt}])

    general_texts = _call_reward_model(general_messages)
    general_rewards = [_get_general_reward(t) for t in general_texts]

    rewards = [risk_rewards[i] + general_rewards[i] for i in range(len(prompts))]

    _save_json.append(
        [
            {
                "prompts": prompts[i],
                "full_completion": full_completions[i],
                "safety_reasoning": safety_reasonings[i],
                "completion": plain_completions[i],
                "harmful_label": int(harmful_label[i]),
                "risk_aware_reward_response": risk_texts[i],
                "general_reward_response": general_texts[i],
                "risk_aware_reward": risk_rewards[i],
                "general_reward": general_rewards[i],
                "reward": rewards[i],
                "str_judge": str_judge(plain_completions[i]),
            }
            for i in range(len(prompts))
        ]
    )
    with open(_log_path, "w") as f:
        json.dump(_save_json, f, indent=2, ensure_ascii=False)

    return rewards


def rl_train(
    model_path: str,
    reward_server_url: str,
    reward_model_name: str,
    dataset_recipe,
    config,
    save_path: str,
    data_dir: Optional[str],
    log_dir: str,
    reward_local_path: Optional[str] = None,
):
    global _reward_client, _reward_model_name, _log_path, _save_json, _reward_concurrency
    global _reward_model_local, _reward_tokenizer_local, _reward_is_openrouter

    try:
        import pandas as pd
        from datasets import Dataset
        from openai import OpenAI
        from trl import GRPOTrainer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `pandas datasets openai trl` before running RL."
        ) from e

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "recipe.json"), "w") as f:
        json.dump(dataset_recipe, f, indent=2, ensure_ascii=False)

    rl_rows = []
    for dataset_name, n in dataset_recipe:
        data = load_data(dataset_name, n, data_dir=data_dir)
        df = pd.DataFrame(data)
        for _, item in df.iterrows():
            rl_rows.append(
                {
                    "prompt": [{"content": item["prompt"], "role": "user"}],
                    "harmful_label": int(item["harmful_label"]),
                }
            )
    rl_dataset = Dataset.from_pandas(pd.DataFrame(rl_rows).sample(frac=1, random_state=0))

    # Load reward model. Prefer in-process (same GPU as training) to keep a
    # single-GPU job above util thresholds and avoid the 2-GPU low-util pattern
    # from a separate vLLM reward server.
    if reward_local_path:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[reward] loading in-process reward model from {reward_local_path}")
        _reward_tokenizer_local = AutoTokenizer.from_pretrained(
            reward_local_path, trust_remote_code=True
        )
        if _reward_tokenizer_local.pad_token is None:
            _reward_tokenizer_local.pad_token = _reward_tokenizer_local.eos_token
        # Explicit left padding for generation.
        _reward_tokenizer_local.padding_side = "left"
        _reward_model_local = AutoModelForCausalLM.from_pretrained(
            reward_local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        _reward_model_local.to("cuda")
        _reward_model_local.eval()
        print("[reward] in-process reward model ready on CUDA")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
        _reward_client = OpenAI(base_url=reward_server_url, api_key=api_key)
        _reward_model_name = reward_model_name
        _reward_concurrency = int(getattr(config, "reward_concurrency", 1))
        _reward_is_openrouter = "openrouter.ai" in reward_server_url.lower()
        if _reward_is_openrouter:
            print(f"[reward] OpenRouter mode: model={reward_model_name}, /no_think + reasoning.exclude enabled")
        else:
            print(f"Reward model server: {reward_server_url}, model: {reward_model_name}")

    os.makedirs(log_dir, exist_ok=True)
    _log_path = os.path.join(log_dir, f"{os.path.basename(save_path.rstrip('/'))}.json")
    _save_json = []

    from peft import LoraConfig as PeftLoraConfig
    peft_config = PeftLoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_func,
        train_dataset=rl_dataset,
        args=config,
        peft_config=peft_config,
    )

    # Auto-detect latest RL checkpoint and resume from it so a daisy chain of
    # SLURM jobs (each <= 8h) can make continuous progress without re-doing work.
    import re
    latest_ckpt = None
    if os.path.isdir(save_path):
        pattern = re.compile(r"^checkpoint-(\d+)$")
        candidates = []
        for name in os.listdir(save_path):
            m = pattern.match(name)
            if m and os.path.isdir(os.path.join(save_path, name)):
                candidates.append((int(m.group(1)), name))
        if candidates:
            candidates.sort()
            latest_ckpt = os.path.join(save_path, candidates[-1][1])
            print(f"[rapo-rl] resuming from {latest_ckpt}")

    if latest_ckpt is not None:
        trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        trainer.train()


def main():
    try:
        from trl import GRPOConfig
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `trl` before running RL."
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--reward-server-url", type=str, default="",
                        help="URL of vLLM reward server, e.g. http://localhost:8001/v1. "
                             "Ignored when --reward-local-path is set.")
    parser.add_argument("--reward-model-name", type=str, default="",
                        help="Model name registered on reward server. Ignored for local mode.")
    parser.add_argument("--reward-local-path", type=str, default="",
                        help="Load reward model from this local path in-process on the "
                             "same GPU as training, instead of calling a remote HTTP server.")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="wildjailbreak:300,star:100,starbenign:400")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="rl_log")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--reward-concurrency", type=int, default=4)
    args = parser.parse_args()

    dataset_recipe = _parse_dataset_recipe(args.datasets)
    config = GRPOConfig(
        output_dir=args.save_path,
        num_train_epochs=args.epochs,
        # save_only_model=False so optimizer/scheduler/RNG state is persisted and
        # daisy-chained SLURM continuations can resume GRPO training mid-run.
        save_only_model=False,
        save_total_limit=3,
        bf16=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        generation_batch_size=args.num_generations,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        temperature=args.temperature,
        top_p=args.top_p,
        save_steps=args.save_steps,
    )
    setattr(config, "reward_concurrency", args.reward_concurrency)

    rl_train(
        model_path=args.model_path,
        reward_server_url=args.reward_server_url,
        reward_model_name=args.reward_model_name,
        dataset_recipe=dataset_recipe,
        config=config,
        save_path=args.save_path,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        reward_local_path=args.reward_local_path or None,
    )


if __name__ == "__main__":
    main()
