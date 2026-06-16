# PROVENANCE: original path defenses/rapo/rapo_sft.py (RAPO stage 1: SFT + datagen).
# The base-model registry now lives in lrm_safety_deliberation.models; eval is pipeline/eval/.
import argparse
import json
import os
from typing import Optional

from load_dataset import load_data
from utils import ADAPTIVE_THINKING_SYSTEM_PROMPT, adaptive_thinking_length


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


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool, enable_thinking: bool):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def create_sft_data(
    model_path: str,
    dataset_recipe,
    save_path: str,
    data_dir: Optional[str],
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    try:
        import gc
        import pandas as pd
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `torch pandas datasets transformers vllm` before running SFT."
        ) from e

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "recipe.json"), "w") as f:
        json.dump(dataset_recipe, f, indent=2, ensure_ascii=False)

    is_ds_model = "deepseek" in model_path.lower()
    is_gptoss = "gpt-oss" in model_path.lower() or "gptoss" in model_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, skip_special_tokens=False)

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    model = LLM(**llm_kwargs)

    sft_dataset = []
    sft_data_prompt = []
    for dataset_name, n in dataset_recipe:
        data = load_data(dataset_name, n, data_dir=data_dir)
        df = pd.DataFrame(data)
        for _, item in df.iterrows():
            level = int(item.get("level", 1))
            level = level if level in adaptive_thinking_length else 1
            messages = [
                {
                    "role": "system",
                    "content": ADAPTIVE_THINKING_SYSTEM_PROMPT.format(
                        level=level, length=adaptive_thinking_length[level]
                    ),
                },
                {"role": "user", "content": f'The user prompt:\n"""{item["prompt"]}"""'},
            ]
            inputs = _apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=True)
            sft_dataset.append(
                {
                    "prompt": item["prompt"],
                    "message": messages,
                    "inputs": inputs,
                    "response": item.get("response", ""),
                    "level": level,
                }
            )
            sft_data_prompt.append(inputs)

    is_phi4 = "phi-4" in model_path.lower()
    # 2026-05-05: Phi-4 + GPT-OSS RAPO datagen validated end-to-end via smokes
    # 7703570 (Phi-4) and 7703088 (GPT-OSS), both 4/4 rows with clean SFT
    # content. Key fixes (all isolated to is_phi4 / is_gptoss branches; Qwen +
    # Olmo paths unchanged): Phi-4 force-closes `</think>` in pass-2 prefix to
    # prevent thinking-loop, slicer consumes `<|im_sep|>`. GPT-OSS extracts
    # ANALYSIS-channel content as safe_reasoning, anchors pass-2 at
    # `<|channel|>final<|message|>` boundary, uses output text directly for
    # final-channel content. Set RAPO_DEBUG_DUMP=path env var for per-row
    # diagnostic dumps to JSON.
    # Loop-truncation heuristic from scripts/phi4_backfill/backfill_phi4_empty.py
    # — Phi-4 fills the budget with repeated phrases when it can't decide. We
    # cut just before the loop and keep the meaningful reasoning prefix.
    import re as _re
    _LOOP_RX = _re.compile(r"(.{30,200}?)(?:\s*\1){2,}", _re.DOTALL)
    def _truncate_loops(text, hard_char_cap=6000):
        if not text:
            return ""
        m = _LOOP_RX.search(text)
        if m:
            text = text[: m.start()]
        if len(text) > hard_char_cap:
            text = text[:hard_char_cap]
        last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"), text.rfind("\n"))
        if last_period > 200:
            text = text[: last_period + 1]
        return text.rstrip()

    debug_dump = os.environ.get("RAPO_DEBUG_DUMP")  # path to write per-row diagnostic dump
    debug_records = []

    initial_outputs = model.generate(sft_data_prompt, sampling_params)
    final_inputs = []
    for idx, output in enumerate(initial_outputs):
        response = output.outputs[0].text or ""
        # Trigger force-close on either max_tokens cap OR finish_reason==length
        # (the latter fires when output hits max_model_len before max_tokens).
        hit_cap = (
            len(output.outputs[0].token_ids) >= max_tokens
            or output.outputs[0].finish_reason == "length"
        )
        if debug_dump:
            debug_records.append({
                "idx": idx,
                "prompt": sft_dataset[idx]["prompt"][:200],
                "pass1_n_tokens": len(output.outputs[0].token_ids),
                "pass1_finish_reason": output.outputs[0].finish_reason,
                "pass1_hit_cap": hit_cap,
                "pass1_response_first_500": response[:500],
                "pass1_response_last_500": response[-500:],
            })
        try:
            if is_gptoss:
                # GPT-OSS harmony format: extract ANALYSIS-channel content (the
                # actual safety reasoning), NOT the final-channel response. This
                # preserves channel semantics for pass-2 prefix injection: we put
                # analysis-shaped content back into the analysis channel.
                # Format: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>
                if "<|channel|>analysis<|message|>" in response:
                    analysis_part = response.split("<|channel|>analysis<|message|>", 1)[1]
                    if "<|end|>" in analysis_part:
                        analysis_part = analysis_part.split("<|end|>", 1)[0]
                    safe_reasoning = analysis_part.strip("\n").strip()
                else:
                    safe_reasoning = ""
            elif is_phi4 and hit_cap and "</think>" not in response:
                # Phi-4-reasoning loops on safety prompts under the RAPO
                # ADAPTIVE_THINKING_SYSTEM_PROMPT and never closes <think>.
                # Force-close: truncate at the first repeated-phrase loop,
                # keeping only the meaningful reasoning prefix.
                trimmed = _truncate_loops(response.strip("\n").strip())
                # Strip leading <think> tag if present (model emits it before reasoning)
                if trimmed.startswith("<think>"):
                    trimmed = trimmed[len("<think>"):].lstrip()
                safe_reasoning = trimmed
            else:
                safe_reasoning = response.split("</think>", 1)[1].strip("\n").strip()
        except Exception:
            safe_reasoning = ""

        # Phi-4 path: even on cap-hit we keep the trace (force-close mode above).
        # For other models, cap-hit means the response was truncated mid-thinking;
        # zero it out as before.
        if hit_cap and not (is_phi4 and len(safe_reasoning) > 0):
            safe_reasoning = ""

        if len(safe_reasoning) == 0 or len(safe_reasoning) < 64:
            final_input = "</s>"
        else:
            final_messages = [{"role": "user", "content": sft_dataset[idx]["prompt"]}]
            final_input = _apply_chat_template(tokenizer, final_messages, add_generation_prompt=True, enable_thinking=True)
            if is_gptoss:
                # Pass analysis-shaped content into analysis channel, then
                # force-prefill the channel transition so the model is anchored
                # at the start of the final channel. Without this, GPT-OSS stops
                # at `<|end|>` (analysis close, treated as EOS by vLLM) and
                # never emits final-channel content. Pre-filling positions the
                # model with no choice but to generate final content.
                final_input += (
                    "<|channel|>analysis<|message|>"
                    + safe_reasoning
                    + "<|end|><|start|>assistant<|channel|>final<|message|>"
                )
            elif is_phi4:
                # Phi-4 loops inside <think> on safety prompts and won't close
                # </think>. Force-close it directly in the prefix so the model
                # is anchored OUT of thinking mode and must generate the final
                # response. Equivalent in spirit to GPT-OSS channel-transition
                # prefill — both prevent a structural infinite-loop in pass 2.
                final_input += "<think>\nOkay, " + safe_reasoning + "</think>\n\n"
            elif not is_ds_model:
                final_input += "<think>\nOkay, " + safe_reasoning + "\n\n"
            else:
                final_input += "Okay, " + safe_reasoning + "\n\n"

        sft_dataset[idx]["final_input"] = final_input
        sft_dataset[idx]["safe_reasoning"] = safe_reasoning
        sft_dataset[idx]["len_safe_reasoning"] = len(safe_reasoning)
        final_inputs.append(final_input)
        if debug_dump and idx < len(debug_records):
            debug_records[idx]["safe_reasoning_len"] = len(safe_reasoning)
            debug_records[idx]["safe_reasoning_first_300"] = safe_reasoning[:300]
            debug_records[idx]["safe_reasoning_last_300"] = safe_reasoning[-300:] if len(safe_reasoning) > 300 else ""
            debug_records[idx]["pass2_prompt_last_400"] = final_input[-400:]

    final_outputs = model.generate(final_inputs, sampling_params)
    sft_training_set = []
    for idx, output in enumerate(final_outputs):
        if debug_dump and idx < len(debug_records):
            debug_records[idx]["pass2_n_tokens"] = len(output.outputs[0].token_ids)
            debug_records[idx]["pass2_finish_reason"] = output.outputs[0].finish_reason
            debug_records[idx]["pass2_text_first_500"] = (output.outputs[0].text or "")[:500]
            debug_records[idx]["pass2_text_last_500"] = (output.outputs[0].text or "")[-500:]
        if len(sft_dataset[idx]["final_input"]) < 10:
            if debug_dump and idx < len(debug_records):
                debug_records[idx]["dropped_reason"] = "final_input_too_short (pass1 failed)"
            continue

        response_ids = output.outputs[0].token_ids
        if len(response_ids) >= max_tokens:
            full_completion = ""
            full_completion_response = ""
            if debug_dump and idx < len(debug_records):
                debug_records[idx]["dropped_reason"] = "pass2_hit_max_tokens"
        else:
            full_completion = (output.prompt or "") + (output.outputs[0].text or "")

            if is_gptoss:
                # Pass-2 prefix anchors the model at `<|channel|>final<|message|>`,
                # so the OUTPUT text IS the final-channel content. Don't try to
                # slice on the assistant boundary — that gets confused by the
                # prefilled analysis `<|end|>`. Just take the output, strip any
                # trailing `<|return|>`/`<|end|>`, and use the pass-1 safe_reasoning
                # as the analysis content (it's what we prefilled).
                gen_text = output.outputs[0].text or ""
                if "<|return|>" in gen_text:
                    gen_text = gen_text.rsplit("<|return|>", 1)[0]
                if "<|end|>" in gen_text:
                    gen_text = gen_text.rsplit("<|end|>", 1)[0]
                final_channel_text = gen_text.strip("\n").strip()
                # full_completion_response is the analysis prefill + final content,
                # serialized in canonical harmony so the downstream split can rebuild
                # the message dict cleanly.
                analysis_prefill = sft_dataset[idx]["safe_reasoning"]
                full_completion_response = (
                    "<|channel|>analysis<|message|>"
                    + analysis_prefill
                    + "<|end|><|start|>assistant<|channel|>final<|message|>"
                    + final_channel_text
                )
                # If the model emitted nothing in final, drop this row
                if not final_channel_text:
                    full_completion = ""
                    full_completion_response = ""
                    if debug_dump and idx < len(debug_records):
                        debug_records[idx]["dropped_reason"] = "pass2_empty_final_channel"
            else:
                if is_ds_model:
                    marker_start = "<｜Assistant｜>"
                    marker_end = ""
                elif is_phi4:
                    # Phi-4 chat template: <|im_start|>assistant<|im_sep|>{content}<|im_end|>
                    # Include the role/content separator so it's consumed too.
                    marker_start = "<|im_start|>assistant<|im_sep|>"
                    marker_end = "<|im_end|>"
                else:
                    marker_start = "<|im_start|>assistant"
                    marker_end = "<|im_end|>"

                if marker_start in full_completion and marker_end in full_completion:
                    start_idx = full_completion.index(marker_start) + len(marker_start)
                    full_completion_response = full_completion[start_idx:].strip("\n").replace(marker_end, "")
                elif is_phi4 and marker_start in full_completion:
                    # Phi-4 may not emit <|im_end|> if it loops in the response.
                    # Take everything from the assistant marker, then loop-truncate.
                    start_idx = full_completion.index(marker_start) + len(marker_start)
                    raw = full_completion[start_idx:].strip("\n")
                    # Split at </think> + loop-truncate the final-response portion only
                    if "</think>" in raw:
                        thinking_part, _, final_part = raw.partition("</think>")
                        final_part = _truncate_loops(final_part.strip("\n").strip())
                        full_completion_response = thinking_part + "</think>\n\n" + final_part if final_part else ""
                    else:
                        full_completion_response = ""
                    if not full_completion_response:
                        full_completion = ""
                        if debug_dump and idx < len(debug_records):
                            debug_records[idx]["dropped_reason"] = "phi4_pass2_no_</think>_in_completion"
                else:
                    full_completion = ""
                    full_completion_response = ""
                    if debug_dump and idx < len(debug_records):
                        debug_records[idx]["dropped_reason"] = (
                            f"marker_slice_failed: marker_start_in={marker_start in full_completion} marker_end_in={marker_end in full_completion}"
                        )

                # Phi-4 safety net: even with marker slice success, the final
                # response may have looped past </think>. Loop-truncate.
                if is_phi4 and full_completion_response and "</think>" in full_completion_response:
                    thinking_part, _, final_part = full_completion_response.partition("</think>")
                    final_part_clean = _truncate_loops(final_part.strip("\n").strip())
                    if final_part_clean:
                        full_completion_response = thinking_part + "</think>\n\n" + final_part_clean
                    else:
                        full_completion_response = ""
                        if debug_dump and idx < len(debug_records):
                            debug_records[idx]["dropped_reason"] = "phi4_pass2_final_loop_collapsed_to_empty"

        sft_dataset[idx]["full_completion"] = full_completion
        sft_dataset[idx]["full_completion_response"] = full_completion_response
        if debug_dump and idx < len(debug_records):
            debug_records[idx]["full_completion_response_first_300"] = full_completion_response[:300]
            debug_records[idx]["full_completion_response_last_300"] = full_completion_response[-300:] if len(full_completion_response) > 300 else ""

        # GPT-OSS: the current `transformers` harmony chat template rejects
        # raw `<|channel|>` tags in the assistant `content` field. The new
        # contract is to put analysis-channel content in `thinking` and
        # final-channel content in `content` and let the template re-emit
        # the channel markers itself.
        if is_gptoss and full_completion_response:
            thinking_text = ""
            final_text = ""
            if "<|channel|>analysis<|message|>" in full_completion_response:
                analysis_part = full_completion_response.split(
                    "<|channel|>analysis<|message|>", 1
                )[1]
                if "<|end|>" in analysis_part:
                    thinking_text = analysis_part.split("<|end|>", 1)[0].strip()
                else:
                    thinking_text = analysis_part.strip()
            if "<|channel|>final<|message|>" in full_completion_response:
                final_part = full_completion_response.split(
                    "<|channel|>final<|message|>", 1
                )[1]
                for marker in ("<|return|>", "<|end|>"):
                    final_part = final_part.split(marker)[0]
                final_text = final_part.strip()
            # Fall back to the whole response as content if no channels found
            # (shouldn't happen for valid harmony output, but defensive).
            if not final_text and not thinking_text:
                final_text = full_completion_response
            assistant_msg = {"role": "assistant", "content": final_text}
            if thinking_text:
                assistant_msg["thinking"] = thinking_text
        else:
            assistant_msg = {"role": "assistant", "content": full_completion_response}

        sft_training_item = {
            "prompt": [{"content": sft_dataset[idx]["prompt"], "role": "user"}],
            "completion": [assistant_msg],
            "level": sft_dataset[idx]["level"],
            "len_safe_thinking": sft_dataset[idx]["len_safe_reasoning"],
            "raw_prompt": sft_dataset[idx]["prompt"],
            "safe_reasoning": sft_dataset[idx]["safe_reasoning"],
        }
        if len(full_completion_response) > 0:
            sft_training_set.append(sft_training_item)

    if debug_dump:
        try:
            with open(debug_dump, "w") as f:
                json.dump(debug_records, f, indent=2, ensure_ascii=False)
            print(f"[debug] wrote {len(debug_records)} records to {debug_dump}")
        except Exception as e:
            print(f"[debug] dump failed: {e}")

    with open(os.path.join(save_path, "json_data.json"), "w") as f:
        json.dump(sft_training_set, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(sft_training_set)
    df.to_csv(os.path.join(save_path, "csv_data.csv"), index=False)
    dataset = Dataset.from_pandas(df)

    if is_ds_model:
        def _concat_prompt_completion(example):
            text = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": example["prompt"][0]["content"]}],
                add_generation_prompt=True,
                enable_thinking=True,
            )
            completion = example["completion"][0]["content"]
            if "<think>\n" in completion:
                completion = completion.split("<think>\n", 1)[1]
            text += completion
            return {"text": text}

        dataset = dataset.map(_concat_prompt_completion, remove_columns=["prompt", "completion"])

    dataset.save_to_disk(save_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def sft_train(base_model_path: str, dataset_path: str, config):
    try:
        from datasets import load_from_disk
        from trl import SFTTrainer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies. Install `datasets trl` before running SFT training."
        ) from e

    ds = load_from_disk(dataset_path)
    trainer = SFTTrainer(model=base_model_path, train_dataset=ds, args=config)
    trainer.train()
    # TRL with save_steps=1000 may not save end-of-training when total_steps is
    # close to but past a save boundary. Force a final checkpoint named
    # `checkpoint-final` so the RL chain has a deterministic path to resume.
    final_ckpt = os.path.join(config.output_dir, "checkpoint-final")
    trainer.save_model(final_ckpt)
    # Marker for dual-submission race: lets a sibling SFT job (e.g. on a
    # different partition) detect that training already finished and exit 0
    # immediately instead of re-running. RL chain depends on afterany of both
    # SFT jobs; the loser no-ops via this marker.
    marker = os.path.join(config.output_dir, ".training_complete")
    try:
        open(marker, "w").close()
    except Exception:
        pass


def main():
    try:
        from trl import SFTConfig
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `trl`. Install it before running SFT."
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-save-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="starbenign:400,stratasword:400")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override vLLM max_model_len (otherwise uses tokenizer config).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for vLLM generation (default 0.0 = greedy).")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling cutoff for vLLM generation (default 1.0).")
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip data generation (assumes data already exists at --data-save-path)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only generate data, skip SFT training")
    args = parser.parse_args()

    if not args.skip_datagen:
        dataset_recipe = _parse_dataset_recipe(args.datasets)
        create_sft_data(
            model_path=args.model_path,
            dataset_recipe=dataset_recipe,
            save_path=args.data_save_path,
            data_dir=args.data_dir,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    if not args.skip_training:
        config = SFTConfig(
            output_dir=args.save_path,
            num_train_epochs=float(args.epochs),
            save_steps=args.save_steps,
            save_only_model=True,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            packing=False,
        )
        sft_train(args.model_path, args.data_save_path, config)


if __name__ == "__main__":
    main()
