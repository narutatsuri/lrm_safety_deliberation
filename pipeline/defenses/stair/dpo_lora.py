"""
STAIR Stage 2b: Step-level DPO training with LoRA.

Trains on preference pairs extracted from SI-MCTS trees. Each pair
consists of (chosen, rejected) partial or complete trajectories that
share a common prefix up to a branching reasoning step.

Paper approach (Section 3.3):
  - Step-DPO: preference pairs at each reasoning step, not just full trajectory
  - LoRA training with TRL DPOTrainer
  - 3 rounds of SI-MCTS + DPO (iterative self-improvement)
  - beta=0.1, lr=5e-6, batch=16, 1 epoch per round
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch

# torch >= 2.6 defaults torch.load to weights_only=True, which refuses to
# unpickle the numpy RNG state inside checkpoint-*/rng_state.pth (saved by the
# HF Trainer's _save_rng_state). The pickle references numpy.core.multiarray
# (numpy 1.x public path) but numpy 2.x renamed it to numpy._core.multiarray;
# torch.serialization.add_safe_globals matches by module+qualname string so the
# 1.x path can never be allowlisted. Force weights_only=False for our trainer
# resume — safe because the only torch.load calls in this script's runtime
# read trusted local checkpoint files we wrote ourselves.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOConfig, DPOTrainer

try:
    from common import (
        LORA_TARGET_MODULES,
        STAIR_SYSTEM_PROMPT,
        add_stair_special_tokens,
        patch_qwen_chat_template,
    )
except ModuleNotFoundError:
    from defenses.stair.train.common import (
        LORA_TARGET_MODULES,
        STAIR_SYSTEM_PROMPT,
        add_stair_special_tokens,
        patch_qwen_chat_template,
    )

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def load_dpo_data(data_path, tokenizer, base_model, max_length=4096):
    """
    Load DPO pairs from JSONL and format for TRL DPOTrainer.

    Returns a HuggingFace Dataset with columns:
      - prompt: the formatted prompt string
      - chosen: the chosen response string
      - rejected: the rejected response string
    """
    pairs = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    logger.info(f"Loaded {len(pairs)} DPO pairs from {data_path}")

    formatted = {"prompt": [], "chosen": [], "rejected": []}

    for pair in pairs:
        # Build the prompt using chat template
        messages = [
            {"role": "system", "content": STAIR_SYSTEM_PROMPT},
            {"role": "user", "content": pair["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # For thinking models, prepend empty think block to responses
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        if base_model in ("Qwen", "Olmo"):
            # Only add think prefix if not already present
            if not chosen.startswith("<think>"):
                chosen = f"<think>\n\n</think>\n\n{chosen}"
            if not rejected.startswith("<think>"):
                rejected = f"<think>\n\n</think>\n\n{rejected}"

        # Filter by length
        chosen_len = len(tokenizer.encode(prompt_text + chosen, add_special_tokens=False))
        rejected_len = len(tokenizer.encode(prompt_text + rejected, add_special_tokens=False))
        if chosen_len > max_length or rejected_len > max_length:
            continue

        formatted["prompt"].append(prompt_text)
        formatted["chosen"].append(chosen)
        formatted["rejected"].append(rejected)

    logger.info(f"After length filtering: {len(formatted['prompt'])} pairs (max_length={max_length})")
    return Dataset.from_dict(formatted)


def train(args):
    logger.info(f"Args: {args}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens_to_add = add_stair_special_tokens(tokenizer)
    if tokens_to_add:
        logger.info(f"Added {len(tokens_to_add)} STAIR special tokens")
    if patch_qwen_chat_template(tokenizer):
        logger.info("Patched Qwen3 chat template")

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))

    # ---- LoRA ----
    target_modules = LORA_TARGET_MODULES[args.base_model]
    logger.info(f"LoRA target modules: {target_modules}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
    )

    # ---- Reference model ----
    # trl>=0.12 requires ref_model=None when using peft_config for LoRA DPO.
    # The LoRA path internally uses the base model (adapters disabled) as reference.
    ref_model = None

    # ---- Dataset ----
    train_dataset = load_dpo_data(
        args.data_path, tokenizer, args.base_model, args.max_seq_len
    )
    logger.info(f"Training dataset: {len(train_dataset)} pairs")

    # ---- DPO training config ----
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bsz_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        warmup_steps=args.warmup_steps,
        beta=args.dpo_beta,
        max_length=args.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        # group_by_length removed in trl >=0.20 (DPOConfig API change). Padded
        # batching is fine; perf cost negligible for our DPO step counts.
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"stair_dpo_{args.model_path.rstrip('/').split('/')[-1]}_round{args.round_num}",
        optim="adamw_torch_fused",
    )

    # ---- Trainer ----
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    # ---- Save ----
    logger.info("Saving final adapter...")
    save_dir = os.path.join(args.output_dir, "checkpoint-final")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved adapter to {save_dir}")

    # ---- Merge if requested ----
    if args.merge_adapter:
        logger.info("Merging LoRA adapter into base model...")
        merged_dir = os.path.join(args.output_dir, "merged-final")
        os.makedirs(merged_dir, exist_ok=True)

        # Load the trained adapter and merge
        trained_model = PeftModel.from_pretrained(model, save_dir)
        merged_model = trained_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Saved merged model to {merged_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAIR Stage 2b: Step-DPO training")

    # Model
    parser.add_argument("--model_path", required=True, type=str,
                        help="Path to SFT model (Stage 1 output, or previous DPO round)")
    parser.add_argument("--base_model", required=True, type=str,
                        choices=["Llama", "Qwen", "Olmo", "Phi4", "GPTOSS"])

    # Data
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to DPO pairs JSONL")

    # DPO hyperparams (paper: beta=0.1, lr=5e-6)
    parser.add_argument("--dpo_beta", type=float, default=0.1)

    # LoRA (same as SFT: r=64, alpha=128)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # Training
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--train_bsz_per_gpu", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round_num", type=int, default=1,
                        help="Current DPO round number (for logging)")
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    model_name = args.model_path.rstrip("/").split("/")[-1]
    if not args.output_dir.endswith(model_name):
        args.output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).exists():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from_checkpoint}")

    set_seed(args.seed)
    train(args)
