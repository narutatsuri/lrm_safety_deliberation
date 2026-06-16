"""
STAIR Stage 1 LoRA SFT training script.

This reuses the proven Accelerate + DeepSpeed LoRA SFT training pattern from
`defenses/r1_act/train/sft_lora.py`, while preserving STAIR's own defaults:
  - STAIR system prompt and special tokens
  - response-only loss masking
  - LoRA target modules per model family
  - lr / batch / warmup / epochs / checkpoint cadence from the STAIR setup
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
import wandb

try:
    from common import (
        LORA_TARGET_MODULES,
        add_stair_special_tokens,
        build_stair_response,
        load_json_records,
        patch_qwen_chat_template,
        render_stair_example,
        stable_hash,
    )
except ModuleNotFoundError:
    from defenses.stair.train.common import (
        LORA_TARGET_MODULES,
        add_stair_special_tokens,
        build_stair_response,
        load_json_records,
        patch_qwen_chat_template,
        render_stair_example,
        stable_hash,
    )

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class StairDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.debug_count = 0

        model_name = config.model_path.rstrip("/").split("/")[-1]
        cache_id = stable_hash(
            {
                "model_path": config.model_path,
                "data_path": config.data_path,
                "base_model": config.base_model,
                "max_seq_len": config.max_seq_len,
                "max_samples": config.max_samples,
            }
        )
        cache_dir = Path(config.cache_dir) / config.experiment_name / f"{model_name}_{cache_id}"

        if cache_dir.exists():
            self.dataset = load_from_disk(str(cache_dir))
            return

        records = load_json_records(
            config.data_path,
            max_samples=config.max_samples,
            seed=config.seed,
        )
        dataset = HFDataset.from_list(records)

        def preprocess(example):
            answer = build_stair_response(example["answer"], config.base_model)
            full_text, query_text = render_stair_example(
                tokenizer,
                example["prompt"],
                answer,
                config.base_model,
            )
            input_ids = tokenizer.encode(full_text, add_special_tokens=False)
            query_ids = tokenizer.encode(query_text, add_special_tokens=False)
            labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
            input_ids = input_ids[-config.max_seq_len :]
            labels = labels[-config.max_seq_len :]
            return {
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
            }

        num_proc = max(1, int(config.preprocessing_num_workers))
        self.dataset = dataset.map(
            preprocess,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenizing STAIR SFT data",
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.save_to_disk(str(cache_dir))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        max_len = min(max(len(ids) for ids in input_ids), self.max_seq_len)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        for ids, lbl in zip(input_ids, labels):
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbl + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        if self.debug_count < 2:
            decoded_input = self.tokenizer.decode(padded_input_ids[-1])
            decoded_labels = self.tokenizer.decode(
                [0 if token == -100 else token for token in padded_labels[-1]]
            )
            print(f"[DEBUG] input_ids: {decoded_input}")
            print(f"[DEBUG] labels:    {decoded_labels}")
            self.debug_count += 1

        return {
            "input_ids": torch.LongTensor(padded_input_ids),
            "attention_mask": torch.LongTensor(attention_masks),
            "labels": torch.LongTensor(padded_labels),
        }


class SFTMetric:
    def __init__(self, device, world_size):
        self.n_step = 0
        self.right = torch.tensor([0.0], device=device)
        self.total = torch.tensor([0.0], device=device)
        self.total_loss = torch.tensor([0.0], device=device)
        self.world_size = world_size

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (
                (shift_preds == shift_labels)
                .masked_fill(shift_labels.eq(-100), 0)
                .sum()
                .item()
            )
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def __call__(self, logits, labels, loss):
        self.update(logits, labels, loss)

    def get_metric(self, reset=True):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.right, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_loss, op=dist.ReduceOp.SUM)

        total = max(self.total.item(), 1.0)
        steps = max(self.n_step, 1)
        acc = (self.right / total).item()
        loss = self.total_loss.item() / (self.world_size * steps)

        if reset:
            self.n_step = 0
            self.right.zero_()
            self.total.zero_()
            self.total_loss.zero_()
        return acc, loss


def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if accelerator.is_main_process:
        model_name = args.model_path.rstrip("/").split("/")[-1]
        wandb_mode = os.environ.get("WANDB_MODE", "")
        if wandb_mode.lower() != "disabled":
            wandb.init(
                project=args.experiment_name,
                config=vars(args),
                dir=args.log_dir,
                name=f"stair_sft_{model_name}",
            )

    accelerator.print(f"args:\n{args}")

    ds_plugin = accelerator.state.deepspeed_plugin
    if ds_plugin is not None:
        ds_cfg = ds_plugin.deepspeed_config
        ds_cfg["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
        ds_cfg["train_batch_size"] = (
            args.train_bsz_per_gpu
            * world_size
            * accelerator.gradient_accumulation_steps
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    patch_qwen_chat_template(tokenizer)
    tokens_to_add = add_stair_special_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_modules = LORA_TARGET_MODULES[args.base_model]
    accelerator.print(f"LoRA target modules for {args.base_model}: {target_modules}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    train_dataset = StairDataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    num_training_steps = (
        int(len(train_dataloader) * args.n_epochs)
        // accelerator.gradient_accumulation_steps
        // world_size
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(num_training_steps, 1),
    )

    accelerator.print(
        f"grad_acc={accelerator.gradient_accumulation_steps}  "
        f"data={args.data_path}  lr={args.learning_rate}  "
        f"total_steps={num_training_steps}  warmup={args.warmup_steps}  "
        f"epochs={args.n_epochs}  "
        f"lora_r={args.lora_r}  lora_alpha={args.lora_alpha}"
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    metric = SFTMetric(device=accelerator.device, world_size=world_size)
    start_epoch = 0
    start_step = 0
    global_step = 0
    last_saved_step = -1

    def save_checkpoint(epoch, step, global_step_value):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step_value}")
        if accelerator.is_main_process:
            checkpoint_files = [
                name for name in os.listdir(args.output_dir) if name.startswith("checkpoint-")
            ]
            if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
                checkpoint_files.sort(
                    key=lambda name: os.path.getctime(os.path.join(args.output_dir, name))
                )
                shutil.rmtree(os.path.join(args.output_dir, checkpoint_files[0]))

            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            accelerator.print(f"Saved LoRA adapter to {save_dir}")

        accelerator.wait_for_everyone()
        accelerator.save_state(save_dir)
        accelerator.save(
            {"epoch": epoch, "step": step, "global_step": global_step_value},
            os.path.join(save_dir, "training_state.pt"),
        )
        accelerator.print(f"checkpoint checkpoint-{epoch}-{global_step_value} is saved...")

    def load_training_state(save_dir):
        state_path = Path(save_dir) / "training_state.pt"
        if not state_path.exists():
            return
        state = torch.load(state_path, map_location="cpu")
        accelerator.load_state(save_dir)
        nonlocal start_epoch, start_step, global_step
        start_epoch = int(state.get("epoch", 0))
        start_step = int(state.get("step", -1)) + 1
        global_step = int(state.get("global_step", 0))
        accelerator.print(
            f"Resumed from {save_dir}: epoch={start_epoch} step={start_step} global_step={global_step}"
        )

    def merge_and_save(tag):
        merged_dir = os.path.join(args.output_dir, f"merged-{tag}")
        os.makedirs(merged_dir, exist_ok=True)

        if accelerator.is_main_process:
            accelerator.print("Merging LoRA adapter into base model...")
            unwrapped = accelerator.unwrap_model(model)
            merged_model = unwrapped.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            accelerator.print(f"Saved merged model to {merged_dir}")

        accelerator.wait_for_everyone()

    if args.resume_checkpoint:
        load_training_state(args.resume_checkpoint)

    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        if accelerator.is_main_process:
            train_iter = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}",
            )
        else:
            train_iter = enumerate(train_dataloader)

        for batch_cnt, batch in train_iter:
            if epoch == start_epoch and batch_cnt < start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
                use_cache=False,
            )
            loss = output.loss

            metric(output.logits, batch["labels"], loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_iter.set_postfix(
                    epoch=epoch,
                    step=batch_cnt,
                    loss=round(train_loss, 4),
                    acc=round(acc, 4),
                    lr=lr_scheduler.get_last_lr()[0],
                    length=int(batch["attention_mask"][0].sum().item()),
                )

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb_mode = os.environ.get("WANDB_MODE", "")
                if wandb_mode.lower() != "disabled":
                    wandb.log(
                        {
                            "loss": train_loss,
                            "acc": acc,
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

            if (
                args.save_every_steps > 0
                and global_step > 0
                and global_step % args.save_every_steps == 0
            ):
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
                last_saved_step = global_step

        if last_saved_step != global_step:
            accelerator.wait_for_everyone()
            save_checkpoint(epoch, batch_cnt, global_step)
            last_saved_step = global_step

    accelerator.print("Training complete. Saving final adapter...")

    if args.merge_adapter:
        merge_and_save("final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAIR Stage 1 LoRA SFT training")

    parser.add_argument("--experiment_name", type=str, default="STAIR-SFT")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument(
        "--base_model",
        required=True,
        type=str,
        choices=["Llama", "Qwen", "Olmo", "Phi4", "GPTOSS"],
    )
    parser.add_argument("--data_path", required=True, type=str)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    parser.add_argument("--output_dir", default="./ckpts", type=str)
    parser.add_argument("--max_ckpts", default=5, type=int)
    parser.add_argument("--log_dir", default="./train_logs", type=str)
    parser.add_argument("--cache_dir", default="./dataset_cache", type=str)
    parser.add_argument("--max_seq_len", default=4096, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--train_bsz_per_gpu", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_steps", default=10, type=int)
    parser.add_argument("--n_epochs", default=3, type=int)
    parser.add_argument("--save_every_steps", default=100, type=int)
    parser.add_argument("--resume_checkpoint", default="", type=str)
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument("--preprocessing_num_workers", default=4, type=int)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge LoRA into base model and save merged weights after training",
    )

    args = parser.parse_args()

    model_name = args.model_path.rstrip("/").split("/")[-1]
    args.log_dir = os.path.join(args.log_dir, args.experiment_name, model_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, model_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)
