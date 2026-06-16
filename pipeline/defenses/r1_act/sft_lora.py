"""
R1-ACT LoRA SFT training script.

Trains a LoRA adapter on the R1-ACT dataset (959 samples: 859 refusal +
100 benign) using the same infrastructure patterns as SafeChain SFT but
with PEFT LoRA instead of full-model fine-tuning.

Paper: R1-Act (2508.00324)
  - LoRA: rank=16, alpha=16, target=attn+MLP projections
  - AdamW (beta1=0.9, beta2=0.95, wd=1e-4), lr=1e-5
  - Batch size 16, 15 epochs, 5 warmup steps, cosine schedule

PROVENANCE: original path defenses/r1_act/train/sft_lora.py (R1-Act LoRA SFT).
The base-model registry now lives in lrm_safety_deliberation.models; eval is pipeline/eval/.
"""

import os
import json
import logging
import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

# Model-family-specific LoRA target modules.
# Each family has different fused/split projection naming.
LORA_TARGET_MODULES = {
    "Qwen": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "Olmo": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "Phi4": [
        # Phi-4 has fused qkv_proj and gate_up_proj
        "qkv_proj", "o_proj",
        "gate_up_proj", "down_proj",
    ],
    "GPTOSS": [
        # GPT-OSS-20B is MoE with 3D packed expert tensors for MLP.
        # Standard LoRA can only target the 2D attention projections.
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class R1ActDataset(torch.utils.data.Dataset):
    """
    Each sample has 'instruction' and 'response'.
    - Refusal samples (859): response is thinking text ending with </think>
      -> full assistant output = "<think>\n{response}"
    - Benign samples (100): response is a normal answer (no think tags)
      -> full assistant output = "<think>\n\n</think>\n\n{response}"

    We mask the query (user turn) tokens with -100 so that the loss is
    computed only on the assistant response.

    Uses tokenizer.apply_chat_template() for template rendering, which
    correctly handles HF-specific Jinja extensions ({% generation %}, etc.)
    used by Phi-4-reasoning and GPT-OSS-20B.

    For Qwen3, we patch the chat_template to remove the clause that strips
    content before </think>, so that thinking traces are preserved.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        with open(config.data_path) as f:
            self.data = json.load(f)

        self.max_seq_len = config.max_seq_len
        self.debug_count = 0

        # Patch Qwen3's template to preserve <think> content in assistant
        # responses. The default template strips everything before </think>.
        if tokenizer.chat_template:
            remove_text = (
                "{% if '</think>' in content %}"
                "{% set content = content.split('</think>')[-1] %}"
                "{% endif %}"
            )
            patched = tokenizer.chat_template.replace(remove_text.strip(), "")
            if patched != tokenizer.chat_template:
                tokenizer.chat_template = patched
                print("[R1ActDataset] Patched Qwen3 chat template to preserve <think> content")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _build_response_segments(self, da):
        """Return (thinking_text, content_text) segments for the assistant turn.

        Preserves any trailing whitespace inside `thinking` so that the
        legacy non-GPT-OSS rendering can reproduce its prior literal bytes
        exactly (the legacy wrap was `f"<think>\\n{response}"` where the
        original `response` ended with `</think>`, so any `\\n` immediately
        before `</think>` belongs to the thinking text, not to the wrapper).

        - Refusal samples (859): response ends with `</think>` and is pure
          thinking; harmony equivalent is `thinking=<text>, content=""`.
        - Benign samples (100): plain answer with no think tags; harmony
          equivalent is `thinking="", content=response`.
        """
        response = da["response"].replace("\\/", "/").strip()
        if response.endswith("</think>"):
            thinking = response[:-len("</think>")]
            content = ""
        else:
            thinking = ""
            content = response
        return thinking, content

    def _apply_template(self, question, thinking, content):
        """Apply chat template via tokenizer and return (full_text, query_text).

        For GPT-OSS we use the chat template's native `thinking`/`content`
        fields, which correctly route to the harmony analysis/final channels.
        For other base models we reconstruct the legacy literal-tag wrapper
        bytes-identically to the prior version of this code.
        """
        if self.config.base_model == "GPTOSS":
            # Strip trailing whitespace on the analysis content (cosmetic;
            # `<|message|>...<|end|>` accepts any text but we don't want a
            # dangling newline before `<|end|>`).
            thinking_render = thinking.rstrip()
            assistant_msg = {
                "role": "assistant",
                "thinking": thinking_render,
                "content": content,
            }
        else:
            # Reproduce the legacy literal bytes exactly. Original code:
            #   refusal: f"<think>\n{response}"  where response ended with </think>
            #   benign:  f"<think>\n\n</think>\n\n{response}"
            if thinking and not content:
                answer = f"<think>\n{thinking}</think>"
            elif not thinking and content:
                answer = f"<think>\n\n</think>\n\n{content}"
            elif thinking and content:
                answer = f"<think>\n{thinking}</think>\n\n{content}"
            else:
                answer = ""
            assistant_msg = {"role": "assistant", "content": answer}

        full = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                assistant_msg,
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        query = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return full, query

    def get_prompt(self, da):
        question = da.get("question", da.get("instruction"))
        thinking, content = self._build_response_segments(da)
        assert question is not None

        full_text, query_text = self._apply_template(question, thinking, content)

        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)
        assert input_ids[:len(query_ids)] == query_ids, (
            "Prefix invariant broke: query_ids is not a token-prefix of input_ids "
            f"(base_model={self.config.base_model})"
        )

        # Mask query tokens with -100
        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids), (
            f"Length mismatch: labels={len(labels)}, input_ids={len(input_ids)}"
        )

        return {
            "input_ids": input_ids[-self.max_seq_len :],
            "labels": labels[-self.max_seq_len :],
        }

    def collate_fn(self, batch):
        data = [self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]

        max_len = min(max(len(x) for x in input_ids), self.max_seq_len)
        pad_id = self.tokenizer.eos_token_id

        input_ids = [
            ids[:max_len] + [pad_id] * (max_len - len(ids)) for ids in input_ids
        ]
        labels = [
            lbl[:max_len] + [-100] * (max_len - len(lbl)) for lbl in labels
        ]

        if self.debug_count < 3:
            decoded_input = self.tokenizer.decode(input_ids[-1])
            decoded_labels = self.tokenizer.decode(
                [0 if x == -100 else x for x in labels[-1]]
            )
            print(f"[DEBUG] input_ids: {decoded_input}")
            print(f"[DEBUG] labels:    {decoded_labels}")
            self.debug_count += 1

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
        }


# ---------------------------------------------------------------------------
# Metric tracker (works with DDP)
# ---------------------------------------------------------------------------

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

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
        return self.update(logits, labels, loss)

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        model_name = args.model_path.rstrip("/").split("/")[-1]
        wandb.init(
            project=args.experiment_name,
            config=vars(args),
            dir=args.log_dir,
            name=f"r1act_{model_name}",
        )

    accelerator.print(f"args:\n{args}")

    # DeepSpeed micro-batch config
    ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
    ds_cfg["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    ds_cfg["train_batch_size"] = (
        args.train_bsz_per_gpu
        * dist.get_world_size()
        * accelerator.gradient_accumulation_steps
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ---- LoRA ----
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

    # Gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- Optimizer (only LoRA params are trainable) ----
    no_decay = ["bias", "LayerNorm.weight"]
    trainable_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
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

    # ---- Dataset ----
    train_dataset = R1ActDataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    num_training_steps = (
        int(len(train_dataloader) * args.n_epochs)
        // accelerator.gradient_accumulation_steps
        // dist.get_world_size()
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
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

    metric = SFTMetric(device=torch.cuda.current_device())
    start_epoch = 0
    start_step = 0
    global_step = 0
    last_saved_step = -1

    # ---- Save helpers ----
    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
                checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                shutil.rmtree(os.path.join(args.output_dir, checkpoint_files[0]))

            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            accelerator.print(f"Saved LoRA adapter to {save_dir}")

        accelerator.wait_for_everyone()
        accelerator.save_state(save_dir)
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f"checkpoint checkpoint-{epoch}-{global_step} is saved...")

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
        """Merge LoRA into base model and save full weights."""
        merged_dir = os.path.join(args.output_dir, f"merged-{tag}")
        os.makedirs(merged_dir, exist_ok=True)

        if accelerator.is_main_process:
            accelerator.print(f"Merging LoRA adapter into base model...")
            unwrapped = accelerator.unwrap_model(model)
            merged_model = unwrapped.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            accelerator.print(f"Saved merged model to {merged_dir}")

        accelerator.wait_for_everyone()

    # ---- Training ----
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

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            output = model(
                input_ids=input_ids,
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
            loss = output.loss

            metric(output.logits, labels, loss)
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
                    length=len(input_ids[0]),
                )

            if global_step % 3 == 0 and accelerator.is_main_process:
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

    # ---- Save final ----
    accelerator.print("Training complete. Saving final adapter...")

    if args.merge_adapter:
        merge_and_save("final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R1-ACT LoRA SFT training")

    # Experiment
    parser.add_argument("--experiment_name", type=str, default="R1-ACT")

    # Model
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument(
        "--base_model",
        required=True,
        type=str,
        choices=["Qwen", "Olmo", "Phi4", "GPTOSS"],
        help="Model family for chat template selection",
    )

    # Data
    parser.add_argument("--data_path", required=True, type=str)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Training
    parser.add_argument("--output_dir", default="./ckpts", type=str)
    parser.add_argument("--max_ckpts", default=5, type=int)
    parser.add_argument("--log_dir", default="./train_logs", type=str)
    parser.add_argument("--max_seq_len", default=8192, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--train_bsz_per_gpu", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--n_epochs", default=15, type=int)
    parser.add_argument("--save_every_steps", default=0, type=int)
    parser.add_argument("--resume_checkpoint", default="", type=str)
    parser.add_argument("--seed", default=1995, type=int)
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
