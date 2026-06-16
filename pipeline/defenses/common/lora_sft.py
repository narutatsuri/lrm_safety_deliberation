"""
Generic LoRA SFT training script for defense methods.

Supports SafeChain, STAR-1, SafeKey, and SafePath defenses across
Qwen, Olmo, Phi4, and GPTOSS model families.

For SafeKey: includes the safety_head and key_sentence_prediction
auxiliary losses from the original SafeKey paper.

Usage (via accelerate launch):
    accelerate launch --config_file configs/deepspeed_zero3.yaml \
        --num_processes 1 lora_sft.py \
        --defense safechain --base_model Phi4 \
        --model_path models/Phi-4-reasoning \
        --data_path defenses/safechain/data/SafeChain-train.json \
        --output_dir models/SafeChain-Phi-4-reasoning-LoRA

PROVENANCE: original path defenses/common/lora_sft.py (shared LoRA SFT harness
used by SafeChain / STAR-1 / SafeKey / SafePath training variants).
The base-model registry now lives in lrm_safety_deliberation.models; eval is pipeline/eval/.
"""

import os
import json
import re
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ---------------------------------------------------------------------------
# Model-family-specific LoRA target modules.
# ---------------------------------------------------------------------------
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
# SafeKey safety head (only used with --defense safekey)
# ---------------------------------------------------------------------------

class SafetyHead(nn.Module):
    """Binary classification head over hidden states for SafeKey."""

    def __init__(self, hidden_size):
        super().__init__()
        self.C_head = nn.Linear(hidden_size, 1)
        self.ctx_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden, c_span):
        """
        hidden : (B, L, H)  final decoder layer hidden states
        c_span : (B, 2)     start/end indices for the summary C tokens
        Returns logits_C, logits_ctx (both shape (B,))
        """
        B, _, H = hidden.shape

        h_C = torch.stack([
            hidden[b, c_span[b, 0]:c_span[b, 1]].mean(0) for b in range(B)
        ], dim=0)  # (B, H)

        h_ctx = torch.stack([
            hidden[b, :c_span[b, 1]].mean(0) for b in range(B)
        ], dim=0)  # (B, H)

        logit_C = self.C_head(h_C).squeeze(-1)      # (B,)
        logit_ctx = self.ctx_head(h_ctx).squeeze(-1)  # (B,)
        return logit_C, logit_ctx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DefenseDataset(torch.utils.data.Dataset):
    """
    Unified dataset for all defense methods.

    Data format per defense:
      - safechain / star1 / safepath:
            {"question": ..., "response": "<think>...</think> ..."}
      - safekey:
            {"question": ..., "response": ...,
             "summary_end_idx": int, "next_sentence_end_idx": int,
             "source": str}

    Chat template handling:
      - Qwen/Olmo: use tokenizer.apply_chat_template() with patched Qwen3
        template that preserves <think> content.
      - Phi4/GPTOSS: use tokenizer.apply_chat_template() which correctly
        handles their Jinja templates ({% generation %}, channel markers, etc.)
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        if config.data_path.endswith(".jsonl"):
            with open(config.data_path) as f:
                self.data = [json.loads(line) for line in f]
        elif config.data_path.endswith(".json"):
            with open(config.data_path) as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {config.data_path}")

        print(f"[DefenseDataset] Loaded {len(self.data)} samples from {config.data_path}")
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
                print("[DefenseDataset] Patched Qwen3 chat template to preserve <think> content")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    # ----- Response formatting (shared across SafeChain / STAR-1 / SafeKey / SafePath) -----

    def _format_response(self, da):
        """
        Format the assistant response with think/no-think wrapping.
        Same logic used by the existing full-fine-tune scripts.
        """
        think_flag = self.config.think_flag

        response = da["response"].replace("\\/", "/").strip()

        match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", response, re.DOTALL)
        if match:
            thinking_trajectory = match.group(1).strip()
            attempt = match.group(2).strip()
        else:
            print(f"Warning: `<think>` parsing failed for response: {response[:80]}...")
            thinking_trajectory = ""
            attempt = response.strip()

        if think_flag:
            return f"<think>\n{thinking_trajectory}\n</think>\n\n{attempt}"
        else:
            return f"<think>\n\n</think>\n\n{attempt}"

    # ----- Chat template application -----

    def _apply_template(self, question, answer):
        """Apply chat template via tokenizer and return (full_text, query_text)."""
        full = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
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

    # ----- Per-defense prompt construction -----

    def _get_prompt_standard(self, da):
        """Standard prompt for SafeChain / STAR-1 / SafePath."""
        q = da.get("question", da.get("instruction"))
        a = self._format_response(da)
        assert q is not None and a is not None

        full_text, query_text = self._apply_template(q, a)

        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)

        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids), (
            f"Length mismatch: labels={len(labels)}, input_ids={len(input_ids)}"
        )

        return {
            "input_ids": input_ids[-self.max_seq_len:],
            "labels": labels[-self.max_seq_len:],
        }

    def _get_prompt_safekey(self, da):
        """
        SafeKey prompt with extra k_mask, attn_mask, c_span, unsafe fields
        needed for the key_sentence_prediction and safety_head losses.
        """
        q = da["question"]
        a = self._format_response(da)

        understanding = a[:da["summary_end_idx"]]
        key_sentence = a[da["summary_end_idx"]:da["next_sentence_end_idx"]]

        assert q is not None and a is not None

        full_text, query_text = self._apply_template(q, a)

        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)

        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)

        # Locate token span of K (key sentence)
        c_ids = self.tokenizer.encode(understanding, add_special_tokens=False)
        k_ids = self.tokenizer.encode(key_sentence, add_special_tokens=False)
        offset = -1 if self.config.base_model in ("Qwen",) else 0
        k_start = len(query_ids) + offset + len(c_ids)
        k_end = k_start + len(k_ids)  # exclusive
        if k_start < 0:
            k_start = 0
            k_end = len(k_ids)

        # Extra mask: 1 = token belongs to K
        k_mask = [0] * len(input_ids)
        for i in range(k_start, min(k_end, len(input_ids))):
            k_mask[i] = 1

        # Extra mask: 0 = token belongs to query
        attn_mask = [1] * len(input_ids)
        attn_mask[:len(query_ids)] = [0] * len(query_ids)

        return {
            "input_ids": input_ids[-self.max_seq_len:],
            "labels": labels[-self.max_seq_len:],
            "k_mask": k_mask[-self.max_seq_len:],
            "attn_mask": attn_mask[-self.max_seq_len:],
            "c_span": [len(query_ids), k_start],
            "unsafe": 0 if da.get("source") == "4o_rewrite" else 1,
        }

    def get_prompt(self, da):
        if self.config.defense == "safekey":
            return self._get_prompt_safekey(da)
        else:
            return self._get_prompt_standard(da)

    # ----- Collation -----

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

        result = {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
        }

        # SafeKey extra fields
        if self.config.defense == "safekey":
            k_masks = [item["k_mask"] for item in data]
            attn_masks = [item["attn_mask"] for item in data]
            c_spans = [item["c_span"] for item in data]
            unsafe = [item["unsafe"] for item in data]

            k_masks = [m[:max_len] + [0] * (max_len - len(m)) for m in k_masks]
            attn_masks = [m[:max_len] + [1] * (max_len - len(m)) for m in attn_masks]

            result["k_mask"] = torch.LongTensor(k_masks)
            result["attn_mask"] = torch.LongTensor(attn_masks)
            result["c_span"] = torch.LongTensor(c_spans)
            result["unsafe"] = torch.LongTensor(unsafe)

        if self.debug_count < 3:
            decoded_input = self.tokenizer.decode(result["input_ids"][-1].tolist())
            decoded_labels = self.tokenizer.decode(
                [0 if x == -100 else x for x in result["labels"][-1].tolist()]
            )
            print(f"[DEBUG] input_ids: {decoded_input[:500]}")
            print(f"[DEBUG] labels:    {decoded_labels[:500]}")
            self.debug_count += 1

        return result


# ---------------------------------------------------------------------------
# Metric tracker (works with DDP)
# ---------------------------------------------------------------------------

class SFTMetric:
    def __init__(self, device, defense="safechain"):
        self.defense = defense
        self.n_step = 0
        self.right = torch.tensor(0.0, device=device)
        self.total = torch.tensor(0.0, device=device)
        self.total_loss = torch.tensor(0.0, device=device)
        self.world_size = dist.get_world_size()

        # SafeKey-specific metrics
        if defense == "safekey":
            self.right_k = torch.tensor(0.0, device=device)
            self.total_k = torch.tensor(0.0, device=device)
            self.total_full_loss = torch.tensor(0.0, device=device)
            self.total_k_loss = torch.tensor(0.0, device=device)
            self.total_g_loss = torch.tensor(0.0, device=device)

    def update(self, logits, labels, loss, **kwargs):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            valid_mask = shift_labels.ne(-100)
            self.right += (shift_preds.eq(shift_labels) & valid_mask).sum()
            self.total += valid_mask.sum()
            self.total_loss += loss.detach()

            if self.defense == "safekey":
                self.total_full_loss += kwargs.get("full_loss", torch.tensor(0.0)).detach()
                self.total_k_loss += kwargs.get("k_loss", torch.tensor(0.0)).detach()
                self.total_g_loss += kwargs.get("g_loss", torch.tensor(0.0)).detach()

                logits_masked = kwargs.get("logits_masked")
                k_mask = kwargs.get("k_mask")
                if logits_masked is not None and k_mask is not None:
                    shift_preds_k = logits_masked[..., :-1, :].argmax(dim=-1)
                    k_mask_shift = k_mask[..., 1:].bool()
                    self.right_k += (shift_preds_k.eq(shift_labels) & k_mask_shift).sum()
                    self.total_k += k_mask_shift.sum()

    def __call__(self, logits, labels, loss, **kwargs):
        return self.update(logits, labels, loss, **kwargs)

    def get_metric(self, reset=True):
        tensors = [self.right, self.total, self.total_loss]
        if self.defense == "safekey":
            tensors += [self.right_k, self.total_k,
                        self.total_full_loss, self.total_k_loss, self.total_g_loss]

        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        acc = (self.right / self.total).item() if self.total > 0 else 0.0
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        extra = {}
        if self.defense == "safekey":
            extra["acc_k"] = (self.right_k / self.total_k).item() if self.total_k > 0 else 0.0
            extra["full_loss"] = self.total_full_loss.item() / (self.world_size * self.n_step)
            extra["k_loss"] = self.total_k_loss.item() / (self.world_size * self.n_step)
            extra["g_loss"] = self.total_g_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            for t in tensors:
                t.zero_()

        return acc, loss, extra


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
            name=f"{args.defense}_{model_name}_lora",
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

    # ---- SafeKey safety head ----
    if args.defense == "safekey" and args.safety_head:
        hidden_size = model.config.hidden_size
        model.safety_head = SafetyHead(hidden_size).to(
            dtype=torch.bfloat16
        )
        accelerator.print(f"[SafeKey] Attached SafetyHead (hidden_size={hidden_size})")

    # ---- Optimizer (only LoRA params + optional safety_head are trainable) ----
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

    # Add safety head params if present
    if args.defense == "safekey" and args.safety_head:
        head_params = list(model.safety_head.named_parameters())
        optimizer_grouped_parameters.append({
            "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        })

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    # ---- Dataset ----
    train_dataset = DefenseDataset(args, tokenizer)
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
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
    )

    accelerator.print(
        f"defense={args.defense}  base_model={args.base_model}  "
        f"grad_acc={accelerator.gradient_accumulation_steps}  "
        f"data={args.data_path}  lr={args.learning_rate}  "
        f"total_steps={num_training_steps}  "
        f"epochs={args.n_epochs}  "
        f"lora_r={args.lora_r}  lora_alpha={args.lora_alpha}"
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    metric = SFTMetric(
        device=torch.cuda.current_device(),
        defense=args.defense,
    )
    global_step = 0

    # ---- Save helpers ----
    def save_lora_adapter(tag):
        """Save the LoRA adapter weights."""
        save_dir = os.path.join(args.output_dir, f"checkpoint-{tag}")
        os.makedirs(save_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            # Save safety head separately if present
            if args.defense == "safekey" and args.safety_head and hasattr(unwrapped, "safety_head"):
                head_path = os.path.join(save_dir, "safety_head.pt")
                torch.save(unwrapped.safety_head.state_dict(), head_path)
                accelerator.print(f"Saved safety head to {head_path}")
            accelerator.print(f"Saved LoRA adapter to {save_dir}")

        accelerator.wait_for_everyone()

    def merge_and_save(tag):
        """Merge LoRA into base model and save full weights."""
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

    # ---- Training ----
    model.train()
    for epoch in range(args.n_epochs):
        if accelerator.is_main_process:
            train_iter = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}",
            )
        else:
            train_iter = enumerate(train_dataloader)

        for batch_cnt, batch in train_iter:
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # Determine whether we need hidden states (SafeKey safety head)
            need_hidden = (
                args.defense == "safekey"
                and args.safety_head
                and epoch > args.n_epochs - args.last_k_epoch - 1
            )

            output = model(
                input_ids=input_ids,
                labels=labels,
                return_dict=True,
                use_cache=False,
                output_hidden_states=need_hidden,
            )
            full_loss = output.loss
            loss = full_loss

            # --- SafeKey: key sentence prediction loss ---
            k_loss_val = torch.tensor(0.0, device=loss.device)
            g_loss_val = torch.tensor(0.0, device=loss.device)
            masked_out_logits = None
            k_mask = None

            if (
                args.defense == "safekey"
                and args.key_sentence_prediction
                and epoch > args.n_epochs - args.last_k_epoch - 1
            ):
                k_mask = batch["k_mask"]
                attn_mask = batch["attn_mask"]

                masked_out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    return_dict=True,
                    use_cache=False,
                )
                masked_out_logits = masked_out.logits

                # Shift for next-token prediction
                shift_logits = masked_out_logits[:, :-1].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                # Token-level NLL
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view_as(shift_labels)

                # Isolate tokens that belong to K
                k_mask_shift = k_mask[:, 1:].float()
                k_loss_val = (token_loss * k_mask_shift).sum() / (k_mask_shift.sum() + 1e-8)
                loss = loss + k_loss_val * args.key_sentence_weight

            # --- SafeKey: safety head loss ---
            if need_hidden:
                hidden = output.hidden_states[-1]
                if args.detach_safety_head:
                    hidden = hidden.detach()

                unwrapped = accelerator.unwrap_model(model)
                logit_C, logit_ctx = unwrapped.safety_head(
                    hidden, batch["c_span"]
                )
                y = batch["unsafe"].float()
                L_head_C = F.binary_cross_entropy_with_logits(logit_C, y)
                L_head_ctx = F.binary_cross_entropy_with_logits(logit_ctx, y)
                g_loss_val = args.head_theta * L_head_C + (1 - args.head_theta) * L_head_ctx

                if args.detach_safety_head:
                    loss = g_loss_val * args.head_weight
                else:
                    loss = loss + g_loss_val * args.head_weight

            # --- Metric update ---
            metric_kwargs = {}
            if args.defense == "safekey":
                metric_kwargs = {
                    "full_loss": full_loss,
                    "k_loss": k_loss_val,
                    "g_loss": g_loss_val,
                    "logits_masked": masked_out_logits,
                    "k_mask": k_mask,
                }
            metric(output.logits, labels, loss, **metric_kwargs)
            acc, train_loss, extra = metric.get_metric()

            accelerator.backward(loss)

            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                postfix = dict(
                    epoch=epoch,
                    step=batch_cnt,
                    loss=round(train_loss, 4),
                    acc=round(acc, 4),
                    lr=lr_scheduler.get_last_lr()[0],
                    length=len(input_ids[0]),
                )
                if args.defense == "safekey":
                    postfix["k_loss"] = round(extra.get("k_loss", 0), 4)
                    postfix["g_loss"] = round(extra.get("g_loss", 0), 4)
                train_iter.set_postfix(**postfix)

            if global_step % 3 == 0 and accelerator.is_main_process:
                log_dict = {
                    "loss": train_loss,
                    "acc": acc,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.defense == "safekey":
                    log_dict.update({
                        "acc_k": extra.get("acc_k", 0),
                        "full_loss": extra.get("full_loss", 0),
                        "k_loss": extra.get("k_loss", 0),
                        "g_loss": extra.get("g_loss", 0),
                    })
                wandb.log(log_dict, step=global_step)

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                accelerator.print(
                    f"Reached max_train_steps={args.max_train_steps}; stopping early."
                )
                break

        # Save at end of each epoch
        accelerator.wait_for_everyone()
        save_lora_adapter(f"epoch{epoch}")

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    # ---- Save final ----
    accelerator.print("Training complete. Saving final adapter...")
    save_lora_adapter("final")

    if args.merge_adapter:
        merge_and_save("final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic LoRA SFT for defense methods"
    )

    # Defense
    parser.add_argument(
        "--defense",
        required=True,
        type=str,
        choices=["safechain", "star1", "safekey", "safepath"],
        help="Defense method (determines data formatting logic)",
    )

    # Experiment
    parser.add_argument("--experiment_name", type=str, default=None)

    # Model
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument(
        "--base_model",
        required=True,
        type=str,
        choices=["Qwen", "Olmo", "Phi4", "GPTOSS"],
        help="Model family for LoRA target module selection",
    )

    # Data
    parser.add_argument("--data_path", required=True, type=str)

    # Think/base flags (for response formatting)
    parser.add_argument("--think_flag", type=int, default=1,
                        help="1=include thinking trajectory, 0=no thinking")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # Training
    parser.add_argument("--output_dir", default="./ckpts", type=str)
    parser.add_argument("--log_dir", default="./train_logs", type=str)
    parser.add_argument("--max_seq_len", default=8192, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--train_bsz_per_gpu", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_rates", default=0.05, type=float)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--max_train_steps", default=0, type=int,
                        help="Stop after this many global steps (0=disabled)")
    parser.add_argument("--seed", default=2002, type=int)
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge LoRA into base model and save merged weights after training",
    )

    # SafeKey-specific
    parser.add_argument("--key_sentence_prediction", action="store_true")
    parser.add_argument("--key_sentence_weight", default=0.2, type=float)
    parser.add_argument("--safety_head", action="store_true")
    parser.add_argument("--head_theta", default=0.5, type=float)
    parser.add_argument("--head_weight", default=0.2, type=float)
    parser.add_argument("--last_k_epoch", default=2, type=int,
                        help="Enable SafeKey aux losses in the last K epochs")
    parser.add_argument("--detach_safety_head", action="store_true")

    args = parser.parse_args()

    # Auto-set experiment name
    if args.experiment_name is None:
        model_name = args.model_path.rstrip("/").split("/")[-1]
        args.experiment_name = f"{args.defense}-lora-{model_name}"

    model_name = args.model_path.rstrip("/").split("/")[-1]
    args.log_dir = os.path.join(args.log_dir, args.experiment_name, model_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name, model_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)
