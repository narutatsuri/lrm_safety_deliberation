# PROVENANCE: original path defenses/safekey/train/sft.py (SafeKey SFT with
# safety-head + key-sentence auxiliary losses).
# The base-model registry now lives in lrm_safety_deliberation.models; eval is pipeline/eval/.
import os
import json
import torch
import logging
import argparse
import gc

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import json
from jinja2 import Template
import time 
import re
import pytz
from datetime import datetime
from pathlib import Path


from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)
get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # if the data is jsonl file, load the data
        if config.data_path.endswith('.jsonl'):
            with open(config.data_path) as f:
                lines = f.readlines()
                self.data = [json.loads(line) for line in lines]
        elif config.data_path.endswith('.json'):
            with open(config.data_path) as f:
                self.data = json.load(f)
        
        newdata = []
        for da in self.data:
            newdata.append(da)
        print('filter out',len(self.data),len(newdata))
        self.data = newdata

        self.max_seq_len = self.config.max_seq_len
        self.debug = 1
        
        default_template = tokenizer.chat_template
        remove_text = """{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"""
        self.template = None
        self._use_apply_chat_template = False
        if default_template:
            new_template = default_template.replace(remove_text.strip(), "")
            try:
                self.template = Template(new_template)
            except Exception:
                self._use_apply_chat_template = True
        if self.config.base_model == 'GPT_OSS':
            self.template = None
            self._use_apply_chat_template = True

    def __getitem__(self, index):
        return self.data[index]

    def _parse_thinking_attempt(self, da):
        """Parse the response into (thinking, attempt) without rebuilding the
        literal-tag wrapper. Used both by `get_response` and the GPT-OSS path."""
        response = da["response"].replace("\\/", "/").strip()
        match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", response, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        print(f"Warning: `<think>` parsing failed for response: {response[:200]}")
        return "", response.strip()

    def get_response(self,da):
        base_flag = self.config.base_flag
        think_flag = self.config.think_flag

        thinking_trajectory, attempt = self._parse_thinking_attempt(da)

        if think_flag:
            return f"<think>\n{thinking_trajectory}\n</think>\n\n{attempt}"
        else:
            if base_flag:
                return attempt
            else:
                return f"<think>\n\n</think>\n\n{attempt}"

    def get_prompt(self,da):
        q = da['question']
        # GPT-OSS uses harmony channel format; the chat template natively accepts
        # an assistant message with separate `thinking` (analysis channel) and
        # `content` (final channel) fields. Bypass the legacy <think>...</think>
        # text wrapper entirely. SafeKey requires locating the "key sentence"
        # token span; under harmony the key sentence lives in the analysis
        # channel content so we use `tokenizer(..., return_offsets_mapping=True)`
        # to convert character offsets to exact token indices and add the
        # constant analysis-marker offset (`<|channel|>analysis<|message|>` = 3).
        if self.config.base_model == 'GPT_OSS':
            thinking_trajectory, attempt = self._parse_thinking_attempt(da)
            think_flag = self.config.think_flag
            if think_flag:
                thinking_render = thinking_trajectory
                content_render = attempt
            else:
                thinking_render = ""
                content_render = attempt

            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "thinking": thinking_render, "content": content_render},
                ],
                tokenize=False, add_generation_prompt=False,
            )
            query_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            query_ids = self.tokenizer.encode(query_text, add_special_tokens=False)
            assert input_ids[:len(query_ids)] == query_ids, (
                "GPT-OSS prefix invariant broke: query_ids is not a token-prefix of input_ids"
            )
            labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
            assert len(labels) == len(input_ids)

            # Locate key-sentence token span. summary_end_idx and
            # next_sentence_end_idx are character offsets into the *original*
            # `<think>\n{thinking}\n</think>\n\n{attempt}` literal string. The
            # SafeKey data has thinking_trajectory positioned exactly 8 chars
            # (len("<think>\n")) into that literal — verified for the full 1915
            # examples (1911 fall inside the think block, 4 outliers have empty
            # thinking and key sentence in the response).
            ANALYSIS_MARKER_LEN = 3  # <|channel|>analysis<|message|> = 3 tokens
            offset_in_literal = len("<think>\n")
            rel_summary = da["summary_end_idx"] - offset_in_literal
            rel_next = da["next_sentence_end_idx"] - offset_in_literal

            k_start = 0
            k_end = 0
            if (think_flag and thinking_render
                    and 0 <= rel_summary <= len(thinking_render)
                    and 0 <= rel_next <= len(thinking_render)):
                enc = self.tokenizer(
                    thinking_render,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = enc["offset_mapping"]
                think_token_count = len(enc["input_ids"])

                k_start_in_think = think_token_count
                k_end_in_think = think_token_count
                for i, (s, _e) in enumerate(offsets):
                    if k_start_in_think == think_token_count and s >= rel_summary:
                        k_start_in_think = i
                    if s >= rel_next:
                        k_end_in_think = i
                        break

                k_start = len(query_ids) + ANALYSIS_MARKER_LEN + k_start_in_think
                k_end   = len(query_ids) + ANALYSIS_MARKER_LEN + k_end_in_think

            k_mask = [0] * len(input_ids)
            for i in range(k_start, k_end):
                if 0 <= i < len(input_ids):
                    k_mask[i] = 1

            attn_mask = [1] * len(input_ids)
            attn_mask[:len(query_ids)] = [0] * len(query_ids)
            return {
                "input_ids": input_ids[-self.max_seq_len:],
                "labels": labels[-self.max_seq_len:],
                "k_mask": k_mask[-self.max_seq_len:],
                "attn_mask": attn_mask[-self.max_seq_len:],
                "c_span": [len(query_ids), max(k_start, len(query_ids))],
                "unsafe": 0 if da["source"] == "4o_rewrite" else 1,
            }

        a = self.get_response(da)
        understanding = a[:da["summary_end_idx"]]
        key_sentence = a[da["summary_end_idx"]:da["next_sentence_end_idx"]]
        # print('understanding:',understanding)
        # print('key_sentence:',key_sentence)
        # time.sleep(100)
        assert q is not None and a is not None, f'q:{q} a:{a}'
        if self.template is not None and self.config.base_model == 'Llama':
            input = self.template.render(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                bos_token=self.tokenizer.bos_token,
                add_generation_prompt=False,
            )
            query = self.template.render(
                messages=[{"role": "user", "content": q}],
                bos_token=self.tokenizer.bos_token,
                add_generation_prompt=True,
            )
        elif self.template is not None and self.config.base_model in ('Qwen', 'Olmo', 'Phi'):
            input = self.template.render(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                add_generation_prompt=False,
            )
            query = self.template.render(
                messages=[{"role": "user", "content": q}],
                add_generation_prompt=True,
            )
        elif self._use_apply_chat_template:
            input = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False,
            )
            query = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            query = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            input = f"{query}{a}"
        input_ids = self.tokenizer.encode(input,add_special_tokens= False)
        query_ids = self.tokenizer.encode(query,add_special_tokens= False)
        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)

        # ─── locate token span of K ─────────────────────────────────
        c_ids = self.tokenizer.encode(understanding, add_special_tokens=False)
        k_ids = self.tokenizer.encode(key_sentence, add_special_tokens=False)
        offset = -1 if self.config.base_model in ('Qwen', 'Llama') else 0
        k_start = len(query_ids) + offset + len(c_ids)
        k_end   = k_start + len(k_ids)               # exclusive
        if k_start < 0:
            k_start = 0
            k_end = len(k_ids)
        
        # extra mask: 1 ⇔ token belongs to K
        k_mask = [0] * len(input_ids)
        for i in range(k_start, k_end):
            k_mask[i] = 1

        # extra mask: 0 ⇔ token belongs to query
        attn_mask = [1] * len(input_ids)
        attn_mask[:len(query_ids)] = [0] * len(query_ids)
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:], "k_mask": k_mask[-self.max_seq_len:], "attn_mask": attn_mask[-self.max_seq_len:], "c_span": [len(query_ids), k_start], "unsafe": 0 if da["source"] == "4o_rewrite" else 1}        

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        k_masks = [item["k_mask"] for item in data]
        attn_masks = [item["attn_mask"] for item in data]
        c_spans = [item["c_span"] for item in data]
        unsafe = [item["unsafe"] for item in data]
        
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        
        k_masks = [item[:max_len] + [0] * (max_len - len(item)) for item in k_masks]
        attn_masks = [item[:max_len] + [1] * (max_len - len(item)) for item in attn_masks]
    
        if self.debug < 3:
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
                "k_mask": torch.LongTensor(k_masks),
                "attn_mask": torch.LongTensor(attn_masks),
                "c_span": torch.LongTensor(c_spans),
                "unsafe": torch.LongTensor(unsafe)
            }
    
    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        # accuracy on masked‑prefix forward (K)
        self.right_k = torch.tensor(0.0, device=device)
        self.total_k = torch.tensor(0.0, device=device)

        # losses
        self.total_loss = torch.tensor(0.0, device=device)
        self.total_full_loss  = torch.tensor(0.0, device=device)
        self.total_k_loss     = torch.tensor(0.0, device=device)
        self.total_g_loss     = torch.tensor(0.0, device=device)
        
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels,                  # original forward
                 loss, full_loss, k_loss,
                 logits_masked, k_mask, g_loss):                # masked‑prefix pass
        return self.update(logits, labels,
                           loss, full_loss, k_loss,
                           logits_masked, k_mask, g_loss)

    def update(self, logits, labels,
               loss, full_loss, k_loss,
               logits_masked, k_mask, g_loss):
        self.n_step += 1
        with torch.no_grad():
            # -------- overall token accuracy (ignore -100 pads) ---------------
            shift_preds  = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            valid_mask   = shift_labels.ne(-100)

            self.right += (shift_preds.eq(shift_labels) & valid_mask).sum()
            self.total += valid_mask.sum()

            if logits_masked is not None:
                # ------------------- accuracy on K‑sentence ------------------------
                shift_preds_k  = logits_masked[..., :-1, :].argmax(dim=-1)
                k_mask_shift   = k_mask[..., 1:].bool()          # align positions

                self.right_k += (shift_preds_k.eq(shift_labels) & k_mask_shift).sum()
                self.total_k += k_mask_shift.sum()
            
            # ------------------------ running losses --------------------------
            self.total_loss      += loss.detach()
            self.total_full_loss += full_loss.detach()
            self.total_k_loss    += k_loss.detach()
            self.total_g_loss    += g_loss.detach()

    def get_metric(self, reset=True):
        # Distributed aggrehead
        for t in (self.right, self.total,
                  self.right_k, self.total_k,
                  self.total_loss,
                  self.total_full_loss,
                  self.total_k_loss):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        acc     = (self.right   / self.total  ).item() if self.total  > 0 else 0.0
        acc_k   = (self.right_k / self.total_k).item() if self.total_k > 0 else 0.0

        mean_train_loss = self.total_loss.item() / (self.world_size * self.n_step)
        mean_full_loss  = self.total_full_loss.item()  / (self.world_size * self.n_step)
        mean_k_loss     = self.total_k_loss.item()     / (self.world_size * self.n_step)
        mean_g_loss     = self.total_g_loss.item()     / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            for t in (self.right, self.total,
                      self.right_k, self.total_k,
                      self.total_loss,
                      self.total_full_loss,
                      self.total_k_loss,
                      self.total_g_loss):
                t.zero_()

        return acc, acc_k, mean_train_loss, mean_full_loss, mean_k_loss, mean_g_loss


class SafetyHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.C_head   = nn.Linear(hidden_size, 1)
        self.ctx_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden, c_span):
        """
        hidden : (B, L, H)  final decoder layer under teacher forcing
        c_span : (B, 2)     start/end indices for the summary C tokens
        returns logits_C, logits_ctx (both shape (B,))
        """
        B, L, H = hidden.shape

        # Defensive: empty/invalid spans → mean() over zero-length dim returns NaN
        # which poisons every downstream gradient. Fall back to a single token
        # (idx 0) for those rows; their head loss is then a finite no-op.
        h_C_list = []
        h_ctx_list = []
        for b in range(B):
            s = int(c_span[b, 0].item())
            e = int(c_span[b, 1].item())
            s = max(0, min(s, L))
            e = max(0, min(e, L))
            if e <= s:
                h_C_list.append(hidden[b, 0])
                h_ctx_list.append(hidden[b, 0])
            else:
                h_C_list.append(hidden[b, s:e].mean(0))
                h_ctx_list.append(hidden[b, :e].mean(0))
        h_C = torch.stack(h_C_list, dim=0)
        h_ctx = torch.stack(h_ctx_list, dim=0)

        logit_C   = self.C_head(h_C).squeeze(-1)
        logit_ctx = self.ctx_head(h_ctx).squeeze(-1)
        return logit_C, logit_ctx


def train(args):

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 
    timestamp = get_time()
    if accelerator.is_main_process:
        model_name = args.model_path.split("/")[-1]
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir, name=f"{model_name}_think_flag{args.think_flag}_{timestamp}")
    
    accelerator.print(f'args:\n{args}')
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps

    # Cap Stage-3 transient buffers — DeepSpeed defaults can sit at multi-GB per
    # rank, which combined with full-FT 14B/20B optimizer state pushes past 80 GB
    # on H100 even at the textbook GPU count. These caps trade a little comm
    # overhead for ~5–10 GB/rank of live-buffer headroom.
    _zero_cfg = accelerator.state.deepspeed_plugin.deepspeed_config.setdefault('zero_optimization', {})
    _zero_cfg.setdefault('stage3_max_live_parameters', int(1e9))
    _zero_cfg.setdefault('stage3_max_reuse_distance', int(1e9))
    _zero_cfg.setdefault('stage3_prefetch_bucket_size', int(1e8))
    _zero_cfg.setdefault('stage3_param_persistence_threshold', int(1e6))
    _zero_cfg.setdefault('reduce_bucket_size', int(1e8))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # open gradient checkpointing
    model.gradient_checkpointing_enable()
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if args.safety_head:
        # attach the Head
        hidden_size        = model.config.hidden_size
        model.safety_head  = SafetyHead(hidden_size).to(model.device)
        
        head_params = list(model.safety_head.named_parameters())
        optimizer_grouped_parameters.append({
            "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        })
        
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas= (0.9, 0.95))

    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    start_epoch = 0
    start_step = 0
    global_step = 0
    last_saved_step = -1

    metric = SFTMetric(device=torch.cuda.current_device())

    def load_weights_only(save_dir):
        weights_dir = Path(save_dir) / "tfmr"
        if not weights_dir.exists():
            raise FileNotFoundError(f"weight-only resume checkpoint missing tfmr dir: {weights_dir}")

        accelerator.print(
            f"Falling back to weight-only resume from {weights_dir} "
            "(optimizer/scheduler state not restored)."
        )
        resumed_model = AutoModelForCausalLM.from_pretrained(
            str(weights_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        missing, unexpected = accelerator.unwrap_model(model).load_state_dict(
            resumed_model.state_dict(),
            strict=False,
        )
        accelerator.print(
            f"Weight-only resume loaded with missing={len(missing)} unexpected={len(unexpected)}"
        )
        del resumed_model
        gc.collect()
        torch.cuda.empty_cache()

    def load_training_state(save_dir):
        state_path = Path(save_dir) / "training_state.pt"
        if not state_path.exists():
            return
        state = torch.load(state_path, map_location="cpu")
        nonlocal start_epoch, start_step, global_step
        global_step = int(state.get("global_step", 0))
        try:
            accelerator.load_state(save_dir)
            start_epoch = int(state.get("epoch", 0))
            start_step = int(state.get("step", -1)) + 1
            accelerator.print(
                f"Resumed from {save_dir}: epoch={start_epoch} step={start_step} global_step={global_step}"
            )
        except Exception as exc:
            accelerator.print(
                "Full-state resume failed; attempting weight-only fallback. "
                f"Original error: {exc}"
            )
            load_weights_only(save_dir)
            # ZeRO optimizer state cannot be repartitioned across a different DP world size.
            # Resume from the next epoch with restored model weights instead.
            start_epoch = int(state.get("epoch", 0)) + 1
            start_step = 0
            accelerator.print(
                f"Weight-only resumed from {save_dir}: epoch={start_epoch} step={start_step} "
                f"global_step={global_step}"
            )

    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts>0:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))        
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            if accelerator.state.deepspeed_plugin.zero_stage!=3:
                model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
            tokenizer.save_pretrained(output_dir)

        if accelerator.state.deepspeed_plugin.zero_stage==3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(os.path.join(save_dir, f'tfmr'),is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
            
        accelerator.wait_for_everyone()
        accelerator.save_state(save_dir)
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    accelerator.print(accelerator.deepspeed_config)
    if args.resume_checkpoint:
        load_training_state(args.resume_checkpoint)
    
    # model.eval()
    # save_checkpoint(0,0,0)
    
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            # vanilla SFT training
            input_ids=batch['input_ids']
            labels=batch['labels']

            if args.safety_head:
                output_hidden_states=True
            else:
                output_hidden_states=False
            
                
            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False, output_hidden_states=output_hidden_states)
            full_loss = output.loss
            loss = full_loss
            
            # key sentence prediction
            if args.key_sentence_prediction and epoch > args.n_epochs - args.last_k_epoch - 1:
                k_mask = batch['k_mask'] # calculate the loss of key sentence prediction
                attn_mask = batch['attn_mask'] # mask the query tokens
                
                if args.key_sentence_prediction_mask_ablation:
                    masked_out = model(input_ids=input_ids,
                                    return_dict=True, use_cache=False)
                else:
                    masked_out = model(input_ids=input_ids,
                                    attention_mask=attn_mask,
                                    return_dict=True, use_cache=False)
                
                
                masked_out_logits = masked_out.logits           # (B, L, V)

                # 3) shift for next‑token prediction
                shift_logits = masked_out_logits[:, :-1].contiguous()               # (B, L‑1, V)
                shift_labels = input_ids[:, 1:].contiguous()             # (B, L‑1)

                # 4) token‑level NLL
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view_as(shift_labels)                                  # (B, L‑1)

                # 5) isolate the tokens that belong to K
                k_mask_shift = k_mask[:, 1:].float()                     # align with labels
                k_loss = (token_loss * k_mask_shift).sum() / (k_mask_shift.sum() + 1e-8)
            
                loss += k_loss * args.key_sentence_weight
            else:
                k_loss = torch.tensor(0.0).to(loss.device)
                masked_out_logits = None
                k_mask = None
                
            # safety Head
            if args.safety_head and epoch > args.n_epochs - args.last_k_epoch - 1:
                # 1) get the hidden states of the last layer
                if args.detach_safety_head:
                    hidden = output.hidden_states[-1].detach()         # (B, L, H)
                else:
                    hidden = output.hidden_states[-1]                     # (B, L, H)
                
                logit_C, logit_ctx = model.safety_head(hidden,
                                               batch["c_span"])
                y = batch["unsafe"].float()
                L_head_C   = F.binary_cross_entropy_with_logits(logit_C,   y)
                L_head_ctx = F.binary_cross_entropy_with_logits(logit_ctx, y)
                head_loss  = args.head_theta * L_head_C + (1 - args.head_theta) * L_head_ctx
                if args.detach_safety_head:
                    loss = head_loss * args.head_weight
                else:
                    loss += head_loss * args.head_weight
            else:
                head_loss = torch.tensor(0.0).to(loss.device)

            metric(output.logits, labels, loss, full_loss, k_loss, masked_out_logits, k_mask, head_loss)
            acc, acc_k, train_loss, full_loss_avg, k_loss_avg, g_loss_avg = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'acc_k': acc_k,
                    'full_loss': full_loss_avg,
                    'k_loss': k_loss_avg,
                    'g_loss': g_loss_avg,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                accelerator.print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
                break

            if (
                not args.skip_checkpoint_save
                and args.save_every_steps > 0
                and global_step > 0
                and global_step % args.save_every_steps == 0
            ):
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
                last_saved_step = global_step

        if not args.skip_checkpoint_save:
            if last_saved_step != global_step:
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
                last_saved_step = global_step
        else:
            accelerator.print("Skipping checkpoint save (--skip_checkpoint_save).")

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model Args
    parser.add_argument('--model_path', required=True, type=str)

    # Data Args
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=1, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int) # train_bsz_per_gpu * num_gpu should be 8
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--max_train_steps', default=0, type=int)
    parser.add_argument('--save_every_steps', default=0, type=int)
    parser.add_argument('--skip_checkpoint_save', action='store_true')
    parser.add_argument('--resume_checkpoint', default='', type=str)
    parser.add_argument('--base_model', required=True, type=str, choices=['Qwen', 'Llama', 'Olmo', 'Phi', 'GPT_OSS'])
    parser.add_argument('--think_flag', required=True, type=int)
    parser.add_argument('--base_flag', required=True, type=int)
    parser.add_argument('--key_sentence_prediction_mask_ablation', action='store_true')
    parser.add_argument('--key_sentence_prediction', action='store_true')
    parser.add_argument('--key_sentence_weight', default=0.2, type=float)
    parser.add_argument('--safety_head', action='store_true')
    parser.add_argument('--head_theta', default=0.5, type=float)
    parser.add_argument('--head_weight', default=0.2, type=float)
    parser.add_argument('--last_k_epoch', default=2, type=int)
    parser.add_argument('--detach_safety_head', action='store_true')

    # Other Args
    parser.add_argument('--seed', default=2002, type=int)

    args = parser.parse_args()
    model_name = args.model_path.split("/")[-1]
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.log_dir = os.path.join(args.log_dir,model_name)
    args.log_dir = os.path.join(args.log_dir, f"think_flag{args.think_flag}")
    
    
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,model_name)
    args.output_dir = os.path.join(args.output_dir, f"think_flag{args.think_flag}")
    

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
