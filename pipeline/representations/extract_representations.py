#!/usr/bin/env python3
"""Extract last-layer hidden states for prefill + thinking-hopping + thinking-pooling.

Reads `classifications/base/{harmful,benign}_guardrails.jsonl` for a model,
forwards each (prompt, raw_response) through the HF model, slices the thinking
span, and persists three tensor sets under `representations/`.

Per-model specs (batch_size, channel vs think, max_model_len) come from
scripts/neurips_final/common.py.

Outputs (fp16 on disk):
  representations/prefill/{harmful,benign}.pt
      {ids: [N], X: [N,D], y: [N], y_per_guardrail: [N,4], trace_lens: [N]}
  representations/thinking_hopping/{harmful,benign}.pt
      {ids: [N], X: [N,100,D], y: [N], trace_lens: [N], method: "hopping"}
  representations/thinking_pooling/{harmful,benign}.pt
      {ids: [N], X: [N,100,D], y: [N], trace_lens: [N], method: "chunk_mean_pool"}

Provenance: copied (logic unchanged) from scripts/neurips_final/extract_representations.py.
The model registry + 4-guardrail majority vote now also live in the shared
`lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`); this driver
keeps importing the bundled `common.py` so it runs standalone. Set
LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402  (bundled copy; see lrm_safety_deliberation.models/guardrails)
    MODELS, OUT_ROOT, SUITES, majority_tie_negative, read_jsonl,
    suite_out_root,
)

GUARDRAIL_ORDER = ("wildguard", "qwen3guard", "granite_guardian", "oss_safeguard")


def guardrail_votes(row: dict) -> list[bool]:
    gr = row.get("guardrails", {})
    votes = []
    for name in GUARDRAIL_ORDER:
        payload = gr.get(name, {})
        if name == "wildguard":
            votes.append(bool(payload.get("response_refusal", False)))
        elif name == "qwen3guard":
            votes.append(str(payload.get("refusal", "")).strip().lower() == "yes")
        elif name == "granite_guardian":
            votes.append(str(payload.get("judgment", "")).strip().lower() == "yes")
        elif name == "oss_safeguard":
            votes.append(bool(payload.get("response_refusal", False)))
    return votes


def majority_refusal(row: dict) -> bool:
    return majority_tie_negative(guardrail_votes(row))


def build_full_text(tokenizer, prompt: str, thinking: str, is_channel: bool,
                    is_pretrained: bool = False, channel_target: str = "analysis") -> str:
    if is_pretrained:
        # Plain User/Assistant prompt matching generation. Thinking IS response.
        return f"User: {prompt}\n\nAssistant:{thinking}"
    messages = [{"role": "user", "content": prompt}]
    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if is_channel:
        if channel_target == "final":
            # SFT-trained channel models (e.g. STAR1-GPT-OSS, SafeKey-GPT-OSS) emit
            # the response on the `final` channel only; no analysis channel exists.
            return chat + "<|channel|>final<|message|>" + thinking + "<|return|>"
        return chat + "<|channel|>analysis<|message|>" + thinking + "<|end|>"
    # <think>-family
    if chat.rstrip().endswith("<think>"):
        return chat + "\n" + thinking + "\n</think>"
    return chat + "<think>\n" + thinking + "\n</think>"


def locate_thinking_span(
    tokenizer, chat_text: str, full_text: str, is_channel: bool, is_pretrained: bool = False,
    channel_target: str = "analysis",
) -> tuple[int, int, int, list[int]]:
    """Return (prefill_pos, thinking_start, thinking_end, full_ids).

    For pretrained models: span is full generation after 'User: ... \\nAssistant:'.
    """
    if is_pretrained:
        prefix = chat_text  # for pretrained, chat_text already = 'User: prompt\n\nAssistant:'
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        thinking_start = len(prefix_ids)
        thinking_end = len(full_ids)
        thinking_end = max(thinking_start + 1, thinking_end)
        prefill_pos = thinking_start
        return prefill_pos, thinking_start, thinking_end, full_ids

    prefix = chat_text
    if is_channel:
        if channel_target == "final":
            prefix = chat_text + "<|channel|>final<|message|>"
        else:
            prefix = chat_text + "<|channel|>analysis<|message|>"
    elif chat_text.rstrip().endswith("<think>"):
        prefix = chat_text + "\n"
    else:
        prefix = chat_text + "<think>\n"

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    thinking_start = len(prefix_ids)

    if is_channel:
        close_token = "<|return|>" if channel_target == "final" else "<|end|>"
        close_ids = tokenizer(close_token, add_special_tokens=False)["input_ids"]
    else:
        close_ids = tokenizer("\n</think>", add_special_tokens=False)["input_ids"]

    thinking_end = len(full_ids) - len(close_ids)
    thinking_end = max(thinking_start + 1, thinking_end)
    prefill_pos = thinking_start  # first thinking token == first decoding token
    return prefill_pos, thinking_start, thinking_end, full_ids


def hopping_100(H: torch.Tensor) -> torch.Tensor:
    """Linear-interpolate sequence [L,D] -> [100,D] by picking one token per chunk."""
    L = H.shape[0]
    if L >= 100:
        idx = np.floor(np.linspace(0, L - 1, 100)).astype(np.int64)
        return H[idx]
    # L < 100: upsample with repetition
    idx = np.floor(np.linspace(0, L - 1, 100)).astype(np.int64)
    return H[idx]


def pooling_100(H: torch.Tensor) -> torch.Tensor:
    """Non-overlapping chunk-mean pool: [L,D] -> [100,D]."""
    L = H.shape[0]
    if L <= 100:
        # Upsample by repetition (fallback; same as hopping)
        idx = np.floor(np.linspace(0, L - 1, 100)).astype(np.int64)
        return H[idx]
    edges = np.linspace(0, L, 101).astype(np.int64)
    parts = [H[edges[i]:edges[i + 1]].mean(dim=0) for i in range(100)]
    return torch.stack(parts, dim=0)


def _resolve_final_norm(model):
    """Return the module whose forward output equals hidden_states[-1].

    For Llama/Qwen/Olmo/DeepSeek/Phi and GPT-OSS this is `model.model.norm`
    (a final RMSNorm applied before the LM head).
    """
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        for norm_attr in ("norm", "final_layernorm", "ln_f"):
            n = getattr(base, norm_attr, None)
            if n is not None:
                return n
    raise AttributeError("Could not locate final norm module on model")


def _get_base_model(model):
    """Return the base transformer (without LM head) when available.

    Avoids computing the full [B,L,V] logits tensor, which OOMs for large
    vocab * long seq (e.g. Qwen3's 150K vocab at seq=8K costs ~40 GB in fp16).
    """
    for attr in ("model", "transformer", "gpt_neox", "backbone"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "forward") and base is not model:
            return base
    return model


@torch.no_grad()
def forward_last_hidden(model, input_ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Run forward on the BASE transformer (skipping lm_head) and return
    last-layer post-norm hidden states [B, L, D] fp16 (CPU)."""
    base = _get_base_model(model)
    out = base(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=False,
    )
    # BaseModelOutputWithPast.last_hidden_state is post-final-norm.
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h.detach().to(torch.float16).cpu()


@torch.no_grad()
def forward_last_hidden_stream(
    model, input_ids: torch.Tensor, attn: torch.Tensor, chunk: int = 128
) -> torch.Tensor:
    """Streaming forward on the BASE transformer (skipping lm_head).

    Pushes chunks through the model with DynamicCache; collects last-layer
    hidden states per chunk on CPU to avoid OOM on long seqs.
    """
    from transformers import DynamicCache

    base = _get_base_model(model)
    chunks_out: list[torch.Tensor] = []
    B, L = input_ids.shape
    dev = model.device
    past = DynamicCache()
    pos = 0
    while pos < L:
        end = min(L, pos + chunk)
        ids = input_ids[:, pos:end].to(dev)
        mask = attn[:, :end].to(dev)
        out = base(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        chunks_out.append(h.detach().to(torch.float16).cpu())
        past = out.past_key_values
        del out, h
        pos = end
    return torch.cat(chunks_out, dim=1)  # [B, L, D]


def collate_pad(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lens = [len(x) for x in batch]
    max_len = max(lens)
    B = len(batch)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn[i, : len(seq)] = 1
    return ids, attn, lens


def run_split(
    rows: list[dict],
    model,
    tokenizer,
    spec: dict,
    out_dir: Path,
    intent: str,
) -> None:
    is_channel = spec.get("is_channel", False)
    is_pretrained = spec.get("is_pretrained", False)
    stream = spec.get("stream_with_cache", False)
    stream_chunk = spec.get("stream_chunk_size", 128)
    batch_size = spec.get("batch_size_hf", 8)
    max_model_len = spec.get("max_model_len", 8192)
    D = getattr(model.config, "hidden_size", None)
    if D is None and hasattr(model.config, "text_config"):
        D = model.config.text_config.hidden_size
    assert D is not None, "cannot determine hidden_size"
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    # Pre-tokenize and locate spans.
    prepared = []
    skipped = 0
    for row in rows:
        prompt = row.get("prompt", "")
        thinking = row.get("thinking_trace", "")
        response = row.get("response", "")
        # SFT-trained channel-format models (e.g. STAR1-GPT-OSS, SafeKey-GPT-OSS)
        # collapse the analysis channel during fine-tuning and emit only the final
        # channel. Use the response text as the trace span instead.
        channel_target = "analysis"
        if is_channel and (not thinking or len(thinking.strip()) < 3) and response and len(response.strip()) >= 3:
            thinking = response
            channel_target = "final"
        if not thinking or len(thinking.strip()) < 3:
            skipped += 1
            continue
        try:
            if is_pretrained:
                chat = f"User: {prompt}\n\nAssistant:"
            else:
                messages = [{"role": "user", "content": prompt}]
                try:
                    chat = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    chat = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            full_text = build_full_text(tokenizer, prompt, thinking, is_channel, is_pretrained,
                                         channel_target=channel_target)
            prefill, t_start, t_end, full_ids = locate_thinking_span(
                tokenizer, chat, full_text, is_channel, is_pretrained,
                channel_target=channel_target,
            )
            if t_end - t_start < 3:
                skipped += 1
                continue
            # Truncate overly long inputs
            if len(full_ids) > max_model_len:
                full_ids = full_ids[:max_model_len]
                t_end = min(t_end, max_model_len - 1)
                if t_end - t_start < 3:
                    skipped += 1
                    continue
            prepared.append({
                "id": row.get("global_id", row.get("sample_id", len(prepared))),
                "input_ids": full_ids,
                "prefill_pos": prefill,
                "t_start": t_start,
                "t_end": t_end,
                "y": int(majority_refusal(row)),
                "y_per_guardrail": guardrail_votes(row),
                "trace_len": t_end - t_start,
            })
        except Exception as e:
            print(f"[{intent}] prep error on id={row.get('global_id')}: {e}")
            skipped += 1
    print(f"[{intent}] prepared={len(prepared)} skipped={skipped}")

    if not prepared:
        print(f"[{intent}] NOTHING TO EXTRACT — aborting split")
        return

    # Accumulators
    N = len(prepared)
    prefill_X = torch.empty(N, D, dtype=torch.float16)
    hopping_X = torch.empty(N, 100, D, dtype=torch.float16)
    pooling_X = torch.empty(N, 100, D, dtype=torch.float16)
    y = torch.empty(N, dtype=torch.long)
    y_pg = torch.empty(N, 4, dtype=torch.long)
    ids = []
    trace_lens = torch.empty(N, dtype=torch.long)

    t0 = time.time()
    for bstart in range(0, N, batch_size):
        chunk = prepared[bstart : bstart + batch_size]
        input_ids, attn, lens = collate_pad([c["input_ids"] for c in chunk], pad_id)
        if stream:
            H = forward_last_hidden_stream(model, input_ids, attn, chunk=stream_chunk)
        else:
            input_ids_dev = input_ids.to(model.device)
            attn_dev = attn.to(model.device)
            H = forward_last_hidden(model, input_ids_dev, attn_dev).cpu()
            del input_ids_dev, attn_dev
        # H shape: [B, max_len, D]
        for bi, item in enumerate(chunk):
            Hi = H[bi, : lens[bi]]  # [L_i, D]
            pi = item["prefill_pos"]
            ts = item["t_start"]
            te = item["t_end"]
            # Prefill token = first thinking token
            prefill_X[bstart + bi] = Hi[pi]
            span = Hi[ts:te]  # [trace_len, D]
            hopping_X[bstart + bi] = hopping_100(span)
            pooling_X[bstart + bi] = pooling_100(span)
            y[bstart + bi] = item["y"]
            y_pg[bstart + bi] = torch.tensor(
                [int(v) for v in item["y_per_guardrail"]], dtype=torch.long
            )
            ids.append(item["id"])
            trace_lens[bstart + bi] = item["trace_len"]
        if (bstart // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            done = min(bstart + batch_size, N)
            rate = done / max(elapsed, 1e-3)
            eta = (N - done) / max(rate, 1e-3)
            print(
                f"[{intent}] {done}/{N} ({100*done/N:.1f}%) "
                f"rate={rate:.2f}/s eta={eta/60:.1f} min"
            )

    (out_dir / "prefill").mkdir(parents=True, exist_ok=True)
    (out_dir / "thinking_hopping").mkdir(parents=True, exist_ok=True)
    (out_dir / "thinking_pooling").mkdir(parents=True, exist_ok=True)
    fname = f"{intent}.pt"
    torch.save(
        {"ids": ids, "X": prefill_X, "y": y, "y_per_guardrail": y_pg, "trace_lens": trace_lens},
        out_dir / "prefill" / fname,
    )
    torch.save(
        {"ids": ids, "X": hopping_X, "y": y, "trace_lens": trace_lens, "method": "hopping"},
        out_dir / "thinking_hopping" / fname,
    )
    torch.save(
        {
            "ids": ids,
            "X": pooling_X,
            "y": y,
            "trace_lens": trace_lens,
            "method": "chunk_mean_pool",
        },
        out_dir / "thinking_pooling" / fname,
    )
    print(f"[{intent}] wrote prefill/{fname}, thinking_hopping/{fname}, thinking_pooling/{fname}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="safety", choices=list(SUITES))
    ap.add_argument("--intents", nargs="+", default=None,
                    help="Override default intents for this suite.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Truncate to first N rows per intent (smoke tests).")
    args = ap.parse_args()

    if args.model not in MODELS:
        raise KeyError(args.model)
    spec = MODELS[args.model]
    model_path = spec["model_path"]

    if args.intents is None:
        if args.suite == "negctrl":
            args.intents = ["math", "gpqa"]
        elif args.suite == "supplementary":
            args.intents = ["phtest", "orfuzz"]
        else:
            args.intents = ["harmful", "benign"]

    out_root = suite_out_root(args.suite)
    cls_dir = out_root / args.model / "classifications" / "base"
    out_dir = out_root / args.model / "representations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()

    for intent in args.intents:
        path = cls_dir / f"{intent}_guardrails.jsonl"
        if not path.exists():
            print(f"[{intent}] missing {path}; skipping")
            continue
        rows = read_jsonl(path)
        if args.limit:
            rows = rows[: args.limit]
        run_split(rows, model, tokenizer, spec, out_dir, intent)

    meta_path = out_root / args.model / "representations" / "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "layer": "last",
                "hidden_dim": int(getattr(model.config, "hidden_size", 0)),
                "dtype": "fp16",
                "guardrail_order": list(GUARDRAIL_ORDER),
            },
            f,
            indent=2,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
