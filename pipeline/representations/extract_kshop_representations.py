#!/usr/bin/env python3
"""Extract K-hop prefill hidden states over the user-content span.

For each item in the eval pool, runs prefill forward (no decoding) over
`chat_text + <think>\\n + first-thinking-content-token` and reads out hidden
states at K+1 anchor positions:

  - k = 0 .. K-1  equally spaced across [user_content_start, last_chat_text_token]
                  (the per-item-variable span; system prompt and opening chat
                  scaffolding are excluded from the sweep)
  - k = K         the existing first-thinking-content-token position
                  (= eval_results/.../representations/prefill/*.pt anchor)

At each anchor we save hidden states at 5 layer indices on the grid
{0, L//4, L//2, 3L//4, L} where L = num_transformer_layers (so hidden_states[i]
for i in {0, L//4, L//2, 3L//4, L}; index 0 = embeddings, index L = post-final-norm).

Output (fp16 on disk):
  experiments/refusal_cliff/artifacts/kshop_K20_user_content/<MODEL>/<intent>.pt
    {
      ids: [N],
      X: [N, K+1, 5, D],            # anchor x layer x hidden
      y: [N],                        # majority refusal label (4-guardrail)
      y_per_guardrail: [N, 4],
      anchor_token_positions: [N, K+1],  # absolute positions used per item
      user_content_start: [N],       # first user-content token index
      last_chat_text_token: [N],     # last formatted-prompt token index
      thinking_start: [N],           # first thinking-content token index (= k=K anchor)
      formatted_prompt_len: [N],     # = last_chat_text_token + 1
      layer_indices: [5],            # which layer indices in hidden_states[] we saved
      meta: {model, K, ...}
    }

Per-model specs (batch_size, channel vs think, max_model_len) come from
scripts/neurips_final/common.py.

Provenance: copied (logic unchanged) from scripts/neurips_final/extract_kshop_representations.py.
The model registry + 4-guardrail majority vote now also live in the shared
`lrm_safety_deliberation` library (`lrm_safety_deliberation.models`, `lrm_safety_deliberation.guardrails`); this driver
keeps importing the bundled `common.py` so it runs standalone. Set
LRM_SAFETY_ARTIFACTS to relocate the artifact tree.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402  (bundled copy; see lrm_safety_deliberation.models/guardrails)
    MODELS,
    SUITES,
    majority_tie_negative,
    read_jsonl,
    suite_out_root,
)

ROOT = Path(os.environ.get("LRM_SAFETY_ARTIFACTS", Path(__file__).resolve().parents[3]))
OUT_ROOT = ROOT / "experiments" / "refusal_cliff" / "artifacts" / "kshop_K20_user_content"
GUARDRAIL_ORDER = ("wildguard", "qwen3guard", "granite_guardian", "oss_safeguard")
K_HOP = 20


def guardrail_votes(row: dict) -> list[bool]:
    gr = row.get("guardrails", {})
    votes = []
    for name in GUARDRAIL_ORDER:
        p = gr.get(name, {})
        if name == "wildguard":
            votes.append(bool(p.get("response_refusal", False)))
        elif name == "qwen3guard":
            votes.append(str(p.get("refusal", "")).strip().lower() == "yes")
        elif name == "granite_guardian":
            votes.append(str(p.get("judgment", "")).strip().lower() == "yes")
        elif name == "oss_safeguard":
            votes.append(bool(p.get("response_refusal", False)))
    return votes


def majority_refusal(row: dict) -> bool:
    return majority_tie_negative(guardrail_votes(row))


def build_prefix_marker(chat_text: str, is_channel: bool, is_pretrained: bool,
                        channel_target: str = "analysis") -> str:
    """Marker appended to chat_text to anchor the first thinking-content token."""
    if is_pretrained:
        return ""  # pretrained: thinking immediately follows 'Assistant:'
    if is_channel:
        if channel_target == "final":
            return "<|channel|>final<|message|>"
        return "<|channel|>analysis<|message|>"
    # <think>-family
    if chat_text.rstrip().endswith("<think>"):
        return "\n"
    return "<think>\n"


def find_user_content_start(tokenizer, chat_text: str, prompt: str) -> int:
    """Return token index where user-message content begins inside chat_text.

    Strategy: find character offset of the LAST occurrence of `prompt` in
    `chat_text` (last because some templates include example prompts in the
    system message); locate first token whose start-offset >= that character.
    """
    enc = tokenizer(chat_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    char_start = chat_text.rfind(prompt)
    if char_start < 0:
        # Prompt not found verbatim (rare — template-side escaping). Fall back
        # to first non-template token after the last role marker.
        # Heuristic: take token index right after the longest contiguous
        # block of special tokens at the start.
        for i, (a, b) in enumerate(offsets):
            if b > a:
                return i
        return 0
    for i, (a, b) in enumerate(offsets):
        if b > a and a >= char_start:
            return i
    return len(enc["input_ids"]) - 1


def compute_anchors(user_start: int, last_prompt_pos: int, thinking_start: int,
                    K: int = K_HOP) -> list[int]:
    """K equally spaced anchors over [user_start, last_prompt_pos] + 1 splice at thinking_start."""
    span = last_prompt_pos - user_start
    if span <= 0:
        # Degenerate: user content collapses or runs past prompt end. All anchors
        # at user_start.
        return [user_start] * K + [thinking_start]
    if K == 1:
        anchors = [last_prompt_pos]
    else:
        anchors = [user_start + round(k * span / (K - 1)) for k in range(K)]
    # Clamp (defensive — should already be in range)
    anchors = [max(user_start, min(last_prompt_pos, a)) for a in anchors]
    anchors.append(thinking_start)
    return anchors


def layer_grid_indices(num_hidden_states: int) -> list[int]:
    """5 layer indices spanning {0, L//4, L//2, 3L//4, L} where L = num_transformer_layers.

    hidden_states tuple from output_hidden_states=True has length L+1
    (1 embedding + L transformer outputs). num_hidden_states arg is L+1.
    """
    L = num_hidden_states - 1
    return sorted({0, L // 4, L // 2, (3 * L) // 4, L})


def _get_base(model):
    base = model
    for attr in ("model", "transformer", "gpt_neox", "backbone"):
        b = getattr(model, attr, None)
        if b is not None and hasattr(b, "forward") and b is not model:
            return b
    return base


@torch.no_grad()
def forward_kept_layers(model, input_ids: torch.Tensor, attn: torch.Tensor,
                        keep_indices: list[int]):
    """Forward pass keeping ONLY the requested hidden-state layer indices.

    Returns dict {layer_idx: [B, L, D] fp16 CPU tensor}. Filtering happens
    BEFORE moving off GPU, so unused layers' fp16 CPU copies are never
    allocated.
    """
    base = _get_base(model)
    out = base(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=True,
    )
    hs = out.hidden_states  # tuple of (L+1) GPU tensors
    kept = {idx: hs[idx].detach().to(torch.float16).cpu() for idx in keep_indices}
    del out, hs
    return kept


@torch.no_grad()
def forward_kept_layers_stream(model, input_ids: torch.Tensor, attn: torch.Tensor,
                                keep_indices: list[int], chunk: int = 128):
    """Streaming forward (DynamicCache) keeping only requested layers."""
    from transformers import DynamicCache

    base = _get_base(model)
    B, L = input_ids.shape
    dev = model.device
    past = DynamicCache()
    pos = 0
    parts: dict[int, list[torch.Tensor]] = {idx: [] for idx in keep_indices}
    while pos < L:
        end = min(L, pos + chunk)
        ids = input_ids[:, pos:end].to(dev)
        mask = attn[:, :end].to(dev)
        out = base(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        for idx in keep_indices:
            parts[idx].append(out.hidden_states[idx].detach().to(torch.float16).cpu())
        past = out.past_key_values
        del out
        pos = end
    return {idx: torch.cat(parts[idx], dim=1) for idx in keep_indices}


def collate_pad(batch: list[list[int]], pad_id: int):
    lens = [len(x) for x in batch]
    max_len = max(lens)
    B = len(batch)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn[i, : len(seq)] = 1
    return ids, attn, lens


def prepare_item(tokenizer, row: dict, spec: dict, max_model_len: int) -> dict | None:
    """Tokenize one row and locate (user_start, last_prompt_pos, thinking_start)."""
    is_channel = spec.get("is_channel", False)
    is_pretrained = spec.get("is_pretrained", False)
    prompt = row.get("prompt", "")
    thinking = row.get("thinking_trace", "")
    response = row.get("response", "")
    channel_target = "analysis"
    if is_channel and (not thinking or len(thinking.strip()) < 3) and response and len(response.strip()) >= 3:
        thinking = response
        channel_target = "final"
    if not thinking or len(thinking.strip()) < 3:
        return None

    if is_pretrained:
        chat = f"User: {prompt}\n\nAssistant:"
    else:
        messages = [{"role": "user", "content": prompt}]
        try:
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except TypeError:
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    marker = build_prefix_marker(chat, is_channel, is_pretrained, channel_target)
    prefix_text = chat + marker

    # Token positions
    chat_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    last_prompt_pos = len(chat_ids) - 1
    thinking_start = len(prefix_ids)

    # User-content start within chat_text (then carried over to prefix_text — same indices)
    if is_pretrained:
        # Locate 'User: <prompt>' content position
        user_start = find_user_content_start(tokenizer, chat, prompt)
    else:
        user_start = find_user_content_start(tokenizer, chat, prompt)

    if user_start > last_prompt_pos:
        return None

    # We need rep at thinking_start; forward over [0, thinking_start] inclusive.
    # That requires feeding `prefix_ids + [first_thinking_content_token_id]`.
    # We obtain that token by tokenizing prefix_text + 1 char of `thinking`,
    # but a robust way: tokenize prefix_text + thinking up to a few chars and
    # take the (thinking_start)-th token.
    raw_prefix_chars = max(1, min(len(thinking), 64))
    for _ in range(3):
        full_text = prefix_text + thinking[:raw_prefix_chars]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > thinking_start:
            break
        raw_prefix_chars *= 2
    else:
        return None

    forward_ids = full_ids[: thinking_start + 1]
    if len(forward_ids) > max_model_len:
        return None  # prompt too long for K-hop

    anchors = compute_anchors(user_start, last_prompt_pos, thinking_start, K=K_HOP)

    return {
        "id": row.get("global_id", row.get("sample_id", -1)),
        "forward_ids": forward_ids,
        "anchors": anchors,
        "user_start": user_start,
        "last_prompt_pos": last_prompt_pos,
        "thinking_start": thinking_start,
        "formatted_prompt_len": last_prompt_pos + 1,
        "y": int(majority_refusal(row)),
        "y_per_guardrail": guardrail_votes(row),
    }


def _batch_size_for_seq(seq_len: int, base_batch: int) -> int:
    """Length-bucketed batch size: keep activations bounded for long-tail items."""
    if seq_len <= 512:
        return base_batch
    if seq_len <= 2048:
        return max(1, min(base_batch, 4))
    if seq_len <= 4096:
        return max(1, min(base_batch, 2))
    return 1


def run_split(rows, model, tokenizer, spec, out_dir: Path, intent: str) -> None:
    max_model_len = int(spec.get("max_model_len", 8192))
    base_batch = int(spec.get("batch_size_hf", 8))
    stream = bool(spec.get("stream_with_cache", False))
    stream_chunk = int(spec.get("stream_chunk_size", 128))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    D = int(getattr(model.config, "hidden_size", None) or
            getattr(getattr(model.config, "text_config", model.config), "hidden_size"))
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    layer_indices = layer_grid_indices(num_hidden_layers + 1)
    print(f"[{intent}] layers={num_hidden_layers + 1} grid_indices={layer_indices}")

    prepared = []
    skipped = 0
    for row in rows:
        try:
            item = prepare_item(tokenizer, row, spec, max_model_len)
        except Exception as e:
            print(f"[{intent}] prepare error id={row.get('global_id')}: {e}")
            item = None
        if item is None:
            skipped += 1
            continue
        prepared.append(item)
    print(f"[{intent}] prepared={len(prepared)} skipped={skipped}")
    if not prepared:
        print(f"[{intent}] nothing to extract; skipping")
        return

    # Sort by forward length to keep length-bucketed batches dense.
    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]["forward_ids"]))
    prepared_sorted = [prepared[i] for i in order]

    N = len(prepared_sorted)
    K_total = K_HOP + 1
    X = torch.empty(N, K_total, len(layer_indices), D, dtype=torch.float16)
    y = torch.empty(N, dtype=torch.long)
    y_pg = torch.empty(N, 4, dtype=torch.long)
    anchor_positions = torch.empty(N, K_total, dtype=torch.long)
    user_starts = torch.empty(N, dtype=torch.long)
    last_prompt_positions = torch.empty(N, dtype=torch.long)
    thinking_starts = torch.empty(N, dtype=torch.long)
    formatted_lens = torch.empty(N, dtype=torch.long)
    ids_list = []

    t0 = time.time()
    cursor = 0
    while cursor < N:
        seq_len = len(prepared_sorted[cursor]["forward_ids"])
        bs = _batch_size_for_seq(seq_len, base_batch)
        end = min(N, cursor + bs)
        chunk = prepared_sorted[cursor:end]
        # Don't mix wildly different lengths in one batch — only group items
        # whose lengths land in the same bucket. The sort above already groups
        # them, but a long item at the bucket boundary could inflate this batch.
        # The recomputed seq_len uses the FIRST item; items grow from there.
        forward_ids_batch = [c["forward_ids"] for c in chunk]
        ids_t, attn_t, lens = collate_pad(forward_ids_batch, pad_id)
        try:
            if stream:
                hs_kept = forward_kept_layers_stream(model, ids_t, attn_t,
                                                     keep_indices=layer_indices,
                                                     chunk=stream_chunk)
            else:
                ids_dev = ids_t.to(model.device)
                attn_dev = attn_t.to(model.device)
                hs_kept = forward_kept_layers(model, ids_dev, attn_dev,
                                              keep_indices=layer_indices)
                del ids_dev, attn_dev
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{intent}] OOM at cursor={cursor} seq_len={seq_len} bs={bs}; retrying bs=1")
            if len(chunk) > 1:
                # Retry one item at a time
                end = cursor + 1
                chunk = chunk[:1]
                ids_t, attn_t, lens = collate_pad([chunk[0]["forward_ids"]], pad_id)
                if stream:
                    hs_kept = forward_kept_layers_stream(model, ids_t, attn_t,
                                                         keep_indices=layer_indices,
                                                         chunk=stream_chunk)
                else:
                    ids_dev = ids_t.to(model.device)
                    attn_dev = attn_t.to(model.device)
                    hs_kept = forward_kept_layers(model, ids_dev, attn_dev,
                                                  keep_indices=layer_indices)
                    del ids_dev, attn_dev
            else:
                raise

        # hs_kept: {layer_idx: [B, L_seq, D] fp16 CPU}
        for bi, item in enumerate(chunk):
            global_idx = cursor + bi
            anchors = item["anchors"]
            L_i = lens[bi]
            anchors_clamped = [min(a, L_i - 1) for a in anchors]
            for ki, anc in enumerate(anchors_clamped):
                for li_out, li_src in enumerate(layer_indices):
                    X[global_idx, ki, li_out] = hs_kept[li_src][bi, anc]
            y[global_idx] = item["y"]
            y_pg[global_idx] = torch.tensor(
                [int(v) for v in item["y_per_guardrail"]], dtype=torch.long
            )
            anchor_positions[global_idx] = torch.tensor(anchors_clamped, dtype=torch.long)
            user_starts[global_idx] = item["user_start"]
            last_prompt_positions[global_idx] = item["last_prompt_pos"]
            thinking_starts[global_idx] = item["thinking_start"]
            formatted_lens[global_idx] = item["formatted_prompt_len"]
            ids_list.append(item["id"])
        del hs_kept
        cursor = end
        if cursor % max(1, (10 * base_batch)) < bs:
            elapsed = time.time() - t0
            rate = cursor / max(elapsed, 1e-3)
            eta = (N - cursor) / max(rate, 1e-3)
            print(
                f"[{intent}] {cursor}/{N} ({100*cursor/N:.1f}%) "
                f"rate={rate:.2f}/s eta={eta/60:.1f} min seq={seq_len} bs={bs}",
                flush=True,
            )
    ids = ids_list  # final ordering follows length sort; that's fine for downstream

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{intent}.pt"
    torch.save(
        {
            "ids": ids,
            "X": X,
            "y": y,
            "y_per_guardrail": y_pg,
            "anchor_token_positions": anchor_positions,
            "user_content_start": user_starts,
            "last_chat_text_token": last_prompt_positions,
            "thinking_start": thinking_starts,
            "formatted_prompt_len": formatted_lens,
            "layer_indices": torch.tensor(layer_indices, dtype=torch.long),
            "K_hop": K_HOP,
            "guardrail_order": list(GUARDRAIL_ORDER),
        },
        fname,
    )
    print(f"[{intent}] wrote {fname}  X.shape={tuple(X.shape)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suites", nargs="+", default=["safety", "supplementary"],
                    help="Which suites to extract over (default: both).")
    ap.add_argument("--intents", nargs="+", default=None,
                    help="Override default intents for the chosen suites.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Truncate to first N rows per intent (smoke tests).")
    args = ap.parse_args()

    if args.model not in MODELS:
        raise KeyError(args.model)
    spec = MODELS[args.model]
    model_path = ROOT / spec["model_path"] if not Path(spec["model_path"]).is_absolute() else Path(spec["model_path"])

    out_dir = OUT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print(f"Loading HF model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    ).eval()
    print(f"Model loaded. hidden_size={model.config.hidden_size if hasattr(model.config, 'hidden_size') else '?'}")

    for suite in args.suites:
        if suite not in SUITES:
            print(f"[skip] unknown suite={suite}")
            continue
        out_root = suite_out_root(suite)
        cls_dir = out_root / args.model / "classifications" / "base"
        if not cls_dir.exists():
            print(f"[skip] missing classifications: {cls_dir}")
            continue
        if args.intents is None:
            if suite == "safety":
                intents = ["harmful", "benign"]
            elif suite == "supplementary":
                intents = ["phtest", "orfuzz"]
            else:
                continue  # skip negctrl/pretrained for K-hop
        else:
            intents = args.intents

        for intent in intents:
            path = cls_dir / f"{intent}_guardrails.jsonl"
            if not path.exists():
                print(f"[{suite}/{intent}] missing {path}; skipping")
                continue
            rows = read_jsonl(path)
            if args.limit:
                rows = rows[: args.limit]
            print(f"[{suite}/{intent}] {len(rows)} rows from {path.name}")
            run_split(rows, model, tokenizer, spec, out_dir, intent)

    meta_path = out_dir / "_meta.json"
    meta_path.write_text(json.dumps({
        "model": args.model,
        "K_hop": K_HOP,
        "layer_grid": "{0, L//4, L//2, 3L//4, L}",
        "dtype": "fp16",
        "guardrail_order": list(GUARDRAIL_ORDER),
    }, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
