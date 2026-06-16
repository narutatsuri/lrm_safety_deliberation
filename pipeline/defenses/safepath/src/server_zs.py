"""
SafePath ZS proxy server (OpenAI-compatible).

Thin proxy that receives chat completions, builds a SafePath prompt
(chat template + "<think>\nLet's think about safety first."), then
forwards to an upstream vLLM as a text completion (/v1/completions),
and wraps the response back as a chat completion.

Only loads the tokenizer (~200MB), no model weights.

Usage:
    PYTHONPATH=. python -m defenses.safepath.src.server_zs \
        --upstream-url http://localhost:8001/v1 \
        --served-model-name Qwen3-8B \
        --tokenizer-path models/Qwen3-8B \
        --port 8002
"""

from __future__ import annotations

import argparse
import time
import uuid

import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from defenses.safepath.src.attack_zs import (
    SAFETY_PRIMER,
    apply_chat_template_safe,
    normalize_to_open_think,
    split_thinking_and_response,
)

app = FastAPI()

# Global state populated by main()
_tokenizer = None
_upstream_client = None
_args = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    extra_body: dict = Field(default_factory=dict)


def _extract_user_content(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": _args.served_model_name, "object": "model"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    user_content = _extract_user_content(request.messages)

    # Handle enable_thinking from extra_body
    enable_thinking = False
    extra = request.extra_body
    ct_kwargs = extra.get("chat_template_kwargs", {})
    if ct_kwargs.get("enable_thinking"):
        enable_thinking = True

    # Handle top_k from extra_body
    top_k = _args.top_k
    if "top_k" in extra:
        top_k = extra["top_k"]

    # Build SafePath prompt: chat template + open <think> + safety primer
    messages = [{"role": "user", "content": user_content}]
    base_prompt = apply_chat_template_safe(
        _tokenizer, messages, enable_thinking=enable_thinking,
    )
    open_think_prompt = normalize_to_open_think(base_prompt, _args.think_open)
    prompt = f"{open_think_prompt}{_args.primer_text}"

    # Forward to upstream vLLM as text completion
    extra_body = {}
    if top_k is not None and top_k > 0:
        extra_body["top_k"] = top_k

    resp = _upstream_client.completions.create(
        model=_args.served_model_name,
        prompt=prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        extra_body=extra_body if extra_body else None,
    )
    raw_text = (resp.choices[0].text or "").strip()

    # Parse thinking and response
    thinking, response = split_thinking_and_response(raw_text)

    # Build response with <think>...</think> tags
    if thinking:
        content = f"<think>{thinking}</think>\n{response}"
    else:
        content = response

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _args.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main():
    global _tokenizer, _upstream_client, _args

    parser = argparse.ArgumentParser(description="SafePath ZS proxy server")
    parser.add_argument("--upstream-url", required=True, help="Upstream vLLM base URL (e.g. http://localhost:8001/v1)")
    parser.add_argument("--served-model-name", required=True, help="Model name (must match upstream)")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--think-open", default="<think>\n")
    parser.add_argument("--primer-text", default=SAFETY_PRIMER)
    parser.add_argument("--top-k", type=int, default=50)
    _args = parser.parse_args()

    print(f"Loading tokenizer from: {_args.tokenizer_path}")
    _tokenizer = AutoTokenizer.from_pretrained(_args.tokenizer_path, trust_remote_code=True)

    _upstream_client = OpenAI(api_key="dummy", base_url=_args.upstream_url)
    print(f"SafePath ZS proxy: upstream={_args.upstream_url}, model={_args.served_model_name}, port={_args.port}")

    uvicorn.run(app, host=_args.host, port=_args.port)


if __name__ == "__main__":
    main()
