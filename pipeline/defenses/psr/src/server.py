"""
PSR proxy server (OpenAI-compatible), vLLM backbone.

Thin proxy that receives chat completions, builds a PSR prompt
(chat template + open <think>), then runs the chunked self-reflection
loop against an upstream vLLM, and wraps the result back as a chat
completion.

Only loads the tokenizer (~200MB), no model weights.

Usage:
    PYTHONPATH=. python -m defenses.psr.src.server \
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

from defenses.psr.src.attack import (
    DEFAULT_REFLECTION_TEMPLATE,
    apply_chat_template_safe,
    generate_with_psr_vllm,
    normalize_to_open_think,
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

    enable_thinking = False
    extra = request.extra_body
    ct_kwargs = extra.get("chat_template_kwargs", {})
    if ct_kwargs.get("enable_thinking"):
        enable_thinking = True

    top_k = _args.top_k
    if "top_k" in extra:
        top_k = extra["top_k"]

    base_prompt = apply_chat_template_safe(
        _tokenizer, user_content, enable_thinking_template=enable_thinking,
    )
    prompt_text = normalize_to_open_think(base_prompt, _args.think_open)

    result = generate_with_psr_vllm(
        client=_upstream_client,
        model_name=_args.served_model_name,
        tokenizer=_tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=top_k,
        K=_args.K,
        N=_args.N,
        reflection_template=_args.reflection_template,
        backtrack_on_unsafe=_args.backtrack_on_unsafe,
        logit_bias=_args.logit_bias,
        reflect_in=_args.reflect_in,
        think_close_tag=_args.think_close_tag,
        max_reflection_tokens=_args.max_reflection_tokens,
        stop_strings=_args.stop_string,
    )

    thinking = result["thinking_trace"]
    response = result["response"]
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
                "finish_reason": result.get("finish_reason") or "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": result.get("tokens_generated", 0), "total_tokens": 0},
    }


def main():
    global _tokenizer, _upstream_client, _args

    parser = argparse.ArgumentParser(description="PSR proxy server (vLLM backbone)")
    parser.add_argument("--upstream-url", required=True, help="Upstream vLLM base URL (e.g. http://localhost:8001/v1)")
    parser.add_argument("--served-model-name", required=True, help="Model name (must match upstream)")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--think-open", default="<think>\n")
    parser.add_argument("--think-close-tag", default="</think>")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--K", type=int, default=32, help="Reflection interval in tokens")
    parser.add_argument("--N", type=int, default=4, help="Max reflection rounds; -1 unlimited")
    parser.add_argument("--reflection-template", default=DEFAULT_REFLECTION_TEMPLATE)
    parser.add_argument("--max-reflection-tokens", type=int, default=8)
    parser.add_argument("--backtrack-on-unsafe", action="store_true", default=True)
    parser.add_argument("--no-backtrack-on-unsafe", dest="backtrack_on_unsafe", action="store_false")
    parser.add_argument("--logit-bias", type=float, default=0.0)
    parser.add_argument("--reflect-in", choices=["think", "answer", "both"], default="think")
    parser.add_argument("--stop-string", action="append", default=["<|im_end|>"])
    _args = parser.parse_args()

    print(f"Loading tokenizer from: {_args.tokenizer_path}")
    _tokenizer = AutoTokenizer.from_pretrained(_args.tokenizer_path, trust_remote_code=True)

    _upstream_client = OpenAI(api_key="dummy", base_url=_args.upstream_url)
    print(f"PSR proxy: upstream={_args.upstream_url}, model={_args.served_model_name}, port={_args.port}")

    uvicorn.run(app, host=_args.host, port=_args.port)


if __name__ == "__main__":
    main()
