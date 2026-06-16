"""Model registry: the single source of truth for which models the paper uses
and how each one is decoded.

This consolidates the ``MODELS`` dict that the research code re-declared in
several ``common.py`` files.  Every model is described by a :class:`ModelSpec`
carrying its weight location, decoding configuration (paper Table
``tab:eval_decoding``), and the handful of runtime flags vLLM needs
(``is_channel`` for harmony-format models, streaming for the 20B model).

Groups
------
``PRIMARY``        the four headline reasoning models.
``SUPPLEMENTARY``  five additional reasoning models (appendix refusal valley).
``PRETRAINED``     base/pretrained models for the data-contamination control.
``DEFENSE_CELLS``  6 training defenses x 4 primary models = 24 finetuned cells,
                   derived from their base model + method (decoding inherited).

The decoding numbers, context budgets, and GPU settings are copied verbatim from
the research tree.  ``hf_repo`` is a best-effort pointer to the public weights
(verify against the paper before relying on it); the library only *needs*
``model_path`` (resolved against :data:`lrm_safety_deliberation.paths.MODELS_ROOT`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from .paths import model_path as _resolve_model_path

__all__ = [
    "Decoding",
    "ModelSpec",
    "MODELS",
    "PRIMARY",
    "SUPPLEMENTARY",
    "PRETRAINED",
    "DEFENSE_CELLS",
    "TRAINING_METHODS",
    "INFERENCE_METHODS",
    "get",
    "resolve_path",
    "decoding_dict",
]


# --------------------------------------------------------------------------- #
# Spec dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Decoding:
    """Sampling configuration (paper Table ``tab:eval_decoding``)."""

    temperature: float
    top_p: float = 1.0
    top_k: Optional[int] = None
    min_p: Optional[float] = None

    def as_dict(self) -> dict:
        d = {"temperature": self.temperature, "top_p": self.top_p}
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.min_p is not None:
            d["min_p"] = self.min_p
        return d


@dataclass(frozen=True)
class ModelSpec:
    """A single evaluable model.

    ``model_path`` is a release-relative location (``models/<name>`` or
    ``models/final/<cell>``); resolve it to an absolute directory with
    :meth:`path` / :func:`resolve_path`.
    """

    name: str
    model_path: str
    decoding: Decoding
    size_class: str = "8b"
    is_channel: bool = False           # harmony channel format (GPT-OSS family)
    is_pretrained: bool = False        # base LM, no chat/thinking template
    max_model_len: int = 24576
    max_new_tokens: int = 16384        # universal thinking budget (2026-05-16)
    gpu_memory_utilization: float = 0.90
    batch_size_hf: int = 8             # batch for HF (non-vLLM) representation extraction
    stream_with_cache: bool = False    # chunked forward to cap KV memory (20B)
    stream_chunk_size: int = 128
    hf_repo: Optional[str] = None      # best-effort public weights pointer
    base_model: Optional[str] = None   # for defense cells: the model that was finetuned
    method: Optional[str] = None       # for defense cells: the defense method

    def path(self):
        """Absolute weight directory, anchored at ``MODELS_ROOT``."""
        return _resolve_model_path(self.model_path)

    def with_method(self, method: str, *, model_path: str) -> "ModelSpec":
        """Derive a defense-cell spec from this base model (inherits decoding)."""
        return replace(
            self,
            name=f"{method}-{self.name}",
            model_path=model_path,
            hf_repo=None,
            base_model=self.name,
            method=method,
        )


# --------------------------------------------------------------------------- #
# Primary reasoning models (the four headline models)
# --------------------------------------------------------------------------- #
_PRIMARY_SPECS = [
    ModelSpec(
        name="Qwen3-8B", model_path="models/Qwen3-8B", size_class="8b",
        decoding=Decoding(0.6, top_p=0.95, top_k=20),
        max_model_len=24576, max_new_tokens=16384, gpu_memory_utilization=0.90,
        hf_repo="Qwen/Qwen3-8B",
    ),
    ModelSpec(
        name="Olmo-3-7B-Think", model_path="models/Olmo-3-7B-Think", size_class="8b",
        decoding=Decoding(0.6, top_p=0.95),
        max_model_len=16384, gpu_memory_utilization=0.90,
        hf_repo="allenai/Olmo-3-7B-Think",
    ),
    ModelSpec(
        name="Phi-4-reasoning", model_path="models/Phi-4-reasoning", size_class="14b",
        decoding=Decoding(0.8, top_p=0.95, top_k=50),
        max_model_len=24576, max_new_tokens=16384, gpu_memory_utilization=0.85,
        batch_size_hf=4, hf_repo="microsoft/Phi-4-reasoning",
    ),
    ModelSpec(
        name="GPT-OSS-20B", model_path="models/GPT-OSS-20B", size_class="20b",
        decoding=Decoding(1.0, top_p=1.0), is_channel=True,
        max_model_len=16384, gpu_memory_utilization=0.88, batch_size_hf=2,
        stream_with_cache=True, stream_chunk_size=128, hf_repo="openai/gpt-oss-20b",
    ),
]

# --------------------------------------------------------------------------- #
# Supplementary reasoning models (appendix refusal-valley figure)
# --------------------------------------------------------------------------- #
_SUPPLEMENTARY_SPECS = [
    ModelSpec(
        name="Qwen3-32B", model_path="models/Qwen3-32B", size_class="32b",
        decoding=Decoding(0.6, top_p=0.95, top_k=20),
        max_model_len=8192, gpu_memory_utilization=0.88, batch_size_hf=1,
    ),
    ModelSpec(
        name="DeepSeek-R1-0528-Qwen3-8B", model_path="models/DeepSeek-R1-0528-Qwen3-8B",
        size_class="8b", decoding=Decoding(0.6, top_p=0.95),
        max_model_len=16384, gpu_memory_utilization=0.90,
    ),
    ModelSpec(
        name="DeepSeek-R1-Distill-Llama-8B", model_path="models/DeepSeek-R1-Distill-Llama-8B",
        size_class="8b", decoding=Decoding(0.6, top_p=0.95),
        max_model_len=16384, gpu_memory_utilization=0.90,
    ),
    ModelSpec(
        name="Phi-4-reasoning-plus", model_path="models/Phi-4-reasoning-plus", size_class="14b",
        decoding=Decoding(0.8, top_p=0.95, top_k=50),
        max_model_len=16384, gpu_memory_utilization=0.75, batch_size_hf=8,
    ),
    ModelSpec(
        name="Qwen3.5-9B", model_path="models/Qwen3.5-9B", size_class="8b",
        decoding=Decoding(0.6, top_p=0.95, top_k=20),
        max_model_len=16384, gpu_memory_utilization=0.85,
    ),
]

# --------------------------------------------------------------------------- #
# Pretrained / base models (data-contamination control, e.g. Pythia)
# --------------------------------------------------------------------------- #
def _pretrained(name, model_path, size_class, *, max_model_len=8192,
                max_new_tokens=16384, hf_repo=None, batch_size_hf=8,
                gpu_memory_utilization=0.90):
    return ModelSpec(
        name=name, model_path=model_path, size_class=size_class,
        decoding=Decoding(0.6, top_p=0.95), is_pretrained=True,
        max_model_len=max_model_len, max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization, batch_size_hf=batch_size_hf,
        hf_repo=hf_repo,
    )


_PRETRAINED_SPECS = [
    _pretrained("Qwen3-8B-Base", "models/Qwen3-8B-Base", "8b"),
    _pretrained("OLMo-3-1025-7B", "models/OLMo-3-1025-7B", "8b"),
    _pretrained("phi-4", "models/phi-4", "14b", batch_size_hf=4),
    _pretrained("pythia-6.9b", "models/pythia-6.9b", "7b",
                max_model_len=2048, max_new_tokens=512),
    _pretrained("Mistral-7B-v0.1", "models/Mistral-7B-v0.1", "7b"),
    _pretrained("Llama-3.1-8B", "models/Llama-3.1-8B", "8b"),
    _pretrained("safelm-1.7b", "models/safelm-1.7b", "2b",
                gpu_memory_utilization=0.85, batch_size_hf=16),
]

# --------------------------------------------------------------------------- #
# Defense methods
# --------------------------------------------------------------------------- #
# Training-time defenses produce a finetuned checkpoint per base model
# (``models/final/<METHOD>-<base>``).  Inference-time defenses wrap a base model
# at decode time and have no checkpoint (see ``pipeline/defenses/``).
TRAINING_METHODS = ["STAR1", "SafeKey", "R1ACT", "ThinkSafe", "STAIR", "RAPO"]
INFERENCE_METHODS = ["SafePath-ZS", "SafeRemind", "PSR"]


# --------------------------------------------------------------------------- #
# Assemble the registry
# --------------------------------------------------------------------------- #
PRIMARY = [s.name for s in _PRIMARY_SPECS]
SUPPLEMENTARY = [s.name for s in _SUPPLEMENTARY_SPECS]
PRETRAINED = [s.name for s in _PRETRAINED_SPECS]

MODELS: dict[str, ModelSpec] = {
    s.name: s for s in (*_PRIMARY_SPECS, *_SUPPLEMENTARY_SPECS, *_PRETRAINED_SPECS)
}

# Derive the 24 training-defense cells from their base model + method.
DEFENSE_CELLS: list[str] = []
for _method in TRAINING_METHODS:
    for _base in _PRIMARY_SPECS:
        _cell = _base.with_method(_method, model_path=f"models/final/{_method}-{_base.name}")
        MODELS[_cell.name] = _cell
        DEFENSE_CELLS.append(_cell.name)


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #
def get(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` (raises ``KeyError`` if absent)."""
    try:
        return MODELS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(MODELS)[:8]}... "
            f"({len(MODELS)} total)"
        ) from exc


def resolve_path(name: str):
    """Absolute weight directory for a registered model."""
    return get(name).path()


def decoding_dict(name: str) -> dict:
    """Sampling kwargs (temperature/top_p/top_k/min_p) for a registered model."""
    return get(name).decoding.as_dict()
