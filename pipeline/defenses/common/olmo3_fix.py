"""
Monkey-patch for transformers 5.6.0.dev0 Olmo3Config RoPE standardization bug.

The issue: Olmo3Config.__getattribute__ raises AttributeError for
max_position_embeddings during __post_init__ -> standardize_rope_params,
even though the value exists in the JSON config.

Import this module BEFORE importing vllm or loading OLMo-3 models.
"""
import transformers.configuration_utils as _cu

_orig_standardize = _cu.PretrainedConfig.standardize_rope_params

def _safe_standardize(self):
    try:
        return _orig_standardize(self)
    except AttributeError as e:
        if "max_position_embeddings" in str(e):
            # Fall back: read from the raw config dict
            import json, os
            mpe = None
            if hasattr(self, '_name_or_path') and self._name_or_path:
                cfg_path = os.path.join(self._name_or_path, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        mpe = json.load(f).get("max_position_embeddings")
            if mpe is None:
                mpe = 65536  # safe default for OLMo-3
            # Set it and retry
            object.__setattr__(self, "max_position_embeddings", mpe)
            return _orig_standardize(self)
        raise

_cu.PretrainedConfig.standardize_rope_params = _safe_standardize
