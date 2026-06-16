"""Base evaluator interface for distillation_safety."""

import abc
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Standard container for evaluation results."""
    metric_name: str
    score: float               # primary scalar metric, in [0, 1]
    num_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)
    per_example: list[dict[str, Any]] = field(default_factory=list)


def strip_thinking(text):
    """Return only the content *after* the last ``</think>`` tag.

    If the model opens ``<think>`` but never closes it (e.g. hit the token
    limit mid-thought), there is no real answer — return empty string.
    If there are no thinking tags at all, return the text unchanged.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        # Opened but never closed — no answer was produced.
        return ""
    return text.strip()


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluators.

    Every evaluator must implement:
      - evaluate(model_path, **kwargs) -> list[EvalResult]

    Convention:
      - model_path is a path to a HF checkpoint directory or model name.
      - The evaluator loads the model, generates responses, scores them,
        and returns one or more EvalResult objects.
      - Evaluators should free GPU memory after evaluation.
    """

    @abc.abstractmethod
    def evaluate(self, model_path: str, **kwargs) -> list[EvalResult]:
        """Run evaluation on a single model/checkpoint.

        Args:
            model_path: Path to HF model checkpoint or model name.
            **kwargs: Evaluator-specific options.

        Returns:
            List of EvalResult, one per metric.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this evaluator (e.g., 'tooluse', 'jailbreakbench')."""
        ...
