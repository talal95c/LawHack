"""Runtime settings for Système 2, read from the environment (`.env`)."""

import os
from dataclasses import dataclass

PROVIDERS = {
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


@dataclass(frozen=True)
class Settings:
    answer_provider: str = "mistral"
    answer_model: str = "mistral-medium-latest"
    baseline_provider: str = "openrouter"
    baseline_model: str = "openai/gpt-6-astra"
    temperature: float = 0.0
    # Honesty policy (README §7), to calibrate on the labelled set.
    confidence_high: float = 0.8
    confidence_low: float = 0.5
    support_min: float = 0.5
    answerable_min: float = 0.35
    route_abstain: float = 0.7
    relevance_min: float = 0.3
    top_k: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            answer_provider=os.environ.get("ANSWER_PROVIDER") or d.answer_provider,
            answer_model=os.environ.get("ANSWER_MODEL") or d.answer_model,
            baseline_provider=os.environ.get("BASELINE_PROVIDER") or d.baseline_provider,
            baseline_model=os.environ.get("BASELINE_MODEL") or d.baseline_model,
            temperature=_float("ANSWER_TEMPERATURE", d.temperature),
            confidence_high=_float("CONFIDENCE_HIGH", d.confidence_high),
            confidence_low=_float("CONFIDENCE_LOW", d.confidence_low),
            support_min=_float("SUPPORT_MIN", d.support_min),
            answerable_min=_float("ANSWERABLE_MIN", d.answerable_min),
            route_abstain=_float("ROUTE_ABSTAIN", d.route_abstain),
            relevance_min=_float("RELEVANCE_MIN", d.relevance_min),
            top_k=_int("RETRIEVAL_TOP_K", d.top_k),
        )
