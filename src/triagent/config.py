"""Runtime configuration, read once from environment variables (.env supported)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Every provider below exposes an OpenAI-compatible Chat Completions endpoint,
# so a single client works for all of them. Switching provider = switching URL.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-flash-latest",
    },
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:3b"},
    "openai": {"base_url": None, "model": "gpt-4o-mini"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-4-6"},
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str | None
    confidence_threshold: float
    encoder: str
    model_path: Path
    db_path: Path
    data_dir: Path

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "none":
            return False
        # Ollama runs locally and needs no key.
        return self.llm_provider == "ollama" or bool(self.llm_api_key)


def _threshold(encoder: str) -> float:
    """Env var wins; otherwise use the value chosen by cross-validation in scripts/train.py."""
    if os.getenv("CONFIDENCE_THRESHOLD"):
        return float(os.environ["CONFIDENCE_THRESHOLD"])
    path = ROOT / "models" / f"threshold_{encoder}.json"
    if path.exists():
        return float(json.loads(path.read_text()).get("threshold", 0.55))
    return 0.55


@lru_cache
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
    defaults = PROVIDERS.get(provider, {"base_url": None, "model": ""})
    encoder = os.getenv("ENCODER", "tfidf").strip().lower()
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL") or defaults["model"] or "",
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL") or defaults["base_url"],
        confidence_threshold=_threshold(encoder),
        encoder=encoder,
        model_path=ROOT / "models" / f"classifier_{encoder}.joblib",
        db_path=ROOT / os.getenv("TRIAGENT_DB", "triagent.sqlite"),
        data_dir=ROOT / "data",
    )
