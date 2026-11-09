"""Environment-backed application settings."""
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _streamlit_secret(name: str) -> str:
    """Read an optional Streamlit-hosted secret without exposing it in logs."""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        return str(value).strip() if value is not None else ""
    except Exception:
        # Outside Streamlit, secrets may not be configured or available.
        return ""


def _setting(name: str, default: str = "") -> str:
    environment_value = os.getenv(name, "").strip()
    return environment_value or _streamlit_secret(name) or default


@lru_cache(maxsize=1)
def get_settings():
    return Settings()


class Settings:
    def __init__(self) -> None:
        self.llm_provider = _setting("LLM_PROVIDER", "gemini").lower()
        self.gemini_api_key = _setting("GEMINI_API_KEY")
        self.gemini_model = _setting("GEMINI_MODEL")
        self.cache_dir = Path(os.getenv("PROMOSCOUT_CACHE_DIR", "data/cache/analysis"))
        self.max_chunk_chars = int(os.getenv("PROMOSCOUT_MAX_CHUNK_CHARS", "24000"))
        self.max_retries = int(os.getenv("PROMOSCOUT_MAX_RETRIES", "3"))

    def gemini_configuration_issues(self) -> list[str]:
        issues = []
        if self.llm_provider != "gemini":
            issues.append("LLM_PROVIDER must be 'gemini' for the configured Gemini client")
        if not self.gemini_api_key or self.gemini_api_key.lower() in {
            "your_api_key_here", "your_gemini_api_key", "changeme", "replace_me",
        }:
            issues.append("Set GEMINI_API_KEY in .env, the environment, or the Streamlit app's Secrets settings")
        if not self.gemini_model or self.gemini_model.lower() in {
            "your_selected_free_tier_model", "your_model_name", "model_name", "your_gemini_model",
        }:
            issues.append("Set GEMINI_MODEL to a supported model identifier in .env, the environment, or Streamlit Secrets")
        return issues

