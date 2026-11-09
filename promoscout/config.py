"""Environment-backed application settings."""
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@lru_cache(maxsize=1)
def get_settings():
    return Settings()


class Settings:
    def __init__(self) -> None:
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "").strip()
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
            issues.append("Set GEMINI_API_KEY in .env or the environment")
        if not self.gemini_model or self.gemini_model.lower() in {
            "your_selected_free_tier_model", "your_model_name", "model_name", "your_gemini_model",
        }:
            issues.append("Set GEMINI_MODEL to a supported model identifier in .env or the environment")
        return issues

