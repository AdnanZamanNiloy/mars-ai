from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.example"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # LLM providers
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    huggingface_api_key: str = ""
    huggingface_model: str = "Qwen/Qwen2.5-7B-Instruct"
    tavily_api_key: str = ""

    # Persistence
    database_url: str = "./research.db"

    # Pipeline limits
    max_parallel_search: int = 2
    max_parallel_agents: int = 3  # hardware cap (8GB host) — raise only after load-testing
    max_iterations: int = 3

    # Timeouts (seconds)
    llm_timeout_sec: float = 25.0
    search_timeout_sec: float = 20.0
    research_timeout_sec: float = 90.0

    # Cache (Phase 1.4)
    cache_size_limit_bytes: int = 250_000_000  # 250MB, diskcache size cap
    cache_ttl_sec: int = 3600  # 1 hour

    # Rate limiting (Phase 1.7)
    rate_limit: str = "5/minute"

    @model_validator(mode="after")
    def _require_llm_provider(self) -> "Settings":
        if not (self.groq_api_key or self.huggingface_api_key):
            raise ValueError(
                "No LLM provider configured. Set GROQ_API_KEY or HUGGINGFACE_API_KEY "
                "in your environment or .env file."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
