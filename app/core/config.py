from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Last file wins: .env (real values) must override .env.example
        # (placeholders). Real environment variables beat both.
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # LLM providers
    groq_api_key: str = ""
    # llama-3.1-8b-instant was decommissioned on Groq (2026); gpt-oss-20b is
    # the verified replacement. Check console.groq.com if this 404s again.
    groq_model: str = "openai/gpt-oss-20b"
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

    # Cost governor (Phase 2.2) — real provider rates, update when providers change
    research_max_cost_usd: float = 0.50
    # Groq llama-3.1-8b-instant: $0.05/1M input, $0.08/1M output
    groq_cost_per_1k_input_tokens: float = 0.00005
    groq_cost_per_1k_output_tokens: float = 0.00008
    # HuggingFace serverless inference — approximate blended rate
    hf_cost_per_1k_tokens: float = 0.0002

    # Dynamic Research Depth (Phase 2.8)
    sufficiency_threshold: float = 0.75
    min_marginal_gain: float = 0.03
    # Hard ceiling on expansion depth; 0 means "use MAX_ITERATIONS".
    max_research_depth: int = 0

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
