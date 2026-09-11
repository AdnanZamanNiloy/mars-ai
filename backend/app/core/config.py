from functools import lru_cache
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _real_key_like(value: Any) -> str:
    """Non-empty, non-placeholder key text (mirrors llm._real_key without
    importing it — config must stay import-cycle free)."""
    text = str(value or "").strip()
    if not text or text.lower().startswith("your_"):
        return ""
    return text


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
    # Custom OpenAI-compatible provider (any host serving /chat/completions:
    # OpenRouter, Together, Ollama+ngrok, vLLM, LM Studio, ...). All three
    # must be set; when present it leads the chain, Groq/HF stay as fallback.
    custom_llm_api_key: str = ""
    custom_llm_base_url: str = ""
    custom_llm_model: str = ""

    # Persistence
    database_url: str = "./research.db"

    # Pipeline limits
    max_parallel_search: int = 2
    max_parallel_agents: int = 3  # hardware cap (8GB host) — raise only after load-testing
    # Concurrent LLM calls across the whole pipeline (planner + N summarizer
    # workers + critic + synthesizer). Low by default: concurrent large
    # prompts are what exhausts free-tier TPM/TPD quotas (observed live:
    # Groq TPD 200K burned by 3 parallel ~6K-token summarizer calls).
    max_parallel_llm: int = 2
    max_iterations: int = 3
    # Retrieval depth: how many top-ranked results per sub-question get full
    # content fetched. Each fetch is ~6KB cleaned text kept only until
    # summarization; 8 pages per angle is the deep-research floor (concurrency
    # bounded by MAX_PARALLEL_SEARCH, content released after verification).
    search_fetch_top_n: int = 8
    # Query fan-out: max search queries issued per pass (questions + their
    # alternate phrasings). Bounds latency when plans carry variants.
    search_max_queries_per_pass: int = 8

    # Timeouts (seconds)
    llm_timeout_sec: float = 25.0
    # Slower OpenAI-compatible providers routinely take 20-30s on
    # planner-sized prompts (measured live: glm-5.3-flash 20-30s where
    # small calls take 2-4s) and stall past 60s under load. The shared
    # 25s budget timed out real calls, tripping the breaker and silently
    # degrading whole runs. Worst case per chain pass stays inside the
    # research budget: 90s custom + 25s groq, one pass, no timeout
    # retries. Applies to the custom provider only; Groq/HF keep
    # llm_timeout_sec.
    custom_llm_timeout_sec: float = 90.0
    search_timeout_sec: float = 20.0
    # Full multi-agent runs take minutes (retrieval + 6 LLM stages), the
    # same as upstream GPT Researcher. Per-provider fail-fasts (auth/402/
    # timeouts) keep doomed calls from eating this budget.
    research_timeout_sec: float = 300.0

    # Cache (Phase 1.4)
    cache_size_limit_bytes: int = 250_000_000  # 250MB, diskcache size cap
    cache_ttl_sec: int = 3600  # 1 hour

    # Rate limiting (Phase 1.7)
    rate_limit: str = "5/minute"

    # Dynamic Research Depth (Phase 2.8)
    sufficiency_threshold: float = 0.75
    min_marginal_gain: float = 0.03
    # Hard ceiling on expansion depth; 0 means "use MAX_ITERATIONS".
    max_research_depth: int = 0

    @model_validator(mode="after")
    def _require_llm_provider(self) -> "Settings":
        custom_ok = bool(
            _real_key_like(self.custom_llm_api_key)
            and self.custom_llm_base_url.strip()
            and self.custom_llm_model.strip()
        )
        if not (self.groq_api_key or self.huggingface_api_key or custom_ok):
            raise ValueError(
                "No LLM provider configured. Set GROQ_API_KEY, HUGGINGFACE_API_KEY, "
                "or the CUSTOM_LLM_* trio in your environment or .env file."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
