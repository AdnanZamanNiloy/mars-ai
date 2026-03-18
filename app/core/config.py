import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    huggingface_api_key: str
    huggingface_model: str
    tavily_api_key: str
    database_url: str
    max_parallel_search: int
    max_iterations: int
    llm_timeout_sec: float
    search_timeout_sec: float


def _clean_env(value: str) -> str:
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def get_settings() -> Settings:
    # Load local env files if present; existing shell env vars keep precedence.
    load_dotenv(".env", override=False)
    load_dotenv(".env.example", override=False)

    return Settings(
        groq_api_key=_clean_env(os.getenv("GROQ_API_KEY", "")),
        groq_model=_clean_env(os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
        huggingface_api_key=_clean_env(os.getenv("HUGGINGFACE_API_KEY", "")),
        huggingface_model=_clean_env(os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
        tavily_api_key=_clean_env(os.getenv("TAVILY_API_KEY", "")),
        database_url=_clean_env(os.getenv("DATABASE_URL", "./research.db")),
        max_parallel_search=int(os.getenv("MAX_PARALLEL_SEARCH", "2")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        llm_timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "25")),
        search_timeout_sec=float(os.getenv("SEARCH_TIMEOUT_SEC", "20")),
    )
