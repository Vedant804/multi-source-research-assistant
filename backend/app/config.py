from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Web search
    tavily_api_key: str = ""

    # Infra
    database_url: str = "postgresql://research:research@localhost:5432/research"
    redis_url: str = "redis://localhost:6379/0"

    # App
    upload_dir: str = "./uploads"
    max_upload_mb: int = 25
    cors_origins: str = "http://localhost:5173"

    # TTLs (seconds)
    web_search_cache_ttl: int = 3600
    session_cache_ttl: int = 60 * 60 * 24 * 7
    run_ttl: int = 60 * 60 * 24

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()