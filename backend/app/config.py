from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gmail IMAP
    gmail_address: str = ""
    gmail_app_password: str = ""
    gmail_label: str = "SmashApply"
    gmail_search_query: str = "label:SmashApply is:unread"

    # Local LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Vector matching
    embedding_model: str = "all-MiniLM-L6-v2"

    # Link validation
    link_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    link_validation_timeout: int = 15

    # Database
    database_url: str = "sqlite:///./data/smashapply.db"

    # CORS
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
