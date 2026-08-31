from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # OpenAI LLM for CV tailoring
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL

    # Database
    database_url: str = "sqlite:///./data/smashapply.db"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # JobSpy live scraping
    jobspy_sites: str = "linkedin,indeed,glassdoor,zip_recruiter"
    jobspy_results_wanted: int = 8
    jobspy_hours_old: int = 168
    jobspy_country_indeed: str = "USA"

    @field_validator("openai_model", mode="before")
    @classmethod
    def default_openai_model(cls, value: object) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_OPENAI_MODEL
        return str(value).strip()

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def default_openai_api_key(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def jobspy_site_list(self) -> list[str]:
        return [site.strip() for site in self.jobspy_sites.split(",") if site.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
