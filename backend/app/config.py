from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Database
    database_url: str = "sqlite:///./data/smashapply.db"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # JobSpy live scraping
    jobspy_sites: str = "linkedin,indeed,glassdoor,zip_recruiter"
    jobspy_results_wanted: int = 8
    jobspy_hours_old: int = 168
    jobspy_country_indeed: str = "USA"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def jobspy_site_list(self) -> list[str]:
        return [site.strip() for site in self.jobspy_sites.split(",") if site.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
