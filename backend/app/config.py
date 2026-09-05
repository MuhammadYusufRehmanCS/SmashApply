from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
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

    # Job ingestion sources. Direct ATS/job-board providers run alongside JobSpy
    # so Indeed is one input to the pipeline instead of the primary feed.
    job_sources: str = "greenhouse,lever,builtin,remotive,themuse,jobspy"

    # JobSpy live scraping
    jobspy_sites: str = "linkedin,indeed,glassdoor,zip_recruiter"
    jobspy_results_wanted: int = 8
    jobspy_total_results_wanted: int = 40
    jobspy_hours_old: int = 168
    jobspy_country_indeed: str = "USA"

    # Direct ATS/job board scraping
    job_http_timeout_seconds: float = 12.0
    job_source_concurrency: int = 4
    ats_results_wanted: int = 40
    themuse_pages: int = 3
    themuse_categories: str = "Software Engineering,Computer and IT"
    greenhouse_board_tokens: str = (
        "airbnb|Airbnb,databricks|Databricks,discord|Discord,figma|Figma,"
        "gitlab|GitLab,instacart|Instacart,reddit|Reddit,stripe|Stripe,"
        "toast|Toast,cloudflare|Cloudflare,mongodb|MongoDB,okta|Okta,"
        "asana|Asana,robinhood|Robinhood,scaleai|Scale AI,peloton|Peloton,"
        "affirm|Affirm,elastic|Elastic,samsara|Samsara,cockroachlabs|Cockroach Labs"
    )
    lever_sites: str = (
        "filevine|Filevine,agiloft|Agiloft,lyrahealth|Lyra Health,"
        "swordhealth|Sword Health,xsolla|Xsolla,everbridge|Everbridge,"
        "businesswire|Business Wire,coalfire|Coalfire,"
        "analyticpartners|Analytic Partners,spotify|Spotify"
    )

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
    def job_source_list(self) -> list[str]:
        return [source.strip().lower() for source in self.job_sources.split(",") if source.strip()]

    @property
    def jobspy_site_list(self) -> list[str]:
        return [site.strip() for site in self.jobspy_sites.split(",") if site.strip()]

    @property
    def themuse_category_list(self) -> list[str]:
        return [category.strip() for category in self.themuse_categories.split(",") if category.strip()]

    @staticmethod
    def _named_tokens(value: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for raw_entry in value.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue

            token = entry
            label = entry
            for separator in ("|", "="):
                if separator in entry:
                    token, label = entry.split(separator, 1)
                    break

            token = token.strip()
            label = label.strip() or token
            if token:
                entries.append((token, label))
        return entries

    @property
    def greenhouse_boards(self) -> list[tuple[str, str]]:
        return self._named_tokens(self.greenhouse_board_tokens)

    @property
    def lever_boards(self) -> list[tuple[str, str]]:
        return self._named_tokens(self.lever_sites)


@lru_cache
def get_settings() -> Settings:
    return Settings()
