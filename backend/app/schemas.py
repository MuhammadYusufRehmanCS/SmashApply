from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: str
    location: str
    job_url: str
    site: str
    role_category: str
    is_primary_role: bool
    description: str
    date_posted: str | None
    tailored_keywords: str | None
    tailored_at: datetime | None
    created_at: datetime


class ScrapeRequest(BaseModel):
    primary_role: str = "Cloud Engineer"
    location: str = "Remote"


class ScrapeResult(BaseModel):
    primary_role: str
    location: str
    roles_queried: list[str]
    total_found: int
    created: int
    skipped: int
    site_errors: list[str] = []


class TailorResult(BaseModel):
    job_id: int
    keywords: list[str]
    tailored_cv: str


class MasterCVOut(BaseModel):
    id: int
    filename: str
    raw_text: str
    sections: list[dict]
    layout: dict
    created_at: datetime
    updated_at: datetime
