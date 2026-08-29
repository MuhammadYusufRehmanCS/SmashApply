from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: str
    raw_url: str
    final_url: str | None
    description: str
    match_score: float
    status: str
    validation_note: str | None
    created_at: datetime
    updated_at: datetime


class MetricsOut(BaseModel):
    smashed: int
    passed: int
    active_pipeline: int
    total: int


class SmashResponse(BaseModel):
    job: JobOut
    tailored_cv: str
    apply_url: str


class IngestResult(BaseModel):
    fetched: int
    created: int
    skipped: int
    errors: list[str] = []


class RevalidateResult(BaseModel):
    checked: int
    expired: int


class CVIn(BaseModel):
    raw_text: str


class CVOut(BaseModel):
    id: int
    raw_text: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
