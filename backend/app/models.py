from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MasterCV(Base):
    """Single-row table holding the uploaded Master CV's text and its parsed
    layout profile (fonts, section order, margins, column structure) so the
    ATS PDF generator can mirror the original design.
    """

    __tablename__ = "master_cv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(300), default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    sections_json: Mapped[str] = mapped_column(Text, default="[]")
    layout_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("title", "company", "job_url", name="uq_job_title_company_url"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    company: Mapped[str] = mapped_column(String(300), default="")
    location: Mapped[str] = mapped_column(String(300), default="")
    job_url: Mapped[str] = mapped_column(Text, default="")
    site: Mapped[str] = mapped_column(String(50), default="")
    description: Mapped[str] = mapped_column(Text, default="")

    # The title (primary or one of the 10 aligned roles) whose search surfaced this job.
    role_category: Mapped[str] = mapped_column(String(200), default="")
    is_primary_role: Mapped[bool] = mapped_column(Boolean, default=False)
    date_posted: Mapped[str | None] = mapped_column(String(50), nullable=True)

    tailored_cv: Mapped[str | None] = mapped_column(Text, nullable=True)
    tailored_keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    tailored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
