import os
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

db_path = settings.database_url.replace("sqlite:///", "")
if db_path.startswith("./"):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def _add_missing_columns() -> None:
    """`create_all` only creates missing tables, it never alters an existing
    one -- so a column added to a model after the sqlite file already exists
    (no Alembic in this project) needs a manual ADD COLUMN pass here.
    """
    inspector = inspect(engine)
    if "jobs" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("jobs")}
    missing_column_ddl = {
        "match_score": "INTEGER",
        "applied": "BOOLEAN DEFAULT 0 NOT NULL",
    }
    with engine.begin() as conn:
        for column, ddl_type in missing_column_ddl.items():
            if column not in existing_columns:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {column} {ddl_type}"))


def init_db() -> None:
    from app import models  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
