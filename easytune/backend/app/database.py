import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
STORAGE_ROOT = Path(os.getenv("EASYTUNE_STORAGE_ROOT", APP_DIR / "storage")).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'easytune.db'}")


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


STORAGE_DIRS = [
    "datasets",
    "llamafactory_data",
    "configs",
    "outputs",
    "logs",
    "reports",
    "registry",
]


def ensure_storage_dirs() -> None:
    for dirname in STORAGE_DIRS:
        (STORAGE_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    ensure_storage_dirs()
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_schema_columns()


def ensure_schema_columns() -> None:
    inspector = inspect(engine)
    if "training_jobs" not in inspector.get_table_names():
        return

    training_job_columns = {column["name"] for column in inspector.get_columns("training_jobs")}
    if "extra_config_json" not in training_job_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE training_jobs ADD COLUMN extra_config_json JSON"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
