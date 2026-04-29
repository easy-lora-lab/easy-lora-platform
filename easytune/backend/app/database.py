import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
