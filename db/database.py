from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


DB_URL = os.getenv('DATABASE_URL', 'sqlite:///./data/agro_ai.db')
if DB_URL.startswith('sqlite:///./'):
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URL,
    connect_args={'check_same_thread': False} if DB_URL.startswith('sqlite') else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from db.models import Base

    Base.metadata.create_all(bind=engine)
