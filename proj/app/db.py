"""Prediction log. PostgreSQL in Docker (DATABASE_URL), SQLite locally as a fallback."""
import json
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, Boolean, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    model_version: Mapped[str] = mapped_column(String(40))
    inputs: Mapped[str] = mapped_column(Text)          # JSON of the raw request
    probability: Mapped[float] = mapped_column(Float)
    risk_tier: Mapped[str] = mapped_column(String(10))
    flagged: Mapped[bool] = mapped_column(Boolean)


def make_engine():
    url = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
    return create_engine(url, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)


def log_prediction(inputs: dict, prob: float, tier: str, flagged: bool, version: str) -> int:
    with SessionLocal() as s:
        row = Prediction(model_version=version, inputs=json.dumps(inputs), probability=prob,
                         risk_tier=tier, flagged=flagged)
        s.add(row)
        s.commit()
        return row.id


def recent(limit: int = 20):
    with SessionLocal() as s:
        rows = s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)).all()
        return [{"id": r.id, "created_at": r.created_at.isoformat(), "model_version": r.model_version,
                 "probability": r.probability, "risk_tier": r.risk_tier, "flagged": r.flagged,
                 "inputs": json.loads(r.inputs)} for r in rows]
