import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import db
from app.model_service import ModelService
from app.schemas import ApplicantIn, PredictionOut

log = logging.getLogger("loan-api")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = ModelService()
    db.init_db()
    yield


app = FastAPI(title="Loan Default Scoring API", version="1.0", lifespan=lifespan,
              description="Decision-support tool. Trained on a public 2011 benchmark, not live bank data.")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["svc"].version}


@app.get("/model-info")
def model_info():
    m = state["svc"].meta
    return {k: m[k] for k in ("model_name", "model_version", "threshold", "risk_tiers",
                              "candidates", "test_metrics_at_threshold", "simulated_npa")}


@app.post("/predict", response_model=PredictionOut)
def predict(applicant: ApplicantIn):
    svc = state["svc"]
    p, tier, flagged, factors, feats = svc.score(applicant)
    pid = None
    try:
        pid = db.log_prediction(applicant.model_dump(by_alias=True), p, tier, flagged, svc.version)
    except Exception:                       # never fail a score because logging failed
        log.exception("could not log prediction")
    return PredictionOut(default_probability=round(p, 4), risk_tier=tier, flagged=flagged,
                         top_factors=factors, model_version=svc.version, prediction_id=pid)


@app.get("/predictions")
def predictions(limit: int = 20):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit must be between 1 and 200")
    return db.recent(limit)
