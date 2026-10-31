import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app                   # noqa: E402

GOOD = {"RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 45, "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2, "MonthlyIncome": 7000, "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 1}
RISKY = {**GOOD, "RevolvingUtilizationOfUnsecuredLines": 1.2, "age": 24, "NumberOfTime30-59DaysPastDueNotWorse": 4,
         "NumberOfTimes90DaysLate": 3, "NumberOfTime60-89DaysPastDueNotWorse": 2, "MonthlyIncome": 1500, "DebtRatio": 0.9}


def test_health_and_info():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        info = c.get("/model-info").json()
        assert 0 < info["threshold"] < 1


def test_predict_orders_risk_sensibly():
    with TestClient(app) as c:
        g = c.post("/predict", json=GOOD).json()
        r = c.post("/predict", json=RISKY).json()
        assert 0 <= g["default_probability"] <= 1
        assert r["default_probability"] > g["default_probability"]
        assert g["risk_tier"] == "Low" and r["risk_tier"] == "High"
        assert len(r["top_factors"]) == 3 and r["prediction_id"]


def test_null_income_allowed():
    with TestClient(app) as c:
        resp = c.post("/predict", json={**GOOD, "MonthlyIncome": None, "NumberOfDependents": None})
        assert resp.status_code == 200


def test_validation_errors():
    with TestClient(app) as c:
        assert c.post("/predict", json={**GOOD, "age": 12}).status_code == 422
        assert c.post("/predict", json={**GOOD, "DebtRatio": -1}).status_code == 422
        bad = dict(GOOD); del bad["age"]
        assert c.post("/predict", json=bad).status_code == 422


def test_predictions_are_logged():
    with TestClient(app) as c:
        c.post("/predict", json=GOOD)
        rows = c.get("/predictions?limit=5").json()
        assert rows and "probability" in rows[0]
