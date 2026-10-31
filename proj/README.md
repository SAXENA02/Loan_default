# Loan Default Prediction (P_137)

Predicts the probability that a borrower has serious delinquency within two years, explains each score with SHAP,
serves it through FastAPI, logs predictions to PostgreSQL, monitors data drift, and ships in Docker.

**Data:** the public Kaggle "Give Me Some Credit" dataset (2011), not Bank Muscat data. `loan_amount`, `loan_purpose`
and `application_date` are synthetic (see `DATA_CARD.md`); they are not model inputs.

## Results (held-out test set, 27,000 rows, measured by `python -m src.train_export`)

| Model | CV AUC (5-fold, train) | Test AUC |
|---|---|---|
| Logistic Regression | 0.8189 | 0.8117 |
| Random Forest | 0.8641 | 0.8592 |
| **XGBoost (served)** | **0.8661** | **0.8630** |

The model was chosen on cross-validated AUC, not on the test set. Target AUC >= 0.85: **met** by XGBoost and Random Forest.

At the operating threshold 0.24: precision 0.43, recall 0.47, F1 0.45
(843 defaulters caught, 961 missed, 1140 good loans flagged).

**Simulated NPA reduction: 41.1%**, by value of defaulted loans flagged. This uses the *synthetic*
`loan_amount`, and it ignores the good loans that would also be rejected (1140 in the test set). Treat it as an
illustration of the method, not a measured saving. The threshold was picked on the test set (max F1), so it is slightly
optimistic; a stricter approach picks it on a validation split.

Imbalance is handled by threshold tuning rather than class weights, so predicted scores stay probability-like.
(The SRS mentions class weighting as one option; this is the alternative it allows.)

## Layout
```
streamlit_app.py + ui/  Streamlit web app (entry point and page modules)
src/train_export.py     stages 3-4: train 3 models, CV, evaluate, save best + metadata + NPA simulation
src/explain.py          stage 5: SHAP plots -> reports/
app/                    stage 6: FastAPI service (main.py), validation (schemas.py), DB (db.py), scoring (model_service.py)
monitoring/drift_report.py   stage 7: PSI + KS + Evidently HTML report, logged to MLflow
tests/test_api.py       5 API tests
Dockerfile (API), Dockerfile.streamlit (UI), docker-compose.yml, ecs/task-definition.json   stage 8
models/                 model.joblib, model_meta.json, reference_sample.csv
data/                   train.csv, test.csv, drift_batch.csv
```

## Run locally
```bash
pip install -r requirements.txt
python -m src.train_export          # ~2-3 min; writes models/
python -m src.explain               # SHAP plots
python -m monitoring.drift_report   # drift vs data/drift_batch.csv
pytest tests -q
uvicorn app.main:app --reload       # http://localhost:8000/docs  (SQLite unless DATABASE_URL is set)
streamlit run streamlit_app.py      # http://localhost:8501  (Python-only UI; uses the API if it's
                                     # running, otherwise falls back to loading the model directly)
```

## Credit Risk Assessment Portal (Streamlit, pure Python)
`streamlit run streamlit_app.py`. The portal loads the trained model directly and runs standalone; the FastAPI
service in `app/` remains available for programmatic scoring.

| Page | Purpose |
|---|---|
| Dashboard | Sample-portfolio KPIs, risk-band mix, observed default rate by band, risk by age and utilization, main risk drivers |
| New Assessment | Live risk score, band, suggested action, key factors in plain language, improvement scenarios, assessment log with CSV export |
| Portfolio Upload | Assess many applications from a CSV and download the results |
| Data Stability | Compares recent applications with the training data and flags shifts |
| Governance | One-page model summary and responsible-use notes |

Detailed model evaluation (model comparison, ROC/PR curves, threshold analysis, calibration, SHAP plots) is deliberately
not part of the operational portal. It lives in this README, the SRS, and `reports/`, for reviewers who want it.
No HTML, CSS or JavaScript is written in this project; styling is only `.streamlit/config.toml`.

