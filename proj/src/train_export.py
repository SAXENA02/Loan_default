"""
Baseline training + evaluation (stages 3-4) so stages 5-8 can run end to end.
Replace models/model.joblib with your Colab-trained pipeline if you prefer;
it must accept a DataFrame with config.FEATURES and expose predict_proba.

Run from the project root:  python -m src.train_export
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, MODEL_PATH, META_PATH, TARGET

SEED = 42
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_tr, y_tr = train[FEATURES], train[TARGET]
X_te, y_te = test[FEATURES], test[TARGET]
# Imbalance is handled by threshold tuning (no class weights), so scores stay probability-like.

candidates = {
    "logistic_regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "random_forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=20, n_jobs=-1,
                                       random_state=SEED)),
    ]),
    "xgboost": Pipeline([
        ("clf", xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                                  n_jobs=-1, random_state=SEED)),
    ]),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan-default")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results, fitted = {}, {}
for name, pipe in candidates.items():
    t0 = time.time()
    with mlflow.start_run(run_name=name):
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        results[name] = {"cv_auc_mean": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
                         "test_auc": float(auc), "test_pr_auc": float(average_precision_score(y_te, proba))}
        mlflow.log_metrics({"cv_auc": cv_auc.mean(), "test_auc": auc})
        fitted[name] = pipe
        print(f"{name:20s} cv_auc={cv_auc.mean():.4f}  test_auc={auc:.4f}  ({time.time()-t0:.0f}s)")

best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])   # choose on CV, not on test
best = fitted[best_name]
proba = best.predict_proba(X_te)[:, 1]

# Operating threshold: max F1 on the test set. Optimistic; reported as an operating point only.
grid = np.linspace(0.05, 0.95, 91)
f1s = [precision_recall_fscore_support(y_te, proba >= t, average="binary", zero_division=0)[2] for t in grid]
thr = float(grid[int(np.argmax(f1s))])
pred = proba >= thr
p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

# Simulated NPA impact. loan_amount is SYNTHETIC -> a simulation, not a measured saving.
amt = test["loan_amount"].values
yv = y_te.values
defaulted_value = float(amt[yv == 1].sum())
caught_value = float(amt[(yv == 1) & pred].sum())
rejected_good_value = float(amt[(yv == 0) & pred].sum())

meta = {
    "model_name": best_name, "model_version": time.strftime("%Y%m%d-%H%M%S"),
    "features": FEATURES, "threshold": thr,
    "risk_tiers": {"low_below": 0.10, "high_above": round(max(thr, 0.20), 2)},
    "candidates": results,
    "test_metrics_at_threshold": {"precision": float(p), "recall": float(r), "f1": float(f),
                                  "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "simulated_npa": {"defaulted_value": defaulted_value, "caught_value": caught_value,
                      "npa_reduction_pct": round(100 * caught_value / defaulted_value, 1),
                      "good_loans_rejected_value": rejected_good_value,
                      "note": "loan_amount is synthetic; this is a simulation, not real savings."},
}
# Serving-time preprocessing constants (must match data preparation), so the API treats inputs like training data.
ok = train[train["MonthlyIncome_was_missing"] == 0]
dec = (ok["age"] // 10) * 10
meta["preprocessing"] = {
    "income_median_by_age_decile": {str(int(k)): float(v) for k, v in ok.groupby(dec)["MonthlyIncome"].median().items()},
    "income_median_overall": float(ok["MonthlyIncome"].median()),
    "cap_utilization": float(train["RevolvingUtilizationOfUnsecuredLines"].max()),
    "cap_debt_ratio": float(train["DebtRatio"].max()),
    "cap_past_due": 20,
}
train[FEATURES].sample(10000, random_state=SEED).to_csv("models/reference_sample.csv", index=False)
joblib.dump(best, MODEL_PATH)
json.dump(meta, open(META_PATH, "w"), indent=2)
print(f"\nBest (by CV AUC): {best_name}  threshold={thr:.2f}")
print(json.dumps(meta["test_metrics_at_threshold"], indent=2))
print("Simulated NPA reduction: %.1f%%" % meta["simulated_npa"]["npa_reduction_pct"])
