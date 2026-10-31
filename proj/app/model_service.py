"""Loads the saved pipeline once, applies the same preprocessing as training, scores, explains."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import FEATURES, META_PATH, MODEL_PATH

_HUMAN = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late (count)",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late (count)",
    "NumberRealEstateLoansOrLines": "Real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late (count)",
    "NumberOfDependents": "Dependents",
    "MonthlyIncome_was_missing": "Income not provided",
}


class ModelService:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        self.pipe = joblib.load(model_path)
        self.meta = json.load(open(meta_path))
        self.pre = self.meta["preprocessing"]
        self.threshold = self.meta["threshold"]
        self.tiers = self.meta["risk_tiers"]
        self.version = self.meta["model_version"]
        self.explainer = self._build_explainer()

    def _build_explainer(self):
        est = self.pipe[-1]
        try:
            return shap.TreeExplainer(est)          # works for XGBoost / Random Forest
        except Exception:
            return None                             # e.g. logistic regression: no explanations

    # ---- preprocessing identical to data preparation -------------------------------------
    def to_frame(self, a) -> pd.DataFrame:
        income_missing = int(a.monthly_income is None)
        if income_missing:
            dec = str(int(a.age // 10 * 10))
            income = self.pre["income_median_by_age_decile"].get(dec, self.pre["income_median_overall"])
        else:
            income = a.monthly_income
        row = {
            "RevolvingUtilizationOfUnsecuredLines": min(a.utilization, self.pre["cap_utilization"]),
            "age": a.age,
            "NumberOfTime30-59DaysPastDueNotWorse": min(a.past_due_30_59, self.pre["cap_past_due"]),
            "DebtRatio": min(a.debt_ratio, self.pre["cap_debt_ratio"]),
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": a.open_credit_lines,
            "NumberOfTimes90DaysLate": min(a.late_90, self.pre["cap_past_due"]),
            "NumberRealEstateLoansOrLines": a.real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": min(a.past_due_60_89, self.pre["cap_past_due"]),
            "NumberOfDependents": 0 if a.dependents is None else a.dependents,
            "MonthlyIncome_was_missing": income_missing,
        }
        return pd.DataFrame([row])[FEATURES]

    def tier(self, p: float) -> str:
        if p < self.tiers["low_below"]:
            return "Low"
        return "High" if p >= self.tiers["high_above"] else "Medium"

    def score(self, a, top_k: int = 3):
        X = self.to_frame(a)
        p = float(self.pipe.predict_proba(X)[:, 1][0])
        factors = []
        if self.explainer is not None:
            Xt = self.pipe[:-1].transform(X) if len(self.pipe.steps) > 1 else X
            sv = self.explainer.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            contrib = sv[0]
            for i in np.argsort(-np.abs(contrib))[:top_k]:
                f = FEATURES[i]
                factors.append({"feature": _HUMAN[f], "value": float(X.iloc[0, i]),
                                "contribution": round(float(contrib[i]), 4),
                                "direction": "increases risk" if contrib[i] > 0 else "decreases risk"})
        return p, self.tier(p), p >= self.threshold, factors, X.iloc[0].to_dict()
