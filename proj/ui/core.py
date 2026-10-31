"""Shared loaders, preprocessing and scoring helpers for the Streamlit app.

Preprocessing here mirrors data preparation + app/model_service.py exactly, so a
score shown in the UI equals the score the FastAPI service would return.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import FEATURES, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts):
    return os.path.join(ROOT, *parts)


# Raw inputs a user supplies (the 11th model feature is derived from MonthlyIncome)
RAW = [f for f in FEATURES if f != "MonthlyIncome_was_missing"]

LABEL = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late",
    "NumberOfDependents": "Dependents",
    "MonthlyIncome_was_missing": "Income not provided",
}

PRESETS = {
    "Low risk": dict(age=55, util=0.10, debt=0.20, income=8000.0, lines=8, re=1, dep=1, l30=0, l60=0, l90=0),
    "Medium risk": dict(age=38, util=0.70, debt=0.50, income=4500.0, lines=10, re=1, dep=2, l30=1, l60=0, l90=0),
    "High risk": dict(age=27, util=1.00, debt=0.80, income=2500.0, lines=6, re=0, dep=2, l30=3, l60=1, l90=2),
}


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    return joblib.load(path("models", "model.joblib"))


@st.cache_resource
def load_meta():
    return json.load(open(path("models", "model_meta.json")))


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(load_model()[-1])


@st.cache_data
def load_df(name):
    return pd.read_csv(path("data", name))


@st.cache_data
def scored_test():
    t = load_df("test.csv")
    return t.assign(p=load_model().predict_proba(t[FEATURES])[:, 1])


@st.cache_data
def shap_sample(n=3000):
    t = load_df("test.csv").sample(n, random_state=0)
    X = t[FEATURES]
    return X, np.asarray(get_explainer().shap_values(X))


# ---------------------------------------------------------- preprocessing
def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw applicant columns -> the 11 model features (caps + imputation as in training)."""
    pre = load_meta()["preprocessing"]
    d = raw[RAW].copy()
    missing = d["MonthlyIncome"].isna()
    decile = (d["age"] // 10 * 10).astype(int).astype(str)
    fill = decile.map(pre["income_median_by_age_decile"]).fillna(pre["income_median_overall"])
    d["MonthlyIncome"] = d["MonthlyIncome"].where(~missing, fill)
    d["MonthlyIncome_was_missing"] = missing.astype(int)
    d["NumberOfDependents"] = d["NumberOfDependents"].fillna(0)
    d["RevolvingUtilizationOfUnsecuredLines"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(upper=pre["cap_utilization"])
    d["DebtRatio"] = d["DebtRatio"].clip(upper=pre["cap_debt_ratio"])
    for c in ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"):
        d[c] = d[c].clip(upper=pre["cap_past_due"])
    return d[FEATURES]


def tier_of(p):
    t = load_meta()["risk_tiers"]
    if p < t["low_below"]:
        return "Low"
    return "High" if p >= t["high_above"] else "Medium"


def predict(raw: pd.DataFrame):
    return load_model().predict_proba(build_frame(raw))[:, 1]


def applicant_from_state(ss) -> pd.DataFrame:
    """Build the one-row raw frame from the Score page widgets stored in session_state."""
    return pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": ss["w_util"],
        "age": ss["w_age"],
        "NumberOfTime30-59DaysPastDueNotWorse": ss["w_l30"],
        "DebtRatio": ss["w_debt"],
        "MonthlyIncome": ss["w_income"] if ss["w_income_known"] else np.nan,
        "NumberOfOpenCreditLinesAndLoans": ss["w_lines"],
        "NumberOfTimes90DaysLate": ss["w_l90"],
        "NumberRealEstateLoansOrLines": ss["w_re"],
        "NumberOfTime60-89DaysPastDueNotWorse": ss["w_l60"],
        "NumberOfDependents": ss["w_dep"] if ss["w_dep_known"] else np.nan,
    }])


def explain_one(X: pd.DataFrame):
    """SHAP contributions (log-odds) for a single prepared row, as a labelled Series."""
    sv = np.asarray(get_explainer().shap_values(X))[0]
    return pd.Series(sv, index=[LABEL[f] for f in FEATURES])


def base_rate_logodds():
    return float(np.ravel(get_explainer().expected_value)[0])


# ================================================================ banking helpers
ACTION = {
    "Low": "Standard processing",
    "Medium": "Enhanced review: verify income and obligations",
    "High": "Refer to senior credit officer",
}
BAND_COLOR = {"Low": "#2e9e5b", "Medium": "#e0a21b", "High": "#d64545"}
_LATE = {"NumberOfTime30-59DaysPastDueNotWorse": "30-59 days", "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days",
         "NumberOfTimes90DaysLate": "90+ days"}


def band_series(p):
    t = load_meta()["risk_tiers"]
    return pd.cut(pd.Series(p), [-1, t["low_below"], t["high_above"], 2], labels=["Low", "Medium", "High"], right=False)


def describe(feature, v, income_missing=False):
    """Plain-language statement of one input value."""
    if feature == "RevolvingUtilizationOfUnsecuredLines":
        return f"Credit utilization at {v:.0%} of limit"
    if feature == "age":
        return f"Applicant age {v:.0f}"
    if feature == "DebtRatio":
        return f"Debt ratio {v:.2f}"
    if feature == "MonthlyIncome":
        return f"Monthly income {v:,.0f}" + (" (not provided, estimated)" if income_missing else "")
    if feature in _LATE:
        return f"{int(v)} payment(s) {_LATE[feature]} late" if v > 0 else f"No payments {_LATE[feature]} late"
    if feature == "NumberOfOpenCreditLinesAndLoans":
        return f"{v:.0f} open credit lines / loans"
    if feature == "NumberRealEstateLoansOrLines":
        return f"{v:.0f} real-estate loans"
    if feature == "NumberOfDependents":
        return f"{v:.0f} dependents"
    return "Income not provided"


def strength(s):
    a = abs(s)
    return "strong" if a >= 0.5 else ("moderate" if a >= 0.2 else "slight")


def key_factors(X: pd.DataFrame, k=3):
    """Top factors raising and lowering risk for one prepared row, in plain language."""
    sv = np.asarray(get_explainer().shap_values(X))[0]
    row = X.iloc[0]
    miss = bool(row["MonthlyIncome_was_missing"])
    items = [(f, float(sv[i]), describe(f, row[f], miss)) for i, f in enumerate(FEATURES)]
    up = sorted([i for i in items if i[1] > 0.02 and i[0] != "MonthlyIncome_was_missing"], key=lambda i: -i[1])[:k]
    down = sorted([i for i in items if i[1] < -0.02 and i[0] != "MonthlyIncome_was_missing"], key=lambda i: i[1])[:k]
    return [(d, strength(s)) for _, s, d in up], [(d, strength(s)) for _, s, d in down]
