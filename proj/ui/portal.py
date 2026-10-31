"""Credit Risk Assessment Portal: pages for loan officers and credit managers."""
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.config import FEATURES, TARGET
from ui import core

TIERS = ["Low", "Medium", "High"]


def _fig(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")


# =================================================================== DASHBOARD
def dashboard():
    t = core.scored_test().copy()
    t["band"] = core.band_series(t["p"].values).values
    st.title("Portfolio Dashboard")
    st.caption("Sample portfolio: 27,000 historical applications. Risk scores are model estimates; "
               "observed outcomes come from the historical record.")

    c = st.columns(4)
    c[0].metric("Applications", f"{len(t):,}")
    c[1].metric("Average risk score", f"{t['p'].mean():.1%}")
    c[2].metric("Medium-risk share", f"{(t.band == 'Medium').mean():.1%}")
    c[3].metric("High-risk share", f"{(t.band == 'High').mean():.1%}")

    l, r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("**Applications by risk band**")
        cnt = t.band.value_counts().reindex(TIERS)
        _fig(px.bar(x=cnt.index, y=cnt.values, color=cnt.index, color_discrete_map=core.BAND_COLOR,
                    labels={"x": "", "y": "Applications"}).update_layout(showlegend=False))
    with r, st.container(border=True):
        st.markdown("**Observed default rate by risk band**")
        obs = t.groupby("band", observed=True)[TARGET].mean().reindex(TIERS) * 100
        fig = px.bar(x=obs.index, y=obs.values, color=obs.index, color_discrete_map=core.BAND_COLOR,
                     labels={"x": "", "y": "Defaulted (%)"}, text=[f"{v:.1f}%" for v in obs.values])
        _fig(fig.update_layout(showlegend=False))
        st.caption("In the historical record, applicants scored High defaulted far more often than those scored Low, "
                   "so the bands separate risk as intended.")

    l, r = st.columns(2)
    with l, st.container(border=True):
        st.markdown("**Average risk score by age group**")
        ag = pd.cut(t.age, [17, 30, 40, 50, 60, 70, 120], labels=["18-30", "31-40", "41-50", "51-60", "61-70", "70+"])
        v = t.groupby(ag, observed=True)["p"].mean() * 100
        _fig(px.bar(x=v.index.astype(str), y=v.values, labels={"x": "", "y": "Average risk (%)"}))
    with r, st.container(border=True):
        st.markdown("**Average risk score by credit utilization**")
        ub = pd.cut(t.RevolvingUtilizationOfUnsecuredLines, [-0.01, 0.1, 0.3, 0.5, 0.8, 1.0, 100],
                    labels=["0-10%", "10-30%", "30-50%", "50-80%", "80-100%", "100%+"])
        v = t.groupby(ub, observed=True)["p"].mean() * 100
        _fig(px.bar(x=v.index.astype(str), y=v.values, labels={"x": "", "y": "Average risk (%)"}))

    with st.container(border=True):
        st.markdown("**What drives risk across the portfolio**")
        X, sv = core.shap_sample()
        imp = pd.Series(np.abs(sv).mean(0), index=[core.LABEL[f] for f in FEATURES]).drop("Income not provided").sort_values()
        _fig(px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Relative influence on risk", "y": ""}), 340)
        st.caption("Payment behaviour and how much of available credit is already used matter most.")


# =================================================================== NEW ASSESSMENT
DEFAULTS = dict(w_ref="", w_age=42, w_util=0.45, w_debt=0.31, w_income=5400.0, w_income_known=True, w_lines=8,
                w_re=1, w_dep=2, w_dep_known=True, w_l30=0, w_l60=0, w_l90=0)
PRESET_MAP = dict(w_age="age", w_util="util", w_debt="debt", w_income="income", w_lines="lines", w_re="re",
                  w_dep="dep", w_l30="l30", w_l60="l60", w_l90="l90")


def _load_profile(name):
    for k, src in PRESET_MAP.items():
        st.session_state[k] = core.PRESETS[name][src]
    st.session_state["w_income_known"] = st.session_state["w_dep_known"] = True


def _gauge(p):
    t = core.load_meta()["risk_tiers"]
    top = max(0.5, min(1.0, p * 1.3)) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, top]}, "bar": {"color": "#1c2430"},
               "steps": [{"range": [0, t["low_below"] * 100], "color": "#bfe6cf"},
                         {"range": [t["low_below"] * 100, t["high_above"] * 100], "color": "#f6df9b"},
                         {"range": [t["high_above"] * 100, top], "color": "#f2b8b8"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0))
    return fig


def _levers(base: pd.DataFrame, p0: float):
    """Actionable scenarios: what the risk would be if the applicant improved a controllable factor."""
    rows = []
    u, d = float(base.RevolvingUtilizationOfUnsecuredLines.iloc[0]), float(base.DebtRatio.iloc[0])
    s = []
    if u > 0.30:
        s.append(("Reduce credit utilization to 30%", {"RevolvingUtilizationOfUnsecuredLines": 0.30}))
    if d > 0.30:
        s.append(("Reduce debt ratio to 0.30", {"DebtRatio": 0.30}))
    if len(s) == 2:
        s.append(("Both of the above", {**s[0][1], **s[1][1]}))
    for name, change in s:
        b = base.copy()
        for k, v in change.items():
            b[k] = v
        p = float(core.predict(b)[0])
        rows.append({"Scenario": name, "Risk score": f"{p:.1%}", "Change": f"{(p - p0) * 100:+.1f} pts",
                     "Risk band": str(core.band_series([p])[0])})
    return pd.DataFrame(rows)


def assessment():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("log", [])
    st.title("New Assessment")
    st.caption("Enter the application details. The risk assessment updates as you type.")

    with st.expander("Load a demo profile"):
        b = st.columns(4)
        for i, n in enumerate(core.PRESETS):
            b[i].button(n, on_click=_load_profile, args=(n,), width="stretch")
        b[3].button("Clear form", on_click=lambda: st.session_state.update(DEFAULTS), width="stretch")

    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True):
        st.markdown("**Application details**")
        st.text_input("Application reference", key="w_ref", placeholder="e.g. APP-2026-0001",
                      help="Use an internal reference. Do not enter names or ID numbers.")
        a, b2 = st.columns(2)
        a.number_input("Applicant age", 18, 110, key="w_age")
        b2.number_input("Open credit lines / loans", 0, 60, key="w_lines")
        st.slider("Credit utilization (share of available credit in use)", 0.0, 2.0, step=0.01, key="w_util")
        st.slider("Debt ratio (monthly debt payments / monthly income)", 0.0, 3.0, step=0.01, key="w_debt")
        c1, c2 = st.columns(2)
        c1.checkbox("Income provided", key="w_income_known")
        c2.checkbox("Dependents provided", key="w_dep_known")
        d1, d2, d3 = st.columns(3)
        d1.number_input("Monthly income", 0.0, 500000.0, step=100.0, key="w_income", disabled=not st.session_state["w_income_known"])
        d2.number_input("Dependents", 0, 15, key="w_dep", disabled=not st.session_state["w_dep_known"])
        d3.number_input("Real-estate loans", 0, 30, key="w_re")
        st.markdown("**Payment history** (number of occurrences)")
        e1, e2, e3 = st.columns(3)
        e1.number_input("30-59 days late", 0, 20, key="w_l30")
        e2.number_input("60-89 days late", 0, 20, key="w_l60")
        e3.number_input("90+ days late", 0, 20, key="w_l90")

    raw = core.applicant_from_state(st.session_state)
    X = core.build_frame(raw)
    p = float(core.load_model().predict_proba(X)[0, 1])
    band = str(core.band_series([p])[0])
    action = core.ACTION[band]

    with right:
        with st.container(border=True):
            st.markdown("**Risk assessment**")
            st.plotly_chart(_gauge(p), width="stretch")
            m = st.columns(2)
            m[0].metric("Risk band", band)
            m[1].metric("Relative to average applicant", f"{p / 0.0668:.1f}x")
            {"Low": st.success, "Medium": st.warning, "High": st.error}[band](f"**Suggested action:** {action}")
        with st.container(border=True):
            st.markdown("**Key factors**")
            up, down = core.key_factors(X)
            u, dn = st.columns(2)
            u.markdown("Increasing risk")
            for txt, s in up or [("None significant", "")]:
                u.markdown(f"- {txt}" + (f" *({s})*" if s else ""))
            dn.markdown("Reducing risk")
            for txt, s in down or [("None significant", "")]:
                dn.markdown(f"- {txt}" + (f" *({s})*" if s else ""))

    st.markdown("**What could lower this risk?**")
    lev = _levers(raw, p)
    if lev.empty:
        st.caption("Utilization and debt ratio are already at or below 30%, so there are no simple levers to show.")
    else:
        st.dataframe(lev, hide_index=True, width="stretch")
        st.caption("Illustrative scenarios from the same model. They are not offers or commitments.")

    st.divider()
    s1, s2 = st.columns([1, 3])
    officer = st.session_state.get("officer", "")
    if s1.button("Save to assessment log", type="primary", width="stretch"):
        rec = {"Reference": st.session_state["w_ref"] or "(none)", "Assessed at": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "Assessed by": officer or "(not entered)", "Risk score (%)": round(p * 100, 1), "Risk band": band,
               "Suggested action": action, "Model version": core.load_meta()["model_version"]}
        rec.update({core.LABEL[c]: raw[c].iloc[0] for c in core.RAW})
        st.session_state["log"].append(rec)
        s2.success("Saved. See the log below.")
    if st.session_state["log"]:
        st.markdown("**Assessment log (this session)**")
        log = pd.DataFrame(st.session_state["log"])
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("Download log (CSV)", log.to_csv(index=False), "assessment_log.csv", "text/csv")


# =================================================================== PORTFOLIO UPLOAD
def portfolio():
    st.title("Portfolio Upload")
    st.caption("Assess many applications at once. Upload a CSV with the 10 application columns. "
               "Income and dependents may be left blank.")
    tpl = core.load_df("test.csv")[core.RAW].head(50).rename(columns=lambda c: c)
    st.download_button("Download CSV template (50 example rows)", tpl.to_csv(index=False), "applications_template.csv", "text/csv")
    up = st.file_uploader("Upload applications (CSV)", type="csv")
    demo = st.checkbox("Use 500 sample applications instead")
    df = pd.read_csv(up) if up is not None else (core.load_df("test.csv")[core.RAW].sample(500, random_state=1).reset_index(drop=True) if demo else None)
    if df is None:
        return
    miss = [c for c in core.RAW if c not in df.columns]
    if miss:
        st.error("Missing columns: " + ", ".join(miss))
        return
    for c in core.RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    req = [c for c in core.RAW if c not in ("MonthlyIncome", "NumberOfDependents")]
    if df[req].isna().any().any():
        st.error("Some required cells are empty or not numeric. Please correct them and upload again.")
        return
    if (df["age"] < 18).any():
        st.error("Every applicant must be 18 or older.")
        return
    p = core.predict(df)
    out = df.copy()
    out.insert(0, "Risk score (%)", (p * 100).round(1))
    out.insert(1, "Risk band", core.band_series(p).astype(str).values)
    out.insert(2, "Suggested action", out["Risk band"].map(core.ACTION))

    c = st.columns(4)
    c[0].metric("Applications", f"{len(out):,}")
    c[1].metric("Average risk score", f"{p.mean():.1%}")
    c[2].metric("Medium risk", int((out["Risk band"] == "Medium").sum()))
    c[3].metric("High risk", int((out["Risk band"] == "High").sum()))
    l, r = st.columns(2)
    with l, st.container(border=True):
        cnt = out["Risk band"].value_counts().reindex(TIERS).fillna(0)
        _fig(px.bar(x=cnt.index, y=cnt.values, color=cnt.index, color_discrete_map=core.BAND_COLOR,
                    labels={"x": "", "y": "Applications"}, title="Risk bands").update_layout(showlegend=False))
    with r, st.container(border=True):
        _fig(px.histogram(out, x="Risk score (%)", nbins=30, title="Score distribution"))
    show = st.multiselect("Show risk bands", TIERS, default=["High", "Medium"])
    st.dataframe(out[out["Risk band"].isin(show)].sort_values("Risk score (%)", ascending=False), hide_index=True, width="stretch")
    st.download_button("Download all results (CSV)", out.to_csv(index=False), "portfolio_assessment.csv", "text/csv")


# =================================================================== DATA STABILITY
def _psi(a, b, bins=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    e[0], e[-1] = -np.inf, np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-6
    pb = np.histogram(b, e)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def stability():
    st.title("Data Stability")
    st.caption("Checks whether recent applications still resemble the data the model was built on. "
               "Large shifts mean scores should be reviewed.")
    ref = pd.read_csv(core.path("models", "reference_sample.csv"))[FEATURES]
    src = st.radio("Compare against", ["Recent batch (sample)", "Historical sample", "Upload CSV"], horizontal=True)
    cur = None
    if src.startswith("Recent"):
        cur = core.load_df("drift_batch.csv")[FEATURES]
    elif src.startswith("Historical"):
        cur = core.load_df("test.csv")[FEATURES]
    else:
        up = st.file_uploader("CSV with the 10 application columns", type="csv")
        if up is not None:
            d = pd.read_csv(up)
            if all(c in d.columns for c in core.RAW):
                cur = core.build_frame(d)
            else:
                st.error("Missing columns. The 10 application columns are required.")
    if cur is None:
        return
    rows = []
    for f in FEATURES:
        if f == "MonthlyIncome_was_missing":
            continue
        ks = stats.ks_2samp(ref[f], cur[f]).statistic
        psi = _psi(ref[f].values, cur[f].values)
        s = "Action needed" if psi > 0.25 else ("Watch" if (psi > 0.10 or ks > 0.10) else "Stable")
        chg = (cur[f].mean() - ref[f].mean()) / (abs(ref[f].mean()) + 1e-9) * 100
        rows.append({"Factor": core.LABEL[f], "Training average": round(ref[f].mean(), 2), "Current average": round(cur[f].mean(), 2),
                     "Change (%)": round(chg, 1), "Status": s, "_f": f, "PSI": round(psi, 3), "KS": round(ks, 3)})
    t = pd.DataFrame(rows)
    flagged = t[t.Status != "Stable"]
    overall = "Action needed" if (t.Status == "Action needed").any() else ("Watch" if len(flagged) else "Stable")
    m = st.columns(3)
    m[0].metric("Applications checked", f"{len(cur):,}")
    m[1].metric("Factors shifted", len(flagged))
    m[2].metric("Overall status", overall)
    if len(flagged):
        st.warning("Shift detected in: " + ", ".join(flagged.Factor) + ". Review recent scores before relying on them.")
    else:
        st.success("Recent applications look consistent with the model's training data.")
    st.dataframe(t[["Factor", "Training average", "Current average", "Change (%)", "Status"]], hide_index=True, width="stretch")
    f = st.selectbox("Compare distribution for", list(t.Factor))
    col = t.loc[t.Factor == f, "_f"].iloc[0]
    hi = max(ref[col].quantile(0.99), cur[col].quantile(0.99))
    d = pd.concat([ref[[col]].assign(Data="Training"), cur[[col]].assign(Data="Recent")])
    d = d[d[col] <= hi]
    _fig(px.histogram(d, x=col, color="Data", barmode="overlay", histnorm="probability density", nbins=40, opacity=0.6,
                      labels={col: f}), 300)
    with st.expander("Technical detail"):
        st.dataframe(t[["Factor", "PSI", "KS"]], hide_index=True, width="stretch")
        st.caption("PSI under 0.10 is stable, 0.10 to 0.25 is a moderate shift, over 0.25 is major. "
                   "A KS statistic over 0.10 is also flagged.")


# =================================================================== GOVERNANCE
def governance():
    meta = core.load_meta()
    x = meta["candidates"][meta["model_name"]]
    st.title("Model Governance")
    st.caption("A short summary for credit managers, auditors and reviewers.")
    st.dataframe(pd.DataFrame([
        ["Purpose", "Estimate the probability that an applicant will become seriously delinquent within two years"],
        ["Output", "Risk score, Low / Medium / High band, suggested action, and the key factors behind each score"],
        ["Intended use", "Decision support. An authorised credit officer makes every final decision"],
        ["Model", f"Gradient-boosted decision trees (XGBoost), version {meta['model_version']}"],
        ["Training data", f"{len(core.load_df('train.csv')):,} historical applications from a public 2011 credit benchmark"],
        ["Ranking power", f"AUC {x['test_auc']:.2f} on {len(core.load_df('test.csv')):,} applications held out from training"],
        ["Reliability of scores", "Predicted risk matches observed default rates within each band (see Dashboard)"],
        ["Explanations", "Every score lists the factors that raised and lowered it"],
        ["Monitoring", "Data Stability page compares recent applications with the training data"],
        ["Audit trail", "Assessments can be saved with reference, time, officer, model version and inputs"],
    ], columns=["Item", "Detail"]), hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown("**Responsible use**")
        st.markdown(
            "- This prototype was built on a **public benchmark dataset**, not on this institution's own lending history. "
            "It must be validated and recalibrated on the institution's data before any live use.\n"
            "- The model uses **age** as an input. Review this against applicable fair-lending rules before deployment.\n"
            "- Scores support, and never replace, credit officer judgement. Applicants with clean records can still default "
            "for reasons the data does not capture.\n"
            "- Do not enter names or national ID numbers. Use an internal application reference.")
