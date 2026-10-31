"""
Stage 7: drift monitoring. Reference = training-time sample, current = a newer batch.
Run: python -m monitoring.drift_report [path/to/current.csv]
Produces reports/drift_report.html (Evidently) and reports/drift_summary.json,
and logs the summary to MLflow.
"""
import json
import sys

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from src.config import FEATURES

cur_path = sys.argv[1] if len(sys.argv) > 1 else "data/drift_batch.csv"
ref = pd.read_csv("models/reference_sample.csv")[FEATURES]
cur = pd.read_csv(cur_path)[FEATURES]


def psi(a, b, bins=10):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


rows = []
for c in FEATURES:
    ks = stats.ks_2samp(ref[c], cur[c])
    rows.append({"feature": c, "ks_stat": round(float(ks.statistic), 4), "ks_p": float(ks.pvalue),
                 "psi": round(psi(ref[c].values, cur[c].values), 4),
                 "ref_mean": round(float(ref[c].mean()), 3), "cur_mean": round(float(cur[c].mean()), 3)})
tbl = pd.DataFrame(rows).sort_values("psi", ascending=False)
# PSI reading: <0.1 stable, 0.1-0.25 moderate, >0.25 major. KS statistic >0.10 also counts as
# drift, because PSI can miss shifts concentrated in the tails of skewed features.
tbl["status"] = np.where(tbl.psi > 0.25, "major",
                 np.where((tbl.psi > 0.10) | (tbl.ks_stat > 0.10), "moderate", "stable"))
print(tbl.to_string(index=False))

evidently_share = None
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    snap = Report([DataDriftPreset()]).run(cur, ref)
    snap.save_html("reports/drift_report.html")
    evidently_share = "saved reports/drift_report.html"
except Exception as e:  # keep the PSI/KS summary even if the HTML report API changes
    print("Evidently report skipped:", repr(e))

summary = {"current_file": cur_path, "n_reference": len(ref), "n_current": len(cur),
           "features_moderate_or_major": tbl[tbl.status != "stable"].feature.tolist(),
           "table": tbl.to_dict(orient="records"), "evidently_html": evidently_share}
json.dump(summary, open("reports/drift_summary.json", "w"), indent=2)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan-default-monitoring")
with mlflow.start_run(run_name=cur_path):
    mlflow.log_metric("features_drifted", len(summary["features_moderate_or_major"]))
    mlflow.log_metric("max_psi", float(tbl.psi.max()))
print("\nDrifted features:", summary["features_moderate_or_major"] or "none")
