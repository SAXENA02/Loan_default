"""Stage 5: SHAP explainability. Writes plots to reports/. Run: python -m src.explain"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FEATURES, MODEL_PATH, TARGET

pipe = joblib.load(MODEL_PATH)
test = pd.read_csv("data/test.csv")
sample = test.sample(3000, random_state=0)
X = sample[FEATURES]
Xt = pipe[:-1].transform(X) if len(pipe.steps) > 1 else X

expl = shap.TreeExplainer(pipe[-1])
sv = expl(Xt)
if sv.values.ndim == 3:
    sv = sv[:, :, 1]

plt.figure(); shap.summary_plot(sv, X, show=False); plt.tight_layout()
plt.savefig("reports/shap_summary_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close()

plt.figure(); shap.plots.bar(sv, show=False); plt.tight_layout()
plt.savefig("reports/shap_importance_bar.png", dpi=130, bbox_inches="tight"); plt.close()

proba = pipe.predict_proba(X)[:, 1]
for label, idx in (("high_risk", int(np.argmax(proba))), ("low_risk", int(np.argmin(proba)))):
    plt.figure(); shap.plots.waterfall(sv[idx], show=False); plt.tight_layout()
    plt.savefig(f"reports/shap_waterfall_{label}.png", dpi=130, bbox_inches="tight"); plt.close()

imp = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
imp.round(4).to_csv("reports/shap_mean_abs.csv", header=["mean_abs_shap"])
print(imp.round(3).to_string())
