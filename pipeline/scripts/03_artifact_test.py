#!/usr/bin/env python3
"""
Step 3 - label-source divergence test.

For each candidate surface feature, compare how well it predicts the PANEL
label against how well it predicts the HUMAN label, on the same 130 reports.
A large positive gap means the feature tracks the judge's decision rule rather
than the phenomenon the humans are scoring.

Also reports panel over-flagging and misses against agreed human labels.

Writes: artifact_test.csv, ARTIFACT_TEST.md
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (H, HNAME, RAW_DIR, OUT_DIR, JOIN, HEDGE_RE,
                    content_tokens, load_benchmark, require, banner)

require("Rater-A-scores-807.xlsx", "Rater-B-scores-807.xlsx",
        "Panel-A-scores-807.xlsx")
banner("STEP 3  artifact test")

df = load_benchmark()
ra = pd.read_excel(RAW_DIR / "Rater-A-scores-807.xlsx")
rb = pd.read_excel(RAW_DIR / "Rater-B-scores-807.xlsx")
pa = pd.read_excel(RAW_DIR / "Panel-A-scores-807.xlsx")

g = pa[JOIN].copy()
for h in H:
    a, b = ra[f"human_{h}"].to_numpy(), rb[f"human_{h}"].to_numpy()
    g[f"panel_{h}"] = pa[f"panel_{h}"].to_numpy()
    g[f"human_{h}"] = a
    g[f"agree_{h}"] = (a == b).astype(int)

m = g.merge(df[JOIN + ["model_output", "ground_truth", "model_output_full_len"]],
            on=JOIN, how="left", validate="one_to_one")
assert m.model_output.notna().all(), "gold reports not found in benchmark table"

rt = [set(content_tokens(x)) for x in m.model_output]
gt = [set(content_tokens(x)) for x in m.ground_truth]
FEATURES = {
    "n_words": m.model_output.str.split().str.len().to_numpy(float),
    "novel_rate": np.array([len(r - q) / max(len(r), 1) for r, q in zip(rt, gt)]),
    "missing_rate": np.array([len(q - r) / max(len(q), 1) for r, q in zip(rt, gt)]),
    "hedge_rate": (m.model_output.str.count(HEDGE_RE)
                   / m.model_output.str.split().str.len().clip(lower=1)).to_numpy(),
}

rows = []
for fname, x in FEATURES.items():
    for h in H:
        yp = m[f"panel_{h}"].to_numpy()
        ok = m[f"agree_{h}"].to_numpy() == 1
        yh = m[f"human_{h}"].to_numpy()[ok]
        ap = roc_auc_score(yp, x) if len(np.unique(yp)) > 1 else np.nan
        ah = roc_auc_score(yh, x[ok]) if len(np.unique(yh)) > 1 else np.nan
        rows.append({"feature": fname, "type": h, "name": HNAME[h],
                     "n_human_agreed": int(ok.sum()),
                     "panel_pos_rate": round(float(yp.mean()), 3),
                     "human_pos_rate": round(float(yh.mean()), 3),
                     "AUC_vs_panel": round(ap, 3), "AUC_vs_human": round(ah, 3),
                     "gap": round(ap - ah, 3)})
res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "artifact_test.csv", index=False)

out = ["# Label-source divergence\n",
       "AUC of each single feature against the panel label and against the "
       "human label, on the 130 human-validated reports. The human column is "
       "restricted to rows where both raters agree; n is reported per type.\n",
       "A large positive gap means the feature predicts the judge better than "
       "it predicts the humans.\n"]
for fname in FEATURES:
    out.append(f"\n## {fname}\n")
    out.append(res[res.feature == fname][
        ["type", "name", "n_human_agreed", "panel_pos_rate", "human_pos_rate",
         "AUC_vs_panel", "AUC_vs_human", "gap"]].to_markdown(index=False))

out.append("\n\n## Panel behaviour against agreed human labels\n")
rows3 = []
for h in H:
    neg = (m[f"agree_{h}"] == 1) & (m[f"human_{h}"] == 0)
    pos = (m[f"agree_{h}"] == 1) & (m[f"human_{h}"] == 1)
    rows3.append({
        "type": h, "name": HNAME[h],
        "n_human_neg": int(neg.sum()),
        "over_flag_rate": round(float(m.loc[neg, f"panel_{h}"].mean()), 3) if neg.sum() else None,
        "n_human_pos": int(pos.sum()),
        "miss_rate": round(1 - float(m.loc[pos, f"panel_{h}"].mean()), 3) if pos.sum() else None})
out.append(pd.DataFrame(rows3).to_markdown(index=False))
out.append("\n\nCAVEAT: n per cell is small (9 to 110). Treat any single gap "
           "below ~0.10 as noise.")
(OUT_DIR / "ARTIFACT_TEST.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
