#!/usr/bin/env python3
"""
Step 1 - baseline detectors.

Feature sets: surface, surface_noid (identity removed), tfidf.
Splits: random, heldout_model, heldout_technique.

Generator identity is dropped automatically on heldout_model and technique
identity on heldout_technique, since those features are constant in training
and unseen at test.

Writes: baseline_results.csv, BASELINE_REPORT.md
"""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (H, TARGETS, SEED, OUT_DIR, MULTI_TURN, HEDGE_RE,
                    TRUNCATION_CAP, load_benchmark, banner)

warnings.filterwarnings("ignore")
banner("STEP 1  baselines")
df = load_benchmark()

txt = df.model_output
w = txt.str.split().str.len().clip(lower=1)
s = txt.str.count(r"[.!?]").clip(lower=1)
surf = pd.DataFrame({
    "n_words": w, "log_words": np.log1p(w), "n_sents": s,
    "mean_sent_len": w / s,
    "hedge_rate": txt.str.count(HEDGE_RE) / w,
    "detail_rate": txt.str.count(r"\b(\d+|\d{1,2}:\d{2})\b") / w,
    "type_token": txt.str.lower().apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1)),
    "truncated": (df.model_output_full_len > TRUNCATION_CAP).astype(int),
    "gt_words": df.ground_truth.str.split().str.len(),
})
id_gen = pd.get_dummies(df.model, prefix="gen").astype(float)
id_tec = pd.get_dummies(df.technique, prefix="tec").astype(float)
id_tec["multi_turn"] = df.technique.isin(MULTI_TURN).astype(float)

print("fitting tf-idf ...")
X_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                          sublinear_tf=True,
                          strip_accents="unicode").fit_transform(txt)

SPLITS = {"random": ("split_random", True, True),
          "heldout_model": ("split_heldout_model", False, True),
          "heldout_technique": ("split_heldout_technique", True, False)}


def build(fs, use_gen, use_tec):
    if fs == "tfidf":
        return X_tfidf
    parts = [surf.to_numpy(float)]
    if fs == "surface":
        if use_gen:
            parts.append(id_gen.to_numpy())
        if use_tec:
            parts.append(id_tec.to_numpy())
    return np.hstack(parts)


rows = []
for sname, (scol, ug, ut) in SPLITS.items():
    tr, te = (df[scol] == "train").to_numpy(), (df[scol] == "test").to_numpy()
    for fs in ["surface", "surface_noid", "tfidf"]:
        X = build(fs, ug, ut)
        if sparse.issparse(X):
            Xtr, Xte = X[tr], X[te]
        else:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for t in TARGETS:
            y = df[t].to_numpy()
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                rows.append({"split": sname, "features": fs, "target": t,
                             "auc": np.nan, "f1": np.nan}); continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            rows.append({"split": sname, "features": fs, "target": t,
                         "auc": roc_auc_score(y[te], p),
                         "f1": f1_score(y[te], (p >= .5).astype(int)),
                         "pos_rate_test": float(y[te].mean())})
        print(f"  {sname} / {fs}")

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "baseline_results.csv", index=False)

out = ["# Baseline sweep\n",
       "Logistic regression, balanced class weights, seed 42. "
       "All splits video-disjoint. `macro_hq` excludes H1 (low tier).\n"]
for fs in ["surface", "surface_noid", "tfidf"]:
    out.append(f"\n## {fs}\n")
    p = res[res.features == fs].pivot(index="split", columns="target",
                                      values="auc")[TARGETS].round(3)
    p["macro"] = p[H].mean(axis=1).round(3)
    p["macro_hq"] = p[[h for h in H if h != "H1"]].mean(axis=1).round(3)
    out.append(p.to_markdown())
(OUT_DIR / "BASELINE_REPORT.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
