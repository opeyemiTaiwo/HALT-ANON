#!/usr/bin/env python3
"""
Step 2 - leave-one-generator-out, with grounded and embedding baselines.

Three folds (hold out Claude, GPT, Gemini in turn). Feature sets:
  tfidf              lexical content, ungrounded
  embed              OpenAI embeddings, ungrounded  (skipped if .npy absent)
  grounded           report-vs-reference features only
  grounded+surface   grounded plus the surface block

Writes: logo_results.csv, LOGO_REPORT.md
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (H, TARGETS, AXES, SEED, RAW_DIR, OUT_DIR, JOIN, HEDGE_RE,
                    TRUNCATION_CAP, content_tokens, load_benchmark, banner)

warnings.filterwarnings("ignore")
banner("STEP 2  leave-one-generator-out")
df = load_benchmark()

print("grounded features ...")
rt = [set(content_tokens(x)) for x in df.model_output]
gt = [set(content_tokens(x)) for x in df.ground_truth]
G = pd.DataFrame([{
    "novel_rate": len(r - q) / max(len(r), 1),
    "missing_rate": len(q - r) / max(len(q), 1),
    "jaccard": len(r & q) / max(len(r | q), 1),
    "ref_coverage": len(r & q) / max(len(q), 1),
    "len_ratio": len(r) / max(len(q), 1),
    "log_novel": np.log1p(len(r - q)),
    "log_missing": np.log1p(len(q - r)),
} for r, q in zip(rt, gt)])

tf = TfidfVectorizer(max_features=30000, min_df=3, sublinear_tf=True,
                     strip_accents="unicode", stop_words="english")
tf.fit(pd.concat([df.model_output, df.ground_truth.drop_duplicates()]))
A, B = tf.transform(df.model_output), tf.transform(df.ground_truth)
num = np.asarray(A.multiply(B).sum(axis=1)).ravel()
den = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()
              * np.asarray(B.multiply(B).sum(axis=1)).ravel()) + 1e-9
G["tfidf_cos"] = num / den

w = df.model_output.str.split().str.len().clip(lower=1)
S = pd.DataFrame({"n_words": w, "log_words": np.log1p(w),
                  "hedge_rate": df.model_output.str.count(HEDGE_RE) / w,
                  "digit_rate": df.model_output.str.count(r"\d") / w,
                  "truncated": (df.model_output_full_len > TRUNCATION_CAP).astype(int)})

X_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                          sublinear_tf=True, strip_accents="unicode"
                          ).fit_transform(df.model_output)

FEATS = {"tfidf": lambda: X_tfidf,
         "grounded": lambda: G.to_numpy(float),
         "grounded+surface": lambda: np.hstack([G.to_numpy(float),
                                                S.to_numpy(float)])}

emb_ok = (RAW_DIR / "embeddings_openai.npy").exists() and \
         (RAW_DIR / "embedding_index.csv").exists()
if emb_ok:
    E = np.load(RAW_DIR / "embeddings_openai.npy")
    ix = pd.read_csv(RAW_DIR / "embedding_index.csv")
    ix["emb_row"] = np.arange(len(ix))
    d2 = df.merge(ix[JOIN + ["emb_row"]], on=JOIN, how="left",
                  validate="one_to_one")
    assert d2.emb_row.notna().all(), "embedding index misses reports"
    E = E[d2.emb_row.to_numpy().astype(int)]
    FEATS["embed"] = lambda: E
    print(f"embeddings aligned: {E.shape}")
else:
    print("embeddings not found -> skipping the `embed` baseline")

rows = []
from scipy import sparse
for fname, build in FEATS.items():
    X = build()
    for held in ["Claude", "GPT", "Gemini"]:
        tr, te = (df.model != held).to_numpy(), (df.model == held).to_numpy()
        if sparse.issparse(X):
            Xtr, Xte = X[tr], X[te]
        else:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for t in TARGETS:
            y = df[t].to_numpy()
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                rows.append({"features": fname, "held_out": held,
                             "target": t, "auc": np.nan}); continue
            clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, y[tr])
            rows.append({"features": fname, "held_out": held, "target": t,
                         "auc": roc_auc_score(y[te],
                                              clf.predict_proba(Xte)[:, 1])})
        print(f"  {fname} / held-out {held}")

r = pd.DataFrame(rows)
r.to_csv(OUT_DIR / "logo_results.csv", index=False)

piv = r.pivot_table(index="features", columns="target",
                    values="auc")[TARGETS].round(3)
for ax, cs in AXES.items():
    piv[ax] = piv[cs].mean(axis=1).round(3)
piv["macro"] = piv[H].mean(axis=1).round(3)
out = ["# Leave-one-generator-out\n",
       "AUC on the held-out generator, averaged over three folds.\n",
       piv.to_markdown(), "\n\n## Per-fold detail\n"]
from tqdm.auto import tqdm
for f in tqdm(FEATS, desc='feature sets', unit='set'):
    out.append(f"\n### {f}\n")
    out.append(r[r.features == f].pivot(index="held_out", columns="target",
                                        values="auc")[TARGETS].round(3).to_markdown())
(OUT_DIR / "LOGO_REPORT.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
