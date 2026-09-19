#!/usr/bin/env python3
"""
Step 8 - does the detector read evidence, or recognise the writer?

Two measurements that test the shortcut directly rather than inferring it from
the transfer result:

  (a) SOURCE CLASSIFICATION. Train a classifier to predict which source LLM
      wrote a report. If this is easy, the shortcut is available.

  (b) BASE-RATE PREDICTOR. A predictor that sees no report text at all: it
      looks up which source LLM wrote the report and returns that source's
      training-set prevalence for the label. Whatever AUC this reaches is
      attributable to source identity alone. Comparing it against the full
      detector, on the above-chance scale, gives the share of the detector's
      signal that source identity supplies.

CPU only, a few minutes. No GPU, no network.

    python3 08_shortcut_test.py

Writes: shortcut_test.csv
"""
import argparse, os
import numpy as np, pandas as pd, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
warnings.filterwarnings('ignore')

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
TARGETS = H + ['any_hallucination']
AXIS = {'H1': 'fabrication', 'H5': 'fabrication', 'H6': 'fabrication',
        'H3': 'omission', 'H2': 'distortion', 'H4': 'distortion',
        'any_hallucination': 'binary'}

ap = argparse.ArgumentParser()
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/shortcut_test.csv')
args = ap.parse_args()

df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')

X = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                    sublinear_tf=True, strip_accents='unicode').fit_transform(df.model_output)
tr = (df.split_random == 'train').to_numpy()
te = (df.split_random == 'test').to_numpy()

# ---------------------------------------------------------------- (a)
print('=== (a) SOURCE-MODEL CLASSIFICATION (video-disjoint split) ===')
clf = LogisticRegression(max_iter=3000).fit(X[tr], df.model[tr])
pred = clf.predict(X[te])
acc = accuracy_score(df.model[te], pred)
mf1 = f1_score(df.model[te], pred, average='macro')
print(f'  accuracy  {acc:.3f}   (chance {1/df.model.nunique():.3f})')
print(f'  macro F1  {mf1:.3f}')
for m in sorted(df.model.unique()):
    sel = (df.model[te] == m).to_numpy()
    print(f'    {m:8s} recall {(pred[sel] == m).mean():.3f}')

# ---------------------------------------------------------------- (b)
print('\n=== (b) BASE-RATE PREDICTOR vs FULL DETECTOR ===')
print(f"{'type':6s}{'axis':13s}{'base rate':>10s}{'tfidf':>8s}{'share':>8s}")
rows = []
for t in TARGETS:
    y = df[t].to_numpy()
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        continue
    # base-rate predictor: no text, just the source LLM's training prevalence
    rate = df.loc[tr].groupby('model')[t].mean()
    p_base = df.model[te].map(rate).to_numpy()
    auc_base = roc_auc_score(y[te], p_base)
    # full lexical detector
    d = LogisticRegression(max_iter=2000, class_weight='balanced',
                           random_state=42).fit(X[tr], y[tr])
    auc_full = roc_auc_score(y[te], d.predict_proba(X[te])[:, 1])
    share = (auc_base - 0.5) / (auc_full - 0.5) if auc_full > 0.5 else float('nan')
    rows.append({'target': t, 'axis': AXIS[t], 'auc_base_rate': round(auc_base, 3),
                 'auc_tfidf': round(auc_full, 3), 'share_of_signal': round(share, 3)})
    print(f'{t:6s}{AXIS[t]:13s}{auc_base:10.3f}{auc_full:8.3f}{share:7.0%}')

res = pd.DataFrame(rows)
res.attrs['source_accuracy'] = acc
res.to_csv(args.out, index=False)

fab = res[res.axis == 'fabrication'].share_of_signal
omi = res[res.axis == 'omission'].share_of_signal
print(f'\n  fabrication: {fab.min():.0%}-{fab.max():.0%} of the signal is source identity')
print(f'  omission   : {omi.mean():.0%}')
print('\n  -> that is why fabrication loses most when the source is withheld')
print('\nwrote ->', args.out)
