#!/usr/bin/env python3
"""
Step 9 - is the degradation difference statistically real?

The paper reports fabrication degrading 1.4 to 3.1 times more than omission and
distortion. A ratio is not a test. This bootstraps the DIFFERENCE

    (fabrication LOSO - fabrication standard) - (other LOSO - other standard)

resampling both the standard-split test set and each leave-one-source-out fold,
and reports a 95% interval. If that interval excludes zero the asymmetry is not
an artifact of sampling.

CPU only, roughly 10 minutes for 1,000 resamples.

    python3 09_bootstrap_diff.py --n-boot 1000

Writes: bootstrap_diff.csv
"""
import argparse, os
import numpy as np, pandas as pd, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
FAB = ['H1', 'H5', 'H6']
OTHER = ['H2', 'H3', 'H4']          # omission + distortion

ap = argparse.ArgumentParser()
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/bootstrap_diff.csv')
ap.add_argument('--n-boot', type=int, default=1000)
ap.add_argument('--drop-h1', action='store_true',
                help='exclude H1, the low-reliability type, from the fabrication axis')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

fab = [t for t in FAB if not (args.drop_h1 and t == 'H1')]
print('fabrication axis:', fab)

df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')
X = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                    sublinear_tf=True, strip_accents='unicode').fit_transform(df.model_output)


def fit_predict(train_mask, test_mask):
    """Return {type: (y_true, y_score)} on the test set."""
    out = {}
    for t in H:
        y = df[t].to_numpy()
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        c = LogisticRegression(max_iter=2000, class_weight='balanced',
                               random_state=42).fit(X[train_mask], y[train_mask])
        out[t] = (y[test_mask], c.predict_proba(X[test_mask])[:, 1])
    return out


print('fitting standard split ...', flush=True)
P_std = fit_predict((df.split_random == 'train').to_numpy(),
                    (df.split_random == 'test').to_numpy())
P_loso = {}
for held in sorted(df.model.unique()):
    print('fitting LOSO fold', held, flush=True)
    P_loso[held] = fit_predict((df.model != held).to_numpy(),
                               (df.model == held).to_numpy())


def axis_auc(P, idx, types):
    """Mean per-type AUC within an axis, on a bootstrap resample."""
    vals = []
    for t in types:
        if t not in P:
            return None
        y, p = P[t]
        i = idx[t]
        if len(np.unique(y[i])) < 2:
            return None
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.mean(vals))


rng = np.random.default_rng(args.seed)
diffs, fab_drops, oth_drops = [], [], []
bar = tqdm(range(args.n_boot), desc='bootstrap', unit='resample')
for b in bar:
    idx_s = {t: rng.choice(len(P_std[t][0]), len(P_std[t][0]), replace=True) for t in P_std}
    f_s, o_s = axis_auc(P_std, idx_s, fab), axis_auc(P_std, idx_s, OTHER)
    if f_s is None or o_s is None:
        continue
    f_l, o_l = [], []
    for held in P_loso:
        idx_l = {t: rng.choice(len(P_loso[held][t][0]), len(P_loso[held][t][0]), replace=True)
                 for t in P_loso[held]}
        a = axis_auc(P_loso[held], idx_l, fab)
        c = axis_auc(P_loso[held], idx_l, OTHER)
        if a is None or c is None:
            break
        f_l.append(a); o_l.append(c)
    if len(f_l) < len(P_loso):
        continue
    fd = np.mean(f_l) - f_s
    od = np.mean(o_l) - o_s
    fab_drops.append(fd); oth_drops.append(od); diffs.append(fd - od)
    if len(diffs) % 50 == 0:
        bar.set_postfix(diff=f'{np.mean(diffs):+.3f}')
bar.close()

d = np.array(diffs)
lo, hi = np.percentile(d, [2.5, 97.5])
print()
print('=== fabrication degradation minus omission/distortion degradation (TF-IDF) ===')
print(f'  fabrication drop      {np.mean(fab_drops):+.3f}')
print(f'  omission/distortion   {np.mean(oth_drops):+.3f}')
print(f'  difference            {d.mean():+.3f}')
print(f'  bootstrap 95%         [{lo:+.3f}, {hi:+.3f}]   ({len(d)} resamples)')
print(f'  excludes zero         {hi < 0}')

pd.DataFrame([{'axis_fabrication': '+'.join(fab),
               'fab_drop': round(float(np.mean(fab_drops)), 4),
               'other_drop': round(float(np.mean(oth_drops)), 4),
               'difference': round(float(d.mean()), 4),
               'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
               'n_resamples': len(d), 'excludes_zero': bool(hi < 0)}]).to_csv(args.out, index=False)
print('\nwrote ->', args.out)
