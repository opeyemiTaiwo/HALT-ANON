#!/usr/bin/env python3
"""
Step 11 - entailment-based grounded features.

The paper's `grounded` detector is lexical: token overlap, Jaccard, TF-IDF
cosine. It recovers binary detection under shift but fails on omission, and the
paper says a semantic version is the most promising direction not tried. This
is that version.

For each report, every sentence is scored against the reference annotation by a
cross-encoder NLI model, giving per-sentence entailment / neutral /
contradiction probabilities. Those are aggregated into report-level features:

  FABRICATION side - report sentences the reference does not support
    mean/max contradiction of report sentences given the reference
    fraction of report sentences with entailment below a threshold
  OMISSION side - reference content the report does not carry
    the same, with the direction reversed (reference sentences as hypotheses)

The reversed direction is the point: lexical coverage failed on omission, and
entailment in that direction is what should capture "the report drops the
criminal act".

Features then feed the same logistic regression as every other baseline, so the
comparison is like-for-like.

    pip install "transformers>=4.48" accelerate
    python3 11_nli_grounding.py --nli cross-encoder/nli-deberta-v3-base

Writes: nli_features.parquet, nli_results.csv

RUNTIME. 19,361 reports at ~20 sentences each, both directions, is ~800k pairs.
On an A100 in bfloat16 at batch 128 that is roughly 2-3 hours. --max-sent caps
sentences per report; 15 keeps it near 2 hours with little loss.
"""
import argparse, os, re
import numpy as np, pandas as pd, torch, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
TARGETS = H + ['any_hallucination']

ap = argparse.ArgumentParser()
ap.add_argument('--nli', default='cross-encoder/nli-deberta-v3-base')
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--feats', default=os.environ.get('HALLUBENCH_OUT', '.') + '/nli_features.parquet')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/nli_results.csv')
ap.add_argument('--max-sent', type=int, default=15, help='sentences per document')
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--reuse-feats', action='store_true', help='skip scoring, load cached features')
ap.add_argument('--ckpt', type=int, default=500, help='checkpoint every N reports')
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    _gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)}  ({_gb:.0f} GB)')
df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')
df['ground_truth'] = df.ground_truth.fillna('')

SENT = re.compile(r'(?<=[.!?])\s+')
def sents(t, cap):
    s = [x.strip() for x in SENT.split(str(t)) if len(x.strip()) > 15]
    return s[:cap] if s else ['']

if args.reuse_feats and os.path.exists(args.feats):
    F = pd.read_parquet(args.feats)
    print('loaded cached features', F.shape)
else:
    print('device:', dev, '| NLI model:', args.nli)
    tok = AutoTokenizer.from_pretrained(args.nli)
    nli = AutoModelForSequenceClassification.from_pretrained(
        args.nli, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
    # label order differs between checkpoints; read it off the config
    id2 = {i: l.lower() for i, l in nli.config.id2label.items()}
    ENT = [i for i, l in id2.items() if 'entail' in l][0]
    CON = [i for i, l in id2.items() if 'contra' in l][0]
    print('  entailment index', ENT, '| contradiction index', CON)

    def score_pairs(premises, hypotheses):
        out = []
        for s in range(0, len(premises), args.batch):
            enc = tok(premises[s:s + args.batch], hypotheses[s:s + args.batch],
                      return_tensors='pt', padding=True, truncation=True,
                      max_length=256).to(dev)
            with torch.no_grad():
                p = torch.softmax(nli(**enc).logits.float(), dim=-1).cpu().numpy()
            out.append(p)
        return np.vstack(out) if out else np.zeros((0, 3))

    # Scoring is the expensive step, so it checkpoints every --ckpt rows.
    # Rerunning picks up where it stopped.
    ckpt = args.feats + '.partial.parquet'
    rows = []
    start = 0
    if os.path.exists(ckpt):
        prev = pd.read_parquet(ckpt)
        rows = prev.to_dict('records'); start = len(rows)
        print(f'resuming feature scoring at row {start:,}')

    bar = tqdm(total=len(df), initial=start, desc='scoring reports', unit='report')
    for n, (_, r) in enumerate(df.iterrows()):
        if n < start:
            continue
        rep_s = sents(r.model_output, args.max_sent)
        ref_s = sents(r.ground_truth, args.max_sent)
        ref_joined = ' '.join(ref_s)[:2000]
        rep_joined = ' '.join(rep_s)[:2000]
        # direction 1: does the reference support each report sentence? (fabrication)
        f = score_pairs([ref_joined] * len(rep_s), rep_s)
        # direction 2: does the report carry each reference sentence? (omission)
        o = score_pairs([rep_joined] * len(ref_s), ref_s)
        rows.append({
            'fab_contra_mean': float(f[:, CON].mean()), 'fab_contra_max': float(f[:, CON].max()),
            'fab_entail_mean': float(f[:, ENT].mean()),
            'fab_unsupported_frac': float((f[:, ENT] < 0.5).mean()),
            'omi_entail_mean': float(o[:, ENT].mean()), 'omi_entail_min': float(o[:, ENT].min()),
            'omi_contra_mean': float(o[:, CON].mean()),
            'omi_dropped_frac': float((o[:, ENT] < 0.5).mean()),
            'n_rep_sent': len(rep_s), 'n_ref_sent': len(ref_s)})
        bar.update(1)
        if (n + 1) % args.ckpt == 0:
            pd.DataFrame(rows).to_parquet(ckpt)
            bar.set_postfix(checkpoint=f'{n+1:,}')
    bar.close()
    F = pd.DataFrame(rows)
    F.to_parquet(args.feats)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print('wrote features ->', args.feats)

Xn = StandardScaler().fit_transform(F.to_numpy(dtype=float))

def run(train_mask, test_mask, tag):
    out = []
    for t in TARGETS:
        y = df[t].to_numpy()
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        c = LogisticRegression(max_iter=3000, class_weight='balanced',
                               random_state=42).fit(Xn[train_mask], y[train_mask])
        out.append({'features': 'nli_grounded', 'split': tag, 'target': t,
                    'auc': roc_auc_score(y[test_mask], c.predict_proba(Xn[test_mask])[:, 1])})
    return out

res = []
res += run((df.split_random == 'train').to_numpy(), (df.split_random == 'test').to_numpy(), 'random')
for held in sorted(df.model.unique()):
    res += run((df.model != held).to_numpy(), (df.model == held).to_numpy(), f'heldout_model_{held}')
res += run((df.split_heldout_technique == 'train').to_numpy(),
           (df.split_heldout_technique == 'test').to_numpy(), 'heldout_technique')

r = pd.DataFrame(res); r.to_csv(args.out, index=False)
print('\n' + r.pivot_table(index='split', columns='target', values='auc')[TARGETS].round(3).to_markdown())
print('\nThe comparison that matters is omission (H3) under leave-one-source-out:')
print('lexical grounding reaches 0.577 there. If entailment beats that, semantic')
print('grounding is the answer to the omission gap; if not, the gap is not lexical.')
print('wrote ->', args.out)
