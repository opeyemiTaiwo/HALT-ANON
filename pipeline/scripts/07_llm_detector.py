#!/usr/bin/env python3
"""
Step 7 - LLM-as-detector baseline. RUN THIS WHERE YOU HAVE API ACCESS.

Prompts a fourth model, NOT one of the three source LLMs, zero-shot with the
report and its reference annotation, and asks for the six binary labels. Using
a non-source model matters: prompting one of the three would reintroduce the
self-preference bias the panel design was built to exclude, and would make the
result uninterpretable.

Zero-shot, so there is no training split. The same test sets as the other
detectors are used, and because this is inference-only the "held-out source"
condition is simply the reports from that source LLM.

    pip install openai            # or anthropic / google-generativeai
    export DETECTOR_API_KEY=...
    python3 07_llm_detector.py --model <a-model-not-in-the-corpus> --n-per-fold 1500

Writes: llm_detector_results.csv (schema matches baseline_results.csv)
        llm_detector_raw.csv     (per-report verdicts, for auditing)

COST WARNING. The full corpus is 19,361 reports. Scoring every test set would
be roughly 25,000 calls. --n-per-fold subsamples each test set, stratified by
crime type, which is enough for AUC at the precision this paper reports.
Start with --n-per-fold 300 to sanity-check the prompt and the parsing before
spending real money.
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score, f1_score

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
TARGETS = H + ['any_hallucination']

RUBRIC = """You are auditing a forensic video report against an expert reference
annotation of the same video. Decide, for each category, whether the report
contains that error.

H1 scene fabrication: describes a setting or event with no source in the footage
H2 crime misclassification: reports the act as a different category of crime
H3 crime missed: fails to report the criminal act under investigation
H4 severity minimization: understates the gravity of the act
H5 entity fabrication: introduces objects or details not present
H6 phantom actors: introduces people who do not appear

REFERENCE ANNOTATION:
{ref}

REPORT:
{rep}

Answer with JSON only, no prose:
{{"H1":0or1,"H2":0or1,"H3":0or1,"H4":0or1,"H5":0or1,"H6":0or1}}"""

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True,
                help='a model NOT among the three source LLMs')
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/llm_detector_results.csv')
ap.add_argument('--raw', default=os.environ.get('HALLUBENCH_OUT', '.') + '/llm_detector_raw.csv')
ap.add_argument('--n-per-fold', type=int, default=1500)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--max-ref', type=int, default=1500)
ap.add_argument('--max-rep', type=int, default=3000)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()


def call(prompt):
    """Replace the body with your provider's SDK. Must return the raw string."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['DETECTOR_API_KEY'])
    r = client.chat.completions.create(
        model=args.model, temperature=0,
        messages=[{'role': 'user', 'content': prompt}])
    return r.choices[0].message.content


def parse(txt):
    """Pull the six labels out of the reply. Returns None if unparseable."""
    m = re.search(r'\{[^{}]*\}', txt or '', re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not all(h in d for h in H):
        return None
    return {h: int(bool(int(d[h]))) for h in H}


def score(frame):
    """Score one test set. Retries twice on failure, then records a miss."""
    def one(i, row):
        p = RUBRIC.format(ref=str(row.ground_truth)[:args.max_ref],
                          rep=str(row.model_output)[:args.max_rep])
        for attempt in range(3):
            try:
                v = parse(call(p))
                if v is not None:
                    return i, v
            except Exception:
                time.sleep(2 ** attempt)
        return i, None

    res = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, i, r) for i, r in frame.iterrows()]
        for k, f in enumerate(as_completed(futs)):
            i, v = f.result(); res[i] = v
            if k % 100 == 0:
                print(f'   {k}/{len(frame)}', flush=True)
    return res


df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')
df['ground_truth'] = df.ground_truth.fillna('')
rng = np.random.default_rng(args.seed)


def subsample(frame):
    if len(frame) <= args.n_per_fold:
        return frame
    per = max(1, args.n_per_fold // frame.crime_type.nunique())
    return (frame.groupby('crime_type', group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed)))


conditions = {'random': df[df.split_random == 'test'],
              'heldout_technique': df[df.split_heldout_technique == 'test']}
for held in ['Claude', 'GPT', 'Gemini']:
    conditions[f'heldout_model_{held}'] = df[df.model == held]

rows, raw = [], []
for tag, frame in conditions.items():
    sub = subsample(frame).copy()
    print(f'\n== {tag}: scoring {len(sub):,} reports', flush=True)
    verdicts = score(sub)
    ok = [i for i, v in verdicts.items() if v is not None]
    print(f'   parsed {len(ok)}/{len(sub)}')
    sub = sub.loc[ok]
    P = pd.DataFrame([verdicts[i] for i in ok], index=ok)
    P['any_hallucination'] = P[H].max(axis=1)
    for h in TARGETS:
        raw.append(pd.DataFrame({'split': tag, 'target': h,
                                 'video': sub.video.values, 'model': sub.model.values,
                                 'technique': sub.technique.values,
                                 'pred': P[h].values, 'gold': sub[h].values}))
    for t in TARGETS:
        y, p = sub[t].to_numpy(), P[t].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({'features': 'llm_detector', 'split': tag, 'target': t,
                     'auc': roc_auc_score(y, p), 'f1': f1_score(y, p),
                     'pos_rate_test': float(y.mean()), 'n': len(y)})

res = pd.DataFrame(rows); res.to_csv(args.out, index=False)
pd.concat(raw).to_csv(args.raw, index=False)
print('\n' + res.pivot_table(index='split', columns='target',
                             values='auc')[TARGETS].round(3).to_markdown())
print('\nNOTE: this detector emits hard 0/1 labels, so its AUC is computed on '
      'binary predictions and is not directly comparable to the probabilistic '
      'baselines. Report F1 alongside it, and say so in the caption.')
print('wrote ->', args.out)
