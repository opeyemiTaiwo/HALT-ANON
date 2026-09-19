#!/usr/bin/env python3
"""
Step 10 - prompted-LLM detector, run LOCALLY on the GPU. No API key, no cost.

Addresses the objection that the paper tests linear and encoder detectors but
not the form a practitioner would actually deploy: a frontier model prompted
zero-shot with the report and its reference.

The model MUST NOT be one of the three that wrote the corpus. Prompting Claude,
GPT or Gemini here would reintroduce the self-preference bias the panel design
exists to exclude, and the number would be uninterpretable. Qwen and Llama are
safe choices and run under `transformers` with no API.

Zero-shot, so there is no training split: the held-out-source condition is
simply the reports from that source LLM.

    pip install "transformers>=4.48" accelerate bitsandbytes
    python3 10_llm_detector_local.py --model Qwen/Qwen2.5-3B-Instruct --n-per-fold 300

Writes: llm_detector_results.csv, llm_detector_raw.csv

RUNTIME. Qwen2.5-3B fits a 16 GB T4 and runs about 1 s per report there; at
--n-per-fold 300 over five conditions that is roughly 25 minutes. The 7B model
needs about 15 GB for weights alone and will offload layers to CPU on a T4,
which is far slower. Start at --n-per-fold 100 to check the prompt and the JSON
parsing before scaling.
"""
import argparse, json, os, re
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
TARGETS = H + ['any_hallucination']

RUBRIC = """You are auditing a forensic video report against an expert reference annotation of the same video. For each category decide whether the report contains that error.

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

Use 1 if the error is present and 0 if it is not. Judge each category
independently; most reports contain some errors and not others.

Answer with JSON only, no prose, in exactly this form:
{{"H1": <0 or 1>, "H2": <0 or 1>, "H3": <0 or 1>, "H4": <0 or 1>, "H5": <0 or 1>, "H6": <0 or 1>}}"""

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct',
                help='must NOT be Claude, GPT or Gemini')
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/llm_detector_results.csv')
ap.add_argument('--raw', default=os.environ.get('HALLUBENCH_OUT', '.') + '/llm_detector_raw.csv')
ap.add_argument('--n-per-fold', type=int, default=300)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--max-ref', type=int, default=1500)
ap.add_argument('--max-rep', type=int, default=3000)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

for banned in ('claude', 'gpt', 'gemini'):
    if banned in args.model.lower():
        raise SystemExit(f'ABORT: {args.model} looks like a source LLM of this corpus. '
                         'Using one as the detector reintroduces self-preference bias.')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    _gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)}  ({_gb:.0f} GB)  '
          f'bf16={torch.cuda.is_bf16_supported()}')
    if _gb < 20:
        print('  WARNING: a 7B model in bf16 needs about 15 GB for weights alone. '
              'On a smaller card use a 3B model or load in 8-bit.')
    if not torch.cuda.is_bf16_supported():
        print('  WARNING: this GPU has no bf16. Falling back to fp16.')
print('device:', dev, '| model:', args.model)
tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=(torch.bfloat16 if dev == 'cuda' and torch.cuda.is_bf16_supported()
                 else torch.float16 if dev == 'cuda' else torch.float32),
    device_map='auto')
model.eval()


def parse(txt):
    """Pull the six labels out of the reply. None if unparseable."""
    m = re.search(r'\{[^{}]*\}', txt or '', re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not all(h in d for h in H):
        return None
    try:
        return {h: int(bool(int(d[h]))) for h in H}
    except Exception:
        return None


def score(frame):
    """Score one test set, batched."""
    prompts = [tok.apply_chat_template(
        [{'role': 'user', 'content': RUBRIC.format(
            ref=str(r.ground_truth)[:args.max_ref],
            rep=str(r.model_output)[:args.max_rep])}],
        tokenize=False, add_generation_prompt=True) for _, r in frame.iterrows()]
    out = {}
    bar = tqdm(range(0, len(prompts), args.batch), desc='  generating',
               unit='batch', leave=False)
    for s in bar:
        chunk = prompts[s:s + args.batch]
        enc = tok(chunk, return_tensors='pt', padding=True,
                  truncation=True, max_length=4096).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for k, g in enumerate(gen):
            reply = tok.decode(g[enc['input_ids'].shape[1]:], skip_special_tokens=True)
            out[frame.index[s + k]] = parse(reply)
        ok = sum(v is not None for v in out.values())
        bar.set_postfix(parsed=f'{ok}/{len(out)}')
    bar.close()
    return out


df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')
df['ground_truth'] = df.ground_truth.fillna('')


def subsample(frame):
    if len(frame) <= args.n_per_fold:
        return frame
    per = max(1, args.n_per_fold // frame.crime_type.nunique())
    return (frame.groupby('crime_type', group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed)))


conditions = {'random': df[df.split_random == 'test'],
              'heldout_technique': df[df.split_heldout_technique == 'test']}
for held in sorted(df.model.unique()):
    conditions[f'heldout_model_{held}'] = df[df.model == held]

# Each condition is saved as it finishes; rerunning skips what is already there.
done = set()
if os.path.exists(args.out):
    done = set(pd.read_csv(args.out).split.unique())
    print('resuming: already done ->', sorted(done))

rows, raw = [], []
for tag, frame in conditions.items():
    if tag in done:
        print(f'skip {tag} (already saved)')
        continue
    sub = subsample(frame).copy()
    print(f'\n== {tag}: scoring {len(sub):,} reports', flush=True)
    verdicts = score(sub)
    ok = [i for i, v in verdicts.items() if v is not None]
    print(f'   parsed {len(ok)}/{len(sub)}')
    if ok:
        import collections
        rates = {h: np.mean([verdicts[i][h] for i in ok]) for h in H}
        print('   positive rate per type:',
              '  '.join(f'{h} {r:.2f}' for h, r in rates.items()))
        flat = [h for h, r in rates.items() if r in (0.0, 1.0)]
        if flat:
            print(f'   WARNING: {flat} are constant. The model is copying the template '
                  f'rather than judging; the AUCs for those types are meaningless.')
    if not ok:
        print('   NOTHING PARSED - check the prompt or the model before scaling up')
        continue
    sub = sub.loc[ok]
    P = pd.DataFrame([verdicts[i] for i in ok], index=ok)
    P['any_hallucination'] = P[H].max(axis=1)
    raw.append(pd.DataFrame({'split': tag, 'video': sub.video.values,
                             'model': sub.model.values, 'technique': sub.technique.values,
                             **{f'pred_{h}': P[h].values for h in TARGETS},
                             **{f'gold_{h}': sub[h].values for h in TARGETS}}))
    for t in TARGETS:
        y, p = sub[t].to_numpy(), P[t].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({'features': 'llm_detector', 'split': tag, 'target': t,
                     'auc': roc_auc_score(y, p), 'f1': f1_score(y, p),
                     'pos_rate_test': float(y.mean()), 'n': len(y)})
    part = pd.DataFrame([r for r in rows if r['split'] == tag])
    part.to_csv(args.out, mode='a', header=not os.path.exists(args.out), index=False)
    raw[-1].to_csv(args.raw, mode='a', header=not os.path.exists(args.raw), index=False)
    print(f'   saved {tag}', flush=True)

res = pd.read_csv(args.out)
print('\n' + res.pivot_table(index='split', columns='target', values='auc')[TARGETS].round(3).to_markdown())
print('\nNOTE: this detector emits hard 0/1 labels, so its AUC is computed on binary')
print('predictions and is NOT directly comparable to the probabilistic baselines.')
print('Report F1 alongside it and say so in the caption.')
print('wrote ->', args.out)
