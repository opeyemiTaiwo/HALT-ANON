#!/usr/bin/env python3
"""
Step 6 - fine-tuned encoder detector. RUN THIS ON A GPU (Colab A100 is enough).

Addresses the strongest objection to the paper: that the transfer collapse is
an artifact of linear detectors rather than a property of the task. Fine-tunes
one encoder per split with six per-type heads plus an ANY head, on
[report] [SEP] [reference], and evaluates on the standard, leave-one-source-out
(all three folds), and held-out-strategy splits.

Requires network access to download model weights, so it cannot run in the
offline analysis container.

    pip install "transformers>=4.48" torch scikit-learn accelerate
    python3 06_encoder_detector.py --model answerdotai/ModernBERT-base

CONTEXT LENGTH MATTERS HERE. Reports average ~1,500 tokens. The judge that
produced the labels saw 3,000 characters of report (~750 tokens) plus 1,500 of
reference (~375), so --max_len 1280 gives the detector exactly the judge's
view. At 512 the detector sees less than the judge did and the comparison is
unfair to it. ModernBERT handles 8,192 tokens, so 1,280 costs nothing.

Writes: encoder_results.csv  (same schema as baseline_results.csv, so the
        existing report and figure code consumes it unchanged)

Runtime guide, A100, ModernBERT-base, max_len 1280, bs 8, 2 epochs:
    standard split          ~60 min
    3 LOSO folds            ~150 min
    held-out strategy       ~60 min
Roughly 4.5 h in total. For the validation pass use
    --model roberta-base --max_len 512 --bs 16 --epochs 1
which takes about 20 minutes and only checks that the loop runs.
"""
import argparse, os
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

H = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
TARGETS = H + ['any_hallucination']

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='answerdotai/ModernBERT-base')
ap.add_argument('--labels', default=os.environ.get('HALLUBENCH_OUT', '.') + '/benchmark_labels.csv.gz')
ap.add_argument('--out', default=os.environ.get('HALLUBENCH_OUT', '.') + '/encoder_results.csv')
ap.add_argument('--max_len', type=int, default=1280,
                help='1280 matches what the judge saw; raise it to test whether '
                     'the detector benefits from more than the judge had')
ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--bs', type=int, default=8,
                help='lower than usual because of the long context')
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    print('WARNING: no GPU visible. This will take many hours.')
else:
    # TF32 is free on Ampere and newer and harmless on older cards.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    _gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)}  ({_gb:.0f} GB)  '
          f'bf16={torch.cuda.is_bf16_supported()}')
    _need = {512: 10, 1280: 14, 2048: 16}.get(args.max_len, 16)
    if _gb < _need:
        print(f'  WARNING: --max_len {args.max_len} at --bs {args.bs} wants about '
              f'{_need} GB. Lower --bs or expect an out-of-memory error.')

df = pd.read_csv(args.labels)
df['model_output'] = df.model_output.fillna('')
df['ground_truth'] = df.ground_truth.fillna('')
tok = AutoTokenizer.from_pretrained(args.model)
_cap = getattr(tok, 'model_max_length', 512)
if _cap and _cap < args.max_len and _cap < 100000:
    print(f'WARNING: {args.model} caps at {_cap} tokens but --max_len is '
          f'{args.max_len}. Reports will be cut below what the judge saw. '
          f'Use a long-context model (ModernBERT, Longformer) or lower '
          f'--max_len and say so in the paper.')


class Reports(Dataset):
    """Report and its reference annotation as a sentence pair."""
    def __init__(self, frame):
        self.a = frame.model_output.tolist()
        self.b = frame.ground_truth.tolist()
        self.y = frame[TARGETS].to_numpy(dtype='float32')

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        # No padding here: the collator pads each batch to its own longest
        # sequence, saving roughly a quarter of the compute at --max_len 2048
        # and costing nothing at shorter windows.
        enc = tok(self.a[i], self.b[i], truncation=True, max_length=args.max_len)
        return ({k: torch.tensor(v) for k, v in enc.items()},
                torch.tensor(self.y[i]))


def collate(batch):
    padded = tok.pad([{k: v.tolist() for k, v in b[0].items()} for b in batch],
                     return_tensors='pt')
    return dict(padded), torch.stack([b[1] for b in batch])


class MultiHead(torch.nn.Module):
    """One shared encoder, seven independent binary heads."""
    def __init__(self, name, n=len(TARGETS)):
        super().__init__()
        # force fp32: some checkpoints (DeBERTa-v3) declare a fp16 dtype in
        # their config, which makes GradScaler refuse to unscale gradients
        self.enc = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        d = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.heads = torch.nn.Linear(d, n)

    def forward(self, **kw):
        h = self.enc(**kw).last_hidden_state[:, 0]     # [CLS]
        return self.heads(self.drop(h))


def run(train_mask, test_mask, tag):
    tr, te = df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    print(f'\n== {tag}: train {len(tr):,}  test {len(te):,}', flush=True)

    model = MultiHead(args.model).to(dev)
    nw = min(4, os.cpu_count() or 2)
    dl_tr = DataLoader(Reports(tr), batch_size=args.bs, shuffle=True, num_workers=nw,
                       pin_memory=(dev == 'cuda'), collate_fn=collate,
                       persistent_workers=nw > 0)
    dl_te = DataLoader(Reports(te), batch_size=args.bs * 2, num_workers=nw,
                       pin_memory=(dev == 'cuda'), collate_fn=collate,
                       persistent_workers=nw > 0)

    # class weights per head, matching the balanced logistic baselines
    pos = tr[TARGETS].mean().to_numpy()
    w = torch.tensor(((1 - pos) / np.clip(pos, 1e-6, None)).astype('float32')).to(dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = len(dl_tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    # bf16 where the GPU supports it (A100 and newer): same dynamic range as
    # fp32, so no loss scaling is needed and GradScaler is skipped entirely.
    use_bf16 = dev == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda' and not use_bf16))
    print(f'   precision: {"bf16" if use_bf16 else ("fp16+scaler" if dev=="cuda" else "fp32")}',
          flush=True)

    model.train()
    for ep in range(args.epochs):
        bar = tqdm(dl_tr, desc=f'{tag} ep{ep+1}/{args.epochs}', unit='batch', leave=False)
        run_loss = None
        for x, y in bar:
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
            y = y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(dev == 'cuda')):
                loss = lossf(model(**x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sch.step()
            l = loss.item()
            run_loss = l if run_loss is None else 0.98 * run_loss + 0.02 * l
            bar.set_postfix(loss=f'{run_loss:.4f}')
        bar.close()
        print(f'   {tag} epoch {ep+1}/{args.epochs} done, loss {run_loss:.4f}', flush=True)

    model.eval(); P = []
    with torch.no_grad():
        for x, _ in tqdm(dl_te, desc=f'{tag} eval', unit='batch', leave=False):
            x = {k: v.to(dev) for k, v in x.items()}
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(dev == 'cuda')):
                P.append(torch.sigmoid(model(**x)).float().cpu().numpy())
    P = np.vstack(P)

    rows = []
    for j, t in enumerate(TARGETS):
        y = te[t].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({'features': 'encoder', 'split': tag, 'target': t,
                     'auc': roc_auc_score(y, P[:, j]),
                     'f1': f1_score(y, (P[:, j] >= 0.5).astype(int)),
                     'pos_rate_test': float(y.mean())})
    del model; torch.cuda.empty_cache()
    return rows


# Each condition is saved as soon as it finishes, and any condition already in
# the output file is skipped. A crash or a Colab disconnect costs only the
# condition that was running: rerun the same command to continue.
done = set()
if os.path.exists(args.out):
    done = set(pd.read_csv(args.out).split.unique())
    print(f'resuming: already done -> {sorted(done)}')

conditions = [(df.split_random == 'train', df.split_random == 'test', 'random')]
for held in sorted(df.model.unique()):
    conditions.append((df.model != held, df.model == held, f'heldout_model_{held}'))
conditions.append((df.split_heldout_technique == 'train',
                   df.split_heldout_technique == 'test', 'heldout_technique'))

for train_mask, test_mask, tag in conditions:
    if tag in done:
        print(f'skip {tag} (already saved)')
        continue
    part = pd.DataFrame(run(train_mask, test_mask, tag))
    part.to_csv(args.out, mode='a', header=not os.path.exists(args.out), index=False)
    print(f'   saved {tag} -> {args.out}', flush=True)

res = pd.read_csv(args.out)
res.to_csv(args.out, index=False)
print('\n' + res.pivot_table(index='split', columns='target',
                             values='auc')[TARGETS].round(3).to_markdown())
print('\nwrote ->', args.out)
