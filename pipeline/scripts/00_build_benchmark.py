#!/usr/bin/env python3
"""
Step 0 - rebuild canonical labels and construct video-disjoint splits.

Labels come from panel_raw_judge_labels_full.csv under the paper's rule:
two-judge majority, ties broken toward no hallucination. Reproduces the
published corpus statistics (ANY 91.1%, 2.58 hallucinations per report).

Writes: benchmark_labels.csv.gz, label_reliability_tiers.csv,
        benchmark_manifest.json, INTEGRITY_REPORT.md
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import (H, KEY, JOIN, AXIS_OF, AXES, SEED, RAW_DIR, OUT_DIR,
                    JUDGE_FAIL, MULTI_TURN, require, banner)

PAPER = {"any": 91.1, "mean": 2.58}
log_lines = []


def log(m=""):
    print(m)
    log_lines.append(m)


require("panel_raw_judge_labels_full.csv", "full_labeled_dataset_full.csv",
        "Rater-A-scores-807.xlsx", "Rater-B-scores-807.xlsx",
        "Panel-A-scores-807.xlsx")

banner("STEP 0  build benchmark")

# ------------------------------------------------------------ integrity
log("## 1. Source integrity\n")
raw = pd.read_csv(RAW_DIR / "panel_raw_judge_labels_full.csv")
log(f"- raw judge rows: {len(raw):,}")
log(f"- reports: {raw.groupby(KEY).ngroups:,}")
log(f"- judges per report: {raw.groupby(KEY).size().value_counts().to_dict()}")
log(f"- self-judging rows (judge == generator): {(raw.judge == raw.model).sum()}")

sent = (raw[H] == JUDGE_FAIL)
n_cells, n_rows = int(sent.sum().sum()), int(sent.any(axis=1).sum())
log(f"- judge-failure sentinels ({JUDGE_FAIL}): {n_cells} cells / {n_rows} rows "
    f"-> treated as missing, excluded from the vote")
raw[H] = raw[H].mask(sent)

# ---------------------------------------------------------- aggregation
log("\n## 2. Label aggregation (two-judge majority, ties -> negative)\n")
grp = raw.groupby(KEY)[H]
pos, n = grp.sum(min_count=1), grp.count()
lab = (pos >= 2).astype(int)
short = n < 2                                   # a judge failed on this type
lab[short] = (pos[short] >= n[short]).astype(int)
lab = lab.reset_index()

lab["hallucination_count"] = lab[H].sum(axis=1)
lab["any_hallucination"] = (lab.hallucination_count > 0).astype(int)
for ax, cols in AXES.items():
    lab[f"axis_{ax}"] = lab[cols].max(axis=1)

got = (100 * lab.any_hallucination.mean(), lab.hallucination_count.mean())
log(f"- ANY hallucination: {got[0]:.2f}%   (paper {PAPER['any']}%)")
log(f"- mean per report:   {got[1]:.3f}   (paper {PAPER['mean']})")
if abs(got[0] - PAPER["any"]) > 0.5 or abs(got[1] - PAPER["mean"]) > 0.05:
    log("  !! WARNING: does not match the published statistics. Check the "
        "aggregation rule and the input file.")
log("\nPer-model per-type rate (%):\n")
log((100 * lab.groupby("model")[H].mean()).round(1).to_markdown())

# ----------------------------------------------------------- join text
txt = pd.read_csv(RAW_DIR / "full_labeled_dataset_full.csv",
                  usecols=KEY + ["ground_truth", "model_output",
                                 "model_output_full_len"])
bench = lab.merge(txt, on=KEY, how="left", validate="one_to_one")
assert bench.model_output.notna().all(), "text join incomplete"

# --------------------------------------------------- reliability tiers
log("\n## 3. Label reliability against human raters (n=130)\n")
ra = pd.read_excel(RAW_DIR / "Rater-A-scores-807.xlsx")
rb = pd.read_excel(RAW_DIR / "Rater-B-scores-807.xlsx")
pa = pd.read_excel(RAW_DIR / "Panel-A-scores-807.xlsx")

rows = []
tiers = {}
for h in H:
    a, b = ra[f"human_{h}"].to_numpy(), rb[f"human_{h}"].to_numpy()
    pan = pa[f"panel_{h}"].to_numpy()
    agreed = a == b                     # paper's consensus definition
    k_hh = cohen_kappa_score(a, b)
    k_pc = cohen_kappa_score(a[agreed], pan[agreed])
    tier = "high" if k_pc >= 0.65 else ("medium" if k_pc >= 0.40 else "low")
    tiers[h] = tier
    rows.append({"type": h, "axis": AXIS_OF[h], "n_consensus": int(agreed.sum()),
                 "kappa_human_human": round(k_hh, 3),
                 "kappa_panel_consensus": round(k_pc, 3), "tier": tier})
tier_df = pd.DataFrame(rows)
log(tier_df.to_markdown(index=False))
log(f"\n- macro kappa: {tier_df.kappa_panel_consensus.mean():.3f}  (paper 0.556)")
log("- NOTE: these kappas are computed only on rows where both raters agree; "
    "n varies by type and is reported above.")
log(f"- low-reliability types (exclude from headline macro): "
    f"{[h for h in H if tiers[h] == 'low']}")

gold = pa[JOIN].copy()
gold["gold"] = 1
bench = bench.merge(gold, on=JOIN, how="left")
bench["gold"] = bench.gold.fillna(0).astype(int)

# --------------------------------------------------------------- splits
log("\n## 4. Splits (video-disjoint)\n")
rng = np.random.default_rng(SEED)
vmeta = bench[["video", "crime_type"]].drop_duplicates().sort_values("video")
assign = {}
for ct, g in vmeta.groupby("crime_type"):
    v = g.video.to_numpy().copy()
    rng.shuffle(v)
    n_tr, n_va = int(0.70 * len(v)), int(0.15 * len(v))
    for i, name in ((slice(0, n_tr), "train"),
                    (slice(n_tr, n_tr + n_va), "val"),
                    (slice(n_tr + n_va, None), "test")):
        assign.update({x: name for x in v[i]})
bench["split_random"] = bench.video.map(assign)
bench["split_heldout_model"] = np.where(bench.model == "Gemini", "test", "train")
bench["split_heldout_technique"] = np.where(
    bench.technique.isin(MULTI_TURN), "test", "train")

for c in ["split_random", "split_heldout_model", "split_heldout_technique"]:
    leak = int(bench.groupby("video")[c].nunique().gt(1).sum())
    log(f"- {c}: {bench[c].value_counts().to_dict()}   "
        f"videos spanning >1 fold: {leak}")
log(f"\n- gold (human-validated) reports: {int(bench.gold.sum())}")

# ---------------------------------------------------------------- write
cols = (KEY + H + ["hallucination_count", "any_hallucination"]
        + [f"axis_{a}" for a in AXES]
        + ["gold", "split_random", "split_heldout_model",
           "split_heldout_technique", "model_output_full_len",
           "ground_truth", "model_output"])
bench[cols].to_csv(OUT_DIR / "benchmark_labels.csv.gz", index=False,
                   compression="gzip")
tier_df.to_csv(OUT_DIR / "label_reliability_tiers.csv", index=False)
json.dump({"seed": SEED,
           "aggregation": "two-judge majority, ties -> negative",
           "n_reports": int(len(bench)), "n_videos": int(bench.video.nunique()),
           "any_rate": round(float(bench.any_hallucination.mean()), 4),
           "mean_per_report": round(float(bench.hallucination_count.mean()), 4),
           "judge_failure_cells": n_cells, "reliability_tiers": tiers},
          open(OUT_DIR / "benchmark_manifest.json", "w"), indent=2)
(OUT_DIR / "INTEGRITY_REPORT.md").write_text(
    "# Benchmark rebuild integrity report\n\n" + "\n".join(log_lines) + "\n")
print(f"\nwrote -> {OUT_DIR}")
