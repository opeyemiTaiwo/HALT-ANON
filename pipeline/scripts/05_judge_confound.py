#!/usr/bin/env python3
"""
Step 5 - judge-pair confound test.

Because no model judges its own output, each generator is scored by a fixed
pair of judges, and those pairs differ sharply in mutual agreement. Under a
conjunctive aggregation rule a disagreeing pair yields fewer positives for
mechanical reasons, so aggregated per-generator rates confound the generator
with its assigned judge pair.

Each judge scores exactly two generators, which permits a within-judge
comparison that holds the judge fixed:

    judge Claude  sees  GPT, Gemini
    judge GPT     sees  Claude, Gemini
    judge Gemini  sees  Claude, GPT

If the generator ordering reported in the aggregated labels is a property of
the generators, every judge should reproduce it on the pair it sees. If it is
a property of the judge pairs, the within-judge orderings will disagree.

Pure analysis of the released per-judge verdicts. No generation, no
re-judging.

Writes: judge_confound.csv, JUDGE_CONFOUND.md
"""
import itertools

import numpy as np
import pandas as pd

from config import H, HNAME, RAW_DIR, OUT_DIR, KEY, JUDGE_FAIL, require, banner

require("panel_raw_judge_labels_full.csv")
banner("STEP 5  judge-pair confound test")

raw = pd.read_csv(RAW_DIR / "panel_raw_judge_labels_full.csv")
raw[H] = raw[H].mask(raw[H] == JUDGE_FAIL)

out = ["# Judge-pair confound test\n",
       "Each generator is scored by a fixed pair of judges. This tests "
       "whether the generator ordering survives when the judge is held "
       "fixed.\n"]

# ---------------------------------------------------- 1. the confound
out.append("\n## 1. Which pair scores which generator\n")
pairs = (raw.groupby("model").judge.unique()
         .apply(lambda a: " + ".join(sorted(a))).rename("judged_by"))
n_rep = raw.groupby("model").size().div(2).astype(int).rename("n_reports")
out.append(pd.concat([pairs, n_rep], axis=1).to_markdown())

# ------------------------------------- 2. aggregated (confounded) view
agg = raw.groupby(KEY)[H]
pos, n = agg.sum(min_count=1), agg.count()
lab = (pos >= 2).astype(int)
short = n < 2
lab[short] = (pos[short] >= n[short]).astype(int)
lab = lab.reset_index()
conf = (100 * lab.groupby("model")[H].mean()).round(1)
out.append("\n\n## 2. Aggregated per-generator rates (confounded)\n")
out.append(conf.to_markdown())

# ------------------------------------------ 3. within-judge comparison
out.append("\n\n## 3. Within-judge rates (%): each judge's own verdicts\n")
wj = (100 * raw.groupby(["judge", "model"])[H].mean()).round(1)
out.append(wj.to_markdown())

# ------------------------------------------------ 4. ordering agreement
out.append("\n\n## 4. Does each judge reproduce the aggregated ordering?\n")
out.append("For every judge and every type, the ordering of the two "
           "generators that judge sees, compared against the ordering the "
           "aggregated labels give for the same two generators.\n")
rows = []
for judge in sorted(raw.judge.unique()):
    seen = sorted(raw.loc[raw.judge == judge, "model"].unique())
    for a, b in itertools.combinations(seen, 2):
        for h in H:
            wa = raw.loc[(raw.judge == judge) & (raw.model == a), h].mean()
            wb = raw.loc[(raw.judge == judge) & (raw.model == b), h].mean()
            ca = lab.loc[lab.model == a, h].mean()
            cb = lab.loc[lab.model == b, h].mean()
            rows.append({
                "judge": judge, "pair": f"{a} vs {b}", "type": h,
                "name": HNAME[h],
                "within_judge": f"{a}" if wa > wb else f"{b}",
                "within_gap_pp": round(100 * abs(wa - wb), 1),
                "aggregated": f"{a}" if ca > cb else f"{b}",
                "agrees": int((wa > wb) == (ca > cb)),
            })
cmp = pd.DataFrame(rows)
cmp.to_csv(OUT_DIR / "judge_confound.csv", index=False)
out.append(cmp[["judge", "pair", "type", "name", "within_judge",
                "within_gap_pp", "aggregated", "agrees"]].to_markdown(index=False))

rate = 100 * cmp.agrees.mean()
out.append(f"\n\n**Orderings preserved: {cmp.agrees.sum()} of {len(cmp)} "
           f"({rate:.0f}%).**\n")
by_h = cmp.groupby("type").agrees.agg(["sum", "count"])
out.append("\nBy type:\n")
out.append(by_h.to_markdown())

# ------------------------------------- 5. does the axis signature hold
out.append("\n\n## 5. Dominant axis, within judge\n")
out.append("Axis rate = mean per-type prevalence within the axis, the "
           "aggregation used for the cross-generation check in the source "
           "study.\n")
AX = {"fabrication": ["H1", "H5", "H6"], "omission": ["H3"],
      "distortion": ["H2", "H4"]}
rows2 = []
for judge in sorted(raw.judge.unique()):
    for gen in sorted(raw.loc[raw.judge == judge, "model"].unique()):
        s = raw[(raw.judge == judge) & (raw.model == gen)]
        r = {"judge": judge, "generator": gen}
        for ax, cs in AX.items():
            r[ax] = round(100 * s[cs].mean().mean(), 1)
        r["dominant"] = max(AX, key=lambda a: r[a])
        rows2.append(r)
ax_df = pd.DataFrame(rows2)
out.append(ax_df.to_markdown(index=False))

out.append("\n\n### Aggregated dominant axis, for comparison\n")
rows3 = []
for gen in sorted(lab.model.unique()):
    s = lab[lab.model == gen]
    r = {"generator": gen}
    for ax, cs in AX.items():
        r[ax] = round(100 * s[cs].mean().mean(), 1)
    r["dominant"] = max(AX, key=lambda a: r[a])
    rows3.append(r)
out.append(pd.DataFrame(rows3).to_markdown(index=False))

(OUT_DIR / "JUDGE_CONFOUND.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
