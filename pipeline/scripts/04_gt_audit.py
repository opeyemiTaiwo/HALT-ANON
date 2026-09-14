#!/usr/bin/env python3
"""
Step 4 - ground-truth audit sample.

UCA annotations are inherited from UCF-Crime's category labels, and at least
one (Abuse002_x264) carries an annotation describing a road-traffic incident.
This script does not decide correctness; it draws a stratified sample for a
human to check, and flags candidates using a crude keyword heuristic whose
false-positive rate is high by design (it is a triage aid, not a measurement).

Writes: gt_audit_sample.csv  (fill in the `verdict` column by hand)
"""
import ast
import json
import numpy as np
import pandas as pd

from config import RAW_DIR, OUT_DIR, SEED, require, banner

banner("STEP 4  ground-truth audit sample")
files = [f for f in ["UCFCrime_Train.json", "UCFCrime_Val.json",
                     "UCFCrime_Test.json"] if (RAW_DIR / f).exists()]
if not files:
    raise SystemExit("No UCFCrime_*.json found in RAW_DIR; skipping audit.")

recs = {}
for f in files:
    d = json.load(open(RAW_DIR / f))
    for vid, v in d.items():
        if isinstance(v, str):
            v = ast.literal_eval(v)
        recs[vid] = {"video": vid,
                     "crime_type": v.get("crime_type", vid.rstrip("0123456789_x264")),
                     "text": " ".join(v.get("sentences", []))}
gtdf = pd.DataFrame(recs.values())
print(f"loaded {len(gtdf)} annotations from {files}")

# Restrict to the eleven crime categories the study actually uses. UCA also
# contains Arrest, Arson and Normal_Videos, which are absent from the 807-video
# corpus and would otherwise be flagged at 100% by the keyword heuristic below
# simply because they have no keyword list.
STUDY = {"Abuse", "Assault", "Burglary", "Explosion", "Fighting",
         "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",
         "Vandalism"}
gtdf = gtdf[gtdf.crime_type.isin(STUDY)].copy()
print(f"{len(gtdf)} in the eleven study classes")

KW = {
 "Abuse": ["abuse","beat","hit","slap","kick","punch","push","shov","strangl","drag","throw"],
 "Assault": ["assault","attack","fight","punch","beat","hit","kick","knock","shov"],
 "Burglary": ["burglar","break","broke","pry","climb","window","intrud","steal","stole","door"],
 "Explosion": ["explo","blast","fire","smoke","flame","burst","bomb"],
 "Fighting": ["fight","punch","kick","brawl","beat","hit","wrestl","knock"],
 "RoadAccidents": ["car","vehicle","road","crash","collid","collision","motorcycle","truck","traffic","intersection"],
 "Robbery": ["rob","gun","knife","threat","snatch","demand","cash","money","register"],
 "Shooting": ["shoot","shot","gun","pistol","fire","weapon"],
 "Shoplifting": ["shop","store","shelf","pocket","conceal","steal","stole","merchandise","cashier"],
 "Stealing": ["steal","stole","theft","took","grab","snatch","bag","wallet","pick"],
 "Vandalism": ["vandal","smash","break","broke","destroy","damage","graffiti","kick","shatter"],
}
gtdf["flagged"] = [
    0 if any(k in t.lower() for k in KW.get(c, [])) else 1
    for c, t in zip(gtdf.crime_type, gtdf.text)]
print("\nheuristic flag rate by crime type (HIGH false-positive rate, triage only):")
print((gtdf.groupby("crime_type").flagged.agg(["sum", "count"])
       .assign(pct=lambda d: (100 * d["sum"] / d["count"]).round(0))).to_markdown())

rng = np.random.default_rng(SEED)
flagged = gtdf[gtdf.flagged == 1]
clean = gtdf[gtdf.flagged == 0]
samp = pd.concat([
    flagged.sample(min(25, len(flagged)), random_state=SEED),
    clean.sample(min(25, len(clean)), random_state=SEED)]).sample(frac=1, random_state=SEED)
samp = samp[["video", "crime_type", "flagged", "text"]].copy()
samp["text"] = samp.text.str.slice(0, 600)
samp["verdict"] = ""          # annotator fills: match / mismatch / unclear
samp["notes"] = ""
samp.to_csv(OUT_DIR / "gt_audit_sample.csv", index=False)
print(f"\nwrote {len(samp)} rows -> {OUT_DIR / 'gt_audit_sample.csv'}")
print("Fill the `verdict` column (match / mismatch / unclear). The sample is "
      "balanced 25 flagged / 25 unflagged so you can estimate both error "
      "directions, not just the flagged ones.")
