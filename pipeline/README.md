# Benchmark pipeline

Rebuilds canonical labels from the raw judge file, constructs video-disjoint
splits, and runs the baseline, transfer, and label-diagnostic analyses.

Everything is deterministic (seed 42). Total runtime is roughly 10 to 20
minutes on a laptop; no GPU and no network access required.

---

## Setup

```bash
pip install -r requirements.txt

export HALLUBENCH_RAW=/absolute/path/to/your/data
export HALLUBENCH_OUT=/absolute/path/to/write/outputs   # optional, defaults to ./out

./run_all.sh
```

Or run the steps individually from `scripts/`:

```bash
cd scripts
python3 00_build_benchmark.py
python3 01_baselines.py
python3 02_logo_grounded.py
python3 03_artifact_test.py
python3 04_gt_audit.py
```

Step 0 must run first; the others read its output.

## Input files

Place these in `$HALLUBENCH_RAW`. Names are as released.

| File | Required | Used for |
|---|---|---|
| `panel_raw_judge_labels_full.csv` | yes | all labels |
| `full_labeled_dataset_full.csv` | yes | report text only |
| `Rater-A-scores-807.xlsx` | yes | human validation |
| `Rater-B-scores-807.xlsx` | yes | human validation |
| `Panel-A-scores-807.xlsx` | yes | panel labels on the gold subset |
| `embeddings_openai.npy` | no | `embed` baseline (skipped if absent) |
| `embedding_index.csv` | no | row alignment for the above |
| `UCFCrime_{Train,Val,Test}.json` | no | ground-truth audit sample |

If a required file is missing, the script names it and exits rather than
failing halfway through.

---

## Two things to know before you read any output

**1. `full_labeled_dataset_full.csv` does not match the published paper.**
It aggregates the two judges with OR, giving 98.46% ANY and 3.550
hallucinations per report. The paper's rule is two-judge majority with ties
broken toward no hallucination, which with two judges is AND, giving 91.1%
and 2.58. This pipeline rebuilds every label from
`panel_raw_judge_labels_full.csv` and uses `full_labeled_dataset_full.csv`
only as a source of report text. Do not use it for labels anywhere.

**2. The raw judge file contains `-1` sentinels.** Twenty-four cells across
four judge rows, all Gemini judging Claude. These are judge failures, not
verdicts. They are treated as missing and excluded from the vote. Any
downstream code you write should do the same.

---

## Expected output (verification targets)

If step 0 prints anything materially different from these, stop and check the
inputs before trusting anything downstream.

| Quantity | Expected | Source |
|---|---|---|
| Reports | 19,361 | corpus |
| Videos | 807 | corpus |
| Judge rows | 38,722 | 2 per report |
| Self-judging rows | 0 | protocol |
| ANY hallucination | 91.13% | paper: 91.1% |
| Mean per report | 2.580 | paper: 2.58 |
| Claude H1 / H5 / H3 | 89.6 / 88.6 / 38.5 | paper |
| GPT H3 / H1 / H6 | 48.4 / 30.9 / 6.3 | paper |
| Gemini H1 / H5 | 70.7 / 81.3 | paper |
| Macro kappa (panel vs human consensus) | 0.556 | paper: 0.56 |

Step 0 warns automatically if the ANY rate or the mean drifts from the
published values.

---

## What each step produces

**00 build.** `benchmark_labels.csv.gz` (19,361 rows: labels, axis rollups,
gold flag, three split columns, report text, reference text),
`label_reliability_tiers.csv`, `benchmark_manifest.json`,
`INTEGRITY_REPORT.md`.

Splits are all video-disjoint. The random split is crime-stratified 70/15/15
over videos, so no video appears in more than one fold. A report-level split
would leak, since each video contributes 24 reports.

**01 baselines.** Surface, surface-without-identity, and TF-IDF across the
three splits. `BASELINE_REPORT.md`.

**02 leave-one-generator-out.** Three folds, plus the grounded
(report-vs-reference) and embedding feature sets. `LOGO_REPORT.md`.

**03 artifact test.** For each candidate feature, AUC against the panel label
versus AUC against the human label on the same 130 reports. A large positive
gap means the feature is tracking the judge rather than the phenomenon.
`ARTIFACT_TEST.md`.

**04 ground-truth audit.** Draws a balanced 50-report sample (25 heuristically
flagged, 25 not) for manual checking. The heuristic has a high false-positive
rate by construction and is a triage aid, not a measurement. Fill the
`verdict` column by hand.

---

## Caveats baked into the analysis

- Reliability kappas are computed only on rows where both raters agree, which
  is the paper's definition. n varies by type from 62 to 110 and is reported
  alongside every kappa. H5's 0.811 rests on 64 of 130 reports.
- The artifact test has 9 to 110 items per cell. Gaps below about 0.10 are
  noise.
- H1 is tier `low` and should be excluded from headline macro metrics; the
  reports print `macro_hq` for this reason.
- The grounded feature block contains no generator identity and no technique
  identity by design, so it is the only feature set whose transfer results are
  not confounded by style.
