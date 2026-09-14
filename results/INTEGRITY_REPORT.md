# Benchmark rebuild integrity report

## 1. Source integrity

- raw judge rows: 38,722
- reports: 19,361
- judges per report: {2: 19361}
- self-judging rows (judge == generator): 0
- judge-failure sentinels (-1): 24 cells / 4 rows -> treated as missing, excluded from the vote

## 2. Label aggregation (two-judge majority, ties -> negative)

- ANY hallucination: 91.13%   (paper 91.1%)
- mean per report:   2.580   (paper 2.58)

Per-model per-type rate (%):

| model   |   H1 |   H2 |   H3 |   H4 |   H5 |   H6 |
|:--------|-----:|-----:|-----:|-----:|-----:|-----:|
| Claude  | 89.6 | 29.9 | 38.5 | 18.9 | 88.6 | 37.6 |
| GPT     | 30.9 | 17.3 | 48.4 | 38.7 | 37.2 |  6.3 |
| Gemini  | 70.7 | 31.6 | 44.1 | 28.3 | 81.3 | 36   |

## 3. Label reliability against human raters (n=130)

| type   | axis        |   n_consensus |   kappa_human_human |   kappa_panel_consensus | tier   |
|:-------|:------------|--------------:|--------------------:|------------------------:|:-------|
| H1     | fabrication |            62 |              -0.012 |                   0.15  | low    |
| H2     | distortion  |            99 |               0.471 |                   0.7   | high   |
| H3     | omission    |           110 |               0.533 |                   0.432 | medium |
| H4     | distortion  |           103 |               0.41  |                   0.583 | medium |
| H5     | fabrication |            64 |               0.083 |                   0.811 | high   |
| H6     | fabrication |            84 |               0.191 |                   0.658 | high   |

- macro kappa: 0.556  (paper 0.556)
- NOTE: these kappas are computed only on rows where both raters agree; n varies by type and is reported above.
- low-reliability types (exclude from headline macro): ['H1']

## 4. Splits (video-disjoint)

- split_random: {'train': 13507, 'test': 3046, 'val': 2808}   videos spanning >1 fold: 0
- split_heldout_model: {'train': 12912, 'test': 6449}   videos spanning >1 fold: 807
- split_heldout_technique: {'test': 9684, 'train': 9677}   videos spanning >1 fold: 807

- gold (human-validated) reports: 130
