# HALT — Hallucination Across LLMs and Techniques

Anonymous artifact for a double-blind submission.

**Paper:** *HALT: Held-Out-Source Evaluation for Hallucination Detection in Forensic Video Reporting*

## What this is

Hallucination detectors are almost always evaluated on text from the same LLMs
that produced their training data. HALT makes the alternative testable, on a
corpus of 19,361 long-form forensic reports written by three frontier
multimodal LLMs under eight prompting strategies over 807 surveillance videos,
labelled for six hallucination types by a cross-judging panel in which no model
scores its own output. Because the corpus is fully crossed, the writing model
can be withheld while the inputs stay fixed.

The corpus, per-judge verdicts and rubrics were released with the prior study
that produced them. New here are the held-out-source split and three other
video-disjoint partitions, per-type reliability tiers, detectors spanning
linear, reference-conditioned, fine-tuned encoder and entailment features, and
the analyses in Section 4.

## Layout

```
paper/         LaTeX source, ICLR 2027 style files, compiled PDF
pipeline/      the analysis, runnable locally or in Colab
results/       every output the paper's numbers come from
figures/       figures as vector PDF, with the scripts that generate them
figures/png/   the same figures as 300 dpi PNG, for viewing here
APPENDIX.md    a convenience copy of the paper's appendix tables
```

## Reproducing the paper

`pipeline/HALT_complete_pipeline.ipynb` runs everything in Colab and is the
easiest entry point. It rebuilds the labels from the raw per-judge verdicts,
creates the splits, and reproduces every table. Parts 1 to 5 need only CPU;
the encoder, prompted-LLM and entailment detectors need a GPU.

Locally:

```
export HALLUBENCH_RAW=/path/to/raw
export HALLUBENCH_OUT=/path/to/output
cd pipeline && ./run_all.sh
```

**One trap worth knowing.** `full_labeled_dataset_full.csv` aggregates the two
judges disjunctively and reports 98.5% any-hallucination. The released label is
the two-judge conjunction and gives 91.1%. `00_build_benchmark.py` derives
labels from `panel_raw_judge_labels_full.csv` only, and uses the other file for
report text alone.

## Where the paper's numbers come from

| Result | File |
|---|---|
| Labels, splits, reliability tiers | `results/benchmark_labels.csv.gz`, `label_reliability_tiers.csv` |
| Same-source and held-out-strategy baselines | `results/baseline_results.csv` |
| Leave-one-source-out, with intervals | `results/logo_results.csv`, `logo_auc_ci.csv` |
| Video-disjoint robustness check | `results/logo_strict_results.csv` |
| Source classification and base-rate predictor | `results/shortcut_test.csv` |
| Bootstrap on the degradation difference | `results/bootstrap_diff.csv`, `bootstrap_diff_noH1.csv` |
| Encoder detectors | `results/encoder_results_*.csv` |
| Entailment grounding | `results/nli_results.csv` |
| Length bias against human labels | `results/artifact_test.csv` |
| Judge-pair confound | `results/judge_confound.csv` |

Retired runs are under `results/retired/` and feed nothing.

## Appendix

`APPENDIX.md` mirrors Appendices A to H of the paper. The paper's own appendix
is in the PDF after the references; this copy is for reading in the browser.
