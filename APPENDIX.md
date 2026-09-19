# Appendix

Supporting tables for *HALT: Held-Out-Source Evaluation for Hallucination Detection in Forensic Video Reporting*.

The paper cites these as "released artifact, Appendix A" through
"Appendix G". Every table is regenerated from the raw per-judge verdicts by
`pipeline/run_all.sh`; the source CSVs are named under each table.

Permanent location: <https://anonymous.4open.science/r/HALT-ANON-7FF2/>

---

## A. Held-out-strategy results

Train on the four single-turn strategies, test on the four multi-turn ones.
Compare the macro column against the standard split (Appendix B) and against
leave-one-source-out (Table 3 of the paper).

| Features | H1 | H2 | H3 | H4 | H5 | H6 | ANY | macro | macro-hq |
|---|---|---|---|---|---|---|---|---|---|
| surface | 0.809 | 0.601 | 0.547 | 0.500 | 0.800 | 0.736 | 0.732 | 0.665 | 0.637 |
| surface-noid | 0.794 | 0.593 | 0.555 | 0.534 | 0.790 | 0.719 | 0.703 | 0.664 | 0.638 |
| tfidf | 0.868 | 0.753 | 0.859 | 0.832 | 0.873 | 0.839 | 0.808 | 0.837 | 0.831 |
| embed | 0.745 | 0.681 | 0.617 | 0.730 | 0.632 | 0.624 | 0.633 | 0.671 | 0.657 |
| grounded | 0.837 | 0.690 | 0.602 | 0.358 | 0.821 | 0.769 | 0.746 | 0.679 | 0.648 |

Source: `results/baseline_results.csv`.

---

## B. Same-source baselines and the identity ablation

Removing the source and strategy identity indicators (`surface-noid`) does not
remove the shortcut, because style is recoverable from the text itself: scene
fabrication still reaches 0.811 under the standard split and still collapses to
0.603 under source-LLM shift.

| Features | Split | H1 | H2 | H3 | H4 | H5 | H6 | ANY | macro-hq |
|---|---|---|---|---|---|---|---|---|---|
| surface | random | 0.858 | 0.690 | 0.646 | 0.653 | 0.865 | 0.772 | 0.765 | 0.725 |
| surface | held-out strategy | 0.809 | 0.601 | 0.547 | 0.500 | 0.800 | 0.736 | 0.732 | 0.637 |
| surface | held-out source | 0.628 | 0.646 | 0.506 | 0.443 | 0.559 | 0.564 | 0.583 | 0.544 |
| surface-noid | random | 0.811 | 0.667 | 0.576 | 0.636 | 0.830 | 0.738 | 0.720 | 0.689 |
| surface-noid | held-out strategy | 0.794 | 0.593 | 0.555 | 0.534 | 0.790 | 0.719 | 0.703 | 0.638 |
| surface-noid | held-out source | 0.603 | 0.614 | 0.470 | 0.538 | 0.567 | 0.632 | 0.582 | 0.564 |
| tfidf | random | 0.882 | 0.773 | 0.854 | 0.822 | 0.888 | 0.859 | 0.820 | 0.839 |
| tfidf | held-out strategy | 0.868 | 0.753 | 0.859 | 0.832 | 0.873 | 0.839 | 0.808 | 0.831 |
| tfidf | held-out source | 0.630 | 0.727 | 0.795 | 0.715 | 0.621 | 0.720 | 0.613 | 0.716 |

Source: `results/baseline_results.csv`.

---

## C. Per-fold leave-one-source-out

### C.1 TF-IDF, with bootstrap intervals

Per fold, with bootstrap 95% confidence intervals (1,000 resamples). The H3
interval does not overlap the H1 or H5 interval in any fold.

| Held-out source LLM | H1 | H2 | H3 | H4 | H5 | H6 | ANY |
|---|---|---|---|---|---|---|---|
| Claude | 0.596 | 0.718 | 0.787 | 0.788 | 0.671 | 0.778 | 0.539 |
| | [.581,.611] | [.704,.732] | [.776,.799] | [.774,.801] | [.658,.686] | [.765,.788] | [.502,.578] |
| GPT | 0.702 | 0.735 | 0.796 | 0.779 | 0.719 | 0.736 | 0.548 |
| | [.689,.714] | [.719,.749] | [.785,.807] | [.767,.790] | [.707,.730] | [.713,.760] | [.531,.566] |
| Gemini | 0.630 | 0.727 | 0.795 | 0.715 | 0.621 | 0.720 | 0.613 |
| | [.613,.644] | [.715,.741] | [.785,.806] | [.702,.729] | [.604,.640] | [.708,.732] | [.585,.636] |
| **Mean** | 0.643 | 0.727 | 0.793 | 0.761 | 0.671 | 0.744 | 0.566 |
| **Random split** | 0.882 | 0.773 | 0.854 | 0.822 | 0.888 | 0.859 | 0.820 |
| **Degradation** | −0.239 | −0.046 | −0.061 | −0.061 | −0.217 | −0.115 | −0.254 |

Source: `results/logo_results.csv`, `results/logo_auc_ci.csv`.

---

### C.2 All feature sets, per fold

AUC on the held-out source LLM, per fold. Table 4 of the paper reports the
three-fold averages.

| Features | Held out | H1 | H2 | H3 | H4 | H5 | H6 | ANY |
|---|---|---|---|---|---|---|---|---|
| tfidf | Claude | 0.596 | 0.718 | 0.787 | 0.788 | 0.671 | 0.778 | 0.539 |
| tfidf | GPT | 0.702 | 0.735 | 0.796 | 0.779 | 0.719 | 0.736 | 0.548 |
| tfidf | Gemini | 0.630 | 0.727 | 0.795 | 0.715 | 0.621 | 0.720 | 0.613 |
| embed | Claude | 0.853 | 0.659 | 0.689 | 0.674 | 0.862 | 0.661 | 0.670 |
| embed | GPT | 0.527 | 0.624 | 0.824 | 0.716 | 0.566 | 0.673 | 0.587 |
| embed | Gemini | 0.671 | 0.627 | 0.645 | 0.644 | 0.655 | 0.560 | 0.528 |
| grounded | Claude | 0.883 | 0.612 | 0.572 | 0.546 | 0.840 | 0.566 | 0.852 |
| grounded | GPT | 0.665 | 0.596 | 0.642 | 0.528 | 0.639 | 0.579 | 0.616 |
| grounded | Gemini | 0.649 | 0.720 | 0.518 | 0.523 | 0.616 | 0.673 | 0.638 |
| grounded+surface | Claude | 0.890 | 0.595 | 0.566 | 0.568 | 0.855 | 0.570 | 0.744 |
| grounded+surface | GPT | 0.692 | 0.588 | 0.663 | 0.598 | 0.676 | 0.581 | 0.573 |
| grounded+surface | Gemini | 0.650 | 0.702 | 0.593 | 0.544 | 0.620 | 0.685 | 0.637 |

Holding out GPT, the only omission-dominant family, drops embedding
scene-fabrication AUC to 0.527 while raising crime-omission AUC to 0.824.
Holding out Claude, leaving the similarly fabrication-dominant Gemini, raises
scene fabrication to 0.853. Detectors generalise to a new source LLM mainly
when the training pool contains one that fails the same way.

Source: `results/logo_results.csv`.

---

## D. The judge-pair confound

**(a) Pair agreement, Cohen's κ.** Each source LLM is scored by a fixed pair of
judges; the pairs differ by a factor of five on the fabrication types.

| Source LLM | Judged by | H1 | H5 | overall |
|---|---|---|---|---|
| Claude | GPT + Gemini | 0.850 | 0.845 | 0.809 |
| GPT | Claude + Gemini | 0.415 | 0.475 | 0.580 |
| Gemini | Claude + GPT | 0.148 | 0.138 | 0.614 |

**(b) Within-judge axis rates (%).** Holding the instrument fixed. The dominant
axis is reproduced by every judge for every source LLM it sees.

| Judge | Source LLM | Fabrication | Omission | Distortion | Dominant |
|---|---|---|---|---|---|
| Claude | GPT | 28.6 | 55.7 | 42.7 | omission |
| Claude | Gemini | 66.7 | 54.3 | 39.8 | fabrication |
| GPT | Claude | 77.8 | 41.7 | 36.0 | fabrication |
| GPT | Gemini | 79.2 | 49.4 | 38.1 | fabrication |
| Gemini | Claude | 72.7 | 44.5 | 26.8 | fabrication |
| Gemini | GPT | 44.1 | 53.1 | 33.1 | omission |

Across 18 judge-by-type-by-pair comparisons the ordering given by the
aggregated labels is reproduced in 16. At the axis level it is reproduced in
all six.

Source: `results/judge_confound.csv`.

---

## E. Label-source divergence, all features

AUC of each single feature against the panel label and against the human label,
on the 130-report subset. Table 5 of the paper reports the report-length
column. Every reference-conditioned feature diverges in the opposite
direction: on grounded quantities the panel does not diverge from the human
raters.

| Type | novel rate (panel) | novel rate (human) | missing rate (panel) | missing rate (human) | hedging (panel) | hedging (human) |
|---|---|---|---|---|---|---|
| H1 Scene Fabrication | 0.535 | 0.546 | 0.300 | 0.536 | 0.551 | 0.451 |
| H2 Crime Misclassification | 0.545 | 0.600 | 0.500 | 0.565 | 0.471 | 0.573 |
| H3 Crime Missed | 0.618 | 0.644 | 0.671 | 0.692 | 0.522 | 0.547 |
| H4 Severity Minimization | 0.547 | 0.661 | 0.608 | 0.730 | 0.616 | 0.636 |
| H5 Entity Fabrication | 0.467 | 0.595 | 0.254 | 0.342 | 0.464 | 0.577 |
| H6 Phantom Actors | 0.567 | 0.607 | 0.408 | 0.450 | 0.461 | 0.450 |

Source: `results/artifact_test.csv`.

---

## F. Feature sets under leave-one-source-out

Averaged over the three folds. Axis scores are the mean per-type AUC within the
axis. Among the linear detectors, reference-conditioned features are the only
ones that recover binary detection; the fine-tuned encoders exceed them without
grounding.

| Features | Fabrication | Omission | Distortion | macro | ANY |
|---|---|---|---|---|---|
| surface | 0.584 | 0.506 | 0.545 | 0.558 | 0.583 |
| embed | 0.670 | 0.719 | 0.657 | 0.674 | 0.595 |
| tfidf | 0.686 | **0.793** | **0.744** | 0.723 | 0.566 |
| grounded | 0.679 | 0.577 | 0.587 | 0.631 | 0.702 |
| grounded+surface | 0.691 | 0.607 | 0.599 | 0.646 | 0.652 |
| DeBERTa-v3, 512 | **0.776** | **0.888** | **0.850** | **0.820** | **0.729** |
| ModernBERT, 1,280 | 0.707 | 0.775 | 0.725 | 0.724 | 0.644 |
| ModernBERT, 2,048 | 0.729 | 0.803 | 0.742 | 0.746 | 0.642 |

Source: `results/logo_results.csv`, `results/encoder_results_*.csv`.

---

## G. Rubrics and full corpus statistics

The verbatim judge rubric, the human-rater instructions, the per-cell
hallucination rates for all 24 source-LLM-by-strategy combinations, the
video-disjoint robustness rerun (`results/logo_strict_results.csv`), and the
bootstrap intervals (`results/logo_auc_ci.csv`) are in `results/`, alongside
`results/EXTENDED_RESULTS.md`, which regenerates every table above from the raw
per-judge verdicts.
