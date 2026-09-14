# Label-source divergence

AUC of each single feature against the panel label and against the human label, on the 130 human-validated reports. The human column is restricted to rows where both raters agree; n is reported per type.

A large positive gap means the feature predicts the judge better than it predicts the humans.


## n_words

| type   | name                    |   n_human_agreed |   panel_pos_rate |   human_pos_rate |   AUC_vs_panel |   AUC_vs_human |    gap |
|:-------|:------------------------|-----------------:|-----------------:|-----------------:|---------------:|---------------:|-------:|
| H1     | Scene Fabrication       |               62 |            0.823 |            0.145 |          0.727 |          0.566 |  0.161 |
| H2     | Crime Misclassification |               99 |            0.354 |            0.273 |          0.621 |          0.608 |  0.013 |
| H3     | Crime Missed            |              110 |            0.392 |            0.155 |          0.484 |          0.39  |  0.093 |
| H4     | Severity Minimization   |              103 |            0.369 |            0.155 |          0.505 |          0.275 |  0.23  |
| H5     | Entity Fabrication      |               64 |            0.808 |            0.672 |          0.784 |          0.8   | -0.016 |
| H6     | Phantom Actors          |               84 |            0.392 |            0.226 |          0.662 |          0.668 | -0.006 |

## novel_rate

| type   | name                    |   n_human_agreed |   panel_pos_rate |   human_pos_rate |   AUC_vs_panel |   AUC_vs_human |    gap |
|:-------|:------------------------|-----------------:|-----------------:|-----------------:|---------------:|---------------:|-------:|
| H1     | Scene Fabrication       |               62 |            0.823 |            0.145 |          0.535 |          0.546 | -0.012 |
| H2     | Crime Misclassification |               99 |            0.354 |            0.273 |          0.545 |          0.6   | -0.055 |
| H3     | Crime Missed            |              110 |            0.392 |            0.155 |          0.618 |          0.644 | -0.026 |
| H4     | Severity Minimization   |              103 |            0.369 |            0.155 |          0.547 |          0.661 | -0.114 |
| H5     | Entity Fabrication      |               64 |            0.808 |            0.672 |          0.467 |          0.595 | -0.127 |
| H6     | Phantom Actors          |               84 |            0.392 |            0.226 |          0.567 |          0.607 | -0.041 |

## missing_rate

| type   | name                    |   n_human_agreed |   panel_pos_rate |   human_pos_rate |   AUC_vs_panel |   AUC_vs_human |    gap |
|:-------|:------------------------|-----------------:|-----------------:|-----------------:|---------------:|---------------:|-------:|
| H1     | Scene Fabrication       |               62 |            0.823 |            0.145 |          0.3   |          0.536 | -0.236 |
| H2     | Crime Misclassification |               99 |            0.354 |            0.273 |          0.5   |          0.565 | -0.066 |
| H3     | Crime Missed            |              110 |            0.392 |            0.155 |          0.671 |          0.692 | -0.021 |
| H4     | Severity Minimization   |              103 |            0.369 |            0.155 |          0.608 |          0.73  | -0.122 |
| H5     | Entity Fabrication      |               64 |            0.808 |            0.672 |          0.254 |          0.342 | -0.088 |
| H6     | Phantom Actors          |               84 |            0.392 |            0.226 |          0.408 |          0.45  | -0.042 |

## hedge_rate

| type   | name                    |   n_human_agreed |   panel_pos_rate |   human_pos_rate |   AUC_vs_panel |   AUC_vs_human |    gap |
|:-------|:------------------------|-----------------:|-----------------:|-----------------:|---------------:|---------------:|-------:|
| H1     | Scene Fabrication       |               62 |            0.823 |            0.145 |          0.551 |          0.451 |  0.101 |
| H2     | Crime Misclassification |               99 |            0.354 |            0.273 |          0.471 |          0.573 | -0.102 |
| H3     | Crime Missed            |              110 |            0.392 |            0.155 |          0.522 |          0.547 | -0.025 |
| H4     | Severity Minimization   |              103 |            0.369 |            0.155 |          0.616 |          0.636 | -0.019 |
| H5     | Entity Fabrication      |               64 |            0.808 |            0.672 |          0.464 |          0.577 | -0.113 |
| H6     | Phantom Actors          |               84 |            0.392 |            0.226 |          0.461 |          0.45  |  0.011 |


## Panel behaviour against agreed human labels

| type   | name                    |   n_human_neg |   over_flag_rate |   n_human_pos |   miss_rate |
|:-------|:------------------------|--------------:|-----------------:|--------------:|------------:|
| H1     | Scene Fabrication       |            53 |            0.623 |             9 |       0     |
| H2     | Crime Misclassification |            72 |            0.153 |            27 |       0.074 |
| H3     | Crime Missed            |            93 |            0.215 |            17 |       0.176 |
| H4     | Severity Minimization   |            87 |            0.161 |            16 |       0.062 |
| H5     | Entity Fabrication      |            21 |            0.238 |            43 |       0     |
| H6     | Phantom Actors          |            65 |            0.123 |            19 |       0.158 |


CAVEAT: n per cell is small (9 to 110). Treat any single gap below ~0.10 as noise.
