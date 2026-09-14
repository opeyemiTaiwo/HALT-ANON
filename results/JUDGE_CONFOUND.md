# Judge-pair confound test

Each generator is scored by a fixed pair of judges. This tests whether the generator ordering survives when the judge is held fixed.


## 1. Which pair scores which generator

| model   | judged_by       |   n_reports |
|:--------|:----------------|------------:|
| Claude  | GPT + Gemini    |        6456 |
| GPT     | Claude + Gemini |        6456 |
| Gemini  | Claude + GPT    |        6449 |


## 2. Aggregated per-generator rates (confounded)

| model   |   H1 |   H2 |   H3 |   H4 |   H5 |   H6 |
|:--------|-----:|-----:|-----:|-----:|-----:|-----:|
| Claude  | 89.6 | 29.9 | 38.5 | 18.9 | 88.6 | 37.6 |
| GPT     | 30.9 | 17.3 | 48.4 | 38.7 | 37.2 |  6.3 |
| Gemini  | 70.7 | 31.6 | 44.1 | 28.3 | 81.3 | 36   |


## 3. Within-judge rates (%): each judge's own verdicts

|                      |   H1 |   H2 |   H3 |   H4 |   H5 |   H6 |
|:---------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| ('Claude', 'GPT')    | 32.8 | 29.7 | 55.7 | 55.7 | 40.3 | 12.8 |
| ('Claude', 'Gemini') | 71.2 | 40.7 | 54.3 | 38.9 | 81.8 | 47.2 |
| ('GPT', 'Claude')    | 92   | 44   | 41.7 | 27.9 | 91.3 | 50.3 |
| ('GPT', 'Gemini')    | 96.1 | 44.1 | 49.4 | 32.2 | 97.7 | 43.7 |
| ('Gemini', 'Claude') | 89.6 | 31   | 44.5 | 22.6 | 88.8 | 39.5 |
| ('Gemini', 'GPT')    | 60.3 | 21.9 | 53.1 | 44.4 | 61.6 | 10.4 |


## 4. Does each judge reproduce the aggregated ordering?

For every judge and every type, the ordering of the two generators that judge sees, compared against the ordering the aggregated labels give for the same two generators.

| judge   | pair             | type   | name                    | within_judge   |   within_gap_pp | aggregated   |   agrees |
|:--------|:-----------------|:-------|:------------------------|:---------------|----------------:|:-------------|---------:|
| Claude  | GPT vs Gemini    | H1     | Scene Fabrication       | Gemini         |            38.4 | Gemini       |        1 |
| Claude  | GPT vs Gemini    | H2     | Crime Misclassification | Gemini         |            11   | Gemini       |        1 |
| Claude  | GPT vs Gemini    | H3     | Crime Missed            | GPT            |             1.4 | GPT          |        1 |
| Claude  | GPT vs Gemini    | H4     | Severity Minimization   | GPT            |            16.8 | GPT          |        1 |
| Claude  | GPT vs Gemini    | H5     | Entity Fabrication      | Gemini         |            41.5 | Gemini       |        1 |
| Claude  | GPT vs Gemini    | H6     | Phantom Actors          | Gemini         |            34.4 | Gemini       |        1 |
| GPT     | Claude vs Gemini | H1     | Scene Fabrication       | Gemini         |             4.1 | Claude       |        0 |
| GPT     | Claude vs Gemini | H2     | Crime Misclassification | Gemini         |             0.1 | Gemini       |        1 |
| GPT     | Claude vs Gemini | H3     | Crime Missed            | Gemini         |             7.7 | Gemini       |        1 |
| GPT     | Claude vs Gemini | H4     | Severity Minimization   | Gemini         |             4.3 | Gemini       |        1 |
| GPT     | Claude vs Gemini | H5     | Entity Fabrication      | Gemini         |             6.5 | Claude       |        0 |
| GPT     | Claude vs Gemini | H6     | Phantom Actors          | Claude         |             6.6 | Claude       |        1 |
| Gemini  | Claude vs GPT    | H1     | Scene Fabrication       | Claude         |            29.3 | Claude       |        1 |
| Gemini  | Claude vs GPT    | H2     | Crime Misclassification | Claude         |             9.1 | Claude       |        1 |
| Gemini  | Claude vs GPT    | H3     | Crime Missed            | GPT            |             8.6 | GPT          |        1 |
| Gemini  | Claude vs GPT    | H4     | Severity Minimization   | GPT            |            21.9 | GPT          |        1 |
| Gemini  | Claude vs GPT    | H5     | Entity Fabrication      | Claude         |            27.2 | Claude       |        1 |
| Gemini  | Claude vs GPT    | H6     | Phantom Actors          | Claude         |            29.1 | Claude       |        1 |


**Orderings preserved: 16 of 18 (89%).**


By type:

| type   |   sum |   count |
|:-------|------:|--------:|
| H1     |     2 |       3 |
| H2     |     3 |       3 |
| H3     |     3 |       3 |
| H4     |     3 |       3 |
| H5     |     2 |       3 |
| H6     |     3 |       3 |


## 5. Dominant axis, within judge

Axis rate = mean per-type prevalence within the axis, the aggregation used for the cross-generation check in the source study.

| judge   | generator   |   fabrication |   omission |   distortion | dominant    |
|:--------|:------------|--------------:|-----------:|-------------:|:------------|
| Claude  | GPT         |          28.6 |       55.7 |         42.7 | omission    |
| Claude  | Gemini      |          66.7 |       54.3 |         39.8 | fabrication |
| GPT     | Claude      |          77.8 |       41.7 |         36   | fabrication |
| GPT     | Gemini      |          79.2 |       49.4 |         38.1 | fabrication |
| Gemini  | Claude      |          72.7 |       44.5 |         26.8 | fabrication |
| Gemini  | GPT         |          44.1 |       53.1 |         33.1 | omission    |


### Aggregated dominant axis, for comparison

| generator   |   fabrication |   omission |   distortion | dominant    |
|:------------|--------------:|-----------:|-------------:|:------------|
| Claude      |          71.9 |       38.5 |         24.4 | fabrication |
| GPT         |          24.8 |       48.4 |         28   | omission    |
| Gemini      |          62.7 |       44.1 |         30   | fabrication |
