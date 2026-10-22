# Current designated final submission

**`submission_v4_1.csv`** is the designated final so far.

Configuration: RRF (`k=60`) of exposure `E_48_uniform` at weight **0.5**
and the `v3_1` ranking at weight **1.0**, using top-200 input rankings.
The `v3_1` component is RRF of PPR alpha **0.5** and popularity at weights
**1.0 / 0.5**. Final features/refitting use viewing from weeks **11–30**;
the **31–42 schedule supplies airings only**. Popularity is the cold-start
fallback, and every submitted row has five distinct in-pool programme IDs.

Validation MAP@5: **0.6789544696811836** (all watched),
**0.2374277627283462** (top five by seconds).

## Kaggle results

User-reported scores; screenshots saved by the user.

| Submission | Public | Private |
|---|---:|---:|
| v4_1 — designated final | 0.35383 | 0.33450 |
| v4_2 | 0.34898 | 0.32602 |
| v4_3 | Not submitted | Not submitted |
| v3_1 | 0.33173 | 0.31829 |
| v3_2 | 0.32686 | 0.31413 |
| v3_3 | 0.32539 | 0.31550 |
| PPR alpha 0.3 | 0.30721 | 0.27476 |
| Popularity | 0.29788 | 0.29442 |
| final_v2 — history + airtime | 0.28189 | 0.25260 |
| History | 0.28079 | 0.25013 |

## Step 6 experiment

`src/step6_ltr.py` will compare five validation target definitions using
Spearman correlation against the **nine public scores** above, excluding the
unsubmitted v4_3. Public scores select only the evaluation definition;
they are not ranker features, labels, weights, or fitting targets. Private
scores are recorded here for results tracking and are not used in calibration.

The ranker uses a seed-12345 80/20 user split for early stopping and reports
held-out MAP@5 against v4_1 on the same users. It then refits on all validation
users at the chosen iteration count. If all-watched is selected, every watched
item is relevance 1 (binary); ranked top-five targets use grades 5 through 1.
Final v5 files are experimental until generated, validated, and evaluated.

Tested dependency: LightGBM **4.7.0**. On macOS its OpenMP runtime is installed
with `brew install libomp`.
