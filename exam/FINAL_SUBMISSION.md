# Current designated final submission

**`submission_v6_1.csv`** is the designated final so far:
**public 0.40131 / private 0.38529**, as reported by the user.

This is the Step 7 **new_features_only** recipe: top-50 candidates per source,
the five additional channel/time/season/recency features, seed **12345**, and
**150 trees** from early stopping. It uses the original single training fold
and raw-unit features, then refits on 100% of that fold's users. The submitted
file uses 11–30 viewing and the 31–42 airing schedule only.

## Previous designated submission: v5_ltr

This is the Step 6 LightGBM LambdaRank model: all-watched binary relevance,
seed **12345**, **127 trees** selected by early stopping on the user-level
holdout, then refitted on all validation users. Candidates are the union of
top-50 history, PPR alpha 0.5, E48_uniform, and popularity. Features for the
submitted file use **11–30 viewing** and **31–42 airings only**.

Step 6 held-out all-watched MAP@5 was **0.7317445023148148**, compared with
**0.6824103009259260** for v4_1 on the same users. This is held-out validation,
not an evaluation of the model after its all-user refit.

## Previous designated submission: v4_1

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
| v6_1 — designated final | 0.40131 | 0.38529 |
| v6_2 | Not reported | Not reported |
| v6_3 — post-hoc later-fold recipe | Pending build | Pending build |
| v5_ltr | 0.39205 | 0.37705 |
| v5_blend | 0.37766 | 0.36082 |
| v4_1 | 0.35383 | 0.33450 |
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

`src/step6_ltr.py` compared five validation target definitions using
Spearman correlation against the **nine pre-v5 public scores** above, excluding
the unsubmitted v4_3. **All-watched** was selected. Public scores selected only the evaluation definition;
they are not ranker features, labels, weights, or fitting targets. Private
scores are recorded here for results tracking and are not used in calibration.

The ranker uses a seed-12345 80/20 user split for early stopping and reports
held-out MAP@5 against v4_1 on the same users. It then refits on all validation
users at the chosen iteration count. If all-watched is selected, every watched
item is relevance 1 (binary); ranked top-five targets use grades 5 through 1.
The two v5 files have now been generated, validated and evaluated on Kaggle.

## Step 7 experiment

`src/step7_ltr_plus.py` tests candidate budget, weekly feature scaling, a second
temporal training fold, channel/time/season/recency features, and five-seed
score averaging. Selection uses validation only; no new public/private scores
are consumed. It preserves a shared user-level holdout across both folds,
retraining the v5 recipe on training users for an honest baseline comparison.
The best two distinct recipes are selected by 11–20 -> 21–30 held-out
all-watched MAP@5, then refitted on 100% of their training queries.

### Additional v6_3: post-hoc later-fold selection

After inspecting the fold5 column, the user chose **all_together** for one
additional final submission. This was selected using the **later fold5**,
not the original fold10 criterion. Fold5 is closer in time to the test weeks.
Its MAP@5 was **0.6401752995066948**, versus **0.6176885130373503** for previous_v5
(delta **+0.0224867864693445**). Fold10 values were **0.7310917245370371** versus
**0.7317445023148148**. The fold5 result is not an untouched-holdout estimate
after making this choice based on that column.

Recipe: size 200 per source, weekly scaling, both stacked folds, new features,
and raw-score averaging over seeds 12345–12349. The original **fold10
early-stopping tree counts** remain fixed at **150, 112, 132, 173, 275**.
`src/step7_refit_all_together.py` rebuilds and persists missing feature caches,
refits those five models on 100%, and writes `submission_v6_3.csv`. The live
log (`out/step7_v6_3_refit.log`) explicitly records this post-hoc choice.
The current designated final remains v6_1 pending further results.

Tested dependency: LightGBM **4.7.0**. On macOS its OpenMP runtime is installed
with `brew install libomp`.
