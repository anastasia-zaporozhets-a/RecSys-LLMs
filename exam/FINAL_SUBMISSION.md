# Current designated final submission

**`submission_v6_3.csv`** is the designated final so far:
**public 0.41907 / private 0.40253**, as reported by the user.

The final comparison retains v6_3: v8_ens scored **public 0.41669 / private
0.39991**, v7_1 **0.41615 / 0.40078**, and v7_2 **0.41519 / 0.39924**.
None beat v6_3. The complete chronological submission table and documented
submission count are in `out/final_summary.md`; the three-user comparisons
are in `out/examples_final.md`.

This is the Step 7 **all_together** recipe: size 200 per source, weekly-scaled
features, both stacked folds, the new channel/time/season/recency features,
and raw-score averaging over seeds 12345–12349. The saved tree counts were
**150, 112, 132, 173, 275**, followed by refitting on 100% of training queries.
It was chosen post-hoc from the later fold5 column, as documented below.

**Selection policy from Step 8 onward:** the unweighted mean of fold10 and
fold5 user-holdout all-watched MAP@5, `(MAP_fold10 + MAP_fold5) / 2`.
This policy applies to early stopping and recipe selection in Step 8.

## Previous designated submission: v6_1

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
| v6_3 — designated final | 0.41907 | 0.40253 |
| v7_1 | 0.41615 | 0.40078 |
| v8_ens | 0.41669 | 0.39991 |
| v7_2 | 0.41519 | 0.39924 |
| v6_1 | 0.40131 | 0.38529 |
| v6_2 | Not reported | Not reported |
| v5_ltr | 0.39205 | 0.37705 |
| v5_blend | 0.37766 | 0.36082 |
| v4_1 | 0.35383 | 0.33450 |
| v4_2 | 0.34898 | 0.32602 |
| v4_3 | Not reported | Not reported |
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
The file has now been generated, validated and evaluated; it supersedes v6_1
as the designated final with the Kaggle scores recorded above.

## Step 8 experiment

`src/step8_ltr_graded.py` starts from the cached v6_3 recipe and fixed candidate
sets. It compares binary relevance with per-user airing/day quartiles (watched
grades 1–4, non-watched 0); adds decayed E168, target-half exposure and candidate
E-rank features; and searches the eight requested LightGBM settings. Channel
share is already in v6_3 and is retained as an unchanged control.

The predeclared bounded search consists of independent one-change ablations,
a baseline hyperparameter grid, then the grid on the best label/exposure
settings from the ablations. It reports the best **tested** combination.
Every recipe averages five seeds. Selection and early stopping use the new
equal-weight mean-fold all-watched MAP@5 rule, even for graded-label training.
The same held-out users are excluded from both training folds. The best two
distinct recipes are refitted on all training queries at the selected tree
counts, producing `submission_v7_1.csv` and `submission_v7_2.csv`.

Both files improved validation by approximately 0.0035 but scored below v6_3
on Kaggle. The designated final therefore remains `submission_v6_3.csv`.

## Final file-only rank-average ensemble

`submission_v8_ens.csv` combines v6_3, v7_1 and v7_2 with equal weights:
`score(item) = sum(1 / (60 + one_based_rank))`. Ties follow v6_3 order;
items absent from v6_3 sort after present items, with ascending programme ID
as the final tie-break. Exact rational arithmetic preserves mathematical ties.

**This ensemble was chosen after seeing Kaggle scores.** It trains no model.
All three input files and the output pass the standard template-order,
five-distinct-ID and candidate-pool checks. The ensemble recommends 135 distinct
programmes across 1,260 users; 568 user rankings differ from v6_3.
The reproducible script is `src/ensemble_final.py`, with input/output hashes
and checks recorded in `out/ensemble_final.md`.
Its Kaggle result was **public 0.41669 / private 0.39991**, also below v6_3.

## Step 9: deadline-limited per-airing coverage experiment

**Idea credit:** the per-airing coverage definition and the clipped airtime
multiplier come from a description of his approach given to the user by
classmate **Vladimir Kuznetsov**. **No code or files were shared.** The same
attribution is printed in `out/step9_stdout.log`, saved with coverage cache
metadata, and recorded in the committed planning log `out/step9_plan.md`.

`src/step9_coverage.py` unions overlapping/adjacent raw view intervals globally
per user/channel, including across CSV chunks, before joining labelled airings.
Coverage is union-covered seconds divided by the **full airing duration**;
the numerator is clipped to each observation window. n50/n80 count airings
with at least 50%/80% coverage. Mean coverage is over positively viewed airings;
an all-scheduled-airing mean is also cached for audit. Programme ID 0 is excluded.

The no-training ranker is `n50 / training_weeks * sqrt(clip(target_weekly_airings /
training_weekly_airings, 0.25, 4))`, with popularity fill/fallback. It writes
`submission_v9_freq.csv` first, and reports both holdout folds next to the saved
v6_3 reference on the same users. The optional LTR stage adds n50/week, n80/week,
mean coverage and the multiplier to the cached v6_3 features, and compares
all-watched versus >=one 80%-covered airing labels. Selection/early stopping
remain mean-fold **all-watched** MAP@5. The winning label variant is refitted
on all queries with five seeds and written to `submission_v9_ltr.csv`.

Deadline: **15:00 today, Europe/Kyiv**, with a default **10-minute upload reserve**
(work cutoff 14:50). C is skipped if its conservative estimate exceeds **40
minutes** or the remaining work time; observed fit times update the estimate,
and callbacks enforce the C/deadline limits. B remains available if C is skipped.
The script builds/checks files; the user uploads them to Kaggle. No 31–42 viewing
is used; the 31–42 schedule supplies airings only. Step 9 has not yet been run.

Tested dependency: LightGBM **4.7.0**. On macOS its OpenMP runtime is installed
with `brew install libomp`.
