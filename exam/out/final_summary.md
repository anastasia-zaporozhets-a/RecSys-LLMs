# Final experiment summary

## Designated final: `submission_v6_3.csv`

**Kaggle private 0.40253 / public 0.41907.** The final decision retains v6_3. v8_ens (private 0.39991 / public 0.41669), v7_1 (private 0.40078 / public 0.41615) and v7_2 (private 0.39924 / public 0.41519) did not beat it.

## Total Kaggle submissions

**16 documented Kaggle submissions with reported scores**, counting each scored file once. There are 18 generated primary submission files. v4_3 and v6_2 have no reported Kaggle result/upload confirmation, so they are excluded from that count. The archived PPR alias is also not an additional submission.

v6_3 is the all_together recipe: 200 candidates/source; weekly-scaled features; stacked folds; channel/time/season/recency features; binary LambdaRank; five-seed raw-score averaging. Seed/tree counts: 12345/150, 12346/112, 12347/132, 12348/173, 12349/275, refit on 100%.

## Full submission table

The single table follows experiment/file-generation chronology (initial PPR refit, baselines, history+airtime, then v3 through v8). Exact Kaggle upload timestamps are unavailable. The archived PPR alias is placed beside its canonical file.

All MAP values below are **validation** measurements, not scores on final 31–42 predictions. The scope column distinguishes full validation (4,799 users) from the 960/946-user holdouts. Full-validation and user-holdout results are not interchangeable. The mean column is exactly `(fold10 + fold5) / 2`, not a user-count-weighted average. The top-five-by-time target is diagnostic.

**— means unavailable/not reported, never zero.** The v5 blend and v8 ensemble have no saved held-out validation measurement. Kaggle scores are user-reported; scores not supplied for v4_3 and v6_2 are left blank. The archived PPR alias shares one result, not a separate trial.

| File | Method | Validation scope | MAP@5 all / fold10 | MAP@5 fold5 | Mean | MAP@5 top-5 target | Kaggle public | Kaggle private |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `submission_ppr_alpha0.3.csv` | PPR alpha 0.3 | Full validation (4,799) | 0.579983 | — | — | 0.247010 | 0.30721 | 0.27476 |
| `out/submission_final.csv` | Archived prediction-identical alias of PPR alpha 0.3 | Full validation (4,799) | 0.579983 | — | — | 0.247010 | 0.30721 | 0.27476 |
| `submission_popularity.csv` | Popularity: distinct training viewers | Full validation (4,799) | 0.610720 | — | — | 0.147777 | 0.29788 | 0.29442 |
| `submission_history.csv` | Personal history by watched seconds | Full validation (4,799) | 0.550115 | — | — | 0.241036 | 0.28079 | 0.25013 |
| `submission_final_v2.csv` | History × airtime ratio^0.5 | Full validation (4,799) | 0.561773 | — | — | 0.247583 | 0.28189 | 0.25260 |
| `submission_v3_1.csv` | RRF_ppr_pop(alpha=0.5,weights=1.0/0.5) | Full validation (4,799) | 0.651722 | — | — | 0.205891 | 0.33173 | 0.31829 |
| `submission_v3_2.csv` | RRF_ppr_pop(alpha=0.7,weights=1.0/0.25) | Full validation (4,799) | 0.651344 | — | — | 0.197329 | 0.32686 | 0.31413 |
| `submission_v3_3.csv` | RRF_ppr_pop(alpha=0.5,weights=1.0/1.0) | Full validation (4,799) | 0.650844 | — | — | 0.188247 | 0.32539 | 0.31550 |
| `submission_v4_1.csv` | rrf_E_48_uniform_v3_wE=0.5 | Full validation (4,799) | 0.678954 | — | — | 0.237428 | 0.35383 | 0.33450 |
| `submission_v4_2.csv` | rrf_E_48_uniform_v3_wE=1.0 | Full validation (4,799) | 0.677782 | — | — | 0.247701 | 0.34898 | 0.32602 |
| `submission_v4_3.csv` | rrf_E_48_recent_v3_wE=0.5 | Full validation (4,799) | 0.677543 | — | — | 0.238381 | — | — |
| `submission_v5_ltr.csv` | Binary LambdaRank; K50/source; 1 seed; 127 trees | User holdout (960) | 0.731745 | — | — | 0.230762 | 0.39205 | 0.37705 |
| `submission_v5_blend.csv` | RRF(k=60), v5_ltr + v4_1, weights 1/1 | Not measured | — | — | — | — | 0.37766 | 0.36082 |
| `submission_v6_1.csv` | LambdaRank new_features_only; K50/source; raw units; single fold; 1 seed(s) | User holdouts (960 / 946) | 0.733172 | 0.617791 | 0.675481 | — | 0.40131 | 0.38529 |
| `submission_v6_2.csv` | LambdaRank candidates_only; K200/source; raw units; single fold; 1 seed(s) | User holdouts (960 / 946) | 0.732229 | 0.611416 | 0.671823 | — | — | — |
| **`submission_v6_3.csv`** | LambdaRank all_together; K200/source; weekly units; stacked fold; 5 seed(s) | User holdouts (960 / 946) | 0.731092 | 0.640175 | 0.685634 | — | 0.41907 | 0.40253 |
| `submission_v7_1.csv` | LambdaRank binary; all Step 8 exposures; leaves=63, lr=0.1, min_child=50; 5 seeds | User holdouts (960 / 946) | 0.732196 | 0.646786 | 0.689491 | — | 0.41615 | 0.40078 |
| `submission_v7_2.csv` | LambdaRank binary; all Step 8 exposures; leaves=63, lr=0.05, min_child=20; 5 seeds | User holdouts (960 / 946) | 0.731826 | 0.646346 | 0.689086 | — | 0.41519 | 0.39924 |
| `submission_v8_ens.csv` | Equal RRF(k=60), v6_3 + v7_1 + v7_2; post-Kaggle choice | Not measured | — | — | — | — | 0.41669 | 0.39991 |

## Why validation improvements did not change the final decision

- v6_3 historical mean: **0.685633512**.
- Step 8 retrained binary baseline with mean-MAP early stopping: **0.685873532**.
- v7_1 mean: **0.689491180**; delta vs Step 8 baseline **+0.003617648**, vs historical v6_3 **+0.003857668**. Kaggle private delta vs v6_3: **-0.00175**.
- v7_2 mean: **0.689085770**; delta vs Step 8 baseline **+0.003212237**, vs historical v6_3 **+0.003452258**. Kaggle private delta vs v6_3: **-0.00329**.
- v8_ens: no saved validation measurement; Kaggle private **0.39991**, delta vs v6_3 **-0.00262**.
- The roughly +0.0035 validation improvement did not transfer to Kaggle. Final choice: v6_3.

## Protocol and selection chronology

- Viewing is available only for 11–30. Final feature windows end before 31; the 31–42 schedule supplies programme/channel/time airings only. No 31–42 viewing is used.
- Validation fold10: 11–20 features → 21–30 labels. Fold5: 11–25 features → 26–30 labels. The same held-out users are excluded from supervised training in both folds.
- Early steps selected by top-five-by-time MAP. Step 4 changed selection to all-watched MAP after comparing Kaggle ordering. Step 6 used nine public scores only to calibrate the target definition; all-watched won. Both Step 8 winning configurations retained binary labels.
- Step 7 initially selected v6_1/v6_2 using fold10. v6_3 was an additional **post-hoc choice after seeing fold5**, using the original fold10-NDCG early-stopping tree counts.
- Step 8 predeclared the mean of fold10/fold5 all-watched MAP for early stopping and selection. These repeatedly used validation folds are selection data, not untouched test estimates.
- v8 is a file-only equal RRF ensemble **chosen after seeing Kaggle scores**. It did not beat v6_3; the designated final is explicitly v6_3.

## Content-based comparison (no Kaggle submission file)

Plain M2: full-validation all-watched MAP@5 **0.132708**, top-five-target MAP@5 **0.049009**, 4,799 users. Its final-profile examples are in `examples_final.md`.

## Submission checks and fingerprints

Every listed submission passed: 1,260 template-ordered unique users; exactly five distinct, nonzero programme IDs per row; all IDs in the 6,636-programme final airing pool.

Fingerprints below identify the archived files exactly:

- `submission_ppr_alpha0.3.csv`: 1260 rows; 412 distinct programmes; SHA-256 `2545429b3a45ab62052a00f44c21b6d39ddc8891829759e491c033f1a8b21d3e`.
- `out/submission_final.csv`: 1260 rows; 412 distinct programmes; SHA-256 `2545429b3a45ab62052a00f44c21b6d39ddc8891829759e491c033f1a8b21d3e`.
- `submission_popularity.csv`: 1260 rows; 5 distinct programmes; SHA-256 `0faec6e600906c1d3b8f15bf98efa851bf6a1202cefa57e4c65a363bfde8ce88`.
- `submission_history.csv`: 1260 rows; 480 distinct programmes; SHA-256 `d09831a97d675a83dbd743e66404a2addc65c07cf1a9185ef2969e654918649f`.
- `submission_final_v2.csv`: 1260 rows; 527 distinct programmes; SHA-256 `20e7e8d9d2e78d0a4d00b502cdb8ccb5b0526eab3e61ae4c82ca22fe48e22a33`.
- `submission_v3_1.csv`: 1260 rows; 102 distinct programmes; SHA-256 `73a6a63a21779c70f8d733dace9c9be4d72c9020e5ca67aa421eceb9fc11885a`.
- `submission_v3_2.csv`: 1260 rows; 107 distinct programmes; SHA-256 `f4ecb5ea5f3122a56d29b144e80437915862a4833779186c6b36d9b4d25a4f14`.
- `submission_v3_3.csv`: 1260 rows; 65 distinct programmes; SHA-256 `63a14c8d2ef61cbfc0b3eb1ffbca0685f4e1aa904155a21c2c340c40ff369eaa`.
- `submission_v4_1.csv`: 1260 rows; 112 distinct programmes; SHA-256 `f01b48c5c96e9aea5f6b6b8a2dd9b1d7ad77b756e24cadf98d2d8da9ca16b950`.
- `submission_v4_2.csv`: 1260 rows; 121 distinct programmes; SHA-256 `0ba5437bf24bee8f38eb1036f630abf18faf0cfe10d0355a8812d020a3ef1ede`.
- `submission_v4_3.csv`: 1260 rows; 113 distinct programmes; SHA-256 `78b92727ba1424f7411fc7316488db9a26b146f1935e4b782b7d0429ae1cc64e`.
- `submission_v5_ltr.csv`: 1260 rows; 121 distinct programmes; SHA-256 `20228703ddaddd888ad6d508f536e2d8216e564ffe6219a2d9de1027e2a4ef40`.
- `submission_v5_blend.csv`: 1260 rows; 107 distinct programmes; SHA-256 `39e6380b29f715cb8209c36f0d73a93164cde992a728054cb9a32aa71c1beea7`.
- `submission_v6_1.csv`: 1260 rows; 126 distinct programmes; SHA-256 `8db9f50798c0bbf0a3767a2a5bd4891ae2d916fcd74820c53af4f3639df70564`.
- `submission_v6_2.csv`: 1260 rows; 127 distinct programmes; SHA-256 `1f15594ff65433b365c72a15416ded1fb625a55b828c2e45e0861f5d564311ed`.
- `submission_v6_3.csv`: 1260 rows; 131 distinct programmes; SHA-256 `411bca1e61ee4b84a960f85bcb937939ab2613214c1c29ecb5207e4677d7cb20`.
- `submission_v7_1.csv`: 1260 rows; 134 distinct programmes; SHA-256 `2e4b8b4aee92eb3b39c68fadc206aed8ea727a649368eba01ba5e4882aa9e37d`.
- `submission_v7_2.csv`: 1260 rows; 139 distinct programmes; SHA-256 `469354257ffa10c77155cd2d9b6564bde6ce2ef65d372f475d28b4891494b66d`.
- `submission_v8_ens.csv`: 1260 rows; 135 distinct programmes; SHA-256 `aebe6792e6bfa4df90d128858761c091d7498b9f2d667d90177d861727cf1e48`.

## Validation metric provenance

- `submission_ppr_alpha0.3.csv`: `out/val_ppr_alpha0.3.csv`.
- `out/submission_final.csv`: `out/val_ppr_alpha0.3.csv`.
- `submission_popularity.csv`: `out/val_popularity.csv`.
- `submission_history.csv`: `out/val_personal_history.csv`.
- `submission_final_v2.csv`: `out/step3_tune_grid.csv`.
- `submission_v3_1.csv`: `out/step4_allwatched_grid.csv`.
- `submission_v3_2.csv`: `out/step4_allwatched_grid.csv`.
- `submission_v3_3.csv`: `out/step4_allwatched_grid.csv`.
- `submission_v4_1.csv`: `out/step5_exposure_grid.csv`.
- `submission_v4_2.csv`: `out/step5_exposure_grid.csv`.
- `submission_v4_3.csv`: `out/step5_exposure_grid.csv`.
- `submission_v5_ltr.csv`: `out/step6_heldout.csv`.
- `submission_v5_blend.csv`: `src/step6_ltr.py (no held-out blend measurement)`.
- `submission_v6_1.csv`: `out/step7_ablations.csv`.
- `submission_v6_2.csv`: `out/step7_ablations.csv`.
- `submission_v6_3.csv`: `out/step7_ablations.csv`.
- `submission_v7_1.csv`: `out/step8_final_configs.json`.
- `submission_v7_2.csv`: `out/step8_final_configs.json`.
- `submission_v8_ens.csv`: `out/ensemble_final.md`.

## Committed experiment logs

- `out/final_reports.log`
- `out/m2_stdout.log`
- `out/step1_stdout.log`
- `out/step2_full_stdout.log`
- `out/step2_stdout.log`
- `out/step3_tune_stdout.log`
- `out/step4_stdout.log`
- `out/step5_stdout.log`
- `out/step6_stdout.log`
- `out/step7_stdout.log`
- `out/step7_v6_3_refit.log`
- `out/step8_stdout.log`

The accompanying experiment Markdown reports, metric/configuration tables and saved validation prediction exports are archived alongside these logs. `out/test_ppr.csv` is an auxiliary multi-column prediction export, not a separate Kaggle submission.
