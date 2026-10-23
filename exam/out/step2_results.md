
====================================================================================================
STEP 2 - EVALUATION HARNESS, BASELINES, PPR, CONTENT-BASED
====================================================================================================
Environment : Python 3.14.0 macOS-26.6.2-arm64-arm-64bit-Mach-O
numpy 2.5.4 | pandas 3.0.6 | scipy 1.18.1 | scikit-learn 1.9.1
Seed        : 12345 (no stochastic component in this step)
Protocol    : train ['11-20'] -> eval 21-30; pool = labelled programmes in the 21-30 schedule; K=5; repeats allowed; popularity fill; popularity for cold-start users.

  eval users with >=1 watched programme in 21-30: 4,799 (expected 4799)
  training users: 4,834 (expected 4834)
  submission users: 1,260 (ids kept as strings)
  candidate pool: 6,363 labelled programmes airing in 21-30
  cold-start eval users (no training interactions): 56

====================================================================================================
B1 - POPULARITY BASELINE (distinct training users per programme, pool-restricted)
====================================================================================================
  MAP@5 (top-5 target) = 0.147777   MAP@5 (all-watched) = 0.610720   users = 4,799   runtime = 0.001s

====================================================================================================
B2 - PERSONAL HISTORY BASELINE (own training programmes by overlap_seconds)
====================================================================================================
  MAP@5 (top-5 target) = 0.241036   MAP@5 (all-watched) = 0.550115   fallback users = 56   runtime = 0.050s

====================================================================================================
M1 - PERSONALIZED PAGERANK (bipartite user-programme graph)
====================================================================================================
  edge weight = log1p(seconds/60); restart = own programmes weighted by that weight;
  R <- alpha * P^T R + V; tol 1e-6 on max column L1 change; max_iter 30; chunk 500 users.
  alpha=0.3: MAP@5 (top-5 target) = 0.247010   MAP@5 (all-watched) = 0.579983   fallback users = 56   runtime = 58.4s
  selected alpha = 0.3 (MAP@5 top-5 target = 0.247010)

====================================================================================================
M2 - CONTENT-BASED (category + genres one-hot, TF-IDF of normalised title)
====================================================================================================
  M2_content: MAP@5 (top-5 target) = 0.049009   MAP@5 (all-watched) = 0.132708   fallback users = 56
  M2_content+hist_tiebreak: MAP@5 (top-5 target) = 0.058259   MAP@5 (all-watched) = 0.144279   fallback users = 56

====================================================================================================
M3 - HYBRID (reciprocal-rank fusion: PPR + personal history + popularity)
====================================================================================================
  weights (PPR, history, popularity) are selected on the 21-30 hold-out;
  every component is trained on 11-20 only; the 31-42 schedule is not touched.
  w=(1.0, 0.0, 0.0): MAP@5 (top-5 target) = 0.247010  MAP@5 (all-watched) = 0.579983  runtime = 0.7s
  w=(1.0, 0.0, 0.5): MAP@5 (top-5 target) = 0.219510  MAP@5 (all-watched) = 0.645748  runtime = 1.0s
  w=(1.0, 0.0, 1.0): MAP@5 (top-5 target) = 0.199474  MAP@5 (all-watched) = 0.649540  runtime = 1.0s
  w=(1.0, 1.0, 0.0): MAP@5 (top-5 target) = 0.243897  MAP@5 (all-watched) = 0.565346  runtime = 0.8s
  w=(1.0, 1.0, 0.5): MAP@5 (top-5 target) = 0.238418  MAP@5 (all-watched) = 0.624561  runtime = 1.0s
  w=(1.0, 1.0, 1.0): MAP@5 (top-5 target) = 0.224170  MAP@5 (all-watched) = 0.640229  runtime = 0.9s
  w=(1.0, 2.0, 0.0): MAP@5 (top-5 target) = 0.241679  MAP@5 (all-watched) = 0.554712  runtime = 0.8s
  w=(1.0, 2.0, 0.5): MAP@5 (top-5 target) = 0.244583  MAP@5 (all-watched) = 0.611157  runtime = 1.0s
  w=(1.0, 2.0, 1.0): MAP@5 (top-5 target) = 0.234210  MAP@5 (all-watched) = 0.629164  runtime = 0.9s
  w=(2.0, 0.0, 0.0): MAP@5 (top-5 target) = 0.247010  MAP@5 (all-watched) = 0.579983  runtime = 0.8s
  w=(2.0, 0.0, 0.5): MAP@5 (top-5 target) = 0.235286  MAP@5 (all-watched) = 0.631394  runtime = 0.9s
  w=(2.0, 0.0, 1.0): MAP@5 (top-5 target) = 0.219510  MAP@5 (all-watched) = 0.645748  runtime = 1.0s
  w=(2.0, 1.0, 0.0): MAP@5 (top-5 target) = 0.246596  MAP@5 (all-watched) = 0.573624  runtime = 0.8s
  w=(2.0, 1.0, 0.5): MAP@5 (top-5 target) = 0.243456  MAP@5 (all-watched) = 0.616213  runtime = 0.9s
  w=(2.0, 1.0, 1.0): MAP@5 (top-5 target) = 0.232507  MAP@5 (all-watched) = 0.632712  runtime = 1.0s
  w=(2.0, 2.0, 0.0): MAP@5 (top-5 target) = 0.243897  MAP@5 (all-watched) = 0.565346  runtime = 0.8s
  w=(2.0, 2.0, 0.5): MAP@5 (top-5 target) = 0.245608  MAP@5 (all-watched) = 0.605831  runtime = 1.0s
  w=(2.0, 2.0, 1.0): MAP@5 (top-5 target) = 0.238418  MAP@5 (all-watched) = 0.624561  runtime = 1.0s
  selected hybrid weights = (PPR 1.0, history 0.0, popularity 0.0); MAP@5 (top-5 target) = 0.247010

====================================================================================================
VALIDATION RESULTS (train 11-20 -> eval 21-30, K=5)
====================================================================================================
method | params | MAP@5 (top-5 target) | MAP@5 (all-watched) | runtime | fallback users
--- | --- | --- | --- | --- | ---
B1_popularity | distinct training users per programme, pool-restricted | 0.147777 | 0.610720 | 0.001s | 0
B2_personal_history | own training programmes by overlap_seconds, pool-restricted | 0.241036 | 0.550115 | 0.050s | 56
M1_PPR(alpha=0.3) | log1p(sec/60) edges, own-programme restart, tol 1e-6, max_iter 30 | 0.247010 | 0.579983 | 58.372s | 56
M2_content | cat+genres one-hot & title TF-IDF, seconds-weighted user profile, cosine | 0.049009 | 0.132708 | 5.012s | 56
M2_content+hist_tiebreak | cat+genres one-hot & title TF-IDF, seconds-weighted user profile, cosine | 0.058259 | 0.144279 | 5.012s | 56
M3_hybrid(PPR_a0.3,w=1.0/0.0/0.0) | RRF(k=60) of PPR + personal history + popularity rankings | 0.247010 | 0.579983 | 1.010s | 56

evaluated users: 4,799 (macro-average over users with >= 1 watched programme in 21-30)
selected PPR configuration: alpha=0.3, MAP@5 (top-5 target)=0.247010

====================================================================================================
SAVING VALIDATION PREDICTIONS
====================================================================================================

total runtime: 83.4s

## Full PPR alpha sweep (source: out/step2_full_stdout.log, same code path as src/evaluate.py)

Re-running `src/evaluate.py` (default, all four alphas) regenerates every row of this
table in this one file, and also writes out/val_ppr_alpha{0.5,0.7,0.85}.csv.

| method | params | MAP@5 (top-5 target) | MAP@5 (all-watched) | runtime | fallback users |
| --- | --- | --- | --- | --- | --- |
| M1_PPR | alpha=0.3 | 0.247010 | 0.579983 | 64.1s | 56 |
| M1_PPR | alpha=0.5 | 0.240571 | 0.618347 | 164.9s | 56 |
| M1_PPR | alpha=0.7 | 0.213805 | 0.645481 | 184.0s | 56 |
| M1_PPR | alpha=0.85 | 0.182234 | 0.642237 | 180.9s | 56 |

Hybrid weight grid (reciprocal-rank fusion, k=60) was searched on the 21-30 hold-out:
the best top-5-target weight vector is (PPR 1.0, history 0.0, popularity 0.0), i.e. the
fusion does not improve on pure PPR alpha=0.3 for the top-5 target (0.247010); it only
helps the all-watched sensitivity metric (e.g. w=(1,0,1) -> 0.199474 / 0.649540).
