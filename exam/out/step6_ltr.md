Target calibration and LightGBM LambdaRank, with training-only features.

Run from exam/:
  .venv/bin/python -m pip install lightgbm
  # macOS runtime dependency, if absent: brew install libomp
  .venv/bin/python -u src/step6_ltr.py

A: Nine submitted configurations, five definitions of relevance, Spearman
against PUBLIC Kaggle scores only. Public-LB scores choose the evaluation
definition; they are never features, labels, sample weights or model targets.
Private scores are not used. v4_1 remains the designated final so far.

Qualifying watched user/item pairs and seconds targets are the existing
harness interactions. Airings, days and mean fractions are reconstructed from
raw view/airing overlap on those pairs. A broadcast is a unique channel/start/
duration/item schedule event; sessions in the same broadcast are combined,
and watched fractions are capped at one. Days are local dates touched by the
overlap. Target ties break by programme ID. Ranked targets have labels 5..1;
all-watched has binary labels 1 for EVERY watched item, 0 otherwise, because
that target has no preference ordering. MAP@5 uses the target's relevant set.

B: Candidates are the union of top-50 history, PPR(0.5), E(48_uniform), and
popularity. No label-positive injection. Features use ONLY 11-20 viewing and
the 21-30 airing schedule. Split users 80/20 with seed 12345; early stopping
on NDCG@5; report held-out MAP@5 versus v4_1 on identical users, including
cold-start fallback. The holdout is isolated from ranker fitting, but target
definition selection in A uses the full calibration population. After honest
holdout scoring, refit on all validation users at the selected iteration.

C: Recompute features using ONLY 11-30 viewing and 31-42 airings. No 31-42
viewing is available or used. Predict with the refitted ranker; also produce
equal-weight RRF(k=60) of its top-200 ranking with the v4_1 top-200 ranking.
Exactly five distinct IDs, pool restriction and existing popularity fallback.
Outputs: out/step6_* reports, model, split, importances and calibration tables;
submission_v5_ltr.csv and submission_v5_blend.csv.

A) RECONSTRUCT OBSERVATIONS AND CALIBRATE THE TARGET
  schedule 11-20: 115,316 distinct labelled broadcasts, clipped to [2020-03-09, 2020-05-18)
  schedule 21-30: 121,238 distinct labelled broadcasts, clipped to [2020-05-18, 2020-07-27)
  raw aggregation chunk 1: 200,000 eligible views, 103,056 labelled overlaps
  raw aggregation chunk 2: 200,000 eligible views, 106,818 labelled overlaps
  raw aggregation chunk 3: 200,000 eligible views, 110,308 labelled overlaps
  raw aggregation chunk 4: 200,000 eligible views, 113,918 labelled overlaps
  raw aggregation chunk 5: 200,000 eligible views, 107,684 labelled overlaps
  raw aggregation chunk 6: 200,000 eligible views, 114,738 labelled overlaps
  raw aggregation chunk 7: 200,000 eligible views, 116,201 labelled overlaps
  raw aggregation chunk 8: 200,000 eligible views, 111,548 labelled overlaps
  raw aggregation chunk 9: 200,000 eligible views, 120,149 labelled overlaps
  raw aggregation chunk 10: 200,000 eligible views, 113,306 labelled overlaps
  raw aggregation chunk 11: 200,000 eligible views, 123,478 labelled overlaps
  raw aggregation chunk 12: 200,000 eligible views, 112,094 labelled overlaps
  raw aggregation chunk 13: 200,000 eligible views, 115,969 labelled overlaps
  raw aggregation chunk 14: 200,000 eligible views, 115,588 labelled overlaps
  raw aggregation chunk 15: 200,000 eligible views, 114,401 labelled overlaps
  raw aggregation chunk 16: 200,000 eligible views, 131,799 labelled overlaps
  raw aggregation chunk 17: 200,000 eligible views, 137,523 labelled overlaps
  raw aggregation chunk 18: 200,000 eligible views, 150,169 labelled overlaps
  raw aggregation chunk 19: 200,000 eligible views, 148,109 labelled overlaps
  raw aggregation chunk 20: 200,000 eligible views, 162,024 labelled overlaps
  raw aggregation chunk 21: 200,000 eligible views, 164,549 labelled overlaps
  raw aggregation chunk 22: 200,000 eligible views, 157,428 labelled overlaps
  raw aggregation chunk 23: 200,000 eligible views, 160,464 labelled overlaps
  raw aggregation chunk 24: 200,000 eligible views, 143,691 labelled overlaps
  raw aggregation chunk 25: 200,000 eligible views, 140,607 labelled overlaps
  raw aggregation chunk 26: 200,000 eligible views, 159,358 labelled overlaps
  raw aggregation chunk 27: 200,000 eligible views, 161,089 labelled overlaps
  raw aggregation chunk 28: 200,000 eligible views, 163,162 labelled overlaps
  raw aggregation chunk 29: 200,000 eligible views, 151,008 labelled overlaps
  raw aggregation chunk 30: 21,987 eligible views, 15,800 labelled overlaps
  aggregated 11-20: 856,917 user/item pairs; 1,823,893 user/broadcast pairs
  aggregated 21-30: 658,373 user/item pairs; 1,188,874 user/broadcast pairs
  E48 features: users 1-250
  E48 features: users 251-500
  E48 features: users 501-750
  E48 features: users 751-1,000
  E48 features: users 1,001-1,250
  E48 features: users 1,251-1,500
  E48 features: users 1,501-1,750
  E48 features: users 1,751-2,000
  E48 features: users 2,001-2,250
  E48 features: users 2,251-2,500
  E48 features: users 2,501-2,750
  E48 features: users 2,751-3,000
  E48 features: users 3,001-3,250
  E48 features: users 3,251-3,500
  E48 features: users 3,501-3,750
  E48 features: users 3,751-4,000
  E48 features: users 4,001-4,250
  E48 features: users 4,251-4,500
  E48 features: users 4,501-4,750
  E48 features: users 4,751-4,799
  reconstruction PPR_0.3: PASS versus step3_tune_grid.csv
  reconstruction history: PASS versus step3_tune_grid.csv
  reconstruction final_v2: PASS versus step3_tune_grid.csv
  reconstruction v3_1: PASS versus step4_allwatched_grid.csv
  reconstruction v3_2: PASS versus step4_allwatched_grid.csv
  reconstruction v3_3: PASS versus step4_allwatched_grid.csv
  reconstruction v4_1: PASS versus step5_exposure_grid.csv
  reconstruction v4_2: PASS versus step5_exposure_grid.csv
  calibration v4_1: {"all_watched": 0.6789544696811836, "kaggle_public": 0.35383, "method": "v4_1", "top5_airings": 0.37187730082656106, "top5_days": 0.3755537612002501, "top5_mean_fraction": 0.041431374591928874, "top5_seconds": 0.23742776272834618}
  calibration v4_2: {"all_watched": 0.6777824083721146, "kaggle_public": 0.34898, "method": "v4_2", "top5_airings": 0.3719992591049061, "top5_days": 0.37452478525618765, "top5_mean_fraction": 0.04004051770044685, "top5_seconds": 0.24770114144150404}
  calibration v3_1: {"all_watched": 0.6517215392095576, "kaggle_public": 0.33173, "method": "v3_1", "top5_airings": 0.34438112106688895, "top5_days": 0.35094151559352643, "top5_mean_fraction": 0.04267937764812114, "top5_seconds": 0.20589115788011392}
  calibration v3_2: {"all_watched": 0.6513435669468176, "kaggle_public": 0.32686, "method": "v3_2", "top5_airings": 0.3371031001829085, "top5_days": 0.3448651339399412, "top5_mean_fraction": 0.04362737607371906, "top5_seconds": 0.19732884165682665}
  calibration v3_3: {"all_watched": 0.6508435785233035, "kaggle_public": 0.32539, "method": "v3_3", "top5_airings": 0.3334038341321109, "top5_days": 0.34131312078905324, "top5_mean_fraction": 0.043821976800722376, "top5_seconds": 0.18824720427866917}
  calibration PPR_0.3: {"all_watched": 0.5799826352712371, "kaggle_public": 0.30721, "method": "PPR_0.3", "top5_airings": 0.31408904632909634, "top5_days": 0.31578801139126206, "top5_mean_fraction": 0.03288601792040008, "top5_seconds": 0.2470097937070223}
  calibration popularity: {"all_watched": 0.6107201731842282, "kaggle_public": 0.29788, "method": "popularity", "top5_airings": 0.2843688499918965, "top5_days": 0.2961726980157903, "top5_mean_fraction": 0.044159084068440185, "top5_seconds": 0.14777719895348568}
  calibration final_v2: {"all_watched": 0.5617734597485587, "kaggle_public": 0.28189, "method": "final_v2", "top5_airings": 0.29610422310203516, "top5_days": 0.29751632284503715, "top5_mean_fraction": 0.03172970063207613, "top5_seconds": 0.24758300340348685}
  calibration history: {"all_watched": 0.5501153017989859, "kaggle_public": 0.28079, "method": "history", "top5_airings": 0.2882176842397722, "top5_days": 0.2882274084878794, "top5_mean_fraction": 0.031400986316593736, "top5_seconds": 0.24103563242342155}

NINE-CONFIGURATION VALIDATION TABLE
    method  kaggle_public  top5_seconds  top5_airings  top5_days  top5_mean_fraction  all_watched
      v4_1        0.35383      0.237428      0.371877   0.375554            0.041431     0.678954
      v4_2        0.34898      0.247701      0.371999   0.374525            0.040041     0.677782
      v3_1        0.33173      0.205891      0.344381   0.350942            0.042679     0.651722
      v3_2        0.32686      0.197329      0.337103   0.344865            0.043627     0.651344
      v3_3        0.32539      0.188247      0.333404   0.341313            0.043822     0.650844
   PPR_0.3        0.30721      0.247010      0.314089   0.315788            0.032886     0.579983
popularity        0.29788      0.147777      0.284369   0.296173            0.044159     0.610720
  final_v2        0.28189      0.247583      0.296104   0.297516            0.031730     0.561773
   history        0.28079      0.241036      0.288218   0.288227            0.031401     0.550115

PUBLIC-LB SPEARMAN CORRELATIONS
            target  spearman_public   pvalue
       all_watched         0.983333 0.000002
         top5_days         0.983333 0.000002
      top5_airings         0.933333 0.000236
top5_mean_fraction         0.333333 0.380713
      top5_seconds         0.016667 0.966055
SELECTED TARGET: all_watched. Public scores choose only the evaluation definition; they do not fit model parameters. Private scores are unused. Ties break by target name.

CATEGORY ONE-HOT MAPPING
{
  "category_0": "__missing__",
  "category_1": "Детям",
  "category_2": "Другое",
  "category_3": "Инфо",
  "category_4": "Познавательное",
  "category_5": "Развлечения",
  "category_6": "Сериалы",
  "category_7": "Спорт",
  "category_8": "Фильмы"
}

B) BUILD TRAINING-ONLY CANDIDATES AND FEATURES
  feature users 250/4,799, candidates 25,789
  feature users 500/4,799, candidates 51,979
  feature users 750/4,799, candidates 77,817
  feature users 1,000/4,799, candidates 103,424
  feature users 1,250/4,799, candidates 128,937
  feature users 1,500/4,799, candidates 154,650
  feature users 1,750/4,799, candidates 180,298
  feature users 2,000/4,799, candidates 206,332
  feature users 2,250/4,799, candidates 232,014
  feature users 2,500/4,799, candidates 257,877
  feature users 2,750/4,799, candidates 283,745
  feature users 3,000/4,799, candidates 309,478
  feature users 3,250/4,799, candidates 335,555
  feature users 3,500/4,799, candidates 362,002
  feature users 3,750/4,799, candidates 387,741
  feature users 4,000/4,799, candidates 413,098
  feature users 4,250/4,799, candidates 438,857
  feature users 4,500/4,799, candidates 464,697
  feature users 4,750/4,799, candidates 490,399
  feature users 4,799/4,799, candidates 495,549
  candidate feature matrix: (495549, 24), nnz=6,492,565
  candidate recall of selected target: mean 0.317085; positive rows 115,049/495,549; no positive injection

TRAIN LAMBDARANK: 3,839 train users / 960 held-out users; seed=12345
  best iteration: 127; held-out user IDs saved for reproducibility

HELD-OUT MAP@5; selected target=all_watched
         method  chosen_target  all_watched  top5_seconds
    LTR_heldout       0.731745     0.731745      0.230762
v4_1_same_users       0.682410     0.682410      0.233153

Refitting on all 4,799 validation users at 127 iterations; features still 11-20, labels still 21-30, no 31-42 outcomes.

FEATURE IMPORTANCES
            feature          gain  split
      exposure_rank 140114.549019    353
  user_item_airings  30813.688684    145
   popularity_users  25563.205156    421
     user_item_days  19864.051047    160
           ppr_rank  17335.664858    195
     target_airings   8187.414547    374
          ppr_score   5516.885341    209
     exposure_score   4240.676842    345
     target_airtime   3760.319559    345
      airtime_ratio   3653.740412    363
global_item_seconds   3344.466601    177
user_category_share   2655.835749    268
 last4weeks_seconds   2351.878924    211
         category_3    899.874899     22
  user_item_seconds    873.588013    141
         category_6    368.298487     27
         category_4    108.507710     13
         category_5     70.430600     16
    is_new_for_user     59.347221      2
         category_1     51.342759     11
         category_8     31.624740      8
         category_2     22.317420      4
         category_0      0.000000      0
         category_7      0.000000      0

C) FINAL FEATURES: 11-30 VIEWS ONLY, 31-42 SCHEDULE ONLY
  E48 features: users 1-250
  E48 features: users 251-500
  E48 features: users 501-750
  E48 features: users 751-1,000
  E48 features: users 1,001-1,250
  E48 features: users 1,251-1,260
  schedule 31-42: 144,517 distinct labelled broadcasts, clipped to [2020-07-27, 2020-10-19)
  feature users 250/1,260, candidates 26,460
  feature users 500/1,260, candidates 52,280
  feature users 750/1,260, candidates 78,681
  feature users 1,000/1,260, candidates 105,070
  feature users 1,250/1,260, candidates 131,125
  feature users 1,260/1,260, candidates 132,151
  candidate feature matrix: (132151, 24), nnz=1,790,558
submission_v5_ltr.csv = LambdaRank, selected target=all_watched
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v5_ltr.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 121
    pool_size: 6636
submission_v5_blend.csv = RRF(k=60), LTR/v4_1 weights 1/1, top-200 input rankings
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v5_blend.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 107
    pool_size: 6636

Completed in 401.0s; designated final so far remains v4_1.
