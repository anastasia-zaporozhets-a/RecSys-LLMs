Validation-only LTR improvements; run from exam/ with python -u.

No public/private LB score is consulted for any Step 7 selection. Labels are
all-watched/binary, the target selected in Step 6. Never read 31-42 viewing.
The 31-42 schedule supplies airings only, after selecting the two recipes.

Recall selection: test 50/100/200 per history/PPR0.5/E48_uniform/popularity.
Attainable means recall at the largest TESTED budget (200); also report the
full airing-pool ceiling. Pick the smallest retaining >=95% of macro recall
on training users in BOTH folds. Save each user's recall at every size.

Independent single-change ablations versus v5: candidates, per-week scaling,
stacked folds, five new features, five-seed bagging; plus all together.
One shared 20% user holdout is excluded from ALL fold training. Reuse the
Step 6 split where present; additional users receive a stable hashed split.
Early stopping uses NDCG@5 on the common 11-20 -> 21-30 holdout. Report MAP@5
there as the PRIMARY selection metric, and on the 26-30 holdout diagnostically.
Retrain the v5 recipe on 80%, never evaluate its already-full-data fitted model.

Counts/seconds use source-appropriate weekly rates: whole-history features
/10,15,20 training weeks; last-four /4, last-two /2; target supply and exposure
/10,5,12 target weeks. H already divides by training weeks. Airtime ratio uses
weekly target/train airtime with +1h/week smoothing. Ranks, shares, indicators
and one-hot categories stay dimensionless. Pop distinct users is /train weeks.

Fold 1: 11-20 features -> 21-30 labels. Fold 2: 11-25 features -> 26-30 labels.
Queries are (fold,user), NOT user across concatenated folds. Short-window
qualifying pairs use the existing >=120s AND >=5% aggregated overlap rule;
original full-block interactions/targets are reused for exact v5 comparability.

Final: refit each winning recipe on 100% of its training queries at each seed's
early-stopped tree count. Average raw scores for seeds 12345..12349 when bagged.
Recompute final features on 11-30 with 31-42 airings, write submission_v6_1.csv
and submission_v6_2.csv with the usual checks. Save ablations, per-user recall,
split, models, feature importances, config manifest and report under out/step7*.

  raw chunk 1: 200,000 views, 103,056 overlaps; 4 windows accumulated
  raw chunk 2: 200,000 views, 106,818 overlaps; 4 windows accumulated
  raw chunk 3: 200,000 views, 110,308 overlaps; 4 windows accumulated
  raw chunk 4: 200,000 views, 113,918 overlaps; 4 windows accumulated
  raw chunk 5: 200,000 views, 107,684 overlaps; 4 windows accumulated
  raw chunk 6: 200,000 views, 114,738 overlaps; 4 windows accumulated
  raw chunk 7: 200,000 views, 116,201 overlaps; 4 windows accumulated
  raw chunk 8: 200,000 views, 111,548 overlaps; 4 windows accumulated
  raw chunk 9: 200,000 views, 120,149 overlaps; 4 windows accumulated
  raw chunk 10: 200,000 views, 113,306 overlaps; 4 windows accumulated
  raw chunk 11: 200,000 views, 123,478 overlaps; 4 windows accumulated
  raw chunk 12: 200,000 views, 112,094 overlaps; 4 windows accumulated
  raw chunk 13: 200,000 views, 115,969 overlaps; 4 windows accumulated
  raw chunk 14: 200,000 views, 115,588 overlaps; 4 windows accumulated
  raw chunk 15: 200,000 views, 114,401 overlaps; 4 windows accumulated
  raw chunk 16: 200,000 views, 131,799 overlaps; 4 windows accumulated
  raw chunk 17: 200,000 views, 137,523 overlaps; 4 windows accumulated
  raw chunk 18: 200,000 views, 150,169 overlaps; 4 windows accumulated
  raw chunk 19: 200,000 views, 148,109 overlaps; 4 windows accumulated
  raw chunk 20: 200,000 views, 162,024 overlaps; 4 windows accumulated
  raw chunk 21: 200,000 views, 164,549 overlaps; 4 windows accumulated
  raw chunk 22: 200,000 views, 157,428 overlaps; 4 windows accumulated
  raw chunk 23: 200,000 views, 160,464 overlaps; 4 windows accumulated
  raw chunk 24: 200,000 views, 143,691 overlaps; 4 windows accumulated
  raw chunk 25: 200,000 views, 140,607 overlaps; 4 windows accumulated
  raw chunk 26: 200,000 views, 159,358 overlaps; 4 windows accumulated
  raw chunk 27: 200,000 views, 161,089 overlaps; 4 windows accumulated
  raw chunk 28: 200,000 views, 163,162 overlaps; 4 windows accumulated
  raw chunk 29: 200,000 views, 151,008 overlaps; 4 windows accumulated
  raw chunk 30: 21,987 views, 15,800 overlaps; 4 windows accumulated
  train10: 856,917 user/item pairs, 1,823,893 watched broadcasts
  train15: 1,090,986 user/item pairs, 2,361,714 watched broadcasts
  train20: 1,370,917 user/item pairs, 3,012,767 watched broadcasts
  label5: 398,586 user/item pairs, 651,317 watched broadcasts
CATEGORY ONE-HOT MAPPING: {"0": "__missing__", "1": "Детям", "2": "Другое", "3": "Инфо", "4": "Познавательное", "5": "Развлечения", "6": "Сериалы", "7": "Спорт", "8": "Фильмы"}
  Reusing Step 6 user split; new fold-only users are assigned by a seed-12345 stable hash.
SHARED USER SPLIT: train=3,839, heldout=960

PREPARE fold10: 10 training weeks -> 10 target weeks
  new channel/time features: users 1-250
  new channel/time features: users 251-500
  new channel/time features: users 501-750
  new channel/time features: users 751-1,000
  new channel/time features: users 1,001-1,250
  new channel/time features: users 1,251-1,500
  new channel/time features: users 1,501-1,750
  new channel/time features: users 1,751-2,000
  new channel/time features: users 2,001-2,250
  new channel/time features: users 2,251-2,500
  new channel/time features: users 2,501-2,750
  new channel/time features: users 2,751-3,000
  new channel/time features: users 3,001-3,250
  new channel/time features: users 3,251-3,500
  new channel/time features: users 3,501-3,750
  new channel/time features: users 3,751-4,000
  new channel/time features: users 4,001-4,250
  new channel/time features: users 4,251-4,500
  new channel/time features: users 4,501-4,750
  new channel/time features: users 4,751-4,799

PREPARE fold5: 15 training weeks -> 5 target weeks
  new channel/time features: users 1-250
  new channel/time features: users 251-500
  new channel/time features: users 501-750
  new channel/time features: users 751-1,000
  new channel/time features: users 1,001-1,250
  new channel/time features: users 1,251-1,500
  new channel/time features: users 1,501-1,750
  new channel/time features: users 1,751-2,000
  new channel/time features: users 2,001-2,250
  new channel/time features: users 2,251-2,500
  new channel/time features: users 2,501-2,750
  new channel/time features: users 2,751-3,000
  new channel/time features: users 3,001-3,250
  new channel/time features: users 3,251-3,500
  new channel/time features: users 3,501-3,750
  new channel/time features: users 3,751-4,000
  new channel/time features: users 4,001-4,250
  new channel/time features: users 4,251-4,500
  new channel/time features: users 4,501-4,735

CANDIDATE RECALL
  fold   split  size  users  macro_recall  pool_ceiling  mean_candidates
fold10 heldout    50    960      0.318189           1.0       102.865625
fold10 heldout   100    960      0.422963           1.0       205.463542
fold10 heldout   200    960      0.536424           1.0       387.131250
fold10   train    50   3839      0.316809           1.0       103.359729
fold10   train   100   3839      0.418233           1.0       206.209690
fold10   train   200   3839      0.532263           1.0       386.821829
 fold5 heldout    50    946      0.382178           1.0       104.226216
 fold5 heldout   100    946      0.499892           1.0       206.608879
 fold5 heldout   200    946      0.636088           1.0       386.913319
 fold5   train    50   3789      0.377663           1.0       104.661916
 fold5   train   100   3789      0.496936           1.0       206.862761
 fold5   train   200   3789      0.632740           1.0       386.250462
CANDIDATE BUDGET: 200/source; >=95% of the 200/source tested ceiling in BOTH training folds.

ABLATION previous_v5: {"new": false, "scale": false, "seeds": [12345], "size": 50, "stack": false}
  fitting seed 12345, rows=396,798, queries=3,839, features=24
  seed 12345: best_iteration=127
  HOLDOUT: {"map_all_fold10": 0.7317445023148148, "map_all_fold5": 0.6176885130373503, "recipe": "previous_v5", "seeds": 1, "size": 50, "users_fold10": 960, "users_fold5": 946}

ABLATION candidates_only: {"new": false, "scale": false, "seeds": [12345], "size": 200, "stack": false}
  fitting seed 12345, rows=1,485,009, queries=3,839, features=24
  seed 12345: best_iteration=163
  HOLDOUT: {"map_all_fold10": 0.7322294560185185, "map_all_fold5": 0.6114159032182288, "recipe": "candidates_only", "seeds": 1, "size": 200, "users_fold10": 960, "users_fold5": 946}

ABLATION weekly_scaling_only: {"new": false, "scale": true, "seeds": [12345], "size": 50, "stack": false}
  fitting seed 12345, rows=396,798, queries=3,839, features=24
  seed 12345: best_iteration=117
  HOLDOUT: {"map_all_fold10": 0.7309007523148148, "map_all_fold5": 0.6295621916842847, "recipe": "weekly_scaling_only", "seeds": 1, "size": 50, "users_fold10": 960, "users_fold5": 946}

ABLATION stacked_folds_only: {"new": false, "scale": false, "seeds": [12345], "size": 50, "stack": true}
  fitting seed 12345, rows=793,362, queries=7,628, features=24
  seed 12345: best_iteration=154
  HOLDOUT: {"map_all_fold10": 0.7285720486111111, "map_all_fold5": 0.6380364693446089, "recipe": "stacked_folds_only", "seeds": 1, "size": 50, "users_fold10": 960, "users_fold5": 946}

ABLATION new_features_only: {"new": true, "scale": false, "seeds": [12345], "size": 50, "stack": false}
  fitting seed 12345, rows=396,798, queries=3,839, features=29
  seed 12345: best_iteration=150
  HOLDOUT: {"map_all_fold10": 0.7331715856481481, "map_all_fold5": 0.6177906976744185, "recipe": "new_features_only", "seeds": 1, "size": 50, "users_fold10": 960, "users_fold5": 946}

ABLATION five_seed_bagging_only: {"new": false, "scale": false, "seeds": [12345, 12346, 12347, 12348, 12349], "size": 50, "stack": false}
  seed 12345: best_iteration=127
  fitting seed 12346, rows=396,798, queries=3,839, features=24
  seed 12346: best_iteration=121
  fitting seed 12347, rows=396,798, queries=3,839, features=24
  seed 12347: best_iteration=160
  fitting seed 12348, rows=396,798, queries=3,839, features=24
  seed 12348: best_iteration=112
  fitting seed 12349, rows=396,798, queries=3,839, features=24
  seed 12349: best_iteration=251
  HOLDOUT: {"map_all_fold10": 0.7307879050925926, "map_all_fold5": 0.6155329457364341, "recipe": "five_seed_bagging_only", "seeds": 5, "size": 50, "users_fold10": 960, "users_fold5": 946}

ABLATION all_together: {"new": true, "scale": true, "seeds": [12345, 12346, 12347, 12348, 12349], "size": 200, "stack": true}
  fitting seed 12345, rows=2,948,512, queries=7,628, features=29
  seed 12345: best_iteration=150
  fitting seed 12346, rows=2,948,512, queries=7,628, features=29
  seed 12346: best_iteration=112
  fitting seed 12347, rows=2,948,512, queries=7,628, features=29
  seed 12347: best_iteration=132
  fitting seed 12348, rows=2,948,512, queries=7,628, features=29
  seed 12348: best_iteration=173
  fitting seed 12349, rows=2,948,512, queries=7,628, features=29
  seed 12349: best_iteration=275
  HOLDOUT: {"map_all_fold10": 0.7310917245370371, "map_all_fold5": 0.6401752995066948, "recipe": "all_together", "seeds": 5, "size": 200, "users_fold10": 960, "users_fold5": 946}

HOLDOUT TABLE — SELECT BY fold10 MAP@5 ONLY
                recipe  size  seeds  map_all_fold10  users_fold10  map_all_fold5  users_fold5  delta_vs_previous_v5
     new_features_only    50      1        0.733172           960       0.617791          946              0.001427
       candidates_only   200      1        0.732229           960       0.611416          946              0.000485
           previous_v5    50      1        0.731745           960       0.617689          946              0.000000
          all_together   200      5        0.731092           960       0.640175          946             -0.000653
   weekly_scaling_only    50      1        0.730901           960       0.629562          946             -0.000844
five_seed_bagging_only    50      5        0.730788           960       0.615533          946             -0.000957
    stacked_folds_only    50      1        0.728572           960       0.638036          946             -0.003172
SELECTED BEST TWO: ["new_features_only", "candidates_only"]

PREPARE final: 20 training weeks -> 12 target weeks
  new channel/time features: users 1-250
  new channel/time features: users 251-500
  new channel/time features: users 501-750
  new channel/time features: users 751-1,000
  new channel/time features: users 1,001-1,250
  new channel/time features: users 1,251-1,260
  final inference features: no target viewing or relevance labels consumed

FINAL rank=1, recipe=new_features_only: {"new": true, "scale": false, "seeds": [12345], "size": 50, "stack": false}
  refit seed 12345 on 100% of queries, trees=150
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v6_1.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 126
    pool_size: 6636
FEATURE IMPORTANCES submission_v6_1.csv
                feature     mean_gain  mean_split
          exposure_rank 115626.811513       307.0
      user_item_airings  29649.488451       129.0
               ppr_rank  26733.622121       236.0
  target_distinct_weeks  21093.698244       133.0
         user_item_days  19166.120060       153.0
       popularity_users  18567.727193       335.0
         exposure_score   8487.989085       216.0
user_item_channel_share   7195.860016       558.0
            E168_recent   5777.225910       205.0
         target_airings   5560.351828       450.0
              ppr_score   3748.704125       200.0
    user_category_share   2546.566942       273.0
         target_airtime   2507.378387       338.0
    global_item_seconds   2182.153089       180.0
          airtime_ratio   2077.477816       291.0
last2weeks_item_seconds   2053.173205       165.0
     last4weeks_seconds   1060.742519       117.0
      user_item_seconds    699.298782       124.0
             category_3    462.961169        17.0
             category_5    168.098900        20.0
             category_4    144.915322        19.0
             category_6     81.369900        10.0
             category_8     44.297020        10.0
        is_new_for_user     42.406830         5.0
             category_1     26.667100         3.0
             category_2     11.043490         3.0
   airs_first_two_weeks      7.234450         3.0
             category_0      0.000000         0.0
             category_7      0.000000         0.0

FINAL rank=2, recipe=candidates_only: {"new": false, "scale": false, "seeds": [12345], "size": 200, "stack": false}
  refit seed 12345 on 100% of queries, trees=163
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v6_2.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 127
    pool_size: 6636
FEATURE IMPORTANCES submission_v6_2.csv
            feature     mean_gain  mean_split
      exposure_rank 345528.216799       488.0
  user_item_airings  37710.312986       139.0
   popularity_users  24944.017460       519.0
     user_item_days  23641.387725       164.0
global_item_seconds  18149.478564       280.0
           ppr_rank  14176.948749       197.0
     target_airings  10307.181969       473.0
     exposure_score   8746.904521       489.0
          ppr_score   6897.733384       262.0
user_category_share   4523.621546       378.0
      airtime_ratio   3646.779815       474.0
     target_airtime   3203.610824       454.0
 last4weeks_seconds   2476.597507       252.0
         category_3   1086.926154        34.0
         category_6    788.811671        27.0
  user_item_seconds    649.255930       130.0
         category_8    560.192110        35.0
         category_4    317.572142        25.0
         category_5    278.350781        28.0
         category_1    118.742031        24.0
    is_new_for_user     70.207781         6.0
         category_2     58.045540        12.0
         category_0      0.000000         0.0
         category_7      0.000000         0.0
Completed in 696.4s. Current designated final remains v5_ltr pending results.
