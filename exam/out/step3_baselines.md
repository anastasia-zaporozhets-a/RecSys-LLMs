
========================================================================================================
C) FINAL SUBMISSIONS FOR THE BASELINES (refit on weeks 11-30, predict 31-42)
========================================================================================================
Environment : Python 3.14.0 macOS-26.6.2-arm64-arm-64bit-Mach-O
Both files use the identical final pipeline as src/step3_final.py / src/step3_tune.py.

========================================================================================================
B1 POPULARITY -> exam/submission_popularity.csv
========================================================================================================
  training pairs: 883,931 over 4,890 users; pool: 6,636; cold-start submission users: 2
  airtime ratio: min 0.005 median 0.967 max 349.250
  fallback users served by the popularity baseline: 2
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_popularity.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 5
    pool_size: 6636
  checks: {'rows': 1260, 'template_rows': 1260, 'five_ids_every_row': True, 'distinct_ids_every_row': True, 'all_ids_in_pool': True, 'order_matches_template': True, 'zero_ids': 0, 'distinct_programmes_used': 5, 'pool_size': 6636}

========================================================================================================
B2 PERSONAL HISTORY -> exam/submission_history.csv
========================================================================================================
  training pairs: 883,931 over 4,890 users; pool: 6,636; cold-start submission users: 2
  airtime ratio: min 0.005 median 0.967 max 349.250
  fallback users served by the popularity baseline: 2
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_history.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 480
    pool_size: 6636
  checks: {'rows': 1260, 'template_rows': 1260, 'five_ids_every_row': True, 'distinct_ids_every_row': True, 'all_ids_in_pool': True, 'order_matches_template': True, 'zero_ids': 0, 'distinct_programmes_used': 480, 'pool_size': 6636}

Total runtime: 14.5s
