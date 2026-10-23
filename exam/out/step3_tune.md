
========================================================================================================
B) VALIDATION TUNING (train 11-20 -> eval 21-30; 31-42 NOT used)
========================================================================================================
Environment : Python 3.14.0 macOS-26.6.2-arm64-arm-64bit-Mach-O; numpy 2.5.4; pandas 3.0.6
Seed        : 12345
Grid        : PPR alpha [0.1, 0.2, 0.3] x beta [0.0, 0.5, 1.0]; hybrid variants
  eval users 4,799; pool 6,363; cold start 56
  airtime ratio (eval 21-30 vs train 11-20, +1h smoothing): min 0.031 median 1.800 max 386.300

========================================================================================================
B2 - PERSONAL HISTORY WITH AIRTIME-RATIO ADJUSTMENT
========================================================================================================
  B2_history(beta=0.0)                           MAP@5(top-5)=0.241036  MAP@5(all)=0.550115  fallback=56  0.4s
  B2_history(beta=0.5)                           MAP@5(top-5)=0.247583  MAP@5(all)=0.561773  fallback=56  0.7s
  B2_history(beta=1.0)                           MAP@5(top-5)=0.226220  MAP@5(all)=0.511400  fallback=56  0.7s

========================================================================================================
M1 - PERSONALIZED PAGERANK WITH AIRTIME-RATIO ADJUSTMENT
========================================================================================================
  PPR alpha=0.1 ...
  M1_PPR(alpha=0.1,beta=0.0)                     MAP@5(top-5)=0.241798  MAP@5(all)=0.553718  fallback=56  65.1s
      (scoring for this beta only: 0.2s)
  M1_PPR(alpha=0.1,beta=0.5)                     MAP@5(top-5)=0.095740  MAP@5(all)=0.276262  fallback=56  78.1s
      (scoring for this beta only: 13.0s)
  M1_PPR(alpha=0.1,beta=1.0)                     MAP@5(top-5)=0.034904  MAP@5(all)=0.161040  fallback=56  90.2s
      (scoring for this beta only: 12.2s)
  PPR alpha=0.2 ...
  M1_PPR(alpha=0.2,beta=0.0)                     MAP@5(top-5)=0.244206  MAP@5(all)=0.564207  fallback=56  166.8s
      (scoring for this beta only: 0.4s)
  M1_PPR(alpha=0.2,beta=0.5)                     MAP@5(top-5)=0.100715  MAP@5(all)=0.285848  fallback=56  191.1s
      (scoring for this beta only: 24.2s)
  M1_PPR(alpha=0.2,beta=1.0)                     MAP@5(top-5)=0.035412  MAP@5(all)=0.162910  fallback=56  214.1s
      (scoring for this beta only: 23.0s)
  PPR alpha=0.3 ...
  M1_PPR(alpha=0.3,beta=0.0)                     MAP@5(top-5)=0.247010  MAP@5(all)=0.579983  fallback=56  289.1s
      (scoring for this beta only: 0.5s)
  M1_PPR(alpha=0.3,beta=0.5)                     MAP@5(top-5)=0.108892  MAP@5(all)=0.303769  fallback=56  321.4s
      (scoring for this beta only: 32.4s)
  M1_PPR(alpha=0.3,beta=1.0)                     MAP@5(top-5)=0.036628  MAP@5(all)=0.166051  fallback=56  352.6s
      (scoring for this beta only: 31.2s)

========================================================================================================
M3 - HYBRID: B2 HISTORY FIRST, PPR FILLS THE REST
========================================================================================================
  best PPR alpha from this run: 0.3
  M3_hybrid_hist_first(alpha=0.3,beta=0.0,n_hist=2) MAP@5(top-5)=0.244195  MAP@5(all)=0.573306  fallback=56  28.1s
  M3_hybrid_hist_first(alpha=0.3,beta=0.5,n_hist=2) MAP@5(top-5)=0.192854  MAP@5(all)=0.419337  fallback=56  30.7s
  M3_hybrid_hist_first(alpha=0.3,beta=1.0,n_hist=2) MAP@5(top-5)=0.158483  MAP@5(all)=0.342331  fallback=56  30.8s
  M3_hybrid_hist_first(alpha=0.3,beta=0.0,n_hist=3) MAP@5(top-5)=0.243466  MAP@5(all)=0.568735  fallback=56  28.1s
  M3_hybrid_hist_first(alpha=0.3,beta=0.5,n_hist=3) MAP@5(top-5)=0.216041  MAP@5(all)=0.468065  fallback=56  30.8s
  M3_hybrid_hist_first(alpha=0.3,beta=1.0,n_hist=3) MAP@5(top-5)=0.189967  MAP@5(all)=0.405789  fallback=56  30.8s

========================================================================================================
M3 - HYBRID: RECIPIPROCAL-RANK FUSION (PPR + HISTORY + POPULARITY)
========================================================================================================
  M3_rrf(a=0.3,w=1.0/0.0/0.0)                    MAP@5(top-5)=0.247010  MAP@5(all)=0.579983  fallback=56  1.4s
  M3_rrf(a=0.3,w=1.0/1.0/0.0)                    MAP@5(top-5)=0.243897  MAP@5(all)=0.565346  fallback=56  1.5s
  M3_rrf(a=0.3,w=1.0/1.0/0.5)                    MAP@5(top-5)=0.238418  MAP@5(all)=0.624561  fallback=56  1.8s
  M3_rrf(a=0.3,w=1.0/0.0/0.5)                    MAP@5(top-5)=0.219510  MAP@5(all)=0.645748  fallback=56  1.6s
  M3_rrf(a=0.3,w=2.0/1.0/0.5)                    MAP@5(top-5)=0.243456  MAP@5(all)=0.616213  fallback=56  1.6s
  M3_rrf(a=0.3,w=1.0/2.0/0.5)                    MAP@5(top-5)=0.244583  MAP@5(all)=0.611157  fallback=56  1.8s

========================================================================================================
ONE TABLE - ALL VALIDATION CONFIGURATIONS (best by MAP@5 top-5 target)
========================================================================================================
                                           method                                                  params  map_top5  map_all    runtime  fallback
                             B2_history(beta=0.5)      train overlap_seconds x ratio^0.5, pool-restricted  0.247583 0.561773   0.651941        56
                      M3_rrf(a=0.3,w=1.0/0.0/0.0)        RRF(k=60) of PPR + history + popularity rankings  0.247010 0.579983   1.431201        56
                       M1_PPR(alpha=0.3,beta=0.0) log1p(sec/60) edges, own-programme restart, x ratio^0.0  0.247010 0.579983 289.056614        56
                      M3_rrf(a=0.3,w=1.0/2.0/0.5)        RRF(k=60) of PPR + history + popularity rankings  0.244583 0.611157   1.807706        56
                       M1_PPR(alpha=0.2,beta=0.0) log1p(sec/60) edges, own-programme restart, x ratio^0.0  0.244206 0.564207 166.845053        56
M3_hybrid_hist_first(alpha=0.3,beta=0.0,n_hist=2)  B2 history takes the first 2 slots, PPR fills the rest  0.244195 0.573306  28.122329        56
                      M3_rrf(a=0.3,w=1.0/1.0/0.0)        RRF(k=60) of PPR + history + popularity rankings  0.243897 0.565346   1.468484        56
M3_hybrid_hist_first(alpha=0.3,beta=0.0,n_hist=3)  B2 history takes the first 3 slots, PPR fills the rest  0.243466 0.568735  28.127231        56
                      M3_rrf(a=0.3,w=2.0/1.0/0.5)        RRF(k=60) of PPR + history + popularity rankings  0.243456 0.616213   1.637474        56
                       M1_PPR(alpha=0.1,beta=0.0) log1p(sec/60) edges, own-programme restart, x ratio^0.0  0.241798 0.553718  65.063214        56
                             B2_history(beta=0.0)      train overlap_seconds x ratio^0.0, pool-restricted  0.241036 0.550115   0.360456        56
                      M3_rrf(a=0.3,w=1.0/1.0/0.5)        RRF(k=60) of PPR + history + popularity rankings  0.238418 0.624561   1.814544        56
                             B2_history(beta=1.0)      train overlap_seconds x ratio^1.0, pool-restricted  0.226220 0.511400   0.689480        56
                      M3_rrf(a=0.3,w=1.0/0.0/0.5)        RRF(k=60) of PPR + history + popularity rankings  0.219510 0.645748   1.603936        56
M3_hybrid_hist_first(alpha=0.3,beta=0.5,n_hist=3)  B2 history takes the first 3 slots, PPR fills the rest  0.216041 0.468065  30.760256        56
M3_hybrid_hist_first(alpha=0.3,beta=0.5,n_hist=2)  B2 history takes the first 2 slots, PPR fills the rest  0.192854 0.419337  30.746392        56
M3_hybrid_hist_first(alpha=0.3,beta=1.0,n_hist=3)  B2 history takes the first 3 slots, PPR fills the rest  0.189967 0.405789  30.757958        56
M3_hybrid_hist_first(alpha=0.3,beta=1.0,n_hist=2)  B2 history takes the first 2 slots, PPR fills the rest  0.158483 0.342331  30.750843        56
                       M1_PPR(alpha=0.3,beta=0.5) log1p(sec/60) edges, own-programme restart, x ratio^0.5  0.108892 0.303769 321.448835        56
                       M1_PPR(alpha=0.2,beta=0.5) log1p(sec/60) edges, own-programme restart, x ratio^0.5  0.100715 0.285848 191.088247        56
                       M1_PPR(alpha=0.1,beta=0.5) log1p(sec/60) edges, own-programme restart, x ratio^0.5  0.095740 0.276262  78.074584        56
                       M1_PPR(alpha=0.3,beta=1.0) log1p(sec/60) edges, own-programme restart, x ratio^1.0  0.036628 0.166051 352.619954        56
                       M1_PPR(alpha=0.2,beta=1.0) log1p(sec/60) edges, own-programme restart, x ratio^1.0  0.035412 0.162910 214.131069        56
                       M1_PPR(alpha=0.1,beta=1.0) log1p(sec/60) edges, own-programme restart, x ratio^1.0  0.034904 0.161040  90.240108        56

  BEST: B2_history(beta=0.5)  MAP@5(top-5)=0.247583  MAP@5(all)=0.561773  fallback=56
  reference: B1 popularity 0.147777 | B2 history (beta=0) 0.241036 | PPR alpha=0.3 beta=0 0.247010

========================================================================================================
FINAL BUILD: refit on weeks 11-30, predict weeks 31-42
========================================================================================================
  chosen configuration: B2_history(beta=0.5) -> method=history, alpha=0.3, beta=0.5
  training pairs: 883,931 over 4,890 users; pool: 6,636; cold-start submission users: 2
  airtime ratio: min 0.005 median 0.967 max 349.250
  fallback users served by the popularity baseline: 2
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_final_v2.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 527
    pool_size: 6636
  checks: {'rows': 1260, 'template_rows': 1260, 'five_ids_every_row': True, 'distinct_ids_every_row': True, 'all_ids_in_pool': True, 'order_matches_template': True, 'zero_ids': 0, 'distinct_programmes_used': 527, 'pool_size': 6636}

Total runtime: 899.3s
