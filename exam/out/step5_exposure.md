STEP 5 — TRAINING-ONLY CHANNEL/TIME EXPOSURE; SELECT BY ALL-WATCHED MAP@5
Training-only channel/time habits projected onto the target airing schedule.

Run from exam/: .venv/bin/python -u src/step5_exposure.py

Validation: raw views strictly clipped to ISO weeks 11-20; airings from 21-30.
Final: raw views strictly clipped to 11-30; airings from 31-42. No 31-42
viewing is read or inferred. Schedule metadata supplies only channel/time/item.

Four variants: channel x 168 hour-of-week slots or channel x 48
weekday/weekend-hour slots, each uniform or last-four-training-weeks x2.
H is watched seconds per training week, including unlabelled airtime. A is
labelled target-airing seconds. E = H @ A.T is an exposure affinity (seconds
squared/week), not a calibrated expected-seconds estimate. Dividing by 3600
would not change any ranking. The 48-slot variant pools days without further
day-count normalization. Recency keeps the same unweighted week denominator.

Each variant tests E alone; E * 1.25 for watched items; RRF of E/v3_1 at
weights (0.5/1, 1/1, 2/1); and rank-normalized linear blends at E weights
0.3, 0.5, 0.7. Both fusion families use top-200 input rankings. Linear rank
scores are (201-rank)/200, absent items zero; ties use ascending programme ID.
v3_1 is RRF(k=60) of PPR(alpha=0.5)/popularity at weights 1/0.5. Its ranking
is extended beyond five for fusion, preserving the original top five exactly.

All 33 configurations (including v3_1 reference) are eligible for selection
by all-watched MAP@5; top-five-target MAP@5 is diagnostic. Saves the grid,
novelty counts, report and final configuration manifest under out/, and writes
submission_v4_1.csv, submission_v4_2.csv, submission_v4_3.csv.


Building target-airing matrix from schedule 21-30 only
  schedule chunk 1: 42,483 labelled airings, 72,329 hour segments
  schedule chunk 2: 16,727 labelled airings, 29,390 hour segments
  schedule chunk 3: 16,495 labelled airings, 24,325 hour segments
  schedule chunk 4: 17,950 labelled airings, 32,209 hour segments
  schedule chunk 5: 11,562 labelled airings, 24,128 hour segments
  schedule chunk 6: 16,288 labelled airings, 29,540 hour segments
  A168 (6363, 31584), nnz=98,409; target seconds=409,495,080

Building habits: [2020-03-09, 2020-05-18), 10 weeks; recent cutoff 2020-04-20 00:00:00
  All raw channel-view seconds retained, including tv_show_id=0 airtime; no label join/filter.
  raw-view chunk 1: 197,601 eligible views, 215,035 hour segments
  raw-view chunk 2: 198,044 eligible views, 211,200 hour segments
  raw-view chunk 3: 197,996 eligible views, 216,734 hour segments
  raw-view chunk 4: 198,026 eligible views, 214,998 hour segments
  raw-view chunk 5: 197,991 eligible views, 215,668 hour segments
  raw-view chunk 6: 198,224 eligible views, 216,602 hour segments
  raw-view chunk 7: 198,374 eligible views, 214,445 hour segments
  raw-view chunk 8: 198,441 eligible views, 217,275 hour segments
  raw-view chunk 9: 198,583 eligible views, 217,904 hour segments
  raw-view chunk 10: 198,648 eligible views, 215,970 hour segments
  raw-view chunk 11: 198,595 eligible views, 215,359 hour segments
  raw-view chunk 12: 198,838 eligible views, 216,181 hour segments
  raw-view chunk 13: 198,890 eligible views, 212,465 hour segments
  raw-view chunk 14: 198,851 eligible views, 218,139 hour segments
  raw-view chunk 15: 199,133 eligible views, 215,744 hour segments
  raw-view chunk 16: 199,257 eligible views, 216,757 hour segments
  raw-view chunk 17: 199,402 eligible views, 216,589 hour segments
  raw-view chunk 18: 199,345 eligible views, 243,667 hour segments
  raw-view chunk 19: 199,526 eligible views, 246,744 hour segments
  raw-view chunk 20: 145,102 eligible views, 187,693 hour segments
  raw-view chunk 21: no eligible training rows
  raw-view chunk 22: no eligible training rows
  raw-view chunk 23: no eligible training rows
  raw-view chunk 24: no eligible training rows
  raw-view chunk 25: no eligible training rows
  raw-view chunk 26: no eligible training rows
  raw-view chunk 27: no eligible training rows
  raw-view chunk 28: no eligible training rows
  raw-view chunk 29: no eligible training rows
  raw-view chunk 30: no eligible training rows
  retained context seconds=4,158,899,004; seconds on channels without target labelled airings=734,715,741 (cannot contribute exposure)
  H 168_uniform: (4799, 31584), nnz=2,424,228, zero-habit users=44
  H 168_recent: (4799, 31584), nnz=2,424,228, zero-habit users=44
  H 48_uniform: (4799, 9024), nnz=1,517,615, zero-habit users=44
  H 48_recent: (4799, 9024), nnz=1,517,615, zero-habit users=44
  phase 21-30: 4,799 users, 6,363 candidates; 56 item-history cold users get popularity for all methods
  v3_1_reference: MAP@5(all)=0.651722, MAP@5(top-5)=0.205891; novel 1,225/23,995 (0.255/user, 5.1%); fallback=56

VALIDATION VARIANT 168_uniform
  exposure 168_uniform: scored users 1-250; sparse H @ A.T
  exposure 168_uniform: scored users 251-500; sparse H @ A.T
  exposure 168_uniform: scored users 501-750; sparse H @ A.T
  exposure 168_uniform: scored users 751-1,000; sparse H @ A.T
  exposure 168_uniform: scored users 1,001-1,250; sparse H @ A.T
  exposure 168_uniform: scored users 1,251-1,500; sparse H @ A.T
  exposure 168_uniform: scored users 1,501-1,750; sparse H @ A.T
  exposure 168_uniform: scored users 1,751-2,000; sparse H @ A.T
  exposure 168_uniform: scored users 2,001-2,250; sparse H @ A.T
  exposure 168_uniform: scored users 2,251-2,500; sparse H @ A.T
  exposure 168_uniform: scored users 2,501-2,750; sparse H @ A.T
  exposure 168_uniform: scored users 2,751-3,000; sparse H @ A.T
  exposure 168_uniform: scored users 3,001-3,250; sparse H @ A.T
  exposure 168_uniform: scored users 3,251-3,500; sparse H @ A.T
  exposure 168_uniform: scored users 3,501-3,750; sparse H @ A.T
  exposure 168_uniform: scored users 3,751-4,000; sparse H @ A.T
  exposure 168_uniform: scored users 4,001-4,250; sparse H @ A.T
  exposure 168_uniform: scored users 4,251-4,500; sparse H @ A.T
  exposure 168_uniform: scored users 4,501-4,750; sparse H @ A.T
  exposure 168_uniform: scored users 4,751-4,799; sparse H @ A.T
  E_168_uniform: MAP@5(all)=0.619211, MAP@5(top-5)=0.261620; novel 3,638/23,995 (0.758/user, 15.2%); fallback=56
  E_168_uniform_history_bonus: MAP@5(all)=0.627116, MAP@5(top-5)=0.263608; novel 2,782/23,995 (0.580/user, 11.6%); fallback=56
  rrf_E_168_uniform_v3_wE=0.5: MAP@5(all)=0.666856, MAP@5(top-5)=0.240719; novel 902/23,995 (0.188/user, 3.8%); fallback=56
  rrf_E_168_uniform_v3_wE=1.0: MAP@5(all)=0.664598, MAP@5(top-5)=0.254196; novel 960/23,995 (0.200/user, 4.0%); fallback=56
  rrf_E_168_uniform_v3_wE=2.0: MAP@5(all)=0.657497, MAP@5(top-5)=0.262493; novel 1,181/23,995 (0.246/user, 4.9%); fallback=56
  linear_E_168_uniform_v3_wE=0.3: MAP@5(all)=0.666406, MAP@5(top-5)=0.238690; novel 863/23,995 (0.180/user, 3.6%); fallback=56
  linear_E_168_uniform_v3_wE=0.5: MAP@5(all)=0.663071, MAP@5(top-5)=0.252900; novel 927/23,995 (0.193/user, 3.9%); fallback=56
  linear_E_168_uniform_v3_wE=0.7: MAP@5(all)=0.655298, MAP@5(top-5)=0.262517; novel 1,067/23,995 (0.222/user, 4.4%); fallback=56

VALIDATION VARIANT 168_recent
  exposure 168_recent: scored users 1-250; sparse H @ A.T
  exposure 168_recent: scored users 251-500; sparse H @ A.T
  exposure 168_recent: scored users 501-750; sparse H @ A.T
  exposure 168_recent: scored users 751-1,000; sparse H @ A.T
  exposure 168_recent: scored users 1,001-1,250; sparse H @ A.T
  exposure 168_recent: scored users 1,251-1,500; sparse H @ A.T
  exposure 168_recent: scored users 1,501-1,750; sparse H @ A.T
  exposure 168_recent: scored users 1,751-2,000; sparse H @ A.T
  exposure 168_recent: scored users 2,001-2,250; sparse H @ A.T
  exposure 168_recent: scored users 2,251-2,500; sparse H @ A.T
  exposure 168_recent: scored users 2,501-2,750; sparse H @ A.T
  exposure 168_recent: scored users 2,751-3,000; sparse H @ A.T
  exposure 168_recent: scored users 3,001-3,250; sparse H @ A.T
  exposure 168_recent: scored users 3,251-3,500; sparse H @ A.T
  exposure 168_recent: scored users 3,501-3,750; sparse H @ A.T
  exposure 168_recent: scored users 3,751-4,000; sparse H @ A.T
  exposure 168_recent: scored users 4,001-4,250; sparse H @ A.T
  exposure 168_recent: scored users 4,251-4,500; sparse H @ A.T
  exposure 168_recent: scored users 4,501-4,750; sparse H @ A.T
  exposure 168_recent: scored users 4,751-4,799; sparse H @ A.T
  E_168_recent: MAP@5(all)=0.618133, MAP@5(top-5)=0.263879; novel 3,770/23,995 (0.786/user, 15.7%); fallback=56
  E_168_recent_history_bonus: MAP@5(all)=0.626914, MAP@5(top-5)=0.266327; novel 2,900/23,995 (0.604/user, 12.1%); fallback=56
  rrf_E_168_recent_v3_wE=0.5: MAP@5(all)=0.667571, MAP@5(top-5)=0.241961; novel 907/23,995 (0.189/user, 3.8%); fallback=56
  rrf_E_168_recent_v3_wE=1.0: MAP@5(all)=0.665336, MAP@5(top-5)=0.257398; novel 965/23,995 (0.201/user, 4.0%); fallback=56
  rrf_E_168_recent_v3_wE=2.0: MAP@5(all)=0.658420, MAP@5(top-5)=0.265005; novel 1,192/23,995 (0.248/user, 5.0%); fallback=56
  linear_E_168_recent_v3_wE=0.3: MAP@5(all)=0.666293, MAP@5(top-5)=0.240026; novel 856/23,995 (0.178/user, 3.6%); fallback=56
  linear_E_168_recent_v3_wE=0.5: MAP@5(all)=0.663937, MAP@5(top-5)=0.255398; novel 927/23,995 (0.193/user, 3.9%); fallback=56
  linear_E_168_recent_v3_wE=0.7: MAP@5(all)=0.655575, MAP@5(top-5)=0.264776; novel 1,079/23,995 (0.225/user, 4.5%); fallback=56

VALIDATION VARIANT 48_uniform
  exposure 48_uniform: scored users 1-250; sparse H @ A.T
  exposure 48_uniform: scored users 251-500; sparse H @ A.T
  exposure 48_uniform: scored users 501-750; sparse H @ A.T
  exposure 48_uniform: scored users 751-1,000; sparse H @ A.T
  exposure 48_uniform: scored users 1,001-1,250; sparse H @ A.T
  exposure 48_uniform: scored users 1,251-1,500; sparse H @ A.T
  exposure 48_uniform: scored users 1,501-1,750; sparse H @ A.T
  exposure 48_uniform: scored users 1,751-2,000; sparse H @ A.T
  exposure 48_uniform: scored users 2,001-2,250; sparse H @ A.T
  exposure 48_uniform: scored users 2,251-2,500; sparse H @ A.T
  exposure 48_uniform: scored users 2,501-2,750; sparse H @ A.T
  exposure 48_uniform: scored users 2,751-3,000; sparse H @ A.T
  exposure 48_uniform: scored users 3,001-3,250; sparse H @ A.T
  exposure 48_uniform: scored users 3,251-3,500; sparse H @ A.T
  exposure 48_uniform: scored users 3,501-3,750; sparse H @ A.T
  exposure 48_uniform: scored users 3,751-4,000; sparse H @ A.T
  exposure 48_uniform: scored users 4,001-4,250; sparse H @ A.T
  exposure 48_uniform: scored users 4,251-4,500; sparse H @ A.T
  exposure 48_uniform: scored users 4,501-4,750; sparse H @ A.T
  exposure 48_uniform: scored users 4,751-4,799; sparse H @ A.T
  E_48_uniform: MAP@5(all)=0.628681, MAP@5(top-5)=0.252847; novel 3,867/23,995 (0.806/user, 16.1%); fallback=56
  E_48_uniform_history_bonus: MAP@5(all)=0.636753, MAP@5(top-5)=0.255183; novel 2,983/23,995 (0.622/user, 12.4%); fallback=56
  rrf_E_48_uniform_v3_wE=0.5: MAP@5(all)=0.678954, MAP@5(top-5)=0.237428; novel 948/23,995 (0.198/user, 4.0%); fallback=56
  rrf_E_48_uniform_v3_wE=1.0: MAP@5(all)=0.677782, MAP@5(top-5)=0.247701; novel 1,037/23,995 (0.216/user, 4.3%); fallback=56
  rrf_E_48_uniform_v3_wE=2.0: MAP@5(all)=0.672967, MAP@5(top-5)=0.256149; novel 1,263/23,995 (0.263/user, 5.3%); fallback=56
  linear_E_48_uniform_v3_wE=0.3: MAP@5(all)=0.677452, MAP@5(top-5)=0.234877; novel 927/23,995 (0.193/user, 3.9%); fallback=56
  linear_E_48_uniform_v3_wE=0.5: MAP@5(all)=0.676979, MAP@5(top-5)=0.246557; novel 1,004/23,995 (0.209/user, 4.2%); fallback=56
  linear_E_48_uniform_v3_wE=0.7: MAP@5(all)=0.671234, MAP@5(top-5)=0.256123; novel 1,181/23,995 (0.246/user, 4.9%); fallback=56

VALIDATION VARIANT 48_recent
  exposure 48_recent: scored users 1-250; sparse H @ A.T
  exposure 48_recent: scored users 251-500; sparse H @ A.T
  exposure 48_recent: scored users 501-750; sparse H @ A.T
  exposure 48_recent: scored users 751-1,000; sparse H @ A.T
  exposure 48_recent: scored users 1,001-1,250; sparse H @ A.T
  exposure 48_recent: scored users 1,251-1,500; sparse H @ A.T
  exposure 48_recent: scored users 1,501-1,750; sparse H @ A.T
  exposure 48_recent: scored users 1,751-2,000; sparse H @ A.T
  exposure 48_recent: scored users 2,001-2,250; sparse H @ A.T
  exposure 48_recent: scored users 2,251-2,500; sparse H @ A.T
  exposure 48_recent: scored users 2,501-2,750; sparse H @ A.T
  exposure 48_recent: scored users 2,751-3,000; sparse H @ A.T
  exposure 48_recent: scored users 3,001-3,250; sparse H @ A.T
  exposure 48_recent: scored users 3,251-3,500; sparse H @ A.T
  exposure 48_recent: scored users 3,501-3,750; sparse H @ A.T
  exposure 48_recent: scored users 3,751-4,000; sparse H @ A.T
  exposure 48_recent: scored users 4,001-4,250; sparse H @ A.T
  exposure 48_recent: scored users 4,251-4,500; sparse H @ A.T
  exposure 48_recent: scored users 4,501-4,750; sparse H @ A.T
  exposure 48_recent: scored users 4,751-4,799; sparse H @ A.T
  E_48_recent: MAP@5(all)=0.627845, MAP@5(top-5)=0.255698; novel 3,942/23,995 (0.821/user, 16.4%); fallback=56
  E_48_recent_history_bonus: MAP@5(all)=0.636355, MAP@5(top-5)=0.258374; novel 3,000/23,995 (0.625/user, 12.5%); fallback=56
  rrf_E_48_recent_v3_wE=0.5: MAP@5(all)=0.677543, MAP@5(top-5)=0.238381; novel 953/23,995 (0.199/user, 4.0%); fallback=56
  rrf_E_48_recent_v3_wE=1.0: MAP@5(all)=0.677040, MAP@5(top-5)=0.249625; novel 1,046/23,995 (0.218/user, 4.4%); fallback=56
  rrf_E_48_recent_v3_wE=2.0: MAP@5(all)=0.672074, MAP@5(top-5)=0.258966; novel 1,284/23,995 (0.268/user, 5.4%); fallback=56
  linear_E_48_recent_v3_wE=0.3: MAP@5(all)=0.675996, MAP@5(top-5)=0.236121; novel 934/23,995 (0.195/user, 3.9%); fallback=56
  linear_E_48_recent_v3_wE=0.5: MAP@5(all)=0.676499, MAP@5(top-5)=0.248347; novel 1,019/23,995 (0.212/user, 4.2%); fallback=56
  linear_E_48_recent_v3_wE=0.7: MAP@5(all)=0.670852, MAP@5(top-5)=0.258725; novel 1,186/23,995 (0.247/user, 4.9%); fallback=56

VALIDATION TABLE — ALL-WATCHED MAP@5 FIRST; NOVEL = NOT WATCHED BY THIS USER IN TRAINING
 rank                         method  map_all  map_top5  fallback  novel_top5_count  novel_per_user  novel_fraction
    1     rrf_E_48_uniform_v3_wE=0.5 0.678954  0.237428        56               948        0.197541        0.039508
    2     rrf_E_48_uniform_v3_wE=1.0 0.677782  0.247701        56              1037        0.216087        0.043217
    3      rrf_E_48_recent_v3_wE=0.5 0.677543  0.238381        56               953        0.198583        0.039717
    4  linear_E_48_uniform_v3_wE=0.3 0.677452  0.234877        56               927        0.193165        0.038633
    5      rrf_E_48_recent_v3_wE=1.0 0.677040  0.249625        56              1046        0.217962        0.043592
    6  linear_E_48_uniform_v3_wE=0.5 0.676979  0.246557        56              1004        0.209210        0.041842
    7   linear_E_48_recent_v3_wE=0.5 0.676499  0.248347        56              1019        0.212336        0.042467
    8   linear_E_48_recent_v3_wE=0.3 0.675996  0.236121        56               934        0.194624        0.038925
    9     rrf_E_48_uniform_v3_wE=2.0 0.672967  0.256149        56              1263        0.263180        0.052636
   10      rrf_E_48_recent_v3_wE=2.0 0.672074  0.258966        56              1284        0.267556        0.053511
   11  linear_E_48_uniform_v3_wE=0.7 0.671234  0.256123        56              1181        0.246093        0.049219
   12   linear_E_48_recent_v3_wE=0.7 0.670852  0.258725        56              1186        0.247135        0.049427
   13     rrf_E_168_recent_v3_wE=0.5 0.667571  0.241961        56               907        0.188998        0.037800
   14    rrf_E_168_uniform_v3_wE=0.5 0.666856  0.240719        56               902        0.187956        0.037591
   15 linear_E_168_uniform_v3_wE=0.3 0.666406  0.238690        56               863        0.179829        0.035966
   16  linear_E_168_recent_v3_wE=0.3 0.666293  0.240026        56               856        0.178370        0.035674
   17     rrf_E_168_recent_v3_wE=1.0 0.665336  0.257398        56               965        0.201084        0.040217
   18    rrf_E_168_uniform_v3_wE=1.0 0.664598  0.254196        56               960        0.200042        0.040008
   19  linear_E_168_recent_v3_wE=0.5 0.663937  0.255398        56               927        0.193165        0.038633
   20 linear_E_168_uniform_v3_wE=0.5 0.663071  0.252900        56               927        0.193165        0.038633
   21     rrf_E_168_recent_v3_wE=2.0 0.658420  0.265005        56              1192        0.248385        0.049677
   22    rrf_E_168_uniform_v3_wE=2.0 0.657497  0.262493        56              1181        0.246093        0.049219
   23  linear_E_168_recent_v3_wE=0.7 0.655575  0.264776        56              1079        0.224839        0.044968
   24 linear_E_168_uniform_v3_wE=0.7 0.655298  0.262517        56              1067        0.222338        0.044468
   25                 v3_1_reference 0.651722  0.205891        56              1225        0.255262        0.051052
   26     E_48_uniform_history_bonus 0.636753  0.255183        56              2983        0.621588        0.124318
   27      E_48_recent_history_bonus 0.636355  0.258374        56              3000        0.625130        0.125026
   28                   E_48_uniform 0.628681  0.252847        56              3867        0.805793        0.161159
   29                    E_48_recent 0.627845  0.255698        56              3942        0.821421        0.164284
   30    E_168_uniform_history_bonus 0.627116  0.263608        56              2782        0.579704        0.115941
   31     E_168_recent_history_bonus 0.626914  0.266327        56              2900        0.604293        0.120859
   32                  E_168_uniform 0.619211  0.261620        56              3638        0.758075        0.151615
   33                   E_168_recent 0.618133  0.263879        56              3770        0.785580        0.157116

FINAL REFIT — 11-30 VIEWS ONLY; 31-42 SCHEDULE ONLY FOR AIRINGS

Building target-airing matrix from schedule 31-42 only
  schedule chunk 1: 40,125 labelled airings, 70,776 hour segments
  schedule chunk 2: 24,328 labelled airings, 40,076 hour segments
  schedule chunk 3: 20,872 labelled airings, 31,196 hour segments
  schedule chunk 4: 17,471 labelled airings, 27,492 hour segments
  schedule chunk 5: 9,999 labelled airings, 22,411 hour segments
  schedule chunk 6: 11,979 labelled airings, 24,261 hour segments
  schedule chunk 7: 20,000 labelled airings, 36,722 hour segments
  A168 (6636, 32088), nnz=113,758; target seconds=487,595,880

Building habits: [2020-03-09, 2020-07-27), 20 weeks; recent cutoff 2020-06-29 00:00:00
  All raw channel-view seconds retained, including tv_show_id=0 airtime; no label join/filter.
  raw-view chunk 1: 71,299 eligible views, 74,959 hour segments
  raw-view chunk 2: 70,232 eligible views, 72,800 hour segments
  raw-view chunk 3: 71,898 eligible views, 76,047 hour segments
  raw-view chunk 4: 71,741 eligible views, 76,140 hour segments
  raw-view chunk 5: 72,941 eligible views, 77,164 hour segments
  raw-view chunk 6: 72,846 eligible views, 77,533 hour segments
  raw-view chunk 7: 72,868 eligible views, 76,782 hour segments
  raw-view chunk 8: 77,749 eligible views, 83,401 hour segments
  raw-view chunk 9: 72,658 eligible views, 77,623 hour segments
  raw-view chunk 10: 72,656 eligible views, 76,707 hour segments
  raw-view chunk 11: 72,204 eligible views, 76,339 hour segments
  raw-view chunk 12: 73,125 eligible views, 77,528 hour segments
  raw-view chunk 13: 73,599 eligible views, 76,770 hour segments
  raw-view chunk 14: 74,199 eligible views, 79,300 hour segments
  raw-view chunk 15: 71,459 eligible views, 75,034 hour segments
  raw-view chunk 16: 71,843 eligible views, 76,403 hour segments
  raw-view chunk 17: 70,242 eligible views, 73,804 hour segments
  raw-view chunk 18: 74,008 eligible views, 85,072 hour segments
  raw-view chunk 19: 73,950 eligible views, 86,106 hour segments
  raw-view chunk 20: 74,765 eligible views, 90,346 hour segments
  raw-view chunk 21: 78,672 eligible views, 94,947 hour segments
  raw-view chunk 22: 78,828 eligible views, 94,733 hour segments
  raw-view chunk 23: 79,763 eligible views, 96,934 hour segments
  raw-view chunk 24: 82,226 eligible views, 91,016 hour segments
  raw-view chunk 25: 83,389 eligible views, 92,283 hour segments
  raw-view chunk 26: 75,696 eligible views, 92,782 hour segments
  raw-view chunk 27: 75,802 eligible views, 91,748 hour segments
  raw-view chunk 28: 77,366 eligible views, 93,014 hour segments
  raw-view chunk 29: 76,921 eligible views, 88,444 hour segments
  raw-view chunk 30: 8,368 eligible views, 8,557 hour segments
  retained context seconds=2,422,888,364; seconds on channels without target labelled airings=508,876,571 (cannot contribute exposure)
  H 168_uniform: (1260, 32088), nnz=1,152,783, zero-habit users=1
  H 168_recent: (1260, 32088), nnz=1,152,783, zero-habit users=1
  H 48_uniform: (1260, 9168), nnz=660,334, zero-habit users=1
  H 48_recent: (1260, 9168), nnz=660,334, zero-habit users=1
  phase 31-42: 1,260 users, 6,636 candidates; 2 item-history cold users get popularity for all methods

submission_v4_1.csv <- rrf_E_48_uniform_v3_wE=0.5; config={"kind": "rrf", "variant": "48_uniform", "weight": 0.5}
  exposure 48_uniform: scored users 1-250; sparse H @ A.T
  exposure 48_uniform: scored users 251-500; sparse H @ A.T
  exposure 48_uniform: scored users 501-750; sparse H @ A.T
  exposure 48_uniform: scored users 751-1,000; sparse H @ A.T
  exposure 48_uniform: scored users 1,001-1,250; sparse H @ A.T
  exposure 48_uniform: scored users 1,251-1,260; sparse H @ A.T
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v4_1.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 112
    pool_size: 6636

submission_v4_2.csv <- rrf_E_48_uniform_v3_wE=1.0; config={"kind": "rrf", "variant": "48_uniform", "weight": 1.0}
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v4_2.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 121
    pool_size: 6636

submission_v4_3.csv <- rrf_E_48_recent_v3_wE=0.5; config={"kind": "rrf", "variant": "48_recent", "weight": 0.5}
  exposure 48_recent: scored users 1-250; sparse H @ A.T
  exposure 48_recent: scored users 251-500; sparse H @ A.T
  exposure 48_recent: scored users 501-750; sparse H @ A.T
  exposure 48_recent: scored users 751-1,000; sparse H @ A.T
  exposure 48_recent: scored users 1,001-1,250; sparse H @ A.T
  exposure 48_recent: scored users 1,251-1,260; sparse H @ A.T
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v4_3.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 113
    pool_size: 6636

Completed in 172.6s
