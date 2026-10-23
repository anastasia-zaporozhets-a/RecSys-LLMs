
==========================================================================================
STEP 1 - DATA AUDIT AND JOIN (READ-ONLY; NO MODEL FITTING)
==========================================================================================
Environment
  Python       : 3.14.0 (macOS-26.6.2-arm64-arm-64bit-Mach-O)
  numpy        : 2.5.4
  pandas       : 3.0.6
  scipy        : 1.18.1
  randomness   : none used in this step (no seed needed)
  timezone     : Europe/Kyiv (inferred from data; see DST evidence)
  exam root    : /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam

==========================================================================================
1. FILE INVENTORY AND SCHEMA
==========================================================================================
Loaded view log in 3.26 s

View log: dataset11-30.csv
  size on disk : 396.2 MB
  rows         : 5,821,987
  columns      : ['user_id', 'vsetv_id', 'start_time', 'stop_time', 'duraton']
  dtypes       : user_id=uint64, vsetv_id=int64, start_time=str, stop_time=str, duraton=int64
  first rows   :
                user_id  vsetv_id           start_time            stop_time  duraton
0   3398461054087191302         6  2020-03-09 00:00:02  2020-03-09 00:01:53      111
1  17226860011138219284       353  2020-03-09 00:00:02  2020-03-09 00:03:26      204
2  18417531283109304442       332  2020-03-09 00:00:05  2020-03-09 00:01:26       81
  time format  : view log = %Y-%m-%d %H:%M:%S (naive local wall clock)
  start range  : 2020-03-09 00:00:02 .. 2020-07-26 23:58:42
  stop range   : 2020-03-09 00:01:26 .. 2020-07-26 23:59:58
  view duration (duraton, seconds):
    duraton: n=5,821,987, mean=1480.459, std=3123.737
      quantile      value
          0.00        61.000
          0.01        62.000
          0.05        69.000
          0.10        80.000
          0.25       133.000
          0.50       396.000
          0.75      1452.000
          0.90      3911.000
          0.95      6344.000
          0.99     14079.140
          1.00    342145.000
  stop_time-start_time == duraton: 5,821,870 / 5,821,987
  wall-clock difference exceeds duraton by exactly 3600 s: 117
  all such rows cross the spring DST jump (start dates): {datetime.date(2020, 3, 28): 2, datetime.date(2020, 3, 29): 115}
  (those sessions last one real hour longer than their local wall-clock timestamps suggest)
  distinct users: 4,914; distinct vsetv_id: 202
  duplicate full rows: 17
  DST evidence (views by hour on 2020-03-29): {0: 1740, 1: 938, 2: 424, 4: 290, 5: 334, 6: 627, 7: 1405, 8: 2343, 9: 3339, 10: 3728, 11: 3655, 12: 3600, 13: 3417, 14: 3236, 15: 3490, 16: 3572, 17: 3870, 18: 4585, 19: 4922, 20: 5879, 21: 6284, 22: 5533, 23: 3883}
  DST evidence (views by hour on 2020-03-28): {0: 1600, 1: 738, 2: 468, 3: 374, 4: 340, 5: 460, 6: 1106, 7: 2027, 8: 2833, 9: 2957, 10: 2725, 11: 2538, 12: 2680, 13: 2594, 14: 2674, 15: 2804, 16: 3169, 17: 3510, 18: 4359, 19: 5390, 20: 5906, 21: 5274, 22: 4075, 23: 3103}
Loaded schedule 11-20 in 1.24 s

Schedule 11-20: export_arh_11-20-final.csv
  size on disk : 153.3 MB
  rows         : 1,074,235
  columns      : ['channel_id', 'channel_title', 'start_time', 'duration', 'tv_show_title', 'tv_show_id', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors', 'block']
  dtypes       : channel_id=int64, channel_title=str, start_time=str, duration=int64, tv_show_title=str, tv_show_id=int64, tv_show_category=str, tv_show_genre_1=str, tv_show_genre_2=str, tv_show_genre_3=str, year_of_production=str, director=str, actors=str, block=str
  null counts  : tv_show_genre_1=620,089, tv_show_genre_2=928,009, tv_show_genre_3=1,054,059, year_of_production=959,500, director=974,050, actors=973,861
  first rows   :
   channel_id channel_title           start_time  duration                           tv_show_title  tv_show_id tv_show_category      tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors  block
0           3           1+1  09.03.2020 06:00:00      6000       'Жизнь известных людей', 2 сезон.           0           Другое                  NaN             NaN             NaN                NaN      NaN    NaN  11-20
1           3           1+1  09.03.2020 07:40:00      5400                          'ТСН. Неделя'.     2400480             Инфо  общество и политика             NaN             NaN               1997      NaN    NaN  11-20
2           3           1+1  09.03.2020 09:10:00      8400  'Голос страны', 10 сезон, 8 эп. 'Бои'.      700475      Развлечения               музыка             NaN             NaN               2010      NaN    NaN  11-20
  time format  : schedule = %d.%m.%Y %H:%M:%S (naive local wall clock)
  start range  : 2020-03-08 23:00:00 .. 2020-05-18 11:45:00
  end range    : 2020-03-08 21:04:00 .. 2020-05-18 09:00:00 (UTC)
  ISO week of airing start (rows): W10=86, W11=106,036, W12=108,177, W13=107,363, W14=106,527, W15=108,126, W16=107,697, W17=107,721, W18=107,505, W19=105,473, W20=107,093, W21=2,431
  channels: 400; labelled rows: 115,613 (10.76%); distinct labelled ids: 5,847
  tv_show_id == 0 rows: 958,622 (89.24%)
  schedule duration quantiles: {0.0: 0.0, 0.01: 60.0, 0.5: 1500.0, 0.99: 12600.0, 1.0: 86400.0}
  adjacent same-channel rows: gap=0 100.00%, gap<0 0.00%, median gap 0 s, gaps>600 s 2
Loaded schedule 21-30 in 1.26 s

Schedule 21-30: export_arh_21-30-final.csv
  size on disk : 154.4 MB
  rows         : 1,074,394
  columns      : ['channel_id', 'channel_title', 'start_time', 'duration', 'tv_show_title', 'tv_show_id', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors', 'block']
  dtypes       : channel_id=int64, channel_title=str, start_time=str, duration=int64, tv_show_title=str, tv_show_id=int64, tv_show_category=str, tv_show_genre_1=str, tv_show_genre_2=str, tv_show_genre_3=str, year_of_production=str, director=str, actors=str, block=str
  null counts  : tv_show_genre_1=616,370, tv_show_genre_2=924,020, tv_show_genre_3=1,050,668, year_of_production=954,382, director=969,487, actors=968,834
  first rows   :
   channel_id channel_title           start_time  duration                                           tv_show_title  tv_show_id tv_show_category           tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors  block
0           3           1+1  18.05.2020 06:30:00      1800  'Завтрак с 1+1'. Информационно-развлекательная прог...       20088      Развлечения  информация (комплексная)             NaN             NaN              1996-      NaN    NaN  21-30
1           3           1+1  18.05.2020 07:00:00       600                                                    ТСН.     2400480             Инфо       общество и политика             NaN             NaN               1997      NaN    NaN  21-30
2           3           1+1  18.05.2020 07:10:00      3000  'Завтрак с 1+1'. Информационно-развлекательная прог...       20088      Развлечения  информация (комплексная)             NaN             NaN              1996-      NaN    NaN  21-30
  time format  : schedule = %d.%m.%Y %H:%M:%S (naive local wall clock)
  start range  : 2020-05-18 00:00:00 .. 2020-07-27 11:55:00
  end range    : 2020-05-17 21:03:00 .. 2020-07-27 09:00:00 (UTC)
  ISO week of airing start (rows): W21=104,798, W22=107,274, W23=107,657, W24=107,449, W25=107,104, W26=107,544, W27=106,878, W28=107,528, W29=107,640, W30=108,093, W31=2,429
  channels: 403; labelled rows: 121,505 (11.31%); distinct labelled ids: 6,363
  tv_show_id == 0 rows: 952,889 (88.69%)
  schedule duration quantiles: {0.0: 60.0, 0.01: 120.0, 0.5: 1500.0, 0.99: 13200.0, 1.0: 86400.0}
  adjacent same-channel rows: gap=0 100.00%, gap<0 0.00%, median gap 0 s, gaps>600 s 2
Loaded schedule 31-42 in 1.51 s

Schedule 31-42: export_arh_31-42-final.csv
  size on disk : 186.0 MB
  rows         : 1,295,378
  columns      : ['channel_id', 'channel_title', 'start_time', 'duration', 'tv_show_title', 'tv_show_id', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors', 'block']
  dtypes       : channel_id=int64, channel_title=str, start_time=str, duration=int64, tv_show_title=str, tv_show_id=int64, tv_show_category=str, tv_show_genre_1=str, tv_show_genre_2=str, tv_show_genre_3=str, year_of_production=str, director=str, actors=str, block=str
  null counts  : tv_show_genre_1=737,871, tv_show_genre_2=1,113,598, tv_show_genre_3=1,266,621, year_of_production=1,151,706, director=1,170,189, actors=1,169,519
  first rows   :
   channel_id channel_title           start_time  duration                                           tv_show_title  tv_show_id tv_show_category           tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors  block
0           3           1+1  27.07.2020 06:30:00      1800  'Завтрак с 1+1'. Информационно-развлекательная прог...       20088      Развлечения  информация (комплексная)             NaN             NaN              1996-      NaN    NaN  31-42
1           3           1+1  27.07.2020 07:00:00       600                                                    ТСН.     2400480             Инфо       общество и политика             NaN             NaN               1997      NaN    NaN  31-42
2           3           1+1  27.07.2020 07:10:00      3000  'Завтрак с 1+1'. Информационно-развлекательная прог...       20088      Развлечения  информация (комплексная)             NaN             NaN              1996-      NaN    NaN  31-42
  time format  : schedule = %d.%m.%Y %H:%M:%S (naive local wall clock)
  start range  : 2020-07-27 00:00:00 .. 2020-10-19 11:55:00
  end range    : 2020-07-26 21:03:00 .. 2020-10-19 09:00:00 (UTC)
  ISO week of airing start (rows): W31=105,719, W32=108,256, W33=107,039, W34=106,862, W35=107,028, W36=107,492, W37=107,558, W38=108,258, W39=108,512, W40=108,763, W41=108,843, W42=108,649, W43=2,399
  channels: 404; labelled rows: 144,774 (11.18%); distinct labelled ids: 6,636
  tv_show_id == 0 rows: 1,150,604 (88.82%)
  schedule duration quantiles: {0.0: 60.0, 0.01: 120.0, 0.5: 1500.0, 0.99: 13500.0, 1.0: 43200.0}
  adjacent same-channel rows: gap=0 100.00%, gap<0 0.00%, median gap 0 s, gaps>600 s 3

  DST evidence (schedule 11-20 rows starting 03:00-03:59 on 2020-03-29): 0
  Example channel 3 rows around the jump (01:25 + 6000 s of real time lands at 04:05 local):
         start_time  duration  tv_show_id                tv_show_title
29.03.2020 01:25:00      6000      700369 'Лига смеха', 6 сезон, 3 эп.
29.03.2020 04:05:00      4500      200344                 'Чистоnews'.
29.03.2020 05:20:00      2400     2400480                         ТСН.

==========================================================================================
2. SUBMISSION USERS VS VIEW LOG
==========================================================================================

Submission template: submission.csv
  size on disk : 0.0 MB
  rows         : 1,260
  columns      : ['user_id', 'tv_show_id']
  dtypes       : user_id=uint64, tv_show_id=str
  first rows   :
               user_id tv_show_id
0  8377619604347126107  0 0 0 0 0
1  8381667675275833309  0 0 0 0 0
2  8387147770138767246  0 0 0 0 0
  submission rows: 1,260; distinct users: 1,260
  template tv_show_id tokens: 1 distinct strings (all 1,260 rows are the literal '0 0 0 0 0')
  views outside weeks 11-30: 0; views in test weeks 31-42: 0
  view-log users: 4,914; in 11-20: 4,876; in 21-30: 4,820; in both: 4,782; 11-20 only: 94; 21-30 only: 38
  submission users present anywhere in view log: 1,260 / 1,260
  submission users in 11-20: 1,256; in 21-30: 1,246; in both: 1,242
  submission users 11-20 only: 14; 21-30 only: 4; no view at all: 0
  NOTE: the view log ends 2020-07-26 (week 30); the test schedule starts 2020-07-27 (week 31). There is no who-watched-what data for weeks 31-42.

==========================================================================================
CHANNEL ID NAMESPACE: vsetv_id vs channel_id
==========================================================================================
  view-log vsetv_id values: 202; schedule channel_id values: 405
  intersection: 197; view ids with no schedule: 5 [149, 1174, 1508, 1509, 1510]; schedule ids never viewed: 208
  views on channels absent from the schedule: 113,579 (1.95% of views), 116,312,068 s
  id ranges: view 3..1531, schedule 3..1541
  Interpretation: the two fields use the same numeric channel namespace where the channel exists in the export (197/202 view channels match exactly); 5 view channels are missing from the schedule files. No remapping is applied.

==========================================================================================
SAME CLOCK CHECK: DO VIEWS OVERLAP PROGRAMMES ON THE SAME CHANNEL?
==========================================================================================
  Examples from the first 200,000 log rows (same channel, positive interval overlap):
             user_id  vsetv_id    view_start_local     view_stop_local programme_start_local  programme_dur_s  tv_show_id  overlap_s                            title
15166841293414410417       945 2020-03-09 00:02:58 2020-03-09 00:06:41   09.03.2020 00:02:00              420           0        223              Календарь рыболова.
 9228678650058957216      1499 2020-03-09 00:03:21 2020-03-09 00:04:36   09.03.2020 00:00:00             1800           0         75 Футбол. Премьер-лига Нетбастерс.
11754466009510647262       298 2020-03-09 00:03:33 2020-03-09 00:14:35   09.03.2020 00:00:00             1200           0        662                         Новости.
 7981779138282146703      1495 2020-03-09 00:40:50 2020-03-09 02:28:55   09.03.2020 01:16:00             1380    12001732       1380  М/с 'Пингвины из 'Мадагаскара'.
 9992380986623445456      1495 2020-03-09 01:35:55 2020-03-09 01:37:07   09.03.2020 01:16:00             1380    12001732         72  М/с 'Пингвины из 'Мадагаскара'.
 2874711542322083256      1495 2020-03-09 01:37:49 2020-03-09 01:39:47   09.03.2020 01:16:00             1380    12001732         71  М/с 'Пингвины из 'Мадагаскара'.
  These are consistent with one local clock: programme starts and view starts fall on the same wall-clock scale. Absolute timezone is inferred as Europe/Kyiv.

==========================================================================================
3. JOIN: CHANNEL + INTERVAL OVERLAP, KEEP tv_show_id != 0
==========================================================================================
  Training schedule rows used: 2,148,629 (blocks 11-20, 21-30)
  Join loop runtime: 21.32 s
  views: 5,821,987; total viewing time: 8,619,215,224 s (99759.4 channel-days)
  views with no schedule channel: 113,579 (1.95%), 116,312,068 s
  views with any schedule overlap: 5,408,690 (92.90%)
  views with at least one labelled overlap: 2,964,378 (50.92%)
  view-airing overlap events: 7,752,561
  sum of overlaps / total viewing time: 94.07%
  views whose summed overlap exceeds their duraton (overlapping schedule rows): 9; excess 4,746 s

  Viewing-time allocation across schedule labels:
    tv_show_id != 0 : 4,575,229,505 s (53.08% of total viewing time)
    tv_show_id == 0 : 3,532,473,459 s (40.98% of total viewing time)
    no schedule overlap: 511,512,260 s (5.93% of total viewing time)
    among covered time, tv_show_id == 0 share: 43.57%

  Aggregated (user, programme, block) records with positive labelled overlap: 1,516,199
  distinct users in these records: 4,895
  distinct labelled programmes touched: 4,363
  records per block: 11-20=857,826, 21-30=658,373

  Distribution of aggregated overlap_seconds (one record = user x programme x block):
overlap_seconds: n=1,516,199, mean=3017.565, std=7677.134
  quantile      value
      0.00         1.000
      0.01        38.000
      0.05        71.000
      0.10        91.000
      0.25       220.000
      0.50       969.000
      0.75      3521.000
      0.90      6821.000
      0.95     10079.000
      0.99     30077.000
      1.00   1985904.000

  Distribution of aggregated overlap_fraction = overlap_seconds / sum(duration of airings touched):
overlap_fraction: n=1,516,199, mean=0.234, std=0.286
  quantile      value
      0.00         0.000
      0.01         0.005
      0.05         0.010
      0.10         0.013
      0.25         0.029
      0.50         0.104
      0.75         0.328
      0.90         0.727
      0.95         1.000
      0.99         1.000
      1.00         1.000

  Distribution of airings touched per record:
airings: n=1,516,199, mean=2.538, std=7.108
  quantile      value
      0.00         1.000
      0.01         1.000
      0.05         1.000
      0.10         1.000
      0.25         1.000
      0.50         1.000
      0.75         2.000
      0.90         4.000
      0.95         7.000
      0.99        26.000
      1.00      1850.000

  Distribution of per-view/per-airing overlap / programme duration (events):
event_fraction: n=7,752,561, mean=0.306, std=0.352
  quantile      value
      0.00         0.000
      0.01         0.004
      0.05         0.010
      0.10         0.014
      0.25         0.034
      0.50         0.128
      0.75         0.497
      0.90         1.000
      0.95         1.000
      0.99         1.000
      1.00         1.000

  Candidate absolute thresholds (surviving records, surviving users):
    >=    30 s :  1,504,365 records (99.22%), 4,895 users
    >=    60 s :  1,492,204 records (98.42%), 4,895 users
    >=    90 s :  1,369,336 records (90.31%), 4,892 users
    >=   120 s :  1,291,684 records (85.19%), 4,891 users
    >=   180 s :  1,187,884 records (78.35%), 4,890 users
    >=   300 s :  1,061,129 records (69.99%), 4,890 users
    >=   600 s :    886,403 records (58.46%), 4,888 users
    >=   900 s :    778,567 records (51.35%), 4,883 users
    >=  1800 s :    583,611 records (38.49%), 4,873 users
  Candidate fraction thresholds (records, users):
    >=   1% :  1,440,161 records (94.98%), 4,895 users
    >=   2% :  1,244,049 records (82.05%), 4,892 users
    >=   5% :    977,986 records (64.50%), 4,891 users
    >=  10% :    768,918 records (50.71%), 4,888 users
    >=  20% :    542,087 records (35.75%), 4,881 users
    >=  30% :    407,666 records (26.89%), 4,871 users
    >=  50% :    249,527 records (16.46%), 4,849 users

  PROPOSED watched rule: overlap_seconds >= 120 s AND overlap_fraction >= 5%
    watched records: 973,912 (64.23% of positive records)
    watched users: 4,890
    records dropped: 542,287
    watched records per block: 11-20=528,524, 21-30=445,388
    Justification from the quantiles above:
      - 120 s is 1.3x the p10 of overlap_seconds (91 s) and 0.55x the p25 (220 s); it removes the shortest zapping/boundary overlaps while still fitting short items.
      - 5% is 1.7x the p10 of overlap_fraction (1.3%) and 0.48x the median (10.4%); it removes accidental crossings of long films/airings.
      - The AND rule is protective in both directions: 5% of a 25-min bulletin is only 75 s, and 120 s of a 2-hour film is only 1.7%; neither alone is a good 'watched' test.
      - The rule keeps 64.2% of positive records and 4,890/4,895 users; the surviving set is not dominated by a handful of heavy users.

  Submission users with >=1 labelled interaction: 1,258 / 1,260
  Submission users with >=1 watched record: 1,258 / 1,260
  Submission users with no labelled interaction (cold start at item level): 2

==========================================================================================
4. PROGRAMME SUPPLY AND REPEAT POTENTIAL
==========================================================================================
  distinct labelled programmes in 11-20: 5,847 (+ the tv_show_id=0 bucket = 5,848 distinct id values)
  distinct labelled programmes in 21-30: 6,363 (+ the tv_show_id=0 bucket = 6,364 distinct id values)
  distinct labelled programmes in 31-42: 6,636 (+ the tv_show_id=0 bucket = 6,637 distinct id values)
  Block dates (ISO weeks): 11-20 = Mon 2020-03-09 .. Sun 2020-05-17; 21-30 = Mon 2020-05-18 .. Sun 2020-07-26; 31-42 = Mon 2020-07-27 .. Sun 2020-10-18.
  labelled programmes in training union (11-20 or 21-30): 8,234
  labelled programmes in TEST schedule 31-42: 6,636
  11-20 programmes that also air in 21-30: 3,976 / 5,847 (68.00%)
  11-20 programmes that also air in TEST 31-42: 4,070 / 5,847
  21-30 programmes that also air in TEST 31-42: 4,389 / 6,363
  programmes present in all three blocks: 3,260

  Per-user repeat potential for users with >=1 watched 11-20 programme (n=4,834):
    fraction of watched 11-20 programmes that air again in 21-30:
      repeat_fraction_21_30: n=4,834, mean=0.836, std=0.073
        quantile      value
            0.00         0.000
            0.01         0.667
            0.05         0.724
            0.10         0.750
            0.25         0.793
            0.50         0.836
            0.75         0.880
            0.90         0.923
            0.95         0.950
            0.99         1.000
            1.00         1.000
    fraction of watched 11-20 programmes that air again in TEST 31-42:
      repeat_fraction_31_42: n=4,834, mean=0.865, std=0.065
        quantile      value
            0.00         0.000
            0.01         0.716
            0.05         0.766
            0.10         0.788
            0.25         0.826
            0.50         0.867
            0.75         0.908
            0.90         0.941
            0.95         0.964
            0.99         1.000
            1.00         1.000
  Pooled user-programme pairs: 432,616 / 528,524 (81.85%) of watched 11-20 pairs have the programme airing again in 21-30.
  Submission users with >=1 watched 11-20 programme: 1,250; median repeat fraction into 21-30: 0.829

==========================================================================================
5. EXPORTS
==========================================================================================
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/out/interactions.csv in 0.77 s
    rows=973,912 (watched only; all positive overlaps=1,516,199), size=40.9 MB
    columns=['user_id', 'tv_show_id', 'block', 'overlap_seconds', 'airings']
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/out/programmes.csv in 0.01 s
    rows=9,671, size=1.0 MB
    columns=['tv_show_id', 'title', 'category', 'genres', 'blocks']
    distinct title variants collapsed: 4,351 shows had >1 title
    shows with >1 genre set across airings: 205
    category is consistent within a show: True

==========================================================================================
SUMMARY / UNVERIFIED POINTS
==========================================================================================
  * Same clock: yes for the overlap join; both files are naive local wall clock.
  * Absolute timezone: inferred as Europe/Kyiv from the DST jump; the files do not state a zone. If the platform actually used a fixed-offset zone, only the 117 DST-spanning sessions would be affected.
  * vsetv_id and channel_id use the same numeric namespace where both exist; 5 view channels have no schedule rows and are treated as uncovered time.
  * Week labels mean ISO weeks: W11 starts 2020-03-09, W20 ends 2020-05-17; W21 starts 2020-05-18, W30 ends 2020-07-26; W31 starts 2020-07-27, W42 ends 2020-10-18. Files include small boundary tails from adjacent weeks (W10/W21, W31, W43).
  * No model fitting and no use of weeks 31-42 viewing data: there are 0 such views. The test schedule is used only for programme supply/repeat counts in this step.
  * tv_show_id == 0 is unlabelled airtime: not an item, not a miss; it is counted only to quantify viewing context and dropped from interactions.
  * WATCHED rule is a proposal, not a ground truth: >= 120 s and >= 5% of the summed aired duration of touched airings. Sensitivity counts are printed above.
  * interactions.csv contains the 973,912 watched records only; the 542,287 positive-but-sub-threshold overlaps are not exported. programmes.csv contains all 9,671 labelled ids seen in any schedule.

Total runtime: 53.77 s
