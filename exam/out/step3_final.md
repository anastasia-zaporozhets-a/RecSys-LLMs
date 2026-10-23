
====================================================================================================
STEP 3 - FINAL FIT (11-20 + 21-30) AND PREDICTION FOR WEEKS 31-42
====================================================================================================
Environment : Python 3.14.0 macOS-26.6.2-arm64-arm-64bit-Mach-O; numpy 2.5.4; pandas 3.0.6
Seed        : 12345 (no stochastic component)
Method      : ppr (alpha=0.3)
  training interactions (11-20 + 21-30): 883,931 user-programme pairs, 4,890 users
  candidate pool (labelled programmes airing in 31-42): 6,636
  submission users: 1,260
  submission users with no training interaction (cold start): 2 -> popularity baseline
  popularity baseline (distinct training users per programme, pool-restricted): top-5 = [2400480, 240081, 2400467, 2400508, 6500479]
  fallback users served by the popularity baseline: 2
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/out/test_ppr.csv (1,260 rows)
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/out/submission_final.csv (Kaggle format: user_id + 5 space-separated ids)

====================================================================================================
EXAMPLE USERS (same two across every method)
====================================================================================================
  heavy viewer : 13571199046623737884 (training seconds = 9,982,268; 775 programmes)
  light viewer : 8150816946177670084 (training seconds = 23,198; 12 programmes)

  ppr final (31-42 pool) - heavy viewer | user 13571199046623737884:
    1. id=2400480      cat=Инфо             title=ТСН.
    2. id=240081       cat=Инфо             title=Факты.
    3. id=2400508      cat=Инфо             title=Сегодня.
    4. id=20088        cat=Развлечения      title='Завтрак с 1+1'. Информационно-развлекательная программа.
    5. id=6500479      cat=Познавательное   title='Мир наизнанку'.

  ppr final (31-42 pool) - light viewer | user 8150816946177670084:
    1. id=90050977     cat=Фильмы           title=Х/ф 'Васаби'.
    2. id=240081       cat=Инфо             title=Факты.
    3. id=90041065     cat=Фильмы           title=Х/ф 'Доспехи Бога 2'.
    4. id=90079612     cat=Фильмы           title=Х/ф 'Зачинщики'.
    5. id=90065424     cat=Фильмы           title=Х/ф 'Мистер Бин на отдыхе'.

  popularity fallback - cold-start user | user 16561743585370579180:
    1. id=2400480      cat=Инфо             title=ТСН.
    2. id=240081       cat=Инфо             title=Факты.
    3. id=2400467      cat=Инфо             title='Викна-Новости'.
    4. id=2400508      cat=Инфо             title=Сегодня.
    5. id=6500479      cat=Познавательное   title='Мир наизнанку'.

Total runtime: 24.9s
