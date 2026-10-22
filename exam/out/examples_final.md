# Final submission and validation examples

**Designated final submission: `submission_v3_1.csv`.**

## Exact winning configuration

- Configuration: `RRF_ppr_pop(alpha=0.5,weights=1.0/0.5)`.
- Reciprocal-rank fusion: `k=60`, one-based ranks, top 200 items from each input ranking.
- PPR: alpha **0.5**, weight **1.0**. Bipartite user–programme graph, edge weight `log1p(overlap_seconds / 60)`, row-normalized transitions; restart on the user's own programmes weighted by those edges. `R <- 0.5 * P.T @ R + V`, with `V = 0.5 * normalized_restart`; max 30 iterations, tolerance 1e-6, chunks of 500.
- Popularity: weight **0.5**, ranked by distinct training viewers, restricted to the airing pool.
- History weight **0.0**; no airtime adjustment (beta **0.0**).
- Fused score: `1 / (60 + rank_PPR) + 0.5 / (60 + rank_popularity)`; an absent item contributes zero for that input. Score ties use ascending programme ID.
- Exactly five distinct IDs; popularity fill and popularity-only cold-start fallback; already-watched programmes allowed.
- Validation MAP@5, **all watched**: **0.6517215392095576**.
- Validation MAP@5, **top-five-by-time target**: **0.2058911578801139**.
- Validation fits on 11–20, predicts the 21–30 airing pool, and averages over 4,799 users. Selection uses all-watched MAP@5. Final submissions refit on 11–30 and use the 31–42 airing pool.

## Kaggle results

User-reported public/private scores (screenshots saved).

| Submission / method | Public | Private |
|---|---:|---:|
| submission_v3_1.csv (designated final) | 0.33173 | 0.31829 |
| submission_v3_2.csv | 0.32686 | 0.31413 |
| submission_v3_3.csv | 0.32539 | 0.31550 |
| PPR alpha 0.3 | 0.30721 | 0.27476 |
| Popularity | 0.29788 | 0.29442 |
| History + airtime (final_v2) | 0.28189 | 0.25260 |
| History | 0.28079 | 0.25013 |

## Submission validation

All files use the 31–42 airing pool; user IDs are read as strings.

- `submission_v3_1.csv`: **PASS** — 1,260 rows, template order, unique users, five distinct nonzero in-pool IDs per row.
  SHA-256: `73a6a63a21779c70f8d733dace9c9be4d72c9020e5ca67aa421eceb9fc11885a`
- `submission_v3_2.csv`: **PASS** — 1,260 rows, template order, unique users, five distinct nonzero in-pool IDs per row.
  SHA-256: `f4ecb5ea5f3122a56d29b144e80437915862a4833779186c6b36d9b4d25a4f14`
- `submission_v3_3.csv`: **PASS** — 1,260 rows, template order, unique users, five distinct nonzero in-pool IDs per row.
  SHA-256: `63a14c8d2ef61cbfc0b3eb1ffbca0685f4e1aa904155a21c2c340c40ff369eaa`

## Side-by-side validation examples

All recommendations below are trained on 11–20 and restricted to programmes airing in 21–30. Actual Top-5 is ranked by 21–30 overlap seconds (ties by programme ID). These are validation predictions, not rows from the 31–42 submission.

### Heavy: `13571199046623737884`

11–20 history: 596 programmes, 9,396,297 seconds. 21–30 actual: 232 programmes.

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v3_1: PPR 0.5 + popularity | Actual Top-5 watched in 21–30 |
|---:|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `90037440`<br>Х/ф &#x27;Инферно&#x27;.<br>(Фильмы) | `2400480`<br>ТСН.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) |
| 2 | `240081`<br>Факты.<br>(Инфо) | `20088`<br>&#x27;Завтрак с 1+1&#x27;. Информационно-развлекательная программа.<br>(Развлечения) | `240081`<br>Факты.<br>(Инфо) | `90045141`<br>Х/ф &#x27;Громобой&#x27;.<br>(Фильмы) | `240081`<br>Факты.<br>(Инфо) | `200352`<br>&#x27;Вечерний квартал&#x27;.<br>(Развлечения) |
| 3 | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `90050977`<br>Х/ф &#x27;Васаби&#x27;.<br>(Фильмы) | `2400508`<br>Сегодня.<br>(Инфо) | `90074629`<br>Х/ф &#x27;Американский пирог: Все в сборе&#x27;.<br>(Фильмы) |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `90068169`<br>Х/ф &#x27;Хэнкок&#x27;.<br>(Фильмы) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) |
| 5 | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `20088`<br>&#x27;Завтрак с 1+1&#x27;. Информационно-развлекательная программа.<br>(Развлечения) | `90071772`<br>Х/ф &#x27;Центурион&#x27;.<br>(Фильмы) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `700369`<br>&#x27;Ліга сміху&#x27;.<br>(Развлечения) |

### Light: `8150816946177670084`

11–20 history: 1 programmes, 2,753 seconds. 21–30 actual: 11 programmes.

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v3_1: PPR 0.5 + popularity | Actual Top-5 watched in 21–30 |
|---:|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `90053098`<br>Х/ф &#x27;Мужество в бою&#x27;.<br>(Фильмы) | `90053098`<br>Х/ф &#x27;Мужество в бою&#x27;.<br>(Фильмы) | `90036309`<br>Х/ф &#x27;Военные потери&#x27;.<br>(Фильмы) | `2400480`<br>ТСН.<br>(Инфо) | `90050977`<br>Х/ф &#x27;Васаби&#x27;.<br>(Фильмы) |
| 2 | `240081`<br>Факты.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `90081821`<br>Х/ф &#x27;Подлежит уничтожению&#x27;, 1-4 с.<br>(Фильмы) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) |
| 3 | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `90053098`<br>Х/ф &#x27;Мужество в бою&#x27;.<br>(Фильмы) | `2400508`<br>Сегодня.<br>(Инфо) | `90041065`<br>Х/ф &#x27;Доспехи Бога 2&#x27;.<br>(Фильмы) |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `90038879`<br>Х/ф &#x27;Щит и меч&#x27;, 1 с.<br>(Фильмы) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `90079612`<br>Х/ф &#x27;Зачинщики&#x27;.<br>(Фильмы) |
| 5 | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `90079336`<br>Х/ф &#x27;Второе дыхание&#x27;, 1-4 с.<br>(Фильмы) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `90065424`<br>Х/ф &#x27;Мистер Бин на отдыхе&#x27;.<br>(Фильмы) |

### Cold start: `16561743585370579180`

11–20 history: 0 programmes, 0 seconds. 21–30 actual: 0 programmes.

Cold start in 11–20: all five methods use the same popularity fallback.

No recorded 21–30 viewing: the actual Top-5 is unavailable, and this user is excluded from validation MAP@5 (which requires a nonempty target).

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v3_1: PPR 0.5 + popularity | Actual Top-5 watched in 21–30 |
|---:|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | — |
| 2 | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | — |
| 3 | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | `2400467`<br>&#x27;Викна-Новости&#x27;.<br>(Инфо) | — |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | — |
| 5 | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | `6500479`<br>&#x27;Мир наизнанку&#x27;.<br>(Познавательное) | — |
