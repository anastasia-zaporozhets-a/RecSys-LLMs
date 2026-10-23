# Final recommendation examples

**Designated final: `submission_v6_3.csv` — private 0.40253 / public 0.41907.**

## Protocol and model

All five recommendation columns below predict **31–42**, using **11–30** training viewing and the 31–42 airing pool. Popularity, history, PPR and v6_3 come directly from the saved final submission files. Plain M2 uses the same catalogue features and seconds-weighted 11–30 profiles, scored only for these three users.

The actual 21–30 and whole-training 11–30 columns are **historical context**, not ground truth for 31–42. In particular, 21–30 is part of the final model's training history. These tables are not held-out validation predictions, and they are not used to calculate MAP@5.

Actual Top-5 means highest summed qualifying watched seconds, ties by programme ID; historical lists are not restricted to the future airing pool. The audited watched-pair rule is at least 120 seconds and at least 5% of summed touched-airing duration per block.

The heavy/light labels refer to the original 11–20 examples. Cold start means no qualifying training programme history; such users receive the same popularity fallback for every method.

v6_3: size 200/source; weekly rates; stacked folds 11–20 → 21–30 and 11–25 → 26–30; channel/time/season/recency features; binary all-watched LambdaRank; five-seed raw-score mean. Saved seed/tree counts: `{"12345": 150, "12346": 112, "12347": 132, "12348": 173, "12349": 275}`.

## Heavy: `13571199046623737884`

Whole 11–30 history: **775 programmes, 9,982,268 seconds**. 21–30 actual: **232 programmes**.

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v6_3 (designated final) | Actual Top-5: 21–30 | Actual Top-5: whole 11–30 history |
|---:|---|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `90068169`<br>Х/ф 'Хэнкок'.<br>(Фильмы) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо)<br>18,605 s | `2400480`<br>ТСН.<br>(Инфо)<br>903,201 s |
| 2 | `240081`<br>Факты.<br>(Инфо) | `20088`<br>'Завтрак с 1+1'. Информационно-развлекательная программа.<br>(Развлечения) | `240081`<br>Факты.<br>(Инфо) | `90045141`<br>Х/ф 'Громобой'.<br>(Фильмы) | `2400480`<br>ТСН.<br>(Инфо) | `200352`<br>'Вечерний квартал'.<br>(Развлечения)<br>17,764 s | `20088`<br>'Завтрак с 1+1'. Информационно-развлекательная программа.<br>(Развлечения)<br>351,085 s |
| 3 | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `90037440`<br>Х/ф 'Инферно'.<br>(Фильмы) | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `90074629`<br>Х/ф 'Американский пирог: Все в сборе'.<br>(Фильмы)<br>10,807 s | `240081`<br>Факты.<br>(Инфо)<br>272,661 s |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `20088`<br>'Завтрак с 1+1'. Информационно-развлекательная программа.<br>(Развлечения) | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы) | `2400508`<br>Сегодня.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо)<br>10,185 s | `2400508`<br>Сегодня.<br>(Инфо)<br>251,641 s |
| 5 | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `90071772`<br>Х/ф 'Центурион'.<br>(Фильмы) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `700369`<br>'Ліга сміху'.<br>(Развлечения)<br>10,024 s | `6500479`<br>'Мир наизнанку'.<br>(Познавательное)<br>245,137 s |

## Light: `8150816946177670084`

Whole 11–30 history: **12 programmes, 23,198 seconds**. 21–30 actual: **11 programmes**.

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v6_3 (designated final) | Actual Top-5: 21–30 | Actual Top-5: whole 11–30 history |
|---:|---|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы) | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы) | `90068169`<br>Х/ф 'Хэнкок'.<br>(Фильмы) | `240081`<br>Факты.<br>(Инфо) | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы)<br>4,345 s | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы)<br>4,345 s |
| 2 | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `90045141`<br>Х/ф 'Громобой'.<br>(Фильмы) | `2400480`<br>ТСН.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо)<br>3,991 s | `240081`<br>Факты.<br>(Инфо)<br>3,991 s |
| 3 | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `90041065`<br>Х/ф 'Доспехи Бога 2'.<br>(Фильмы) | `90041065`<br>Х/ф 'Доспехи Бога 2'.<br>(Фильмы) | `90050977`<br>Х/ф 'Васаби'.<br>(Фильмы) | `200432`<br>Дизель-шоу. Дайджест.<br>(Развлечения) | `90041065`<br>Х/ф 'Доспехи Бога 2'.<br>(Фильмы)<br>3,868 s | `90041065`<br>Х/ф 'Доспехи Бога 2'.<br>(Фильмы)<br>3,868 s |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `90079612`<br>Х/ф 'Зачинщики'.<br>(Фильмы) | `90079612`<br>Х/ф 'Зачинщики'.<br>(Фильмы) | `90037440`<br>Х/ф 'Инферно'.<br>(Фильмы) | `300358`<br>Скетч-шоу 'На троих', 4 сезон.<br>(Развлечения) | `90079612`<br>Х/ф 'Зачинщики'.<br>(Фильмы)<br>2,084 s | `90053098`<br>Х/ф 'Мужество в бою'.<br>(Фильмы)<br>2,753 s |
| 5 | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `90065424`<br>Х/ф 'Мистер Бин на отдыхе'.<br>(Фильмы) | `90065424`<br>Х/ф 'Мистер Бин на отдыхе'.<br>(Фильмы) | `90082810`<br>Х/ф 'Шазам!'.<br>(Фильмы) | `2500413`<br>Гражданская оборона.<br>(Инфо) | `90065424`<br>Х/ф 'Мистер Бин на отдыхе'.<br>(Фильмы)<br>2,082 s | `90079612`<br>Х/ф 'Зачинщики'.<br>(Фильмы)<br>2,084 s |

## Cold start: `16561743585370579180`

Whole 11–30 history: **0 programmes, 0 seconds**. 21–30 actual: **0 programmes**.

All methods use the popularity fallback. No recorded qualifying viewing exists in 21–30 or 11–30, so both actual-watch columns are empty.

| Rank | Popularity | History | PPR alpha 0.3 | Content-based M2 | v6_3 (designated final) | Actual Top-5: 21–30 | Actual Top-5: whole 11–30 history |
|---:|---|---|---|---|---|---|---|
| 1 | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | `2400480`<br>ТСН.<br>(Инфо) | — | — |
| 2 | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | `240081`<br>Факты.<br>(Инфо) | — | — |
| 3 | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `2400467`<br>'Викна-Новости'.<br>(Инфо) | `2400467`<br>'Викна-Новости'.<br>(Инфо) | — | — |
| 4 | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | `2400508`<br>Сегодня.<br>(Инфо) | — | — |
| 5 | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | `6500479`<br>'Мир наизнанку'.<br>(Познавательное) | — | — |
