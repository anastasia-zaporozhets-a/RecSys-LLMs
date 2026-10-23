# Final rank-average ensemble

**Selection note: this ensemble was chosen after seeing Kaggle scores.**

- Components: submission_v6_3, submission_v7_1, submission_v7_2; equal weights 1/1/1.
- Score: sum of `1 / (60 + rank)` across the three files, with one-based ranks.
- Ties: v6_3 order first; items absent from v6_3 follow those present, with ascending ID as the remaining tie-break.
- Exact rational scores; five distinct IDs; no model training.

## Source validation
- `submission_v6_3.csv`: PASS — 1,260 ordered unique users; five distinct nonzero in-pool IDs per row.
  SHA-256: `411bca1e61ee4b84a960f85bcb937939ab2613214c1c29ecb5207e4677d7cb20`
- `submission_v7_1.csv`: PASS — 1,260 ordered unique users; five distinct nonzero in-pool IDs per row.
  SHA-256: `2e4b8b4aee92eb3b39c68fadc206aed8ea727a649368eba01ba5e4882aa9e37d`
- `submission_v7_2.csv`: PASS — 1,260 ordered unique users; five distinct nonzero in-pool IDs per row.
  SHA-256: `469354257ffa10c77155cd2d9b6564bde6ce2ef65d372f475d28b4891494b66d`

## Output checks
```text
  wrote /Users/anastasiazaporozec/Desktop/RecSys-LLMs/exam/submission_v8_ens.csv
    rows: 1260
    template_rows: 1260
    five_ids_every_row: True
    distinct_ids_every_row: True
    all_ids_in_pool: True
    order_matches_template: True
    zero_ids: 0
    distinct_programmes_used: 135
    pool_size: 6636
```
SHA-256: `aebe6792e6bfa4df90d128858761c091d7498b9f2d667d90177d861727cf1e48`

## Diversity versus inputs
- `submission_v6_3.csv`: 568/1,260 user rankings differ from the ensemble.
- `submission_v7_1.csv`: 302/1,260 user rankings differ from the ensemble.
- `submission_v7_2.csv`: 247/1,260 user rankings differ from the ensemble.
