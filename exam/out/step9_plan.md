# Step 9 planning log — 2026-10-24

Status: **implementation and synthetic checks only; the long job has not been run**.

## Idea attribution

The **per-airing coverage definition** and the **clipped airtime multiplier**
come from a description of his approach given to the user by classmate
**Vladimir Kuznetsov**. **No code or files were shared.**

This attribution is printed by the job and stored in its cache/result metadata.

## Deadline and command

Finish/upload deadline: **15:00 today, Europe/Kyiv**. Default upload reserve:
10 minutes, so computation stops at **14:50**.

Run from `exam/`:

```bash
.venv/bin/python -u src/step9_coverage.py --deadline 15:00 --timezone Europe/Kyiv --reserve-minutes 10
```

The user performs the Kaggle upload. Progress and skip reasons are written live
to `out/step9_stdout.log`; status is saved to `out/step9_result.json`.

## Computation plan

- Reuse Step 7 cached users, holdout assignment, candidate features, popularity
  and relevant sets. New exact coverage cannot be recovered from older summed
  overlaps, so raw intervals are unioned once and persisted under `out/cache/step9/`.
- Clip raw viewing strictly before 31–42; join only the 11–20 and 21–30 schedules
  for coverage. Target 31–42 supplies scheduled airings only.
- Sparse coverage storage materializes positive user/airing overlaps; omitted
  rows have coverage zero. Full airing duration is the denominator even at
  window boundaries. Mean coverage is conditional on some viewing; an overall
  scheduled-airing mean is cached separately. ID 0 is never an item.
- Write/check `submission_v9_freq.csv` before starting LTR, and compare its
  all-watched fold10/fold5 MAP@5 with the saved v6_3 holdout scores.
- Initial conservative C estimate: `3 * 618.8 + 60 = 1916.4` seconds, about
  **31.94 minutes**, using the earlier complete feature-rebuild/five-refit log
  as an upper-bound-style reference, not a measured Step 9 runtime.
- C runs only if its estimate fits both the **40-minute cap** and time left
  before the upload reserve. Pilot fit timings update the projection; per-tree
  checks stop work on the cap/deadline. A partial ensemble is never written.
- C evaluates all-watched versus coverage80 fitting labels, always selecting
  by the mean of both **all-watched** holdout MAP@5 scores. User holdouts remain
  excluded from supervised fitting in both folds, followed by a 100% refit.
- If C is skipped, the log says so explicitly and the frequency file is ready.

Current designated final remains `submission_v6_3.csv` until new Kaggle results.
