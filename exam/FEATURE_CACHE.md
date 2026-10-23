# Reusable Step 7 feature caches

Build/reuse them with:

```bash
.venv/bin/python -u src/step7_refit_all_together.py
```

This command rebuilds **missing features only**, then refits the fixed
`all_together` recipe on 100% of both training folds using the saved five
tree counts. It does not rerun recall selection, ablations or early stopping.
Output: `submission_v6_3.csv`; live log: `out/step7_v6_3_refit.log`.

## Layout

`out/cache/` is ignored by Git through the existing `out/` rule.

```text
out/cache/step7/
  fold10/                   # 11-20 features -> 21-30 labels
  fold5/                    # 11-25 features -> 26-30 labels
  final/                    # 11-30 features -> 31-42 candidates, no viewing labels
  windows/{train10,train15,train20,label5}/
  user_split.json           # original Step 7 user-level holdout assignment
```

Each fold directory contains:

- `base_features.npz`: CSR matrix of the original v5 features, in raw units.
- `new_features.npz`: CSR matrix of the five Step 7 additions.
- `aligned_arrays.npz`: row user codes/item IDs, candidate depths, binary labels,
  training airtime, relevant sets, pool, cold users, and ordered popularity.
  User IDs are stored as strings, never floating point. Relevant sets retain
  out-of-pool items so evaluation denominators remain correct.
- `training_pairs.csv.gz`: full fold training pairs for histories and fallback.
- `manifest.json`: feature names, category mapping, window lengths, row counts,
  cache version, source fingerprint and role descriptions.

All files use non-pickle formats. Sparse features and numeric row arrays
are aligned by row position. There are at most 200 candidates per source
(history, PPR 0.5, E48_uniform, popularity), up to 800 distinct items per user.
Raw habit/airing matrices are not stored; their computed candidate features
are stored. Final labels are zero placeholders and are never used for fitting.

Window directories store compressed pair/channel aggregate tables, allowing
interrupted rebuilds to resume without another raw overlap-join pass.
Completed entries are published atomically. Cache loading checks the schema,
input sizes/mtimes and feature-code SHA-256 digests. An incompatible cache is
refused; `--cache-dir` can point to a new cache root.

## Step 8 reuse without feature recomputation

From a script in `src/`, reuse the existing loaders and matrix transforms:

```python
import harness as H
import step7_feature_cache as C
import step7_ltr_plus as P

root = H.OUT / "cache" / "step7"
fingerprint = C.source_fingerprint()
folds = [C.load_fold(root / name, fingerprint) for name in ("fold10", "fold5")]
final = C.load_fold(root / "final", fingerprint)
config = P.recipes(200)["all_together"]

# Applies the existing weekly-rate transform to cached feature values.
X, y, groups, names = P.training_queries(folds, config, users=None)
indices = P.indices_for(final, config)
X_final, final_names = P.design(final, config, indices)
assert names == final_names
```

This path loads cached arrays and applies matrix-level transforms only: no
view-log join, PPR, exposure computation or new candidate generation. For
held-out evaluation, pass the original training-user set from `user_split.json`
instead of `users=None`.

## Step 8 additions and resumable fits

Run the new experiment with:

```bash
.venv/bin/python -u src/step8_ltr_graded.py
```

It reads the Step 7 caches without modifying their feature code or data.
Additional features and quartile labels are cached separately under
`out/cache/step8/features/{fold10,fold5,final}/`:

- `additional.npz`: E168 half-life-4/8 exposure, first/second-target-half
  exposure, and E48 rank within the existing candidate set.
- `labels.npz`: binary, airing-quartile and day-quartile labels plus per-user
  quartile cutpoints. Final relevance arrays are unused zero placeholders.
- `manifest.json`: alignment/source fingerprints, feature names and tie rule.

The new time features require one training-only raw-log pass on the first run.
PPR and candidate generation are reused. Subsequent runs load these additions.
Decay uses exact interval-integrated weights divided by effective weighted
training weeks; half-schedule exposure uses each half's week count. Zero-E
items receive candidate rank `candidate_count + 1`. Channel share is reused
from the existing base features rather than duplicated.

Grading uses all actual watched items, not only retrieved candidates. Quartile
boundary ties take the lower bucket; missing clipped counts remain in the
lowest positive grade. Fold10 counts are obtained from cached train20 minus
train10 aggregates; fold5 uses cached label5 aggregates directly.

Completed seed fits persist under `out/cache/step8/seed_fits/`, namespaced by
the experiment, split and feature fingerprints. They include best iterations
and held-out predictions, allowing interrupted grid searches to resume.
The predeclared search evaluates at most 24 distinct five-seed configurations
(duplicate controls/configurations reuse fits). Progress is written live to
`out/step8_stdout.log`; trials, selection, importances and final configurations
are saved as `out/step8_*` files. The selected models are in `out/step8_models/`.
