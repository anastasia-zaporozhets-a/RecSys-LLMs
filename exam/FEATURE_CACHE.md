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
