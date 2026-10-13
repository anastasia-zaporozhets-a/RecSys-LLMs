# HW5 — From Ratings to Recommendations (Matrix Factorization)

**Course**: LLM4Rec, HSE University
**Instructor**: Seungmin Jin (sedzhin@hse.ru)
**Repository**: https://github.com/dryjins/RecSys-LLMs/tree/main/week5
**Task type**: Assignment (individual)

---

## 1. Learning goals

By the end of this assignment you should be able to:

- Turn a sparse rating log into a **user–movie rating matrix** and explain why an
  unobserved rating is **missing data, not a zero**.
- Split ratings **per user, by timestamp**, into an **80% train / 10% validation /
  10% test** protocol that mimics "predict the future from the past".
- Implement **matrix factorization** with **16-dimensional** user and movie
  vectors and a **dot-product** prediction rule.
- Train the factors with **batched tensor operations in TensorFlow.js**, using
  an optimizer (`tf.train.adam` / `tf.train.sgd`) instead of hand-written
  per-rating JavaScript loops.
- Select a checkpoint by **validation RMSE** and report **test RMSE** against a
  **training-set mean-rating baseline**.
- Generate **Top-10 recommendation lists** that exclude movies the user already
  rated in training or validation.
- Evaluate retrieval with **Recall@10**, treating test ratings **>= 4.0** as
  relevant, and report the evaluation **exclusions** that make the number
  interpretable.
- Reason about whether a recommender helps users **discover** new movies or mainly
  **repeats** their existing preferences, and what that means for a streaming
  platform.

---

## 2. Run instructions

Serve the folder over HTTP and open `week5/index.html` in a modern browser:

```bash
cd week5
python3 -m http.server 8000
# then open http://localhost:8000/
```

The page needs an **HTTP server** (because `data.js` uses `fetch()` to read
`u.data` and `u.item`) and **network access** on first load (the browser fetches
the TensorFlow.js CDN script). It also runs on static hosting such as GitHub
Pages, which serves files over HTTP.

Keep `week5/u.data`, `week5/u.item`, `week5/data.js`, `week5/style.css`, and
`week5/script.js` next to `index.html`. Press **Run tests** to execute the
self-checks, fill in the six `TODO(hw5)` stubs, then press **Train**,
**Recommend Top-10**, and **Evaluate Recall@10**. The backend that TensorFlow.js
selected is shown in the **TensorFlow.js status** panel (`#tf-status`).

---

## 3. Dataset provenance

- **Source**: MovieLens 100k (GroupLens / University of Minnesota).
  - Dataset page: `https://grouplens.org/datasets/movielens/100k/`
- **Files used** (byte-identical to Week 2 and Week 3, so the **algorithm** is the
  only variable between the assignments):
  - `u.data` — **100,000 ratings**, tab-separated
    `userId | movieId | rating | timestamp`.
  - `u.item` — **1,682 movies**, pipe-separated
    `id | title | release | video | url | 18 genre flags`.
- **Shapes in this dataset**: **943 users**, **1,682 movies**, **100,000 ratings**.
  The highest user id is **943** and the highest movie id is **1,682**, so the
  dense rating matrix is `(943 + 1) x (1682 + 1)`.
- **Timestamps**: every rating in `u.data` carries a Unix timestamp; the split in
  `data.js` sorts each user's ratings by ascending timestamp (ties broken by
  `movieId`).
- **Citation** (as required by the source): F. Maxwell Harper and Joseph A.
  Konstan, "The MovieLens Datasets: History and Context", *ACM Transactions on
  Interactive Intelligent Systems*, 2015.

---

## 4. Tensors and GPU acceleration

### 4.1 What TensorFlow.js is

**TensorFlow.js** is a JavaScript library for **tensor computation and automatic
differentiation**: tensors are multi-dimensional arrays, operations over them are
expressed as library calls, and each operation records enough information to
compute gradients automatically. It runs entirely **in the browser** and can use
the **WebGL** backend to execute tensor operations on the **GPU**, falling back
to the pure-JavaScript **CPU** backend when WebGL is unavailable.

`week5/index.html` already loads the runtime from a CDN before the page scripts:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="data.js"></script>
<script src="script.js"></script>
```

Because this is a plain (classic) script load, TensorFlow.js exposes a global
**`tf`** object that `week5/script.js` can use directly — no module import is
needed.

### 4.2 Use TensorFlow.js (required)

Implement the **matrix-factorization training** and the **batched scoring** with
TensorFlow.js tensor operations rather than hand-written JavaScript loops over
rating arrays. TensorFlow.js is the GPU-accelerated path expected for this
assignment. The six stubs expect these APIs:

| Purpose | API |
|---|---|
| Learnable factor matrices | `tf.variable` |
| Small random initialization | `tf.randomNormal(shape, mean, std)` |
| Gather rows for a batch of user/movie ids | `tf.gather` |
| Score many (user, movie) pairs at once | `tf.matMul` (and element-wise `mul` / `sum`) |
| Loss for a training batch | `tf.losses.meanSquaredError` |
| Optimizer and in-place update | `tf.train.adam` / `tf.train.sgd` + `optimizer.minimize` |
| Top-K over all item scores | `tf.topk` |
| Automatic release of intermediate tensors | `tf.tidy` |

The result is the same algorithm as a handwritten implementation, but each
mini-batch is one tensor operation executed on the GPU where possible.

### 4.3 Practical points

- **Select the backend and report it.** `initTf()` in `script.js` prefers WebGL,
  falls back to CPU, and writes the chosen backend into `#tf-status` via
  `tf.getBackend()`. State the printed backend in your report: if it says `cpu`,
  the GPU path was not used and your timings are host-CPU numbers.
- **Await readiness once.** Call `await tf.ready()` before the first training
  step so the backend is fully initialized (the provided `initTf()` already does
  this).
- **Manage memory.** Wrap per-step work in `tf.tidy` and **dispose** long-lived
  tensors you no longer need. TensorFlow.js tensors live outside the JavaScript
  heap; undisposed tensors leak GPU memory, especially inside an epoch loop.
  Re-initializing the model (repeated **Train** clicks) must also dispose the
  previous `tf.Variable`s.
- **The first CDN load needs network access.** On a machine without network the
  `tf` global is undefined; the page shows a clear message instead of throwing
  (see `initTf()`).

---

## 5. Required student work

### 5.1 Implementation Task

- **Dataset**: MovieLens. Use `userId` and `movieId` to identify users and movies;
  use `title` and `genres` to describe movies. Reuse the same `u.data` / `u.item`
  files as Week 2 and Week 3.
- **Data preparation**: build a user–movie rating matrix. Treat missing ratings as
  **unobserved, not zeros**.
- **Data splitting**: sort each user's ratings by **timestamp** and use
  **80% train / 10% validation / 10% test** (provided by `data.js`).
- **Matrix factorization**: implement JavaScript matrix factorization with
  **16-dimensional** user and movie vectors; the predicted rating is their
  **dot product**.
- **Model evaluation**: select the checkpoint using **validation RMSE**. Compare
  **test RMSE** against a **training-set mean-rating baseline**.
- **Recommendation**: generate **Top-10** lists for **two users**. Show movie
  titles and scores, and **exclude movies rated in training or validation**.
- **Retrieval evaluation**: treat test ratings **>= 4.0** as relevant and report
  **Recall@10** using **training-set movies as candidates**. Report the
  **evaluation exclusions** for unseen ids (relevant test movies absent from the
  training pool) and for users without an eligible relevant test movie.

You implement **six stubs**, each marked `TODO(hw5)`, in `week5/script.js`:
`initializeFactors`, `trainMF`, `predictRating`, `selectBestCheckpoint`,
`recommendTopK`, and `recallAtK`.

### 5.2 Business & Algorithmic Analysis (exactly ONE question)

Answer **only** the following question (in the course report, using numbers and
titles produced by your own code):

**Familiar Choices vs. New Discoveries.** Do your recommendations help users
**discover** movies, or do they mainly **repeat** their existing preferences?
Compare two users' Top-10 lists with their past ratings. Use specific movies or
genres to explain what the lists are worth to a streaming platform.

---

## 6. Definitions

Let a rating be `r(u, i)`, the rating user `u` gave movie `i`.

The model stores a **global training mean** `mu` and two **16-dimensional** factor
vectors per id: `p_u` for user `u` and `q_i` for movie `i`. The predicted rating is

```
r_hat(u, i) = mu + p_u . q_i
```

where `.` is the dot product. The result is clamped to `[1, 5]`.

The dense factor matrices are indexed by the **raw** user and movie ids:

- `userFactors`: shape `[numUsers + 1, 16]` = `[944, 16]`, row index = `userId`.
- `itemFactors`: shape `[maxMovieId + 1, 16]` = `[1683, 16]`, row index = `movieId`.
- `globalMean`: the mean rating over the **training** split.

**RMSE** (root-mean-square error) over a split of `N` ratings:

```
RMSE = sqrt( (1 / N) * SUM_over_ratings ( r_hat(u, i) - r(u, i) )^2 )
```

**Mean-rating baseline**: predict the training-set mean `mu` for every pair and
compute the RMSE on the same split. It is the number your factorization must beat.
The provided `computeBaselineRMSE` implements it.

**Recall@K**: with **relevant** = test ratings with `rating >= 4.0` and
**candidates** = the set of movie ids appearing anywhere in the training split,
for each evaluable user

```
recall_u = ( number of that user's relevant candidate movies in the Top-K list ) / ( number of that user's relevant candidate movies )
```

and `Recall@K` is the **mean of `recall_u` over the evaluable users**. Movies
rated in training or validation are excluded from a user's list, so a relevant
test movie that the user already rated in training cannot be recommended (and is
counted in the **unseen ids** exclusion report).

---

## 7. Implementation & analysis tasks

1. Read `week5/script.js` and identify the six `TODO(hw5)` stubs:
   `predictRating`, `initializeFactors`, `trainMF`, `selectBestCheckpoint`,
   `recommendTopK`, and `recallAtK`. The rest of the file (TensorFlow.js
   bootstrap, `computeRMSEFor`, `computeBaselineRMSE`, formatting helpers, all DOM
   renderers, the hyper-parameter reader, the pipeline wrapper, and the test
   harness) is scaffolding and should be left as-is.
2. Run the page, press **Run tests**, and record the baseline counts. With the
   stubs untouched the harness reports **4 passed / 0 failed / 7 pending** (the
   provided helpers pass; the seven stub-backed checks are reported as PENDING
   rather than FAIL). After a correct implementation all **11** checks pass.
3. Implement the six stubs with TensorFlow.js tensor operations, following the
   API guide in §4.2 and the per-stub contract notes in the code.
4. Press **Train** with the default hyper-parameters and report the **validation
   RMSE** at the selected checkpoint and the **test RMSE** of that checkpoint.
   Compare the test RMSE with the **training-set mean-rating baseline** from
   `computeBaselineRMSE(testSet)` and state whether the factorization beats it.
5. Generate the **Top-10** list for **two users** and record each movie's **title**
   and **score**. Confirm that none of the listed movies was rated by that user in
   training or validation.
6. Evaluate **Recall@10** on the test split and report the overall recall together
   with the **exclusions**: the number of unseen relevant test items and the
   number of users without a relevant candidate.
7. Answer the single **§5.2** analysis question, comparing the two users' Top-10
   lists with their past ratings and citing specific movies or genres.
8. Verify every citation you use (paper, URL, authors, year) before submitting,
   and export the unmodified `session.json` for the report (see §8).

---

## 8. Submission instructions

Submit the **modified `week5/` directory** (or your own repository) containing the
files below, plus the course report.

| File | Role |
|---|---|
| `week5/script.js` | your implementation of the six `TODO(hw5)` stubs |
| `week5/data.js` | data layer (provided, do not modify) |
| `week5/index.html` | page structure (provided, do not modify) |
| `week5/style.css` | styling (provided, do not modify) |
| `week5/readme.md` | this file |

Plus:

- `report.pdf` — IEEE-aligned report per the course homework guidelines
  (`docs/homework-guidelines/guidelines.md`), answering §5.2.
- `session.json` — the **unmodified** export from your AI tool
  (`opencode export <sessionID>`).

- **No Jupyter notebook** (`.ipynb`) is part of this deliverable.
- **No separate memo** is required; the analysis belongs in the report.
- **Late submissions are not accepted.**

---

## 9. Grading criteria

Grading follows the course homework guidelines, **§8 — Rubric Criteria in Detail**
(`docs/homework-guidelines/guidelines.md`). The two criteria are **binary** (0 or
1) and combine into a per-assignment score of **0, 1, or 2**:

- **c1 — Problem & References.** Clear problem statement; all references valid;
  attribution accurate. A hallucinated citation, a broken reference URL, or a
  fabricated author/year is a **Gate 0** failure of c1.
- **c2 — Solution & Evidence.** The solution works (the page runs and the numbers
  shown match the code's own output); the reasoning is accurate; and verification
  is cited (for example a hand-computed RMSE on one pair, a re-run, a source read
  beyond the abstract, or a cross-check of a reported number against the code).

Read **§10** of the same guidelines for the **Gate 0** failure modes that apply
most directly here:

- a **hallucinated citation** (a paper or URL that does not exist);
- **fabricated verification** — claiming you hand-computed or re-ran something you
  did not;
- a **broken solution** — the page does not run, or the numbers shown do not match
  the code's own output;
- a **hallucinated API call** — reporting use of a TensorFlow.js function that
  does not exist or is not actually used.

If you cannot verify a critical output, say so honestly in the report's AI-usage
section rather than claiming verification you did not perform.

---

*Generated 2026-10-06 from HW5 work order.*

---

## 10. Corrected prompt (A05)

This section documents the fixes applied to the original assignment prompt to resolve the defects identified in the audit.

### 10.1 Per-batch L2 regularization on gathered rows only
The original suggested `trainMF` loss applied L2 over the **full** user and item factor matrices (`model.userFactors.square().sum() + model.itemFactors.square().sum()`) once per mini-batch. This caused the regularization to dominate the loss (≈3.6× batch MSE at initialization) and drove all factors to zero within the first fraction of an epoch, making the model collapse to the global-mean predictor.

**Fix**: L2 is now computed **only on the factor rows gathered for the current batch** and averaged over the batch size:
```js
const l2 = u.square().sum().add(i.square().sum()).div(batch.userIdx.shape[0]).mul(reg);
```
where `u = tf.gather(model.userFactors, batch.userIdx)` and `i = tf.gather(model.itemFactors, batch.itemIdx)`. This matches the intended semantics of per-sample regularization and prevents the collapse.

### 10.2 Best-checkpoint restore and test RMSE of the restored model
The original `runPipeline` selected the best epoch by validation RMSE but never restored those weights — the model held after training was the last epoch's weights. The status line only showed the best validation RMSE, not the test RMSE of the checkpoint actually used.

**Fix**: During training, when a new best validation RMSE is observed, clones of `userFactors` and `itemFactors` are saved. After the last epoch, the best clones are assigned back into the model variables with `assign()`, so the model held after `runPipeline` is exactly the selected checkpoint. The status line now reports:
- Backend used (WebGL/CPU)
- Restored epoch number and its validation RMSE
- **Test RMSE of the restored model** (via `computeRMSEFor(testSet)`)
- Baseline validation and test RMSE for comparison

### 10.3 Sparsity over real IDs (943 × 1682)
The original readme stated the dense matrix as `(943 + 1) × (1682 + 1)` = 944 × 1683, including the unused row/column 0. Sparsity computed over this padded matrix is 93.71%.

**Correction**: The real ID space is 943 users × 1682 movies = 1,586,126 cells. With 100,000 observed ratings:
- Density = 6.3047%
- Sparsity = 93.6953%

### 10.4 Candidate-pool conflict with checks 9/10 and its resolution
The readme §5.1/§6 requires `recommendTopK` to use **training-set movies as candidates** (Recall@10 candidate pool). However, self-checks 9 and 10 use fixtures where the test-rated movie (id 3 in check 9, id 2 in check 10) is **not in the training set**, and non-training movies (ids 4–12 in check 10) must be scored to fill the Top-K and push the relevant item out at rank 11. A strict training-pool-only implementation would fail both checks.

**Resolution**: The implementation follows the self-checks (which are the executable contract) and scores **all movies present in `movies[]`**, excluding only (a) the user's train/val items, (b) duplicate-title copies of movies the user already rated, and (c) movie 267 `"unknown"`. A node-only function `recallAtKTrainOnly` is provided for comparison, which restricts ranking to the training-set pool as the readme originally specified.

### 10.5 ISO-8859-1 titles (Latin-1)
The `u.item` file is encoded in ISO-8859-1 (Latin-1). The original `data.js` used `fetch().text()` which decodes as UTF-8, corrupting 9 titles containing accented characters (e.g., `Mis�rables` instead of `Misérables`).

**Fix**: After `loadData()` completes, the page re-fetches `u.item` as an `ArrayBuffer`, decodes it with `new TextDecoder('iso-8859-1')`, and overwrites `movies[].title` by id. This restores all accented titles correctly.

### 10.6 Duplicate-title and "unknown" exclusion in recommendations
The dataset contains **18 duplicate-title pairs** (e.g., "Chasing Amy (1997)" at ids 246 and 268) and one movie (id 267) titled literally `"unknown"`. The original `recommendTopK` only excluded by `movieId`, so a user who rated one duplicate could be recommended the other copy of the same film, and `"unknown"` could appear in recommendations.

**Fix**: `recommendTopK` now additionally excludes:
- (a) Any movie whose **title matches** the title of a movie the user rated in train or val (same-title duplicate protection).
- (b) Movie 267 with title `"unknown"`.

This ensures recommendations do not repeat the same film under a different id and do not surface the placeholder title.

### 10.7 u.item has 19 genre flags (not 18)
The original readme §3 stated `u.item` has "18 genre flags". The file actually has **24 pipe-separated fields**: indices 0–4 are id/title/release/video/url, index 5 is the legacy **"unknown" genre flag**, and indices 6–23 are the **18 real genre flags** (Action, Adventure, …, Western). Total genre columns = **19**.

**Correction**: The readme now states the format as `id | title | release | video | url | unknown | 18 genre flags` (24 fields total), matching the actual file and `data.js` parsing logic (which correctly uses `fields.slice(6, 24)` for the 18 real genres).
