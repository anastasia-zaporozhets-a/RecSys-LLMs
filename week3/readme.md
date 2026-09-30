You are an expert full-stack web developer who creates robust, well-commented, and modular web applications using only vanilla HTML, CSS, and JavaScript.

Your task is to generate the complete code for a "Collaborative Filtering Movie Recommender" web application based on the detailed specifications below. The application logic will be split into two separate JavaScript files: `data.js` for data loading and parsing, and `script.js` for UI and recommendation logic. Please provide the code for each of the four files—`index.html`, `style.css`, `data.js`, and `script.js`—separately and clearly labeled.

---

### **Project Specification: Collaborative Filtering Movie Recommender (Modular)**

#### **1. Overall Goal**

Build a single-page web application that recommends movies using **collaborative filtering**. The application will use `data.js` to load and parse the same MovieLens 100K files as the previous exercise (`u.item`, `u.data`)—the dataset is deliberately unchanged so that the **algorithm** is the only thing that changes between the Content-Based assignment and this one.

Unlike the Content-Based version, which compared movie **genres**, this version uses the **rating patterns of users**. It must produce a Top-5 recommendation list **two ways** for the same active user—**User-Based CF** and **Item-Based CF**—so the two lists can be compared side by side.

#### **2. File `index.html` - The Application Structure**

-   **DOCTYPE and Language:** The document should start with `<!DOCTYPE html>` and the `<html>` tag should specify `lang="en"`.
-   **Title:** The page title should be "Collaborative Filtering Movie Recommender".
-   **Main Heading:** Include an `<h1>` with the text "Collaborative Filtering Movie Recommender".
-   **Instructions:** Add a `<p>` tag explaining that the user picks a user and receives two Top-5 lists (one per CF approach).
-   **User Dropdown:** Include a `<select>` element with the ID `user-select`. It will be populated dynamically with one option per user ID present in `u.data`.
-   **Button:** Include a `<button>` with the text "Get Recommendations". When clicked, it must call the `getRecommendations()` JavaScript function.
-   **Result Display Areas:** Include a `<div>` with the ID `result-box`. Inside it, provide two clearly labelled sections:
    -   `<div id="user-based-result">` — for the User-Based CF Top-5
    -   `<div id="item-based-result">` — for the Item-Based CF Top-5

    Each section should show the recommended movie titles together with their predicted score (or similarity), so the two approaches can be compared directly.
-   **File Linking:** Link `data.js` and `script.js` at the end of the `<body>`. `data.js` must be loaded **before** `script.js`.
    ```
    <script src="data.js"></script>
    <script src="script.js"></script>
    ```

#### **3. File `style.css` - The Application Design**

-   **Layout:** Create a professional, modern, and user-friendly layout. All content should be centered on the page within a main container.
-   **Background:** The `<body>` should have a light, neutral background color (e.g., `#f4f7f6`).
-   **Container:** The main container holding all elements should have a white background, rounded corners (`border-radius`), and a subtle box shadow.
-   **Typography:** Use a clean, sans-serif font like 'Helvetica' or 'Arial'.
-   **Controls:** The `<select>` dropdown and `<button>` should have consistent styling.
-   **Button:** Distinct background colour (e.g., a shade of blue), white text, hover effect.
-   **Result Areas:** `#user-based-result` and `#item-based-result` should be visually separated (e.g., two columns on wide screens, stacked on narrow screens) with a light background and a clear heading each.

#### **4. File `data.js` - The Data Handling Module**

This file is responsible only for fetching and parsing the data from local files, and for building the rating structures.

1.  **Global Variables:** Declare `let movies = [];`, `let ratings = [];`, `let numUsers = 0;`, `let numMovies = 0;`, and `let ratingMatrix = null;`.

2.  **Primary Function: `loadData()`**
    -   Must be `async`.
    -   Uses `fetch()` to read `u.item` and `u.data` (same directory as `index.html`).
    -   Uses `try...catch`; on failure, display an error message in the result area.
    -   Awaits `u.item` first, then `u.data`, passing the text to the parsers.
    -   After parsing: set `numUsers` (max user ID in `ratings`), `numMovies` (number of parsed movies), and call `buildRatingMatrix()`.

3.  **Parsing Function: `parseItemData(text)`**
    -   Defines the 18 genre names ("Action" ... "Western").
    -   Splits by lines; each line split by `|`.
    -   Extracts `id` (field 0) and `title` (field 1); builds a `genres` array from the last 19 fields where the value is `'1'`.
    -   Pushes `{ id, title, genres }` to `movies`.

4.  **Parsing Function: `parseRatingData(text)`**
    -   Splits by lines; each line split by `\t`.
    -   Pushes `{ userId, itemId, rating, timestamp }` (numbers) to `ratings`.

5.  **Matrix Function: `buildRatingMatrix()`**
    -   Builds a 2-D structure of shape `(numUsers + 1) × (numMovies + 1)`, where a missing rating is represented by `0`.
    -   Also build a parallel "rated" boolean mask (or use `0` as "not rated") so the similarity function can distinguish *not rated* from *rated 0*.
    -   Store the result in the global `ratingMatrix`.

#### **5. File `script.js` - The UI and Logic Module**

This file handles the user interface and the collaborative-filtering logic.

1.  **Initialization Logic:**
    -   Use `window.onload` with an `async` function.
    -   `await loadData()`, then call `populateUserDropdown()` and set an initial status message.

2.  **UI Function: `populateUserDropdown()`**
    -   Gets the `#user-select` element.
    -   Adds one `<option>` per user ID from `1` to `numUsers`, with the value set to the integer user ID.

3.  **Similarity Function: `cosineSimilarity(a, b)`**
    -   Computes the cosine similarity between two vectors, **using only co-rated (non-zero) entries**.
    -   Must guard against a zero denominator (return `0` in that case).
    -   Include a comment explaining the missing-value convention chosen here and why (see "Missing Value Handling" below).

4.  **Core Logic - User-Based: `getUserBasedRecommendations(activeUserId, topK)`**
    -   Step 1: For every other user, compute `cosineSimilarity` against the active user's rating vector.
    -   Step 2: Select the `N` most similar users (e.g., `N = 20`) with positive similarity.
    -   Step 3: For each movie the active user has **not** rated, compute a predicted score as the similarity-weighted average of the similar users' ratings.
    -   Step 4: Sort the candidates by predicted score (descending) and take the top `topK` (default `5`).
    -   Return an array of `{ title, score }`.

5.  **Core Logic - Item-Based: `getItemBasedRecommendations(activeUserId, topK)`**
    -   Step 1: For each movie the active user has rated, compute item-to-item `cosineSimilarity` between that movie's rating column and every other movie's rating column.
    -   Step 2: For each candidate movie the user has **not** rated, aggregate the similarities from the user's rated movies, weighted by the user's rating.
    -   Step 3: Sort by the aggregated score (descending) and take the top `topK` (default `5`).
    -   Return an array of `{ title, score }`.

6.  **Display Function: `getRecommendations()`**
    -   Reads `#user-select`, converted to an integer.
    -   Calls both `getUserBasedRecommendations()` and `getItemBasedRecommendations()`.
    -   Renders each list into its own section, in the form *"Because you are similar to other users, we recommend: ..."* / *"Because you liked ... we recommend: ..."*.
    -   Handle the empty case gracefully (a user with too few ratings) with a clear message.

#### **6. Missing Value Handling**

The rating matrix is sparse. Pick **exactly one** strategy and apply it consistently in `cosineSimilarity`. State the choice in a comment at the top of `script.js`:

-   **Use co-rated items only** (ignore missing values): the default and simplest.
-   **Mean imputation** — replace missing entries with the row/column average.
-   **Weighted approach** — weight the similarity by the number of co-rated items.

Do not mix strategies.

#### **7. Notes on This Exercise**

-   Do **not** change the dataset files.
-   Keep the modular split (`data.js` / `script.js`); do not move logic between them.
-   The code must run offline from `file://`-like static hosting (GitHub Pages); no build step and no external libraries.

---
Please now generate the complete code for the `index.html`, `style.css`, `data.js`, and `script.js` files based on these final, detailed specifications.

---
---

# CORRECTED SPECIFICATION (binding — overrides sections 2–7 where they conflict)

Sections 1–7 above describe the *original* assignment and contain several
defects that a regenerated app would reproduce. The decisions below are
**binding**. Where they contradict sections 2–7, they win.

## C1. Genre names must be 19, not 18

`u.item` has 24 pipe-separated fields: `id | title | release date | video release
date | IMDb URL | <19 genre flags>`. The **first** genre flag is `unknown`.
Defining only the 18 real genres shifts every flag by one, mislabels `unknown`
as `Action`, and silently drops `Western` (26 films in this dataset).

-   Declare **19** names in file order, starting with `"unknown"`:
    `unknown, Action, Adventure, Animation, Children's, Comedy, Crime,
    Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance,
    Sci-Fi, Thriller, War, Western`.
-   Extract them from `fields.slice(5, 24)` and filter on `=== 1`, so the
    name list and the flag list align 1:1.

## C2. `u.item` is ISO-8859-1, not UTF-8

`fetch().text()` always decodes as UTF-8 and would replace the accented titles
(`Misérables`, `Cérémonie`, `Á köldum klaka`, …) with `U+FFFD`. Neither a
`<meta charset>` nor a server header can change this.

-   Read the bytes: `new TextDecoder('iso-8859-1').decode(await response.arrayBuffer())`.
-   `u.data` is pure ASCII and may use `text()`.

## C3. Drop the placeholder movie

`u.item` row 267 has the title `unknown`, an empty release date and an empty
IMDb URL, yet **9 real users rated it**. It must never be recommended.

-   Skip rows whose title is empty or case-insensitively `unknown`.
-   Drop any rating whose `itemId` is not a declared movie.

## C4. Collapse duplicate titles onto a canonical id

18 films are listed twice in `u.item` (e.g. `246`/`268` *Chasing Amy (1997)*),
and up to **112 users rated both copies**. Keyed by raw id, copy B is a legal
candidate for a user who already rated copy A, so the same film can be
recommended twice under one title; and item-based CF would count it as two
independent anchors, because the two columns have cosine ≈ 0.996.

-   Map every duplicate id to the **lowest id with the same title** (canonical id).
-   Keep **one** movie record per title.
-   In the matrix, if a user rated both copies, store the **mean** of the two
     ratings in the single canonical cell.

## C5. `numMovies` is the max movie id, not the movie count

The matrix is indexed by raw id, so its width must be `max(movieId) + 1`. Using
`movies.length` silently breaks indexing as soon as any row is skipped or
deduplicated.

-   `numMovies = max movie id seen in u.item`.
-   Expose a `movieById` map from raw id → movie record (duplicates resolve to
    the canonical record) and resolve every displayed title through it.

## C6. Load errors appear in both panels, in red

-   Write the failure message into **both** `#user-based-result` and
    `#item-based-result`; writing only one leaves the other stuck on
    "Loading…" forever.
-   The error class must **outrank** the generic paragraph rule. `.error` alone
    (specificity 0,1,0) loses to `.result-column p` (0,1,1). Use
    `.result-column p.error`.

## C7. Missing-value strategy: significance weighting

State exactly this one option in the header comment of `script.js`, and no
other:

```
cosineSimilarity(a, b) = cos(a, b) restricted to co-rated entries
                          * min(n, GAMMA) / GAMMA,   n = co-rated count
```

Reason: raw cosine over 1–5 ratings is **exactly 1.0 for any single co-rated
movie**, so an unweighted top-N collapses into a tie of one-movie
"neighbours" and the neighbourhood is decided by tie-break order. Divide by the
sum of the weights instead of averaging over the *neighbour count*: the divisor
`sum(s)` is per-candidate, so dividing by it reorders the results, and when only
one anchor is non-zero it collapses to that anchor's raw rating.

## C8. Shrink both predictions toward the user's own mean

With `mean_u` = mean of the active user's ratings and `LAMBDA = 1`:

```
score = (sum(s * r) + LAMBDA * mean_u) / (sum(s) + LAMBDA)
```

-   **User-based:** sum over the `N = 20` neighbours **who rated that movie**
    (ties in similarity broken by the lower user id).
-   **Item-based:** sum over **all** movies the user rated, with `s = s(i, j)`.
-   A candidate needs `sum(s) > 0`.
-   Sort by score descending, ties on the **lower movie id**.
-   Use the named constants `GAMMA = 50`, `LAMBDA = 1`, `N = 20`.

## C9. Item-based CF must be precomputed and cached

Computing a cosine per (rated movie, candidate) pair and re-slicing the matrix
each time is ~1.17 × 10⁹ operations for a 737-rating user (≈ 42 s per click
under JavaScriptCore; 738× the cost of user-based CF).

-   Build each movie's rating column **once** and reuse it.
-   Memoise item-item similarities in a `Map`.
-   Invalidate both caches whenever `ratingMatrix` is rebuilt (e.g. key the cache
    on the `ratingMatrix` reference).

## C10. Render per section 5.6

-   User-based panel, as a lead-in above the list:
    `Because you are similar to other users, we recommend:`
-   Item-based panel, **per recommendation** (the anchor differs per item):
    `Because you liked <the rated movie contributing most to this
    recommendation>, we recommend: <title> — <score>`
-   Replace the placeholder empty-state text with a real message for users with
    too few ratings.
-   The cosine helper returns a value in **[0, 1]**; floating point can still
    yield `1.0000000000000002`, so do not assert on it.
