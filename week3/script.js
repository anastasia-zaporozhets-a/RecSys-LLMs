// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6):
//
//   [x] use co-rated entries only
//
// ---------------------------------------------------------------------------

// Initialize the application when the window loads
window.onload = async function() {
    const userBased = document.getElementById('user-based-result');
    const itemBased = document.getElementById('item-based-result');

    try {
        userBased.innerHTML = '<p>Loading movie data...</p>';
        itemBased.innerHTML = '<p>Loading movie data...</p>';

        await loadData();

        populateUserDropdown();

        userBased.innerHTML = '<p>Data loaded. Select a user.</p>';
        itemBased.innerHTML = '<p>Data loaded. Select a user.</p>';
    } catch (error) {
        console.error('Initialization error:', error);
        // The error message is already shown by data.js
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }
}

// ---------------------------------------------------------------------------
// TODO (HW3) — cosine similarity between two rating vectors.
//
// Compare only co-rated (non-zero) entries, per the missing-value strategy
// you chose above. Return 0 when the denominator is 0 (that is, when the two
// vectors share no rated items). See week3/readme.md section 5.3.
//
// Inputs: two arrays of equal length (slice the rating matrix column or row).
// Output: a number in [0, 1].
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    // Co-rated entries only: an entry counts only when BOTH vectors are non-zero,
    // so "not rated" (0) is never mistaken for a real score.
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        const x = a[i];
        const y = b[i];
        if (x === 0 || y === 0) continue;
        dot += x * y;
        normA += x * x;
        normB += y * y;
    }

    const denominator = Math.sqrt(normA * normB);
    if (denominator === 0) return 0;   // the two vectors share no rated items
    return dot / denominator;
}

// ---------------------------------------------------------------------------
// TODO (HW3) — User-Based CF.
//
// Return the top-K recommendations for the active user as an array of
// { title, score }, sorted by score descending.
//
// Suggested steps (week3/readme.md section 5.4):
//   1. compare the active user's rating vector against every other user
//   2. take the N most similar users with positive similarity (e.g. N = 20)
//   3. for each movie the active user has NOT rated, predict a score as the
//      similarity-weighted average of those users' ratings
//   4. sort and take the top K
// ---------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const N = 20;
    const activeRow = ratingMatrix[activeUserId];

    // Step 1 + 2: rank every other user by similarity, keep the N most similar
    // ones with a strictly positive similarity. Ties break on the lower user id.
    const ranked = [];
    for (let otherId = 1; otherId <= numUsers; otherId++) {
        if (otherId === activeUserId) continue;
        const similarity = cosineSimilarity(activeRow, ratingMatrix[otherId]);
        if (similarity > 0) ranked.push({ userId: otherId, similarity });
    }
    ranked.sort((a, b) => (b.similarity - a.similarity) || (a.userId - b.userId));
    const neighbours = ranked.slice(0, N);

    // Step 3: similarity-weighted average over the neighbours that rated it.
    const candidates = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) continue;   // already rated -> not a candidate

        let weightedSum = 0;
        let similaritySum = 0;
        for (const neighbour of neighbours) {
            const rating = ratingMatrix[neighbour.userId][movieId];
            if (rating === 0) continue;
            weightedSum += neighbour.similarity * rating;
            similaritySum += neighbour.similarity;
        }
        if (similaritySum === 0) continue;         // nobody in the neighbourhood rated it

        candidates.push({
            id: movieId,
            title: movies[movieId - 1].title,
            score: weightedSum / similaritySum
        });
    }

    // Step 4: highest predicted score first, ties on the lower movie id.
    candidates.sort((a, b) => (b.score - a.score) || (a.id - b.id));
    return candidates.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// ---------------------------------------------------------------------------
// TODO (HW3) — Item-Based CF.
//
// Return the top-K recommendations for the active user as an array of
// { title, score }, sorted by score descending.
//
// Suggested steps (week3/readme.md section 5.5):
//   1. for each movie the active user has rated, compute the item-item
//      similarity against every other movie's rating column
//   2. for each candidate movie the active user has NOT rated, aggregate the
//      similarities from the rated movies, weighted by the user's rating
//   3. sort and take the top K
// ---------------------------------------------------------------------------
function getItemBasedRecommendations(activeUserId, topK = 5) {
    const activeRow = ratingMatrix[activeUserId];

    // Step 1: the rating "column" of each movie, i.e. one entry per user.
    const columns = new Array(numMovies + 1);
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        const column = new Array(numUsers + 1);
        for (let userId = 0; userId <= numUsers; userId++) {
            column[userId] = ratingMatrix[userId][movieId];
        }
        columns[movieId] = column;
    }

    const ratedMovieIds = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) ratedMovieIds.push(movieId);
    }

    // Step 2: aggregate, weighted by the user's own rating, the similarities
    // between every movie the user rated and the candidate.
    const candidates = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) continue;   // already rated -> not a candidate

        let score = 0;
        for (const ratedId of ratedMovieIds) {
            score += cosineSimilarity(columns[ratedId], columns[movieId]) * activeRow[ratedId];
        }

        candidates.push({ id: movieId, title: movies[movieId - 1].title, score: score });
    }

    // Step 3: highest aggregated score first, ties on the lower movie id.
    candidates.sort((a, b) => (b.score - a.score) || (a.id - b.id));
    return candidates.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// Provided — read the selected user and render both recommendation lists
function getRecommendations() {
    const selectElement = document.getElementById('user-select');
    const userId = parseInt(selectElement.value, 10);

    if (isNaN(userId)) {
        renderList('user-based-result', [], 'Please select a user first.');
        renderList('item-based-result', [], 'Please select a user first.');
        return;
    }

    renderList('user-based-result', getUserBasedRecommendations(userId));
    renderList('item-based-result', getItemBasedRecommendations(userId));
}

// Provided — render a list of { title, score } into the given element
function renderList(elementId, items, message) {
    const el = document.getElementById(elementId);

    if (message) {
        el.innerHTML = `<p>${message}</p>`;
        return;
    }

    if (!items || items.length === 0) {
        el.innerHTML = '<p>No recommendations. (Implement the TODO above.)</p>';
        return;
    }

    const entries = items
        .map(item => `<li>${item.title} &mdash; ${Number(item.score).toFixed(3)}</li>`)
        .join('');
    el.innerHTML = `<ul>${entries}</ul>`;
}
