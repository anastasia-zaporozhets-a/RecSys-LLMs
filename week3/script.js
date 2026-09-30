// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// Missing-value strategy (see week3/readme.md section 6):
//
//   [x] significance weighting: cos * min(n, GAMMA) / GAMMA, n = co-rated items
//
// Raw cosine over 1-5 ratings is exactly 1.0 for any single co-rated movie, so an
// unweighted top-N collapses into a tie of one-movie "neighbours"; scaling by the
// confidence in the estimate keeps genuinely co-rated pairs on top.
// ---------------------------------------------------------------------------

// A pair with >= GAMMA co-rated movies is weighted at full strength.
const GAMMA = 50;
// Shrinkage weight: predictions with little support fall back to the user's mean.
const LAMBDA = 1;
// How many neighbours user-based CF aggregates over.
const N = 20;

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
// Cosine similarity between two rating vectors, significance-weighted.
//
// Missing values are 0 and are skipped, so "not rated" is never treated as a
// score. The raw cosine is then down-weighted by how many entries it rests on
// (significance weighting, Herlocker et al. 1999).
//
// Inputs: two arrays of equal length (a row, or a precomputed movie column).
// Output: a number in [0, 1]; 0 when the two vectors share no rated items.
// ---------------------------------------------------------------------------
function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let coRated = 0;

    for (let i = 0; i < a.length; i++) {
        const x = a[i];
        const y = b[i];
        if (x === 0 || y === 0) continue;
        dot += x * y;
        normA += x * x;
        normB += y * y;
        coRated++;
    }

    const denominator = Math.sqrt(normA * normB);
    if (denominator === 0) return 0;   // the two vectors share no rated items
    return (dot / denominator) * (Math.min(coRated, GAMMA) / GAMMA);
}

// Mean of the ratings actually present in a rating row (0 = not rated).
function meanRating(row) {
    let sum = 0;
    let count = 0;
    for (let movieId = 1; movieId < row.length; movieId++) {
        if (row[movieId] === 0) continue;
        sum += row[movieId];
        count++;
    }
    return count === 0 ? 0 : sum / count;
}

// Title for a raw movie id; duplicate ids resolve to the canonical record.
function movieTitle(movieId) {
    const movie = movieById.get(movieId);
    return movie === undefined ? `Movie ${movieId}` : movie.title;
}

// ---------------------------------------------------------------------------
// User-Based CF.
//
// Step 1: cosineSimilarity between the active user's row and every other row.
// Step 2: keep the N most similar users with a positive similarity (ties break
//         on the lower user id).
// Step 3: for every movie the user has not rated, predict over the neighbours
//         who rated it, shrunk toward the user's own mean:
//             score = (sum(s*r) + LAMBDA*mean_u) / (sum(s) + LAMBDA)
//         A candidate needs sum(s) > 0.
// Step 4: sort by score descending, ties on the lower movie id.
// Returns an array of { title, score }.
// ---------------------------------------------------------------------------
function getUserBasedRecommendations(activeUserId, topK = 5) {
    const activeRow = ratingMatrix[activeUserId];
    const userMean = meanRating(activeRow);

    const ranked = [];
    for (let otherId = 1; otherId <= numUsers; otherId++) {
        if (otherId === activeUserId) continue;
        const similarity = cosineSimilarity(activeRow, ratingMatrix[otherId]);
        if (similarity > 0) ranked.push({ userId: otherId, similarity });
    }
    ranked.sort((a, b) => (b.similarity - a.similarity) || (a.userId - b.userId));
    const neighbours = ranked.slice(0, N);

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
        if (similaritySum <= 0) continue;         // nobody in the neighbourhood rated it

        candidates.push({
            id: movieId,
            title: movieTitle(movieId),
            score: (weightedSum + LAMBDA * userMean) / (similaritySum + LAMBDA)
        });
    }

    candidates.sort((a, b) => (b.score - a.score) || (a.id - b.id));
    return candidates.slice(0, topK).map(({ title, score }) => ({ title, score }));
}

// ---------------------------------------------------------------------------
// Item-Based CF.
//
// The rating column of every movie is built once and cached, and item-item
// similarities are memoised, so the matrix is never re-sliced per comparison.
// For every movie the user has not rated:
//     score = (sum over rated i of s*rating(u,i) + LAMBDA*mean_u)
//             / (sum over rated i of s + LAMBDA),   s = cosineSimilarity(i, j)
// A candidate needs sum(s) > 0. Sort by score descending, ties on the lower
// movie id. Returns an array of { title, score, because }.
// ---------------------------------------------------------------------------
let columnCache = null;
let columnCacheFor = null;
let itemSimilarityCache = new Map();
let itemSimilarityCacheFor = null;

// Rating column per movie (one entry per user), built at most once per matrix.
function getMovieColumns() {
    if (columnCacheFor === ratingMatrix) return columnCache;
    const columns = new Array(numMovies + 1);
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        const column = new Array(numUsers + 1);
        for (let userId = 0; userId <= numUsers; userId++) {
            column[userId] = ratingMatrix[userId][movieId];
        }
        columns[movieId] = column;
    }
    columnCache = columns;
    columnCacheFor = ratingMatrix;
    return columnCache;
}

// Memoised item-item similarity, invalidated whenever the matrix is rebuilt.
function itemSimilarity(ratedId, candidateId) {
    if (itemSimilarityCacheFor !== ratingMatrix) {
        itemSimilarityCache = new Map();
        itemSimilarityCacheFor = ratingMatrix;
    }
    const key = ratedId * (numMovies + 1) + candidateId;
    let similarity = itemSimilarityCache.get(key);
    if (similarity === undefined) {
        const columns = getMovieColumns();
        similarity = cosineSimilarity(columns[ratedId], columns[candidateId]);
        itemSimilarityCache.set(key, similarity);
    }
    return similarity;
}

function getItemBasedRecommendations(activeUserId, topK = 5) {
    const activeRow = ratingMatrix[activeUserId];
    const userMean = meanRating(activeRow);

    const ratedMovieIds = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) ratedMovieIds.push(movieId);
    }

    const candidates = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) continue;   // already rated -> not a candidate

        let weightedSum = 0;
        let similaritySum = 0;
        let bestAnchorId = 0;
        let bestContribution = -Infinity;

        for (const ratedId of ratedMovieIds) {
            const similarity = itemSimilarity(ratedId, movieId);
            const contribution = similarity * activeRow[ratedId];
            weightedSum += contribution;
            similaritySum += similarity;
            if (contribution > bestContribution) {
                bestContribution = contribution;
                bestAnchorId = ratedId;
            }
        }
        if (similaritySum <= 0) continue;

        candidates.push({
            id: movieId,
            title: movieTitle(movieId),
            score: (weightedSum + LAMBDA * userMean) / (similaritySum + LAMBDA),
            because: movieTitle(bestAnchorId)
        });
    }

    candidates.sort((a, b) => (b.score - a.score) || (a.id - b.id));
    return candidates.slice(0, topK).map(({ title, score, because }) => ({ title, score, because }));
}

// Provided — read the selected user and render both recommendation lists
function getRecommendations() {
    const selectElement = document.getElementById('user-select');
    const userId = parseInt(selectElement.value, 10);

    if (isNaN(userId)) {
        renderList('user-based-result', [], '', 'Please select a user first.');
        renderList('item-based-result', [], '', 'Please select a user first.');
        return;
    }

    renderList(
        'user-based-result',
        getUserBasedRecommendations(userId),
        'Because you are similar to other users, we recommend:',
        TOO_FEW_RATINGS
    );
    renderList('item-based-result', getItemBasedRecommendations(userId), '', TOO_FEW_RATINGS);
}

const TOO_FEW_RATINGS = 'This user has too few ratings for a reliable recommendation list.';

// Provided — render a list of { title, score } into the given element.
// `message` is an optional lead-in shown above the list; `emptyMessage` is shown
// instead of the list when there is nothing to recommend.
function renderList(elementId, items, message, emptyMessage) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (!items || items.length === 0) {
        el.innerHTML = `<p>${emptyMessage || TOO_FEW_RATINGS}</p>`;
        return;
    }

    const entries = items
        .map(item => {
            const lead = item.because
                ? `Because you liked ${item.because}, we recommend: `
                : '';
            return `<li>${lead}${item.title} &mdash; ${Number(item.score).toFixed(3)}</li>`;
        })
        .join('');
    el.innerHTML = (message ? `<p>${message}</p>` : '') + `<ul>${entries}</ul>`;
}
