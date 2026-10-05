// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // highest movie id found in u.item (bounds raw-id indexing)
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"

// raw movie id -> lowest movie id carrying the same title (canonical id)
let canonicalId = new Map();
// raw movie id -> movie record; duplicates resolve to the canonical record
let movieById = new Map();
// highest movie id seen while parsing, including duplicates and the placeholder
let maxMovieId = 0;

// The ml-100k placeholder row carries no metadata and is never recommendable.
const PLACEHOLDER_TITLE = 'unknown';

// Genre names in the exact order of the 19 genre flags in u.item. The first flag
// is "unknown", so 19 names are required to align with fields 5..23.
const genreNames = [
    "unknown",
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// u.item is ISO-8859-1, not UTF-8. Response.text() always decodes as UTF-8 and
// would replace the accented titles with U+FFFD, so the bytes are decoded here.
const ITEM_ENCODING = 'iso-8859-1';

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = new TextDecoder(ITEM_ENCODING).decode(await moviesResponse.arrayBuffer());
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = maxMovieId;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        showLoadError(error);
        throw error; // Re-throw so script.js can handle the error
    }
}

// Report a load failure in both result panels, never just one of them.
function showLoadError(error) {
    const message = `Error: ${error.message}. This page must be served over HTTP: run ` +
        `"python3 -m http.server 8000" in the week3 folder and open ` +
        `http://localhost:8000, or publish the folder to GitHub Pages. ` +
        `Opening index.html directly as a file:// URL blocks fetch().`;
    for (const elementId of ['user-based-result', 'item-based-result']) {
        const target = document.getElementById(elementId);
        if (target) {
            target.innerHTML = `<p class="error">${message}</p>`;
        }
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    const idByTitle = new Map();

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1].trim();
        if (Number.isNaN(id)) continue;
        maxMovieId = Math.max(maxMovieId, id);

        // Drop the metadata-free placeholder row; its ratings are dropped too.
        if (title === '' || title.toLowerCase() === PLACEHOLDER_TITLE) continue;

        // Extract genres (last 19 fields, aligned 1:1 with genreNames)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        const existingId = idByTitle.get(title);
        if (existingId === undefined) {
            const movie = { id, title, genres };
            movies.push(movie);          // one entry per title
            idByTitle.set(title, id);
            canonicalId.set(id, id);
            movieById.set(id, movie);
        } else {
            // A duplicate listing of a film we already have: keep the lower id
            // as canonical so both copies collapse onto one recommendation.
            canonicalId.set(id, existingId);
            movieById.set(id, movieById.get(existingId));
        }
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines

        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);

        // Ignore ratings for the placeholder / any movie u.item did not declare.
        if (!movieById.has(itemId)) continue;

        ratings.push({ userId, itemId, canonical: canonicalId.get(itemId), rating, timestamp });
    }
}

// ---------------------------------------------------------------------------
// Build the user-item rating matrix.
//
// Shape: (numUsers + 1) x (numMovies + 1), indexed by raw id, so that
//   ratingMatrix[userId][movieId] === rating
// and a missing entry is 0. MovieLens ratings are 1-5, so 0 is unambiguous.
//
// A film listed twice in u.item collapses onto its canonical (lowest) id; if a
// user rated both copies the two ratings are averaged into the single cell.
// ---------------------------------------------------------------------------
function buildRatingMatrix() {
    const width = numMovies + 1;
    const sums = [];
    const counts = [];
    for (let userId = 0; userId <= numUsers; userId++) {
        sums[userId] = new Array(width).fill(0);
        counts[userId] = new Array(width).fill(0);
    }

    for (const r of ratings) {
        sums[r.userId][r.canonical] += r.rating;
        counts[r.userId][r.canonical] += 1;
    }

    ratingMatrix = [];
    for (let userId = 0; userId <= numUsers; userId++) {
        const row = new Array(width).fill(0);
        const sumRow = sums[userId];
        const countRow = counts[userId];
        for (let movieId = 1; movieId < width; movieId++) {
            const count = countRow[movieId];
            row[movieId] = count === 0 ? 0 : sumRow[movieId] / count;
        }
        ratingMatrix[userId] = row;
    }
}
