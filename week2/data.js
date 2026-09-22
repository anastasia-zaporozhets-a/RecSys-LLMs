// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in the u.item file (official ML-100K order, 19 genres)
const genreNames = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data (decode u.item as ISO-8859-1, not UTF-8)
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. This page must be served over HTTP: run "python3 -m http.server 8000" in the week2 folder and open http://localhost:8000.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    const seenTitles = new Set();

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 24) continue; // Skip invalid lines (missing genre flags)

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Skip the placeholder row id 267 (title "unknown", empty release date/URL)
        if (id === 267) continue;

        // Duplicate titles: keep only the first (lowest id) occurrence
        if (seenTitles.has(title)) continue;
        seenTitles.add(title);

        // Extract all 19 genre flags (official order: unknown, Action, ... , Western)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = [];
        const vector = genreValues.map((value, index) => {
            if (value === 1) genres.push(genreNames[index]);
            return value;
        });

        movies.push({ id, title, genres, vector });
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

        ratings.push({ userId, itemId, rating, timestamp });
    }
}
