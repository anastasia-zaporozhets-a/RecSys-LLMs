// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load data
        await loadData();

        // Populate dropdowns and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Cosine similarity over two numeric vectors of equal length (guarded for zero norms)
function cosineSimilarity(a, b) {
    let dot = 0;
    let norm2A = 0;
    let norm2B = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        norm2A += a[i] * a[i];
        norm2B += b[i] * b[i];
    }
    if (norm2A === 0 || norm2B === 0) return 0;
    return dot / (Math.sqrt(norm2A) * Math.sqrt(norm2B));
}

// Populate all movie dropdowns with sorted movie titles
function populateMoviesDropdown() {
    const selectIds = ['movie-select', 'profile-select-1', 'profile-select-2', 'profile-select-3'];

    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    selectIds.forEach(selectId => {
        const selectElement = document.getElementById(selectId);

        // Clear existing options except the first placeholder
        while (selectElement.options.length > 1) {
            selectElement.remove(1);
        }

        // Add movies to dropdown
        sortedMovies.forEach(movie => {
            const option = document.createElement('option');
            option.value = movie.id;
            option.textContent = movie.title;
            selectElement.appendChild(option);
        });
    });
}

// Format the top recommendations as a numbered list with scores
function formatRecommendations(topRecommendations) {
    return topRecommendations.map((movie, index) =>
        `${index + 1}. ${movie.title} (${movie.score.toFixed(2)})`
    ).join('\n');
}

// Main item-to-item recommendation function (Top-5, cosine over genre vectors)
function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);

        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }

        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }

        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';

        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

                // Step 4: Calculate cosine similarity scores against the liked movie's vector
                const scoredMovies = candidateMovies.map(candidate => ({
                    ...candidate,
                    score: cosineSimilarity(likedMovie.vector, candidate.vector)
                }));

                // Step 5: Sort by score (descending), then by title (ascending)
                scoredMovies.sort((a, b) =>
                    b.score - a.score ||
                    (a.title < b.title ? -1 : a.title > b.title ? 1 : 0)
                );

                // Step 6: Select the top 5 recommendations
                const topRecommendations = scoredMovies.slice(0, 5);

                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const lines = formatRecommendations(topRecommendations);
                    resultElement.innerText = `Because you liked "${likedMovie.title}", we recommend:\n${lines}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}

// Profile-based recommendation function (average of 3 watched movies, Top-5)
function getProfileRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        // Step 1: Get the three selected watched movies
        const selectIds = ['profile-select-1', 'profile-select-2', 'profile-select-3'];
        const selectedIds = selectIds.map(selectId =>
            parseInt(document.getElementById(selectId).value)
        );

        if (selectedIds.some(isNaN)) {
            resultElement.textContent = "Please select three movies first.";
            resultElement.className = 'error';
            return;
        }
        if (new Set(selectedIds).size !== 3) {
            resultElement.textContent = "Error: the same movie was selected more than once. Please choose three distinct movies.";
            resultElement.className = 'error';
            return;
        }

        const watched = selectedIds.map(id => movies.find(movie => movie.id === id));
        if (watched.some(movie => !movie)) {
            resultElement.textContent = "Error: a selected movie was not found in the database.";
            resultElement.className = 'error';
            return;
        }

        // Step 2: Profile = element-wise mean of the 3 watched movie vectors
        const profileVector = watched[0].vector.map((_, i) =>
            (watched[0].vector[i] + watched[1].vector[i] + watched[2].vector[i]) / 3
        );

        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';

        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Exclude the 3 watched movies from the candidates
                const watchedIds = new Set(selectedIds);

                // Step 4: Calculate cosine similarity of each candidate to the profile vector
                const scoredMovies = movies
                    .filter(movie => !watchedIds.has(movie.id))
                    .map(candidate => ({
                        ...candidate,
                        score: cosineSimilarity(profileVector, candidate.vector)
                    }));

                // Step 5: Sort by score (descending), then by title (ascending)
                scoredMovies.sort((a, b) =>
                    b.score - a.score ||
                    (a.title < b.title ? -1 : a.title > b.title ? 1 : 0)
                );

                // Step 6: Select the top 5 recommendations
                const topRecommendations = scoredMovies.slice(0, 5);

                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const watchedTitles = watched.map(movie => movie.title).join(', ');
                    const lines = formatRecommendations(topRecommendations);
                    resultElement.innerText = `Based on ${watchedTitles}, we recommend:\n${lines}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = "No recommendations found for the selected movies.";
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in profile recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getProfileRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
