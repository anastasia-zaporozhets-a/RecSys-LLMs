'use strict';
// Runs the REAL week3/data.js and week3/script.js inside a single Node vm context.
// Nothing is re-implemented here: every similarity and every recommendation comes
// from the actual globals those two files define. This file only supplies the
// browser environment (fetch / document / window) and instruments the results.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEEK3 = path.join(__dirname, '..');
const ACTIVE_USER = 1;
const N = 20;

// ---------------------------------------------------------------------------
// Browser stubs
// ---------------------------------------------------------------------------

// Minimal element. A <select> needs options[] / value / appendChild / remove,
// which is exactly what populateUserDropdown() in script.js touches.
function makeElement(id, tag) {
    return {
        id: id,
        tagName: tag,
        innerHTML: '',
        value: '',
        options: [],
        appendChild(child) {
            this.options.push(child);
            if (this.options.length === 1) this.value = child.value;
        },
        remove(index) {
            this.options.splice(index, 1);
        }
    };
}

const userSelect = makeElement('user-select', 'select');
// Exactly the placeholder declared in index.html:15-17.
userSelect.options.push({ value: '', textContent: 'Select a user', disabled: true, selected: true });
userSelect.value = '';

const elements = {
    'user-select': userSelect,
    'user-based-result': makeElement('user-based-result', 'div'),
    'item-based-result': makeElement('item-based-result', 'div')
};

const documentStub = {
    getElementById(id) {
        return Object.prototype.hasOwnProperty.call(elements, id) ? elements[id] : null;
    },
    createElement(tag) {
        return { tagName: tag, value: '', textContent: '', appendChild(c) { this.child = c; } };
    }
};

// fetch() -> Response-like object exposing BOTH text() and arrayBuffer().
function makeResponse(file) {
    const buffer = fs.readFileSync(file);
    return {
        ok: true,
        status: 200,
        url: 'file://' + file,
        text() {
            return Promise.resolve(buffer.toString('utf8'));
        },
        arrayBuffer() {
            return Promise.resolve(
                buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
            );
        },
        json() {
            return Promise.resolve(JSON.parse(buffer.toString('utf8')));
        }
    };
}

function fetchStub(url) {
    const file = path.join(WEEK3, String(url));
    if (!fs.existsSync(file)) {
        return Promise.resolve({
            ok: false, status: 404, url: String(url),
            text: () => Promise.resolve(''),
            arrayBuffer: () => Promise.resolve(new ArrayBuffer(0))
        });
    }
    return Promise.resolve(makeResponse(file));
}

// ---------------------------------------------------------------------------
// One vm context, real sources
// ---------------------------------------------------------------------------

const sandbox = { console: console, fetch: fetchStub, document: documentStub };
sandbox.window = sandbox;          // script.js does `window.onload = ...`
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

for (const file of ['data.js', 'script.js']) {
    vm.runInContext(fs.readFileSync(path.join(WEEK3, file), 'utf8'), sandbox, { filename: file });
}

// `let`/`const` top-level bindings live in the context's global lexical scope,
// not on the sandbox object, so re-export them from inside the context. They are
// exposed as getters because `let` bindings are copied by value at export time
// and would otherwise be frozen at their pre-loadData() initialisers.
vm.runInContext(
    'globalThis.__api = { ' +
    '  get movies() { return movies; }, get ratings() { return ratings; },' +
    '  get numUsers() { return numUsers; }, get numMovies() { return numMovies; },' +
    '  get ratingMatrix() { return ratingMatrix; },' +
    '  loadData, buildRatingMatrix, cosineSimilarity, populateUserDropdown,' +
    '  getUserBasedRecommendations, getItemBasedRecommendations, getRecommendations, renderList };',
    sandbox
);
const api = sandbox.__api;

// ---------------------------------------------------------------------------
// Boot through the real initialisation path
// ---------------------------------------------------------------------------

function heading(text) {
    console.log('\n' + '='.repeat(78));
    console.log(text);
    console.log('='.repeat(78));
}

(async function main() {
    heading('BOOT  (real window.onload -> loadData() -> populateUserDropdown())');
    await sandbox.onload();

    const { movies, ratings, numUsers, numMovies, ratingMatrix, cosineSimilarity } = api;
    console.log('movies parsed          :', movies.length);
    console.log('ratings parsed         :', ratings.length);
    console.log('numUsers / numMovies   :', numUsers, '/', numMovies);
    console.log('ratingMatrix shape     :', ratingMatrix.length, 'x', ratingMatrix[0].length);
    console.log('dropdown options       :', userSelect.options.length,
        '(1 placeholder + ' + (userSelect.options.length - 1) + ' users)');
    console.log('#user-based-result     :', JSON.stringify(elements['user-based-result'].innerHTML));

    // Drive the real UI entry point exactly as the button does.
    userSelect.value = String(ACTIVE_USER);
    api.getRecommendations();
    console.log('\nreal getRecommendations() rendered:');
    console.log('  #user-based-result ->', elements['user-based-result'].innerHTML.slice(0, 150) + '...');
    console.log('  #item-based-result ->', elements['item-based-result'].innerHTML.slice(0, 150) + '...');

    const activeRow = ratingMatrix[ACTIVE_USER];

    // -----------------------------------------------------------------------
    heading(`(1) USER-BASED step 1-2: top-8 neighbours of user ${ACTIVE_USER} (sim desc, ties -> lower user id)`);
    // -----------------------------------------------------------------------
    const ranked = [];
    for (let otherId = 1; otherId <= numUsers; otherId++) {
        if (otherId === ACTIVE_USER) continue;
        const similarity = cosineSimilarity(activeRow, ratingMatrix[otherId]);
        if (similarity > 0) ranked.push({ userId: otherId, similarity });
    }
    ranked.sort((a, b) => (b.similarity - a.similarity) || (a.userId - b.userId));

    console.log(`active user ${ACTIVE_USER} rated ${activeRow.filter(v => v !== 0).length} of ${numMovies} movies`);
    console.log('users with similarity > 0 :', ranked.length);
    console.log('\n rank  user  similarity   co-rated');
    for (let i = 0; i < 8; i++) {
        const n = ranked[i];
        let coRated = 0;
        for (let j = 1; j <= numMovies; j++) {
            if (activeRow[j] !== 0 && ratingMatrix[n.userId][j] !== 0) coRated++;
        }
        console.log(
            String(i + 1).padStart(5),
            String(n.userId).padStart(6),
            n.similarity.toFixed(10).padStart(12),
            String(coRated).padStart(10)
        );
    }
    const top20 = ranked.slice(0, N);
    console.log(`\nN = ${N} neighbours actually used -> user ids: [${top20.map(n => n.userId).join(', ')}]`);
    const minCo = Math.min.apply(null, top20.map(n => {
        let c = 0;
        for (let j = 1; j <= numMovies; j++) if (activeRow[j] !== 0 && ratingMatrix[n.userId][j] !== 0) c++;
        return c;
    }));
    console.log('similarity of the 20th neighbour :', top20[N - 1].similarity.toFixed(10));
    console.log('fewest co-rated movies among those 20 :', minCo);

    // -----------------------------------------------------------------------
    heading(`(2) USER-BASED step 3-4: top-5 for user ${ACTIVE_USER}  [sum(s*r)/sum(s)]`);
    // -----------------------------------------------------------------------
    const ub = api.getUserBasedRecommendations(ACTIVE_USER);
    console.log(' score       #neighbours  #ratings in u.data  title');
    ub.forEach((item, i) => {
        const movie = movies.find(m => m.title === item.title);
        let voters = 0;
        for (const n of top20) if (ratingMatrix[n.userId][movie.id] !== 0) voters++;
        const total = ratings.filter(r => r.itemId === movie.id).length;
        console.log(
            item.score.toFixed(6).padStart(10),
            String(voters).padStart(14),
            String(total).padStart(20),
            ' ' + item.title
        );
        if (i === ub.length - 1) return;
    });

    // -----------------------------------------------------------------------
    heading(`(3) ITEM-BASED step 2-3: top-5 for user ${ACTIVE_USER}  [sum over rated i of sim(i,j)*rating(u,i)]`);
    // -----------------------------------------------------------------------
    const ib = api.getItemBasedRecommendations(ACTIVE_USER);
    console.log(' score       #ratings in u.data  title');
    ib.forEach(item => {
        const movie = movies.find(m => m.title === item.title);
        const total = ratings.filter(r => r.itemId === movie.id).length;
        console.log(
            item.score.toFixed(6).padStart(10),
            String(total).padStart(20),
            ' ' + item.title
        );
    });

    // -----------------------------------------------------------------------
    heading(`(4) FOR COMPARISON ONLY: same item-based scores divided by sum of similarities`);
    // -----------------------------------------------------------------------
    const columns = new Array(numMovies + 1);
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        const column = new Array(numUsers + 1);
        for (let userId = 0; userId <= numUsers; userId++) column[userId] = ratingMatrix[userId][movieId];
        columns[movieId] = column;
    }
    const ratedIds = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) if (activeRow[movieId] !== 0) ratedIds.push(movieId);

    const normalised = [];
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        if (activeRow[movieId] !== 0) continue;
        let sum = 0, simSum = 0;
        for (const ratedId of ratedIds) {
            const s = cosineSimilarity(columns[ratedId], columns[movieId]);
            sum += s * activeRow[ratedId];
            simSum += s;
        }
        normalised.push({ id: movieId, title: movies[movieId - 1].title, score: simSum === 0 ? 0 : sum / simSum });
    }
    normalised.sort((a, b) => (b.score - a.score) || (a.id - b.id));

    console.log(' score       #ratings in u.data  title');
    normalised.slice(0, 5).forEach(item => {
        const total = ratings.filter(r => r.itemId === item.id).length;
        console.log(
            item.score.toFixed(6).padStart(10),
            String(total).padStart(20),
            ' ' + item.title
        );
    });
    const ibIds = ib.map(t => movies.find(m => m.title === t.title).id);
    const normIds = normalised.slice(0, 5).map(t => t.id);
    console.log('\nun-normalised top-5 ids :', ibIds.join(', '));
    console.log('normalised    top-5 ids :', normIds.join(', '));
    console.log('overlap                  :', ibIds.filter(id => normIds.includes(id)).length, 'of 5');

    console.log('\ndone.');
})().catch(err => {
    console.error('CHECK FAILED:', err);
    process.exit(1);
});
