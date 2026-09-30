'use strict';
// Runs the REAL week3/data.js and week3/script.js inside a single Node vm context.
// Nothing is re-implemented here: every similarity and every recommendation comes
// from the actual globals those two files define. This file only supplies the
// browser environment (fetch / document / window) and instruments the results.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEEK3 = path.join(__dirname, '..');
const ACTIVE_USERS = [1, 405];   // 1 = ordinary, 405 = largest profile (737 ratings)

// ---------------------------------------------------------------------------
// Browser stubs
// ---------------------------------------------------------------------------

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
        remove(index) { this.options.splice(index, 1); }
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
        return { tagName: tag, value: '', textContent: '' };
    }
};

// fetch() -> Response-like object exposing BOTH text() and arrayBuffer().
function makeResponse(file) {
    const buffer = fs.readFileSync(file);
    return {
        ok: true,
        status: 200,
        url: 'file://' + file,
        text() { return Promise.resolve(buffer.toString('utf8')); },
        arrayBuffer() {
            return Promise.resolve(
                buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
            );
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

const sandbox = {
    console: console,
    fetch: fetchStub,
    document: documentStub,
    // Provided by the browser (and by Node) but not by a bare vm context.
    TextDecoder: TextDecoder,
    TextEncoder: TextEncoder,
    URL: URL
};
sandbox.window = sandbox;          // script.js does `window.onload = ...`
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

for (const file of ['data.js', 'script.js']) {
    vm.runInContext(fs.readFileSync(path.join(WEEK3, file), 'utf8'), sandbox, { filename: file });
}

// `let`/`const` top-level bindings live in the context's global lexical scope,
// not on the sandbox object, so re-export them from inside the context. They are
// exposed as getters: a `let` binding is copied by value at export time and
// would otherwise be frozen at its pre-loadData() initialiser.
vm.runInContext(
    'globalThis.__api = { ' +
    '  get movies() { return movies; }, get ratings() { return ratings; },' +
    '  get numUsers() { return numUsers; }, get numMovies() { return numMovies; },' +
    '  get ratingMatrix() { return ratingMatrix; },' +
    '  get canonicalId() { return canonicalId; }, get movieById() { return movieById; },' +
    '  get genreNames() { return genreNames; },' +
    '  get GAMMA() { return GAMMA; }, get LAMBDA() { return LAMBDA; }, get N() { return N; },' +
    '  loadData, buildRatingMatrix, cosineSimilarity, meanRating, populateUserDropdown,' +
    '  getMovieColumns, itemSimilarity, movieTitle,' +
    '  getUserBasedRecommendations, getItemBasedRecommendations, getRecommendations, renderList };',
    sandbox
);
const api = sandbox.__api;

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

function rule(char) { return char.repeat(78); }
function heading(text) { console.log('\n' + rule('=') + '\n' + text + '\n' + rule('=')); }
const pad = (s, n) => String(s).padStart(n);

(async function main() {
    // -----------------------------------------------------------------------
    heading('BOOT  (real window.onload -> loadData() -> populateUserDropdown())');
    // -----------------------------------------------------------------------
    await sandbox.onload();

    const { movies, ratings, numUsers, numMovies, ratingMatrix,
            canonicalId, movieById, genreNames, GAMMA, LAMBDA, N } = api;

    console.log('movies (one per title)   :', movies.length);
    console.log('ratings (placeholder out):', ratings.length);
    console.log('numUsers / numMovies     :', numUsers, '/', numMovies, '  (numMovies = max movie id)');
    console.log('ratingMatrix shape       :', ratingMatrix.length, 'x', ratingMatrix[0].length);
    console.log('genreNames declared      :', genreNames.length, '->', genreNames[0], '...', genreNames[genreNames.length - 1]);
    console.log('canonical id != raw id   :', [...canonicalId.entries()].filter(([k, v]) => k !== v).length, 'ids remapped');
    console.log('movieById size           :', movieById.size);
    console.log('dropdown options         :', userSelect.options.length, '(1 placeholder + ' + (userSelect.options.length - 1) + ' users)');
    console.log('GAMMA / LAMBDA / N       :', GAMMA, '/', LAMBDA, '/', N);

    // --- C1: genre alignment spot-check on the real parser output -----------
    const accented = movies.filter(m => m.title !== Buffer.from(m.title, 'utf8').toString('latin1')
        && /[^\x00-\x7F]/.test(m.title));
    console.log('\nC1/C2 spot-check  movie 543 :', JSON.stringify(movieById.get(543).title));
    console.log('C1/C2 spot-check  movie 1633:', JSON.stringify(movieById.get(1633).title));
    console.log('C2 titles with a replacement char (U+FFFD):',
        movies.filter(m => m.title.includes('�')).length);
    console.log('C3 movie 267 present in movieById:', movieById.has(267),
        '| ratings for 267:', ratings.filter(r => r.itemId === 267).length);
    const dupIds = [...canonicalId.entries()].filter(([k, v]) => k !== v);
    const canon268 = canonicalId.get(268);
    console.log('C4 duplicate ids remapped :', dupIds.length, '(e.g.', dupIds.slice(0, 3).map(([k, v]) => `${k}->${v}`).join(', ') + ')');
    console.log('C4 canonicalId(268) =', canon268, '| canonicalId(246) =', canonicalId.get(246),
        '| same movie record:', movieById.get(246) === movieById.get(268));
    const dupRaw = ratings.filter(r => r.itemId === 246 || r.itemId === 268);
    const merged = new Map();
    for (const r of dupRaw) {
        if (!merged.has(r.userId)) merged.set(r.userId, []);
        merged.get(r.userId).push(r);
    }
    const mergedUsers = [...merged.entries()].filter(([, v]) => v.length === 2);
    console.log('C4 users who rated BOTH copies:', mergedUsers.length);
    if (mergedUsers.length) {
        const [u0, pair] = mergedUsers[0];
        const mean = (pair[0].rating + pair[1].rating) / 2;
        console.log('C4 example user', u0, 'rated', pair[0].rating, 'and', pair[1].rating,
            '-> matrix cell =', ratingMatrix[u0][canon268], '(mean =', mean + ')');
    }

    // -----------------------------------------------------------------------
    for (const user of ACTIVE_USERS) {
        const activeRow = ratingMatrix[user];
        const userMean = api.meanRating(activeRow);
        const ratedCount = activeRow.filter(v => v !== 0).length;

        heading(`USER ${user}  (${ratedCount} ratings, mean ${userMean.toFixed(4)})`);

        // -------------------------------------------------------------------
        console.log(`\n--- (1) top-8 neighbours (significance-weighted sim desc, ties -> lower user id)`);
        // -------------------------------------------------------------------
        const ranked = [];
        for (let otherId = 1; otherId <= numUsers; otherId++) {
            if (otherId === user) continue;
            const similarity = api.cosineSimilarity(activeRow, ratingMatrix[otherId]);
            if (similarity > 0) ranked.push({ userId: otherId, similarity });
        }
        ranked.sort((a, b) => (b.similarity - a.similarity) || (a.userId - b.userId));
        const top20 = ranked.slice(0, N);

        const coRatedOf = id => {
            let c = 0;
            for (let j = 1; j <= numMovies; j++) {
                if (activeRow[j] !== 0 && ratingMatrix[id][j] !== 0) c++;
            }
            return c;
        };

        console.log(' rank  user  similarity   co-rated');
        for (let i = 0; i < 8; i++) {
            const n = ranked[i];
            console.log(pad(i + 1, 5), pad(n.userId, 6), pad(n.similarity.toFixed(10), 12),
                pad(coRatedOf(n.userId), 10));
        }
        console.log(`\nN = ${N} neighbours used -> [${top20.map(n => n.userId).join(', ')}]`);
        console.log('similarity of the 20th :', top20[N - 1].similarity.toFixed(10));

        // Corrected tally (this is what the earlier summary got wrong)
        const sim1 = top20.filter(n => n.similarity === 1);
        const byCo = {};
        for (const n of sim1) {
            const c = coRatedOf(n.userId);
            byCo[c] = (byCo[c] || 0) + 1;
        }
        console.log(`neighbours with similarity === 1 exactly : ${sim1.length} of ${N}`);
        console.log('  ... grouped by co-rated count           :', JSON.stringify(byCo));
        console.log('co-rated counts of all 20               :', top20.map(n => coRatedOf(n.userId)).join(', '));
        console.log('weighted sim of the 20th is strictly < 1 :', top20[N - 1].similarity < 1);

        // -------------------------------------------------------------------
        console.log(`\n--- (2) USER-BASED top-5   score = (sum s*r + LAMBDA*mean_u) / (sum s + LAMBDA)`);
        // -------------------------------------------------------------------
        const ub = api.getUserBasedRecommendations(user);
        console.log('    score   #nbrs that rated it   title');
        ub.forEach(item => {
            const id = [...movieById.entries()].find(([, m]) => m.title === item.title)[0];
            const voters = top20.filter(n => ratingMatrix[n.userId][id] !== 0).length;
            console.log(pad(item.score.toFixed(6), 10), pad(voters, 20), ' ' + item.title);
        });

        // -------------------------------------------------------------------
        console.log(`\n--- (3) ITEM-BASED top-5  (columns + similarities cached)`);
        // -------------------------------------------------------------------
        const t0 = Date.now();
        const ib = api.getItemBasedRecommendations(user);
        const ibMs = Date.now() - t0;
        console.log('    score   #ratings in u.data  because (top anchor)            title');
        ib.forEach(item => {
            const id = [...movieById.entries()].find(([, m]) => m.title === item.title)[0];
            const total = ratings.filter(r => canonicalId.get(r.itemId) === canonicalId.get(id)).length;
            console.log(pad(item.score.toFixed(6), 10), pad(total, 20), ' ' +
                pad(item.because.slice(0, 32), 34), item.title);
        });
        const t1 = Date.now();
        api.getItemBasedRecommendations(user);
        console.log('item-based timing: first call ' + ibMs + ' ms, second (fully cached) ' + (Date.now() - t1) + ' ms');

        // -------------------------------------------------------------------
        console.log(`\n--- (4) FOR COMPARISON ONLY: the same item-based sums divided by sum of similarities`);
        // -------------------------------------------------------------------
        const t2 = Date.now();
        const columns = api.getMovieColumns();
        const raw = [];
        for (let movieId = 1; movieId <= numMovies; movieId++) {
            if (activeRow[movieId] !== 0) continue;
            let num = 0, simSum = 0;
            for (let ratedId = 1; ratedId <= numMovies; ratedId++) {
                if (activeRow[ratedId] === 0) continue;
                const s = api.cosineSimilarity(columns[ratedId], columns[movieId]);
                num += s * activeRow[ratedId];
                simSum += s;
            }
            if (simSum <= 0) continue;
            raw.push({ id: movieId, title: api.movieTitle(movieId), score: num / simSum });
        }
        raw.sort((a, b) => (b.score - a.score) || (a.id - b.id));
        console.log('    score   #ratings in u.data  title   (raw sum / sum of similarities)');
        raw.slice(0, 5).forEach(item => {
            const total = ratings.filter(r => canonicalId.get(r.itemId) === canonicalId.get(item.id)).length;
            console.log(pad(item.score.toFixed(6), 10), pad(total, 20), ' ' + item.title);
        });
        console.log('(recomputed from the cached columns in ' + (Date.now() - t2) + ' ms)');

        const shipped = new Set(ib.map(i => i.title));
        const alt = raw.slice(0, 5).map(r => r.title);
        console.log('\nshipped item-based top-5 :', ib.map(i => i.title).join(' | '));
        console.log('alternative   top-5      :', alt.join(' | '));
        console.log('overlap                  :', alt.filter(t => shipped.has(t)).length, 'of 5');

        // -------------------------------------------------------------------
        console.log('\n--- rendered panels (real getRecommendations() -> renderList)');
        // -------------------------------------------------------------------
        userSelect.value = String(user);
        api.getRecommendations();
        console.log('#user-based-result:');
        console.log('  ' + elements['user-based-result'].innerHTML.replace(/<\/li>/g, '</li>\n  '));
        console.log('#item-based-result:');
        console.log('  ' + elements['item-based-result'].innerHTML.replace(/<\/li>/g, '</li>\n  '));
    }

    // -----------------------------------------------------------------------
    heading('EMPTY-STATE MESSAGE (no user selected)');
    // -----------------------------------------------------------------------
    userSelect.value = '';
    api.getRecommendations();
    console.log('#user-based-result:', elements['user-based-result'].innerHTML);
    console.log('#item-based-result:', elements['item-based-result'].innerHTML);

    heading('LOAD-ERROR MESSAGE (both panels)');
    // force a failure through the real showLoadError path
    const originalFetch = sandbox.fetch;
    sandbox.fetch = (url) => (String(url) === 'u.item'
        ? Promise.resolve({ ok: false, status: 404, text: () => Promise.resolve(''), arrayBuffer: () => Promise.resolve(new ArrayBuffer(0)) })
        : originalFetch(url));
    try {
        await api.loadData();
    } catch (e) {
        console.log('loadData() rethrew as expected:', e.message);
    }
    console.log('#user-based-result:', elements['user-based-result'].innerHTML);
    console.log('#item-based-result:', elements['item-based-result'].innerHTML);
    console.log('\n.error specificity in style.css: .result-column p.error (0,2,1) beats .result-column p (0,1,1)');

    console.log('\ndone.');
})().catch(err => {
    console.error('CHECK FAILED:', err);
    process.exit(1);
});
