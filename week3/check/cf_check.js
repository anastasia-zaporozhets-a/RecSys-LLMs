'use strict';
// Runs the REAL week3/data.js and week3/script.js inside Node vm contexts.
//
// Nothing in week3/ is re-implemented. Every matrix, every similarity, every
// neighbour search and every ranking comes from the committed files. This harness
// only supplies the browser environment (fetch / document / window / TextDecoder
// / performance) and instruments the results.
//
// Two knobs cannot be flipped from outside a context:
//   * LAMBDA is a `const` in script.js, so the LAMBDA=0 arm re-evaluates the
//     committed script.js source with that one constant substituted.
//   * cosineSimilarity is the experimental variable in section A, so the harness
//     swaps that binding in place (a function declaration is a mutable global
//     binding). Everything downstream - candidate selection, prediction, caching,
//     tie-breaks - stays the real code.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEEK3 = path.join(__dirname, '..');
const REPORTS = [1, 405];        // ordinary user / largest profile
const BIG = 1e9;                 // "return every candidate", not just top-5
const SMOKE = Number(process.env.CF_MAX_USERS || 0);   // >0 caps section A, for smoke tests

// ===========================================================================
// Browser stubs
// ===========================================================================

function makeElement(id, tag) {
    return {
        id: id, tagName: tag, innerHTML: '', value: '', options: [],
        appendChild(child) {
            this.options.push(child);
            if (this.options.length === 1) this.value = child.value;
        },
        remove(index) { this.options.splice(index, 1); }
    };
}

function makeDom() {
    const userSelect = makeElement('user-select', 'select');
    // Exactly the placeholder declared in index.html:15-17.
    userSelect.options.push({ value: '', textContent: 'Select a user', disabled: true, selected: true });
    userSelect.value = '';
    const elements = {
        'user-select': userSelect,
        'user-based-result': makeElement('user-based-result', 'div'),
        'item-based-result': makeElement('item-based-result', 'div')
    };
    return {
        userSelect, elements,
        document: {
            getElementById(id) {
                return Object.prototype.hasOwnProperty.call(elements, id) ? elements[id] : null;
            },
            createElement(tag) { return { tagName: tag, value: '', textContent: '' }; }
        }
    };
}

// fetch() -> Response-like object exposing BOTH text() and arrayBuffer().
function makeResponse(file) {
    const buffer = fs.readFileSync(file);
    return {
        ok: true, status: 200, url: 'file://' + file,
        text() { return Promise.resolve(buffer.toString('utf8')); },
        arrayBuffer() {
            return Promise.resolve(
                buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
        }
    };
}

const notFound = url => Promise.resolve({
    ok: false, status: 404, url: String(url),
    text: () => Promise.resolve(''),
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(0))
});

function makeFetch(breakItem) {
    return function fetchStub(url) {
        const file = path.join(WEEK3, String(url));
        if (String(url) === 'u.item' && breakItem) return notFound(url);
        if (!fs.existsSync(file)) return notFound(url);
        return Promise.resolve(makeResponse(file));
    };
}

// ===========================================================================
// Context construction
// ===========================================================================

function makeContext({ lamb, keepIds } = {}) {
    const { userSelect, elements, document } = makeDom();

    const sandbox = {
        console: console,
        fetch: makeFetch(false),
        document: document,
        TextDecoder: TextDecoder,
        TextEncoder: TextEncoder,
        performance: performance
    };
    sandbox.window = sandbox;
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);

    const itemSource = fs.readFileSync(path.join(WEEK3, 'data.js'), 'utf8');
    let scriptSource = fs.readFileSync(path.join(WEEK3, 'script.js'), 'utf8');
    if (!/const LAMBDA = 1;/.test(scriptSource)) throw new Error('script.js: "const LAMBDA = 1;" not found');
    if (lamb !== undefined) {
        scriptSource = scriptSource.replace(/const LAMBDA = 1;/, `const LAMBDA = ${lamb};`);
    }
    if (keepIds) {
        // The public functions project candidates down to { title, score }, so a
        // caller cannot tell which raw id was recommended. For measurement the
        // projection keeps `id` as well. Only the destructuring changes - every
        // candidate, score and sort is still produced by the committed code.
        const before = scriptSource;
        scriptSource = scriptSource
            .replace('.map(({ title, score }) => ({ title, score }))',
                     '.map(({ id, title, score }) => ({ id, title, score }))')
            .replace('.map(({ title, score, because }) => ({ title, score, because }))',
                     '.map(({ id, title, score, because }) => ({ id, title, score, because }))');
        if (scriptSource === before) throw new Error('script.js: projection rewrite did not apply');
    }

    vm.runInContext(itemSource, sandbox, { filename: 'data.js' });
    vm.runInContext(scriptSource, sandbox, { filename: 'script.js' });

    // Re-export everything. `let` bindings live in the context's global lexical
    // scope, so they are exposed as getters; the two helpers that must mutate
    // those bindings have to be defined inside the context.
    vm.runInContext(`
        globalThis.__api = {
            get movies() { return movies; }, get ratings() { return ratings; },
            get numUsers() { return numUsers; }, get numMovies() { return numMovies; },
            get ratingMatrix() { return ratingMatrix; },
            get canonicalId() { return canonicalId; }, get movieById() { return movieById; },
            get maxMovieId() { return maxMovieId; }, get genreNames() { return genreNames; },
            get GAMMA() { return GAMMA; }, get LAMBDA() { return LAMBDA; }, get N() { return N; },
            loadData, buildRatingMatrix, cosineSimilarity, meanRating, populateUserDropdown,
            getMovieColumns, itemSimilarity, movieTitle,
            getUserBasedRecommendations, getItemBasedRecommendations, getRecommendations, renderList,
            __all: null,
            __realCosine: cosineSimilarity,
            __patchCosine(fn) { cosineSimilarity = fn; },
            __restoreCosine() { cosineSimilarity = globalThis.__api.__realCosine; },
            __useFull() {
                ratings = globalThis.__api.__all.slice();
                numUsers = 0;
                for (const r of globalThis.__api.__all) numUsers = Math.max(numUsers, r.userId);
                numMovies = maxMovieId;
                buildRatingMatrix();
            },
            __useRows(rows) {
                ratings = rows;
                numUsers = 0;
                for (const r of rows) numUsers = Math.max(numUsers, r.userId);
                numMovies = maxMovieId;
                buildRatingMatrix();
            }
        };
    `, sandbox);

    return { sandbox, api: sandbox.__api, userSelect, elements, document };
}

async function boot(ctx) {
    await ctx.sandbox.onload();
    ctx.api.__all = ctx.api.ratings.slice();
    return ctx.api;
}

// buildRatingMatrix() rebinds `ratingMatrix`, so any value destructured earlier
// is stale. Always re-read through the api after an in-context rebuild.
function snap(api) {
    return {
        ratings: api.ratings,
        numUsers: api.numUsers,
        numMovies: api.numMovies,
        ratingMatrix: api.ratingMatrix
    };
}

// ===========================================================================
// Output helpers
// ===========================================================================

const pad = (s, n) => String(s).padStart(n);
const padL = (s, n) => String(s).padEnd(n);
const rule = c => c.repeat(100);
const title = t => console.log('\n' + rule('=') + '\n' + t + '\n' + rule('='));
const sub = t => console.log('\n' + t + '\n' + '-'.repeat(t.length));

function ltable(headers, rows, widths) {
    const line = cells => '  ' + cells.map((c, i) => padL(String(c), widths[i])).join('  ');
    console.log(line(headers));
    console.log('  ' + widths.map(w => '-'.repeat(w)).join('  '));
    for (const r of rows) console.log(line(r));
}

function errors(pairs) {
    let sae = 0, sse = 0, n = 0, bad = 0;
    for (const p of pairs) {
        if (!Number.isFinite(p.predicted)) { bad++; continue; }
        const e = p.actual - p.predicted;
        sae += Math.abs(e); sse += e * e; n++;
    }
    return { n, bad, mae: n ? sae / n : NaN, rmse: n ? Math.sqrt(sse / n) : NaN };
}

function median(values) {
    if (!values.length) return NaN;
    const s = values.slice().sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// The real recommendation functions return { title, score } with no id, so the
// harness maps titles back to canonical ids. data.js keeps one record per title
// in `movies`, each carrying the canonical (lowest) id, so this is 1:1.
// movieById is NOT usable here: it still holds both raw ids of a duplicated
// listing, and the lower one is not necessarily the first inserted.
function titleIndex(records) {
    const byTitle = new Map();
    const collisions = [];
    for (const m of records) {
        if (byTitle.has(m.title)) collisions.push([m.title, byTitle.get(m.title), m.id]);
        else byTitle.set(m.title, m.id);
    }
    return { byTitle, collisions, idOf: title => byTitle.get(title) };
}

// One row per (userId, canonical movie id), rating = mean of the rows that landed
// there. parseRatingData keeps every u.data line, so a user who rated both copies
// of a duplicated listing contributes two rows; the matrix averages them, but a
// train/test split has to see them as the single observation they represent.
function mergedRatings(all) {
    const acc = new Map();
    for (const r of all) {
        const key = r.userId * 1000000 + r.canonical;
        const e = acc.get(key);
        if (e === undefined) acc.set(key, { userId: r.userId, canonical: r.canonical, sum: r.rating, n: 1 });
        else { e.sum += r.rating; e.n++; }
    }
    const rows = [...acc.values()];
    return { rows: rows.map(e => ({ userId: e.userId, canonical: e.canonical, rating: e.sum / e.n })),
             collapsed: rows.filter(e => e.n > 1).length };
}

// A candidate id the evaluation must ignore: a non-canonical duplicate listing,
// or the metadata-free placeholder. Under mean imputation these ids acquire a
// real cosine, so the filter is applied to the returned candidate list of every
// variant, not just to the shipped arms.
function excludedId(canonicalId, id) {
    return id === 267 || canonicalId.get(id) !== id;
}

// Plain co-rated cosine, used only as an evaluated ALTERNATIVE in section A.
function coRatedOnlyCosine() {
    return function (a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) {
            const x = a[i], y = b[i];
            if (x === 0 || y === 0) continue;
            dot += x * y; na += x * x; nb += y * y;
        }
        const d = Math.sqrt(na * nb);
        return d === 0 ? 0 : dot / d;
    };
}

// Mean imputation. Every missing entry takes the train mean of the user at that
// position: in a row index i is user i, and in a column entry i is also user i.
// One filled vector per row and per column, memoised on the input array identity
// (the real code hands over stable row / column objects).
//
// Note: once nothing is 0, the shipped cosine's co-rated counter reaches the full
// vector length, so min(coRated, 50) / 50 is exactly 1 and the significance
// weighting becomes inactive. This arm therefore measures plain cosine on filled
// data, which is the point of including it.
function makeImputedCosine(userMeanTrain, numMovies) {
    const rowLen = numMovies + 1;
    const cache = new Map();
    function fill(vec) {
        let out = cache.get(vec);
        if (out !== undefined) return out;
        out = new Array(vec.length);
        out[0] = 0;                                  // index 0 is padding, never filled
        for (let i = 1; i < vec.length; i++) out[i] = vec[i] === 0 ? userMeanTrain[i] : vec[i];
        cache.set(vec, out);
        return out;
    }
    return function (a, b) {
        if (a.length !== b.length) throw new Error('vector length mismatch');
        const A = fill(a);
        const B = fill(b);
        let dot = 0, na = 0, nb = 0;
        for (let i = 1; i < A.length; i++) { dot += A[i] * B[i]; na += A[i] * A[i]; nb += B[i] * B[i]; }
        const d = Math.sqrt(na * nb);
        return d === 0 ? 0 : dot / d;
    };
}

// ===========================================================================
// main
// ===========================================================================

// Per-section wall clock, plus a hard 2-minute budget per section so a slow
// section stops the run instead of hanging it.
const LIMIT_MS = Number(process.env.CF_LIMIT_MS || 120000);
const timings = [];
let sectionName = 'startup';
let sectionT0 = performance.now();

class SectionTimeout extends Error {}

function watch() {
    if (performance.now() - sectionT0 > LIMIT_MS) {
        throw new SectionTimeout(`section ${sectionName} exceeded the ${LIMIT_MS / 1000} s budget`);
    }
}

function beginSection(name) {
    timings.push([sectionName, performance.now() - sectionT0]);
    sectionName = name;
    sectionT0 = performance.now();
}

function endSection() {
    watch();
    timings.push([sectionName, performance.now() - sectionT0]);
    const [name, ms] = timings[timings.length - 1];
    console.log(`\n[${name}] ${(ms / 1000).toFixed(1)} s`);
    sectionName = 'between sections';
    sectionT0 = performance.now();
}

(async function main() {
    const t0 = performance.now();

    // =======================================================================
    beginSection('0. load error text + baseline top-5');
    title('(a)+(b) LOAD ERROR TEXT, AND COMMITTED BASELINE TOP-5 FOR USERS 1 AND 405');
    // =======================================================================
    const ctx = makeContext({ keepIds: true });
    const api = await boot(ctx);
    const { movies, ratings, numUsers, numMovies, ratingMatrix, canonicalId, movieById,
            genreNames, GAMMA, LAMBDA, N } = api;
    const T = titleIndex(movies);

    sub('environment');
    console.log(`  movies (one per title)   ${movies.length}`);
    console.log(`  merged ratings           ${ratings.length}`);
    console.log(`  numUsers / numMovies     ${numUsers} / ${numMovies}   (numMovies = max movie id)`);
    console.log(`  ratingMatrix             ${ratingMatrix.length} x ${ratingMatrix[0].length}`);
    console.log(`  genreNames               ${genreNames.length} -> ${genreNames[0]} ... ${genreNames[genreNames.length - 1]}`);
    console.log(`  duplicate ids remapped   ${[...canonicalId].filter(([k, v]) => k !== v).length}`);
    console.log(`  movieById size           ${movieById.size}  (placeholder 267 excluded: ${!movieById.has(267)})`);
    console.log(`  titles with U+FFFD       ${movies.filter(m => m.title.includes('�')).length}`);
    console.log(`  title -> id collisions   ${T.collisions.length}`);
    console.log(`  constants                GAMMA=${GAMMA} LAMBDA=${LAMBDA} N=${N}`);

    sub('(a) load error text, both panels, forced with a 404 on u.item');
    {
        const errCtx = makeContext();
        errCtx.sandbox.fetch = makeFetch(true);
        // data.js logs the failure itself; keep the stack trace out of the report.
        const logged = [];
        errCtx.sandbox.console = Object.assign({}, console, {
            error: (...a) => logged.push(a.map(String).join(' '))
        });
        try { await errCtx.sandbox.onload(); } catch (e) { /* already reported by data.js */ }
        console.log(`  console.error calls from data.js: ${logged.length} (message starts "${logged[0] ? logged[0].slice(0, 34) : ''}")`);
        console.log('  #user-based-result: ' + errCtx.elements['user-based-result'].innerHTML);
        console.log('  #item-based-result: ' + errCtx.elements['item-based-result'].innerHTML);
        console.log('  .result-column p.error (0,2,1) outranks .result-column p (0,1,1) -> red');
    }

    for (const u of REPORTS) {
        const row = ratingMatrix[u];
        let rated = 0;
        for (let j = 1; j <= numMovies; j++) if (row[j] !== 0) rated++;
        const mean = api.meanRating(row);

        const ranked = [];
        for (let other = 1; other <= numUsers; other++) {
            if (other === u) continue;
            const s = api.cosineSimilarity(row, ratingMatrix[other]);
            if (s > 0) ranked.push({ userId: other, s });
        }
        ranked.sort((a, b) => (b.s - a.s) || (a.userId - b.userId));
        const nbrs = ranked.slice(0, N);
        const coRated = id => {
            let c = 0;
            for (let j = 1; j <= numMovies; j++) if (row[j] !== 0 && ratingMatrix[id][j] !== 0) c++;
            return c;
        };
        const canonCount = id => ratings.filter(r => r.canonical === id).length;

        sub(`(b) user ${u}: ${rated} ratings, mean ${mean.toFixed(4)}, ${ranked.length} neighbours with sim>0`);
        console.log(`  top ${N} = [${nbrs.map(n => n.userId).join(', ')}]`);
        console.log(`  sim range ${nbrs[nbrs.length - 1].s.toFixed(6)} .. ${nbrs[0].s.toFixed(6)} | ` +
            `co-rated range ${Math.min(...nbrs.map(n => coRated(n.userId)))}..${Math.max(...nbrs.map(n => coRated(n.userId)))} | ` +
            `sim === 1 exactly: ${nbrs.filter(n => n.s === 1).length}`);

        const ub = api.getUserBasedRecommendations(u);
        ltable(['UB rank', 'score', 'support (nbrs)', 'ratings in u.data', 'title'],
            ub.map((it, i) => [i + 1, it.score.toFixed(6),
                nbrs.filter(n => ratingMatrix[n.userId][T.idOf(it.title)] !== 0).length,
                canonCount(T.idOf(it.title)), it.title]),
            [8, 10, 14, 17, 48]);

        const ib = api.getItemBasedRecommendations(u);
        ltable(['IB rank', 'score', 'ratings in u.data', 'title', 'because (top anchor)'],
            ib.map((it, i) => [i + 1, it.score.toFixed(6),
                canonCount(T.idOf(it.title)), it.title, it.because]),
            [8, 10, 17, 44, 44]);
    }

    // =======================================================================
    beginSection('A. offline accuracy');
    title('A. OFFLINE ACCURACY  (train/test split on the merged ratings)');
    sub('collapse first: one row per (userId, canonical id), rating = mean of the copies');
    const isTest = (u, c) => (u + c) % 5 === 0;
    const merged = mergedRatings(api.__all);
    const trainRows = merged.rows.filter(r => !isTest(r.userId, r.canonical));
    const testRows = merged.rows.filter(r => isTest(r.userId, r.canonical));
    console.log(`  parsed rows                ${api.__all.length}   (100000 less 9 ratings of movie 267)`);
    console.log(`  (user, canonical) pairs rated both copies  ${merged.collapsed}`);
    console.log(`  merged ratings             ${merged.rows.length}`);
    console.log(`  test pairs ((u+c) % 5==0)  ${testRows.length}`);
    console.log(`  train pairs                ${trainRows.length}`);
    for (const [label, got, want] of [['merged', merged.rows.length, 99684],
                                      ['test', testRows.length, 20123],
                                      ['train', trainRows.length, 79561]]) {
        if (got !== want) throw new Error(`${label} count ${got}, expected ${want}`);
    }
    console.log('  counts match the expected 99684 / 20123 / 79561.');
    console.log('  the matrix is rebuilt from the TRAIN rows through the real buildRatingMatrix, so');
    console.log('  mean_u and every similarity use train ratings only. A test movie with sum(s) = 0');
    console.log('  falls back to mean_u. Candidate ids that are non-canonical duplicates or movie 267');
    console.log('  are dropped from every variant, including mean imputation.');
    if (SMOKE) console.log(`  *** SMOKE MODE: only the first ${SMOKE} test users are scored ***`);

    const byUser = new Map();
    for (const r of testRows) {
        if (!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push(r);
    }
    const testUsers = [...byUser.keys()].sort((a, b) => a - b);
    const aRows = [];
    for (const lamb of [0, 1]) {
        const arm = makeContext({ lamb, keepIds: true });
        const A = await boot(arm);

        for (const variant of ['co-rated only', 'weighted gamma 50', 'mean imputation']) {
            const ts = performance.now();
            A.__useRows(trainRows);

            if (variant === 'mean imputation') {
                const us = new Float64Array(A.numUsers + 1), uc = new Int32Array(A.numUsers + 1);
                for (const r of A.ratings) { us[r.userId] += r.rating; uc[r.userId]++; }
                const uMean = new Float64Array(A.numUsers + 1);
                for (let u = 1; u <= A.numUsers; u++) uMean[u] = uc[u] ? us[u] / uc[u] : 0;
                A.__patchCosine(makeImputedCosine(uMean, A.numMovies));
            } else if (variant === 'co-rated only') {
                A.__patchCosine(coRatedOnlyCosine());
            } else {
                A.__restoreCosine();
            }

            const ubErr = [], ibErr = [], baseErr = [];
            const users = SMOKE ? testUsers.slice(0, SMOKE) : testUsers;
            let dropped = 0;
            for (const u of users) {
                const row = A.ratingMatrix[u];
                const meanU = A.meanRating(row);
                if (!Number.isFinite(meanU) || meanU === 0) continue;
                // topK = BIG hands back every surviving candidate with its id, so
                // each test movie's score is a map lookup. One call per user.
                const ubAll = A.getUserBasedRecommendations(u, BIG);
                const ibAll = A.getItemBasedRecommendations(u, BIG);
                const ubList = ubAll.filter(c => !excludedId(canonicalId, c.id));
                const ibList = ibAll.filter(c => !excludedId(canonicalId, c.id));
                dropped += (ubAll.length - ubList.length) + (ibAll.length - ibList.length);
                const ubMap = new Map(ubList.map(c => [c.id, c.score]));
                const ibMap = new Map(ibList.map(c => [c.id, c.score]));
                for (const r of byUser.get(u)) {
                    if (row[r.canonical] !== 0) continue;      // would mean the split leaked
                    ubErr.push({ actual: r.rating, predicted: ubMap.has(r.canonical) ? ubMap.get(r.canonical) : meanU });
                    ibErr.push({ actual: r.rating, predicted: ibMap.has(r.canonical) ? ibMap.get(r.canonical) : meanU });
                    baseErr.push({ actual: r.rating, predicted: meanU });
                }
                watch();
            }
            if (variant === 'co-rated only') {
                const e = errors(baseErr);
                aRows.push({ lamb, arm: 'BASELINE: predict the user mean', n: e.n, mae: e.mae, rmse: e.rmse, drop: 0 });
            }
            const eu = errors(ubErr), ei = errors(ibErr);
            aRows.push({ lamb, arm: variant + ' -> UB', n: eu.n, mae: eu.mae, rmse: eu.rmse, drop: dropped });
            aRows.push({ lamb, arm: variant + ' -> IB', n: ei.n, mae: ei.mae, rmse: ei.rmse, drop: dropped });
            // Printed per arm so a budget stop still leaves usable numbers.
            console.log(`    LAMBDA=${lamb} ${variant.padEnd(18)} ${ubErr.length} pairs, ` +
                `${dropped} excluded cands  ` +
                `UB MAE ${eu.mae.toFixed(4)} RMSE ${eu.rmse.toFixed(4)}  ` +
                `IB MAE ${ei.mae.toFixed(4)} RMSE ${ei.rmse.toFixed(4)}  ` +
                `${((performance.now() - ts) / 1000).toFixed(1)} s`);
            A.__restoreCosine();
        }
    }
    console.log('');
    ltable(['LAMBDA', 'arm', 'test pairs', 'MAE', 'RMSE', 'excluded cands'],
        aRows.map(r => [r.lamb, r.arm, r.n, r.mae.toFixed(4), r.rmse.toFixed(4), r.drop]),
        [7, 32, 11, 9, 9, 15]);
    console.log('  excluded cands = non-canonical duplicate ids and movie 267, removed from the candidate');
    console.log('  list of every variant before scoring. 0 for the shipped arms; mean imputation fills');
    console.log('  an empty column with user means, so only there do such ids score at all.');
    endSection();

    // =======================================================================
    beginSection('B. full data, 943 users');
    title('B. FULL DATA, SHIPPED SETTINGS, ALL 943 USERS');
    // =======================================================================
    api.__useFull();
    const catalogCounts = new Map();
    for (const r of api.__all) catalogCounts.set(r.canonical, (catalogCounts.get(r.canonical) || 0) + 1);
    const catalogMedian = median([...catalogCounts.values()]);
    const twinOf = new Map();
    for (const [raw, canon] of canonicalId) if (raw !== canon) twinOf.set(canon, raw);
    console.log(`  catalog: ${catalogCounts.size} films, median ratings per film = ${catalogMedian}`);
    console.log(`  ${twinOf.size} films have a duplicate listing in u.item`);

    const blank = () => ({ overlap: 0, dupRaw: 0, twinTitle: 0, placeholder: 0,
                           idMismatch: 0, ratings: [], distinct: new Set() });
    const stats = { UB: blank(), IB: blank() };
    const tB = performance.now();
    for (let u = 1; u <= numUsers; u++) {
        const row = ratingMatrix[u];
        for (const [algo, fn] of [['UB', api.getUserBasedRecommendations], ['IB', api.getItemBasedRecommendations]]) {
            const s = stats[algo];
            for (const item of fn(u)) {
                // A raw duplicate id would be canonicalId.get(id) !== id: that is
                // the defect the spec forbids. Recommending the canonical id of a
                // film that happens to be listed twice is not a defect.
                if (canonicalId.get(item.id) !== item.id) s.dupRaw++;
                if (twinOf.has(item.id)) s.twinTitle++;
                if (T.idOf(item.title) !== item.id) s.idMismatch++;
                if (item.id === 267) s.placeholder++;
                if (row[item.id] !== 0) s.overlap++;
                s.ratings.push(catalogCounts.get(item.id) || 0);
                s.distinct.add(item.id);
            }
        }
        if (u % 50 === 0) {
            console.log(`    ...user ${u}/${numUsers} at ${((performance.now() - tB) / 1000).toFixed(1)} s`);
            watch();
        }
    }
    console.log(`  scored all ${numUsers} users in ${((performance.now() - tB) / 1000).toFixed(1)} s`);
    console.log('');
    const below = s => (s.ratings.filter(v => v < catalogMedian).length / s.ratings.length * 100).toFixed(1) + '%';
    ltable(['check', 'UB', 'IB', 'required'],
        [
            ['Top-5 items the user had already rated', stats.UB.overlap, stats.IB.overlap, '0'],
            ['duplicate raw ids recommended', stats.UB.dupRaw, stats.IB.dupRaw, '0'],
            ['placeholder movie 267 recommended', stats.UB.placeholder, stats.IB.placeholder, '0'],
            ['title/id disagreement (harness sanity)', stats.UB.idMismatch, stats.IB.idMismatch, '0'],
            ['recs of a film listed twice (canonical)', stats.UB.twinTitle, stats.IB.twinTitle, 'allowed'],
            ['distinct movies recommended', stats.UB.distinct.size, stats.IB.distinct.size, '-'],
            ['median #ratings of recommended movies', median(stats.UB.ratings), median(stats.IB.ratings),
                `catalog median ${catalogMedian}`],
            ['share below the catalog median', below(stats.UB), below(stats.IB), '-'],
            ['recommendation slots', stats.UB.ratings.length, stats.IB.ratings.length, `${numUsers * 5} each`]
        ],
        [40, 12, 12, 22]);
    endSection();

    // =======================================================================
    beginSection('C. cold start');
    title('C. COLD START');
    // =======================================================================
    sub('C1. synthetic user 999 with exactly one rating: movie 50 = 5');
    // Injected through the real pipeline: a rating row, then the real matrix build.
    vm.runInContext(`
        ratings.push({ userId: 999, itemId: 50, canonical: 50, rating: 5, timestamp: 0 });
        numUsers = Math.max(numUsers, 999);
        buildRatingMatrix();
    `, ctx.sandbox);
    const s = snap(api);
    const sRow = s.ratingMatrix[999];
    let sRated = 0;
    for (let j = 1; j <= s.numMovies; j++) if (sRow[j] !== 0) sRated++;
    console.log(`  synthetic row: ${sRated} rating(s) -> movie 50 = ${sRow[50]}, user mean = ${api.meanRating(sRow).toFixed(4)}`);
    console.log(`  matrix is now ${s.ratingMatrix.length} x ${s.ratingMatrix[0].length}, numUsers = ${s.numUsers}`);

    const rated50 = api.__all.filter(r => r.canonical === 50).length;
    let coOne = 0, wOne = 0, coPos = 0, wPos = 0;
    const coFn = coRatedOnlyCosine();
    const coRanked = [], wRanked = [];
    for (let v = 1; v < s.numUsers; v++) {                 // v < 999: the synthetic user is not its own neighbour
        if (s.ratingMatrix[v][50] === 0) continue;
        const co = coFn(sRow, s.ratingMatrix[v]);
        const wt = api.__realCosine(sRow, s.ratingMatrix[v]);
        if (co > 0) { coPos++; coRanked.push({ v, s: co }); }
        if (wt > 0) { wPos++; wRanked.push({ v, s: wt }); }
        if (co === 1) coOne++;
        if (wt === 1) wOne++;
    }
    const topOf = list => list.sort((a, b) => (b.s - a.s) || (a.v - b.v)).slice(0, N);
    const coTop = topOf(coRanked), wTop = topOf(wRanked);
    const coIds = new Set(coTop.map(x => x.v)), wIds = new Set(wTop.map(x => x.v));
    const shared = [...coIds].filter(x => wIds.has(x)).length;

    console.log(`  real users who rated movie 50: ${rated50}`);
    console.log(`  similarity to the 1-rating user, over those ${rated50} users:`);
    console.log(`    co-rated only : sim > 0 for ${coPos}, sim === 1.0 for ${coOne}  (min ${Math.min(...coRanked.map(x => x.s)).toFixed(4)}, max ${Math.max(...coRanked.map(x => x.s)).toFixed(4)})`);
    console.log(`    weighted g=50 : sim > 0 for ${wPos}, sim === 1.0 for ${wOne}  (min ${Math.min(...wRanked.map(x => x.s)).toFixed(4)}, max ${Math.max(...wRanked.map(x => x.s)).toFixed(4)})`);
    console.log('  one co-rated film gives raw cosine exactly 1.0; significance weighting multiplies it');
    console.log('  by min(1,50)/50 = 0.02, so no pair can reach 1.0 and the ranking below cannot be a tie.');
    console.log(`  top ${N} neighbourhood: co-rated [${coTop.map(x => x.v).join(', ')}]`);
    console.log(`  top ${N} neighbourhood: weighted [${wTop.map(x => x.v).join(', ')}]`);
    console.log(`  neighbourhoods overlap in ${shared} of ${N} users`);

    api.__patchCosine(coFn);
    const ubCo = api.getUserBasedRecommendations(999);
    const ibCo = api.getItemBasedRecommendations(999);
    api.__restoreCosine();
    const ubW = api.getUserBasedRecommendations(999);
    const ibW = api.getItemBasedRecommendations(999);

    console.log('');
    ltable(['arm', 'algo', 'rank', 'score', 'title'],
        [
            ...ubCo.map((x, i) => ['co-rated only', 'UB', i + 1, x.score.toFixed(4), x.title]),
            ...ibCo.map((x, i) => ['co-rated only', 'IB', i + 1, x.score.toFixed(4), x.title]),
            ...ubW.map((x, i) => ['weighted', 'UB', i + 1, x.score.toFixed(4), x.title]),
            ...ibW.map((x, i) => ['weighted', 'IB', i + 1, x.score.toFixed(4), x.title])
        ],
        [14, 5, 5, 9, 54]);
    console.log(`  co-rated-only: ${ubCo.length} UB items, ${ibCo.length} IB items`);
    console.log(`  weighted     : ${ubW.length} UB items, ${ibW.length} IB items`);
    const span = list => `${Math.min(...list.map(x => x.score)).toFixed(4)}..${Math.max(...list.map(x => x.score)).toFixed(4)}`;
    console.log(`  score span  co-rated: UB ${span(ubCo)}  IB ${span(ibCo)}`);
    console.log(`  score span  weighted: UB ${span(ubW)}  IB ${span(ibW)}`);
    console.log(`  the single rating is ${sRow[50]}, so mean_u = ${api.meanRating(sRow).toFixed(1)} and LAMBDA=1 drags every`);
    console.log('  prediction onto that value. The arms differ in WHO the neighbours are, not in the scores.');

    sub('C2. synthetic film with zero ratings');
    const zid = api.maxMovieId + 1;
    vm.runInContext(`
        movieById.set(maxMovieId + 1, { id: maxMovieId + 1, title: 'Synthetic Film (0 ratings)', genres: [] });
        canonicalId.set(maxMovieId + 1, maxMovieId + 1);
        numMovies = maxMovieId + 1;
    `, ctx.sandbox);
    api.buildRatingMatrix();          // new object identity -> the real caches invalidate
    console.log(`  added id ${zid} ("Synthetic Film (0 ratings)"), numMovies is now ${api.numMovies}`);

    const zcols = api.getMovieColumns();
    let zCosNonZero = 0, zListed = 0;
    for (let j = 1; j <= 200; j++) {
        if (j === zid) continue;
        if (api.cosineSimilarity(zcols[zid], zcols[j]) !== 0) zCosNonZero++;
    }
    const ZT = titleIndex([...api.movies, { id: zid, title: 'Synthetic Film (0 ratings)' }]);
    for (let u = 1; u <= 25; u++) {
        if (api.getUserBasedRecommendations(u, BIG).some(c => c.id === zid)) zListed++;
        if (api.getItemBasedRecommendations(u, BIG).some(c => c.id === zid)) zListed++;
    }
    let zRated = 0;
    for (let u = 0; u <= api.numUsers; u++) if (api.ratingMatrix[u][zid] !== 0) zRated++;
    console.log(`  users with a non-zero cell in its column           : ${zRated}`);
    console.log(`  cosineSimilarity(zeroColumn, 200 real columns) != 0 : ${zCosNonZero}  (denominator 0 -> returns 0)`);
    console.log(`  listed as a candidate for 25 users (UB or IB)      : ${zListed}  (sum(s) can never be > 0)`);
    console.log(`  its title is resolvable via the same index: ${ZT.idOf('Synthetic Film (0 ratings)') === zid}`);
    console.log('  -> an unrated film can never be recommended by either algorithm.');

    vm.runInContext('numMovies = maxMovieId;', ctx.sandbox);
    api.buildRatingMatrix();
    api.__useFull();
    vm.runInContext('ratings = globalThis.__api.__all.filter(r => r.userId !== 999); numUsers = 943; buildRatingMatrix();', ctx.sandbox);
    api.__restoreCosine();
    console.log(`  restored: numUsers=${api.numUsers} numMovies=${api.numMovies} ratings=${api.ratings.length}`);
    endSection();

    // =======================================================================
    beginSection('D. timing');
    title('D. TIMING (performance.now)');
    // =======================================================================
    sub('IB memoises the rating columns and the item-item similarities; UB caches nothing,');
    console.log('  so its two calls cost the same. Each measurement starts from a freshly built matrix.');
    for (const u of REPORTS) {
        const rows = [];
        for (const [algo, fn] of [['UB', api.getUserBasedRecommendations], ['IB', api.getItemBasedRecommendations]]) {
            api.__useFull();                     // rebuild -> new matrix -> caches cleared
            let t = performance.now(); fn(u); const first = performance.now() - t;
            t = performance.now(); fn(u); const second = performance.now() - t;
            rows.push([algo, u, first.toFixed(1), second.toFixed(1), (first / second).toFixed(1) + 'x']);
        }
        ltable(['algo', 'user', 'first call (ms)', 'second call (ms)', 'speedup'], rows, [6, 6, 17, 18, 9]);
        watch();
    }
    endSection();

    // =======================================================================
    title('SECTION TIMINGS');
    // =======================================================================
    ltable(['section', 'seconds'],
        timings.filter(([n]) => n !== 'between sections' && n !== 'startup')
               .map(([n, ms]) => [n, (ms / 1000).toFixed(1)]),
        [34, 9]);
    console.log(`\ntotal harness wall clock: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    console.log('\ndone.');
})().catch(err => {
    if (err instanceof SectionTimeout) {
        console.error('\nSTOPPED: ' + err.message);
        console.error('elapsed in that section: ' + ((performance.now() - sectionT0) / 1000).toFixed(1) + ' s');
        console.error('completed sections:');
        for (const [n, ms] of timings) {
            if (n === 'between sections' || n === 'startup') continue;
            console.error(`  ${padL(n, 34)} ${(ms / 1000).toFixed(1)} s`);
        }
        process.exit(2);
    }
    console.error('CHECK FAILED:', err);
    process.exit(1);
});
