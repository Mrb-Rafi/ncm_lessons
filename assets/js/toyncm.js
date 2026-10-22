/* ToyNCM — a tiny, fully working multiscale residual VQ + coarse-to-fine sampler.
   It mirrors the structure of HazyResearch/ncm (models/quant.py, models/next_concept.py)
   at toy scale so it can run live in the browser:
     - encoder  : word -> 2-D embedding (a lookup table, like the repo's enc_layers=0 setting)
     - Down_k   : area (average) pooling to l_k positions   (F.interpolate(mode="area"))
     - R_k      : linear upsampling back to L positions     (F.interpolate(mode="linear"))
     - codebooks: one per level, fitted with k-means (the repo uses EMA k-means updates)
     - decoder  : nearest word embedding (softmax over negative squared distance)
     - generator: a non-parametric stand-in for the transformer — p(r_k | z^(k-1)) is
                  estimated from training sentences with similar cumulative reconstructions,
                  and all l_k codes of a level are sampled independently (paper eq. 5).
*/
(function (global) {
  'use strict';

  // ---------- seeded RNG ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- vocabulary with hand-placed 2-D "meaning" coordinates ----------
  // Similar words live near each other, just like learned embeddings.
  const GROUPS = {
    det:   { c: [-3.2, 2.6], words: ['the', 'a', 'my', 'one'] },
    adj:   { c: [-0.4, 3.4], words: ['big', 'small', 'happy', 'red', 'old', 'sleepy'] },
    animal:{ c: [3.0, 2.4],  words: ['cat', 'dog', 'bird', 'fox', 'frog', 'cow'] },
    verb:  { c: [3.2, -1.0], words: ['sat', 'slept', 'ran', 'jumped', 'hid', 'played'] },
    prep:  { c: [0.2, -3.0], words: ['on', 'under', 'near', 'in', 'by'] },
    place: { c: [-3.2, -1.6],words: ['mat', 'tree', 'box', 'bed', 'park', 'house'] },
    end:   { c: [-0.6, 0.0], words: ['.'] }
  };
  const VOCAB = [], EMB = [], GROUP_OF = [];
  Object.keys(GROUPS).forEach(function (g) {
    const G = GROUPS[g], n = G.words.length;
    G.words.forEach(function (w, i) {
      const ang = (2 * Math.PI * i) / Math.max(n, 1) + 0.4;
      const rad = n > 1 ? 0.95 : 0;
      VOCAB.push(w); GROUP_OF.push(g);
      EMB.push([G.c[0] + rad * Math.cos(ang), G.c[1] + rad * Math.sin(ang)]);
    });
  });
  const IDX = {}; VOCAB.forEach(function (w, i) { IDX[w] = i; });

  const L = 8; // sentence length in tokens
  // template: det adj animal verb prep det place .
  const TEMPLATE = ['det', 'adj', 'animal', 'verb', 'prep', 'det', 'place', 'end'];

  function makeCorpus(n, seed) {
    const rnd = mulberry32(seed || 7), out = [];
    for (let s = 0; s < n; s++) {
      out.push(TEMPLATE.map(function (g) {
        const ws = GROUPS[g].words; return IDX[ws[Math.floor(rnd() * ws.length)]];
      }));
    }
    return out;
  }

  // ---------- 1-D resampling over the sequence axis (each of d channels) ----------
  function downArea(z, n) {           // z: L x d  ->  n x d (average pooling, like mode="area")
    const Lz = z.length, d = z[0].length, out = [];
    for (let j = 0; j < n; j++) {
      const a = Math.floor((j * Lz) / n), b = Math.ceil(((j + 1) * Lz) / n);
      const v = new Array(d).fill(0);
      for (let i = a; i < b; i++) for (let c = 0; c < d; c++) v[c] += z[i][c];
      for (let c = 0; c < d; c++) v[c] /= (b - a);
      out.push(v);
    }
    return out;
  }
  function upLinear(q, n) {           // q: m x d -> n x d (linear, align_corners=False)
    const m = q.length, d = q[0].length, out = [];
    for (let i = 0; i < n; i++) {
      let src = ((i + 0.5) * m) / n - 0.5;
      if (src < 0) src = 0; if (src > m - 1) src = m - 1;
      const i0 = Math.floor(src), i1 = Math.min(i0 + 1, m - 1), t = src - i0;
      const v = new Array(d);
      for (let c = 0; c < d; c++) v[c] = q[i0][c] * (1 - t) + q[i1][c] * t;
      out.push(v);
    }
    return out;
  }
  function d2(a, b) { let s = 0; for (let c = 0; c < a.length; c++) { const t = a[c] - b[c]; s += t * t; } return s; }
  function nearest(v, book) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < book.length; j++) { const dd = d2(v, book[j]); if (dd < bd) { bd = dd; best = j; } }
    return best;
  }
  function kmeans(points, k, iters, rnd) {
    const P = points.length;
    let cent = [];
    // k-means++ style seeding
    cent.push(points[Math.floor(rnd() * P)].slice());
    while (cent.length < k) {
      const ds = points.map(function (p) { let m = Infinity; cent.forEach(function (c) { m = Math.min(m, d2(p, c)); }); return m; });
      const tot = ds.reduce(function (a, b) { return a + b; }, 0);
      if (tot < 1e-12) { cent.push(points[Math.floor(rnd() * P)].slice()); continue; }
      let r = rnd() * tot, i = 0;
      while (r > ds[i] && i < P - 1) { r -= ds[i]; i++; }
      cent.push(points[i].slice());
    }
    for (let it = 0; it < iters; it++) {
      const sum = cent.map(function (c) { return new Array(c.length).fill(0); }), cnt = new Array(k).fill(0);
      points.forEach(function (p) { const j = nearest(p, cent); cnt[j]++; for (let c = 0; c < p.length; c++) sum[j][c] += p[c]; });
      cent = cent.map(function (c, j) { return cnt[j] ? sum[j].map(function (s) { return s / cnt[j]; }) : c; });
    }
    return cent;
  }

  // ---------- the multiscale residual VQ tokenizer ----------
  function Tokenizer(opts) {
    opts = opts || {};
    this.levels = opts.levels || [1, 2, 4, 8];
    this.sizes = opts.sizes || [8, 24, 48, 64];
    this.seed = opts.seed || 3;
    this.books = [];
  }
  Tokenizer.prototype.embed = function (ids) { return ids.map(function (i) { return EMB[i].slice(); }); };

  // Encode: returns codes per level and every intermediate quantity (for visualisation).
  Tokenizer.prototype.encode = function (ids, maxLevels) {
    const z = this.embed(ids), K = maxLevels || this.levels.length;
    let rest = z.map(function (v) { return v.slice(); });
    let zhat = z.map(function (v) { return v.map(function () { return 0; }); });
    const trace = [];
    for (let k = 0; k < K; k++) {
      const lk = this.levels[k];
      const u = lk === L ? rest.map(function (v) { return v.slice(); }) : downArea(rest, lk);
      const book = this.books[k];
      const codes = u.map(function (v) { return nearest(v, book); });
      const q = codes.map(function (j) { return book[j]; });
      const contrib = lk === L ? q.map(function (v) { return v.slice(); }) : upLinear(q, L);
      zhat = zhat.map(function (v, i) { return v.map(function (x, c) { return x + contrib[i][c]; }); });
      rest = z.map(function (v, i) { return v.map(function (x, c) { return x - zhat[i][c]; }); });
      trace.push({ level: k, lk: lk, down: u, codes: codes, q: q, contrib: contrib,
                   zhat: zhat.map(function (v) { return v.slice(); }),
                   residual: rest.map(function (v) { return v.slice(); }) });
    }
    return { z: z, trace: trace, codes: trace.map(function (t) { return t.codes; }) };
  };
  // Cumulative reconstruction from codes of the first k levels (eq. 8 / Alg. 2 line 7)
  Tokenizer.prototype.zhatFromCodes = function (codes) {
    let zhat = []; for (let i = 0; i < L; i++) zhat.push([0, 0]);
    const out = [];
    for (let k = 0; k < codes.length; k++) {
      const book = this.books[k], lk = this.levels[k];
      const q = codes[k].map(function (j) { return book[j]; });
      const contrib = lk === L ? q : upLinear(q, L);
      zhat = zhat.map(function (v, i) { return [v[0] + contrib[i][0], v[1] + contrib[i][1]]; });
      out.push(zhat.map(function (v) { return v.slice(); }));
    }
    return out;
  };
  Tokenizer.prototype.decode = function (zhat) { return zhat.map(function (v) { return nearest(v, EMB); }); };

  // Greedy level-by-level training (the real model learns encoder, decoder and EMA codebooks jointly)
  Tokenizer.prototype.train = function (corpus) {
    const rnd = mulberry32(this.seed);
    this.books = [];
    let rests = corpus.map(function (s) { return s.map(function (i) { return EMB[i].slice(); }); });
    for (let k = 0; k < this.levels.length; k++) {
      const lk = this.levels[k], pts = [];
      const downs = rests.map(function (r) { return lk === L ? r : downArea(r, lk); });
      downs.forEach(function (u) { u.forEach(function (v) { pts.push(v); }); });
      const book = kmeans(pts, this.sizes[k], 12, rnd);
      this.books.push(book);
      rests = rests.map(function (r, s) {
        const q = downs[s].map(function (v) { return book[nearest(v, book)]; });
        const contrib = lk === L ? q : upLinear(q, L);
        return r.map(function (v, i) { return [v[0] - contrib[i][0], v[1] - contrib[i][1]]; });
      });
    }
    return this;
  };
  Tokenizer.prototype.accuracy = function (corpus, uptoLevel) {
    let ok = 0, tot = 0; const self = this;
    corpus.forEach(function (s) {
      const enc = self.encode(s, uptoLevel);
      const dec = self.decode(enc.trace[enc.trace.length - 1].zhat);
      dec.forEach(function (w, i) { tot++; if (w === s[i]) ok++; });
    });
    return ok / tot;
  };

  // ---------- toy generator: p(r_k | z^(k-1)) from neighbours, sampled in parallel ----------
  function Generator(tok, corpus) {
    this.tok = tok;
    this.data = corpus.map(function (s) {
      const e = tok.encode(s);
      return { codes: e.codes, zhats: e.trace.map(function (t) { return t.zhat; }) };
    });
    const K = tok.levels.length, c0 = {};
    this.data.forEach(function (d) { const c = d.codes[0][0]; c0[c] = (c0[c] || 0) + 1; });
    this.unigram0 = c0;                 // level-1 code frequencies (cf. repo's level0_unigram)
    this.K = K;
  }
  // returns per-position probability tables for level k (0-based, k>=1) given zhat^(k-1)
  Generator.prototype.levelProbs = function (k, zhatPrev, tau) {
    const tok = this.tok, lk = tok.levels[k], V = tok.books[k].length;
    const dists = this.data.map(function (d) { let s = 0; for (let i = 0; i < L; i++) s += d2(d.zhats[k - 1][i], zhatPrev[i]); return s; });
    const m = Math.min.apply(null, dists);
    const w = dists.map(function (x) { return Math.exp(-(x - m) / Math.max(tau, 1e-3)); });
    const probs = [];
    for (let i = 0; i < lk; i++) probs.push(new Array(V).fill(1e-6));
    this.data.forEach(function (d, s) { if (w[s] < 1e-9) return; for (let i = 0; i < lk; i++) probs[i][d.codes[k][i]] += w[s]; });
    return probs.map(function (p) { const t = p.reduce(function (a, b) { return a + b; }, 0); return p.map(function (x) { return x / t; }); });
  };
  function sampleFrom(p, rnd, temperature) {
    if (temperature === 0) { let b = 0; p.forEach(function (x, j) { if (x > p[b]) b = j; }); return b; }
    const q = p.map(function (x) { return Math.pow(x, 1 / temperature); });
    const t = q.reduce(function (a, b) { return a + b; }, 0);
    let r = rnd() * t;
    for (let j = 0; j < q.length; j++) { r -= q[j]; if (r <= 0) return j; }
    return q.length - 1;
  }
  // Algorithm 2 (toy): returns per-level codes, zhats and decoded words
  Generator.prototype.generate = function (seed, opts) {
    opts = opts || {};
    const rnd = mulberry32(seed), tok = this.tok, tau = opts.tau == null ? 0.6 : opts.tau;
    const temp = opts.temperature == null ? 1 : opts.temperature;
    const keys = Object.keys(this.unigram0), tot = keys.reduce((a, c) => a + this.unigram0[c], 0);
    let r = rnd() * tot, c0 = +keys[0];
    for (const c of keys) { r -= this.unigram0[c]; if (r <= 0) { c0 = +c; break; } }
    const codes = [opts.fixedFirst ? opts.fixedFirst : [c0]];
    const steps = [];
    let zhat = tok.zhatFromCodes(codes)[0];
    steps.push({ codes: codes[0], zhat: zhat, words: tok.decode(zhat) });
    for (let k = 1; k < this.K; k++) {
      let rk;
      if (opts.fixed && opts.fixed[k]) rk = opts.fixed[k];
      else {
        const probs = this.levelProbs(k, zhat, tau);
        rk = probs.map(function (p) { return sampleFrom(p, rnd, temp); }); // all l_k codes in parallel
      }
      codes.push(rk);
      zhat = tok.zhatFromCodes(codes)[k];
      steps.push({ codes: rk, zhat: zhat, words: tok.decode(zhat) });
    }
    return { codes: codes, steps: steps };
  };

  global.ToyNCM = {
    VOCAB: VOCAB, EMB: EMB, GROUP_OF: GROUP_OF, GROUPS: GROUPS, IDX: IDX, L: L,
    mulberry32: mulberry32, makeCorpus: makeCorpus, downArea: downArea, upLinear: upLinear,
    nearest: nearest, kmeans: kmeans, d2: d2, Tokenizer: Tokenizer, Generator: Generator,
    words: function (ids) { return ids.map(function (i) { return VOCAB[i]; }); },
    ids: function (s) { return s.trim().split(/\s+/).map(function (w) { return IDX[w]; }); }
  };
})(window);
