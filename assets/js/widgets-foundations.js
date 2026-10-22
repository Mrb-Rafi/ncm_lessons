/* Part I widgets: language models, tokens, vectors, probability, learning, transformers, paradigms. */
(function () {
  'use strict';
  const NCM = window.NCM, W = NCM.widgets, el = NCM.el, S = NCM.svg;

  // shared toy model (trained once, lazily)
  NCM.toy = function () {
    if (NCM._toy) return NCM._toy;
    const T = window.ToyNCM;
    const corpus = T.makeCorpus(600, 11), test = T.makeCorpus(200, 99);
    const tok = new T.Tokenizer({ levels: [1, 2, 4, 6, 8], sizes: [8, 16, 32, 48, 64], seed: 3 }).train(corpus);
    const gen = new T.Generator(tok, corpus);
    NCM._toy = { T: T, corpus: corpus, test: test, tok: tok, gen: gen };
    return NCM._toy;
  };
  NCM.groupColor = function (g) {
    return { det: '#718096', adj: '#d53f8c', animal: '#2b6cb0', verb: '#c0392b', prep: '#2f855a', place: '#b7791f', end: '#6b46c1' }[g] || '#888';
  };

  // ---------------------------------------------------------------- hero
  W.heroDemo = function (host) {
    const toy = NCM.toy(), T = toy.T;
    const box = el('div', null, host);
    let seed = 5, timer = null;
    function run() {
      const out = toy.gen.generate(seed++, { tau: 0.5 });
      box.innerHTML = '';
      const rows = out.steps.map(function (st, k) {
        const r = el('div', { class: 'row' }, box);
        el('div', { class: 'lab', text: 'level ' + (k + 1) + ' · ' + toy.tok.levels[k] + (toy.tok.levels[k] === 1 ? ' code' : ' codes') }, r);
        const t = el('div', { class: 'tokrow' }, r);
        st.words.forEach(function (w) { el('span', { class: 'chip', text: T.VOCAB[w], style: 'border-left:3px solid ' + NCM.levelColor(k, 5) }, t); });
        r.style.opacity = 0.15; return r;
      });
      let k = 0;
      (function reveal() {
        if (k > 0) rows[k - 1].classList.remove('cur');
        if (k < rows.length) { rows[k].style.opacity = 1; rows[k].classList.add('cur'); k++; timer = setTimeout(reveal, 900); }
        else timer = setTimeout(run, 2600);
      })();
    }
    run();
    const c = el('div', { class: 'note', html: 'A real (toy-sized) NCM running in your browser: each line decodes the text after one more <b>level</b> of codes is generated. Early levels are blurry “concepts”; later levels add detail. You will build this exact machine, piece by piece.' }, host);
    c.style.marginTop = '10px';
  };

  // ---------------------------------------------------------------- next-word model (bigram)
  W.nextWord = function (host) {
    NCM.head(host, 'A tiny language model: guess the next word', 'This model learned from a handful of sentences by counting which word follows which (a <i>bigram</i> model). Click <b>Predict next word</b> repeatedly. Each click is one step: the model looks at the text so far and outputs a probability for every possible next word, then one word is chosen.');
    const corpus = ['the cat sat on the mat .', 'the dog sat on the rug .', 'the cat ate the fish .', 'a dog ran to the park .', 'the cat ran to the tree .', 'the bird sat on the tree .', 'a cat slept on the mat .', 'the dog ate the bone .'];
    const counts = {};
    corpus.forEach(function (s) { const w = ['<start>'].concat(s.split(' ')); for (let i = 0; i < w.length - 1; i++) { counts[w[i]] = counts[w[i]] || {}; counts[w[i]][w[i + 1]] = (counts[w[i]][w[i + 1]] || 0) + 1; } });
    let text = [], rnd = NCM.toy().T.mulberry32(Date.now() % 100000);
    const ctr = el('div', { class: 'controls' }, host);
    const nb = NCM.button(ctr, 'Predict next word', step, 'primary');
    const greedy = el('label', null, ctr); const gcb = el('input', { type: 'checkbox' }, greedy); greedy.appendChild(document.createTextNode(' always pick the most likely (greedy)'));
    NCM.button(ctr, 'Reset', function () { text = []; draw(); });
    const sent = el('div', { class: 'out' }, host);
    const bars = el('div', null, host);
    function dist() { const prev = text.length ? text[text.length - 1] : '<start>'; const c = counts[prev] || {}; const tot = Object.values(c).reduce(function (a, b) { return a + b; }, 0); return Object.keys(c).map(function (w) { return { w: w, p: c[w] / tot }; }).sort(function (a, b) { return b.p - a.p; }); }
    function draw() {
      sent.textContent = 'Text so far: ' + (text.length ? text.join(' ') : '(empty)') + (text[text.length - 1] === '.' ? '   ← finished!' : '');
      bars.innerHTML = '';
      const d = dist();
      nb.disabled = !d.length || text[text.length - 1] === '.';
      el('div', { class: 'note', text: d.length ? 'Model output — probability of each possible next word:' : '' }, bars);
      d.forEach(function (x) {
        const r = el('div', { style: 'display:grid;grid-template-columns:70px 1fr 50px;gap:8px;align-items:center;margin:3px 0' }, bars);
        el('code', { text: x.w }, r);
        const b = el('div', { style: 'background:var(--bg-soft);border-radius:4px' }, r); el('div', { class: 'bar-h', style: 'width:' + (x.p * 100) + '%' }, b);
        el('span', { class: 'small', text: (x.p * 100).toFixed(0) + '%' }, r);
      });
    }
    function step() {
      const d = dist(); if (!d.length) return;
      let pick = d[0].w;
      if (!gcb.checked) { let r = rnd(); for (const x of d) { r -= x.p; if (r <= 0) { pick = x.w; break; } } }
      text.push(pick); draw();
    }
    draw();
  };

  // ---------------------------------------------------------------- tokenizer playground
  W.tokenizer = function (host) {
    NCM.head(host, 'Tokenizer playground', 'Type anything. A <b>tokenizer</b> chops text into pieces (tokens) and gives each piece a number (its ID in the vocabulary). Compare three strategies.');
    const ta = el('textarea', null, host); ta.value = 'The unhappy cat jumped over the sleeping dogs.';
    const ctr = el('div', { class: 'controls' }, host);
    let mode = 'subword';
    NCM.select(ctr, 'Strategy', [{ value: 'word', label: 'Words' }, { value: 'char', label: 'Characters' }, { value: 'subword', label: 'Subwords (BPE-like)' }], mode, function (v) { mode = v; draw(); });
    const out = el('div', null, host), stats = el('div', null, host);
    const pieces = ['un', 'happy', 'happi', 'ness', 'jump', 'ed', 'ing', 'sleep', 'dog', 's', 'cat', 'the', 'over', 'er', 'est', 'ly', 're', 'play', 'tion', 'a', 'an', 'in', 'on', 'at', 'is', 'it', 'and', 'to', 'of'];
    pieces.sort(function (a, b) { return b.length - a.length; });
    const ids = {}; let next = 1000;
    function id(t) { if (!(t in ids)) { let h = 0; for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) % 50000; ids[t] = h; } return ids[t]; }
    function sub(word) {
      const res = []; let w = word.toLowerCase(), first = true;
      while (w.length) {
        let m = null;
        for (const p of pieces) if (w.startsWith(p)) { m = p; break; }
        if (!m) m = w[0];
        res.push((first ? 'Ġ' : '') + m); first = false; w = w.slice(m.length);
      }
      return res;
    }
    function draw() {
      const s = ta.value; let toks = [];
      if (mode === 'word') toks = s.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9]/g) || [];
      else if (mode === 'char') toks = s.split('').map(function (c) { return c === ' ' ? '␣' : c; });
      else (s.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9]/g) || []).forEach(function (w) { if (/^[A-Za-z]/.test(w)) toks = toks.concat(sub(w)); else toks.push(w); });
      out.innerHTML = '';
      const row = el('div', { class: 'tokrow' }, out);
      toks.forEach(function (t, i) { const c = el('span', { class: 'chip', title: 'ID ' + id(t) }, row); c.innerHTML = t.replace(/</g, '&lt;') + ' <span class="small">' + id(t) + '</span>'; c.style.borderColor = NCM.palette[i % NCM.palette.length]; });
      stats.innerHTML = '';
      const st = el('div', null, stats);
      st.innerHTML = '<div class="stat"><b>' + toks.length + '</b><span>tokens (sequence length L)</span></div><div class="stat"><b>' + s.length + '</b><span>characters</span></div>';
      el('div', { class: 'note', html: mode === 'subword' ? '“Ġ” marks the start of a new word, as in GPT-2’s tokenizer. Rare words split into common pieces, so a fixed vocabulary can spell anything.' : mode === 'char' ? 'Characters need a tiny vocabulary but make sequences very long.' : 'Whole words give short sequences but need a gigantic vocabulary, and unseen words have no ID.' }, stats);
    }
    ta.addEventListener('input', draw); draw();
  };

  // ---------------------------------------------------------------- embedding map
  W.embedMap = function (host) {
    const toy = NCM.toy(), T = toy.T;
    NCM.head(host, 'Words as points in space (embeddings)', 'Each word of our toy vocabulary is a 2-D vector — two numbers. Similar words sit close together. <b>Drag the black probe</b> anywhere: the nearest word is found by the smallest squared distance. <b>Click two words</b> to compare them.');
    const Wd = 620, H = 400, sc = 52, ox = Wd / 2, oy = H / 2;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, style: 'touch-action:none;background:var(--bg-soft);border-radius:10px' }, host);
    const X = function (v) { return ox + v * sc; }, Y = function (v) { return oy - v * sc; };
    for (let g = -5; g <= 5; g++) { S('line', { x1: X(g), x2: X(g), y1: 0, y2: H, stroke: 'var(--line)', 'stroke-width': g === 0 ? 1.5 : .6 }, svg); S('line', { y1: Y(g), y2: Y(g), x1: 0, x2: Wd, stroke: 'var(--line)', 'stroke-width': g === 0 ? 1.5 : .6 }, svg); }
    const link = S('line', { stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '4 3' }, svg);
    const sel = [];
    const dots = T.VOCAB.map(function (w, i) {
      const e = T.EMB[i], g = S('g', { style: 'cursor:pointer' }, svg);
      S('circle', { cx: X(e[0]), cy: Y(e[1]), r: 5, fill: NCM.groupColor(T.GROUP_OF[i]) }, g);
      S('text', { x: X(e[0]) + 7, y: Y(e[1]) + 4, 'font-size': 12, class: 'svg-text', text: w }, g);
      g.addEventListener('click', function (ev) { ev.stopPropagation(); sel.push(i); if (sel.length > 2) sel.shift(); info(); });
      return g;
    });
    const probe = { x: 0.8, y: 0.6 };
    const pl = S('line', { stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-dasharray': '2 3' }, svg);
    const pc = S('circle', { r: 8, fill: 'var(--ink)', stroke: 'var(--panel)', 'stroke-width': 2, style: 'cursor:grab' }, svg);
    const out = el('div', { class: 'out' }, host);
    let drag = false;
    pc.addEventListener('pointerdown', function (e) { drag = true; pc.setPointerCapture(e.pointerId); });
    pc.addEventListener('pointerup', function () { drag = false; });
    pc.addEventListener('pointermove', function (e) {
      if (!drag) return;
      const r = svg.getBoundingClientRect(), sx = Wd / r.width;
      probe.x = ((e.clientX - r.left) * sx - ox) / sc; probe.y = (oy - (e.clientY - r.top) * sx) / sc; info();
    });
    function info() {
      pc.setAttribute('cx', X(probe.x)); pc.setAttribute('cy', Y(probe.y));
      const n = T.nearest([probe.x, probe.y], T.EMB), e = T.EMB[n];
      pl.setAttribute('x1', X(probe.x)); pl.setAttribute('y1', Y(probe.y)); pl.setAttribute('x2', X(e[0])); pl.setAttribute('y2', Y(e[1]));
      let s = 'Probe = (' + probe.x.toFixed(2) + ', ' + probe.y.toFixed(2) + ')   nearest word: “' + T.VOCAB[n] + '”   squared distance = ' + T.d2([probe.x, probe.y], e).toFixed(3);
      if (sel.length === 2) {
        const a = T.EMB[sel[0]], b = T.EMB[sel[1]];
        link.setAttribute('x1', X(a[0])); link.setAttribute('y1', Y(a[1])); link.setAttribute('x2', X(b[0])); link.setAttribute('y2', Y(b[1]));
        s += '\n\n“' + T.VOCAB[sel[0]] + '” = (' + a.map(function (v) { return v.toFixed(2); }).join(', ') + ')   “' + T.VOCAB[sel[1]] + '” = (' + b.map(function (v) { return v.toFixed(2); }).join(', ') + ')' +
          '\nsquared distance ‖a−b‖² = (' + (a[0] - b[0]).toFixed(2) + ')² + (' + (a[1] - b[1]).toFixed(2) + ')² = ' + T.d2(a, b).toFixed(3) +
          '\ndot product a·b = ' + a[0].toFixed(2) + '×' + b[0].toFixed(2) + ' + ' + a[1].toFixed(2) + '×' + b[1].toFixed(2) + ' = ' + (a[0] * b[0] + a[1] * b[1]).toFixed(3);
      } else if (sel.length === 1) s += '\n\nSelected “' + T.VOCAB[sel[0]] + '”. Click one more word.';
      out.textContent = s;
    }
    const lg = el('div', { class: 'legend' }, host);
    Object.keys(T.GROUPS).forEach(function (g) { el('span', { html: '<i style="background:' + NCM.groupColor(g) + '"></i>' + ({ det: 'determiners', adj: 'adjectives', animal: 'animals', verb: 'verbs', prep: 'prepositions', place: 'places', end: 'full stop' })[g] }, lg); });
    info();
  };

  // ---------------------------------------------------------------- sentence matrix
  W.sentenceMatrix = function (host) {
    const toy = NCM.toy(), T = toy.T;
    NCM.head(host, 'A sentence becomes a matrix z ∈ ℝ^{L×d}', 'Pick words. Each token becomes one row of numbers; stacking the rows gives a matrix with L rows (positions) and d columns (dimensions). Here L = 8 and d = 2. In the paper’s models d is 256–384 and L is up to 1,024.');
    const ctr = el('div', { class: 'controls' }, host);
    const tmpl = ['det', 'adj', 'animal', 'verb', 'prep', 'det', 'place', 'end'];
    let ids = T.ids('the small dog ran under a tree .');
    tmpl.forEach(function (g, i) {
      if (g === 'end') return;
      NCM.select(ctr, '', T.GROUPS[g].words, T.VOCAB[ids[i]], function (v) { ids[i] = T.IDX[v]; draw(); });
    });
    const out = el('div', { class: 'scroll-x' }, host);
    function draw() {
      out.innerHTML = '';
      const t = el('table', null, out);
      t.innerHTML = '<tr><th>position i</th><th>token</th><th>z<sub>i,1</sub></th><th>z<sub>i,2</sub></th><th>colour = value</th></tr>';
      ids.forEach(function (id, i) {
        const e = T.EMB[id], tr = el('tr', null, t);
        el('td', { text: i + 1 }, tr); el('td', { html: '<code>' + T.VOCAB[id] + '</code>' }, tr);
        e.forEach(function (v) { el('td', { text: v.toFixed(2), style: 'font-family:var(--font-mono)' }, tr); });
        const td = el('td', null, tr);
        e.forEach(function (v) { const a = Math.min(1, Math.abs(v) / 4); el('span', { style: 'display:inline-block;width:38px;height:16px;margin-right:4px;border-radius:3px;background:' + (v >= 0 ? 'rgba(183,121,31,' + a + ')' : 'rgba(43,108,176,' + a + ')') }, td); });
      });
    }
    draw();
  };

  // ---------------------------------------------------------------- softmax
  W.softmax = function (host) {
    NCM.head(host, 'Softmax: turning scores into probabilities', 'A model outputs one raw score (a <i>logit</i>) per option. Softmax exponentiates and normalises them so they are positive and sum to 1. Move the sliders, then change the <b>temperature</b>.');
    const names = ['cat', 'dog', 'mat', 'tree', 'banana'];
    let logits = [2.0, 1.5, 0.5, 0.2, -1.5], T = 1;
    const grid = el('div', { class: 'grid2' }, host);
    const left = el('div', null, grid), right = el('div', null, grid);
    logits.forEach(function (v, i) { const c = el('div', { class: 'controls', style: 'margin:4px 0' }, left); NCM.slider(c, 'logit “' + names[i] + '”', -4, 4, 0.1, v, function (x) { logits[i] = x; draw(); }, function (x) { return x.toFixed(1); }); });
    const c2 = el('div', { class: 'controls' }, left);
    NCM.slider(c2, 'temperature T', 0.1, 3, 0.05, 1, function (x) { T = x; draw(); }, function (x) { return x.toFixed(2); });
    const bars = el('div', null, right), form = el('div', { class: 'out' }, right);
    const sc = el('div', { class: 'controls' }, right);
    let counts = names.map(function () { return 0; }), rnd = Math.random;
    NCM.button(sc, 'Sample 100 times', function () { const p = probs(); for (let n = 0; n < 100; n++) { let r = rnd(); for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) { counts[i]++; break; } } } draw(); });
    NCM.button(sc, 'Clear counts', function () { counts = counts.map(function () { return 0; }); draw(); });
    function probs() { const m = Math.max.apply(null, logits); const e = logits.map(function (l) { return Math.exp((l - m) / T); }); const s = e.reduce(function (a, b) { return a + b; }); return e.map(function (x) { return x / s; }); }
    function draw() {
      const p = probs(), tot = counts.reduce(function (a, b) { return a + b; }, 0);
      bars.innerHTML = '';
      p.forEach(function (x, i) {
        const r = el('div', { style: 'display:grid;grid-template-columns:62px 1fr 92px;gap:8px;align-items:center;margin:4px 0' }, bars);
        el('code', { text: names[i] }, r);
        const b = el('div', { style: 'background:var(--bg-soft);border-radius:4px;position:relative' }, r);
        el('div', { class: 'bar-h', style: 'width:' + (x * 100) + '%' }, b);
        if (tot) el('div', { style: 'position:absolute;top:0;height:16px;width:2px;background:var(--ink);left:' + (100 * counts[i] / tot) + '%' }, b);
        el('span', { class: 'small', text: (x * 100).toFixed(1) + '%' + (tot ? ' · seen ' + counts[i] : '') }, r);
      });
      const am = p.indexOf(Math.max.apply(null, p));
      form.textContent = 'p_i = exp(ℓ_i / T) / Σ_j exp(ℓ_j / T)\nargmax (greedy choice) = “' + names[am] + '”' + (tot ? '\nblack ticks = how often each word was actually sampled' : '');
    }
    draw();
  };

  // ---------------------------------------------------------------- cross-entropy
  W.crossEntropy = function (host) {
    NCM.head(host, 'Cross-entropy: the price of being surprised', 'During training we know the correct answer. The loss is −log(probability the model gave to it). Slide the probability: confident and right ⇒ loss near 0; confident and wrong ⇒ huge loss.');
    const ctr = el('div', { class: 'controls' }, host);
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    let p = 0.3;
    NCM.slider(ctr, 'p(correct answer)', 0.01, 1, 0.01, p, function (x) { p = x; draw(); }, function (x) { return x.toFixed(2); });
    function draw() {
      const pts = []; for (let x = 0.01; x <= 1.0001; x += 0.01) pts.push([x, -Math.log(x)]);
      const ch = NCM.chart.lines(plot, [{ color: 'var(--purple)', points: pts, marker: 'none' }, { color: 'var(--accent)', points: [[p, -Math.log(p)]], r: 7, name: 'you are here' }], { xmin: 0, xmax: 1, ymin: 0, ymax: 4.8, xticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'probability given to the correct answer', ylabel: 'loss = −log p', height: 230 });
      out.textContent = 'loss = −log(' + p.toFixed(2) + ') = ' + (-Math.log(p)).toFixed(3) + ' nats\nAveraged over many examples, this is the cross-entropy loss used to train language models — and the NCM generator (paper eq. 10).';
    }
    draw();
  };

  // ---------------------------------------------------------------- sequence probability (chain rule)
  W.seqProb = function (host) {
    NCM.head(host, 'Probability of a whole sentence = a product of steps', 'An autoregressive model scores a sentence by multiplying the probability of each token given the ones before it. Adjust each conditional probability.');
    const words = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
    const p = [0.20, 0.10, 0.30, 0.60, 0.70, 0.40];
    const ctr = el('div', null, host), out = el('div', { class: 'out' }, host);
    words.forEach(function (w, i) { const c = el('div', { class: 'controls', style: 'margin:2px 0' }, ctr); NCM.slider(c, 'p(“' + w + '” | ' + (i ? '“' + words.slice(0, i).join(' ') + '”' : 'start') + ')', 0.01, 1, 0.01, p[i], function (x) { p[i] = x; draw(); }, function (x) { return x.toFixed(2); }); });
    function draw() {
      const prod = p.reduce(function (a, b) { return a * b; }, 1), lg = p.reduce(function (a, b) { return a + Math.log(b); }, 0);
      out.textContent = 'p(sentence) = ' + p.map(function (x) { return x.toFixed(2); }).join(' × ') + ' = ' + prod.toExponential(3) +
        '\nlog p(sentence) = ' + p.map(function (x) { return Math.log(x).toFixed(2); }).join(' + ') + ' = ' + lg.toFixed(3) +
        '\nLogs turn tiny products into manageable sums — that is why papers write Σ log p instead of Π p.';
    }
    draw();
  };

  // ---------------------------------------------------------------- gradient descent
  W.gradDescent = function (host) {
    NCM.head(host, 'Learning = rolling downhill on the loss', 'One parameter w, one loss curve. Each <b>step</b> computes the slope (gradient) and moves w a little against it: w ← w − η · slope. Try a tiny, a good, and a too-large learning rate η.');
    const f = function (w) { return 0.15 * (w - 3) * (w - 3) + 0.6 * Math.sin(1.4 * w) + 1.2; };
    const df = function (w) { return 0.3 * (w - 3) + 0.84 * Math.cos(1.4 * w); };
    let w = -2.5, lr = 0.6, hist = [];
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'learning rate η', 0.05, 5, 0.05, lr, function (x) { lr = x; }, function (x) { return x.toFixed(2); });
    NCM.button(ctr, 'Step', function () { step(); draw(); }, 'primary');
    NCM.button(ctr, 'Run 20 steps', function () { for (let i = 0; i < 20; i++) step(); draw(); });
    NCM.button(ctr, 'Reset', function () { w = -2.5; hist = []; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function step() { hist.push(w); w = NCM.clamp(w - lr * df(w), -4, 9); }
    function draw() {
      const pts = []; for (let x = -4; x <= 9; x += 0.05) pts.push([x, f(x)]);
      NCM.chart.lines(plot, [{ color: 'var(--purple)', points: pts, marker: 'none' }, { color: 'var(--muted)', points: hist.map(function (h) { return [h, f(h)]; }), r: 3, dash: '3 3', width: 1 }, { color: 'var(--accent)', points: [[w, f(w)]], r: 8 }], { xmin: -4, xmax: 9, ymin: 0, ymax: 6.5, xticks: [-4, -2, 0, 2, 4, 6, 8], xlabel: 'parameter w', ylabel: 'loss', height: 240 });
      out.textContent = 'step ' + hist.length + ':  w = ' + w.toFixed(3) + '   loss = ' + f(w).toFixed(3) + '   slope = ' + df(w).toFixed(3) + '\nReal networks do exactly this, but with millions of parameters at once (NCM generators: ~25M–725M).';
    }
    draw();
  };

  // ---------------------------------------------------------------- neuron
  W.neuron = function (host) {
    NCM.head(host, 'One artificial neuron', 'A neuron multiplies inputs by weights, adds a bias, and passes the sum through a non-linear function (here GELU, used in the NCM transformer’s feed-forward layers).');
    const v = { x1: 1, x2: -0.5, w1: 0.8, w2: 1.2, b: 0.1 };
    const g = el('div', { class: 'grid2' }, host), l = el('div', null, g), r = el('div', null, g);
    Object.keys(v).forEach(function (k) { const c = el('div', { class: 'controls', style: 'margin:3px 0' }, l); NCM.slider(c, k, -2, 2, 0.05, v[k], function (x) { v[k] = x; draw(); }, function (x) { return x.toFixed(2); }); });
    const out = el('div', { class: 'out' }, r), plot = el('div', null, r);
    const gelu = function (x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); };
    function draw() {
      const s = v.w1 * v.x1 + v.w2 * v.x2 + v.b;
      out.textContent = 's = w1·x1 + w2·x2 + b = ' + s.toFixed(3) + '\noutput = GELU(s) = ' + gelu(s).toFixed(3);
      const pts = []; for (let x = -4; x <= 4; x += 0.1) pts.push([x, gelu(x)]);
      NCM.chart.lines(plot, [{ color: 'var(--teal)', points: pts, marker: 'none' }, { color: 'var(--accent)', points: [[s, gelu(s)]], r: 7 }], { xmin: -4, xmax: 4, ymin: -0.5, ymax: 4, xticks: [-4, -2, 0, 2, 4], xlabel: 's', ylabel: 'GELU(s)', height: 200 });
    }
    draw();
  };

  // ---------------------------------------------------------------- attention
  W.attention = function (host) {
    const toy = NCM.toy(), T = toy.T;
    NCM.head(host, 'Attention: every token looks at the others', 'Click a word (the <i>query</i>). Its attention weights over all words are softmax(q·k/√d) — here q and k are just the toy embeddings rotated by a small learned-looking matrix. The new representation of the query is the weighted average of the others’ values.');
    const words = ['the', 'small', 'cat', 'sat', 'on', 'the', 'mat', '.'];
    const E = words.map(function (w) { return T.EMB[T.IDX[w]]; });
    const Wq = [[0.9, 0.5], [-0.4, 0.8]], Wk = [[0.8, -0.3], [0.6, 0.9]];
    const mul = function (M, v) { return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]; };
    let qi = 2, mode = 'full';
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Mask', [{ value: 'full', label: 'none (bidirectional)' }, { value: 'causal', label: 'causal (only look left)' }], mode, function (v) { mode = v; draw(); });
    const row = el('div', { class: 'tokrow' }, host), svgHost = el('div', null, host), out = el('div', { class: 'out' }, host);
    function weights(i) {
      const q = mul(Wq, E[i]); const s = E.map(function (e, j) { const k = mul(Wk, e); return (mode === 'causal' && j > i) ? -Infinity : (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2; });
      const m = Math.max.apply(null, s); const ex = s.map(function (x) { return Math.exp(x - m); }); const z = ex.reduce(function (a, b) { return a + b; }); return ex.map(function (x) { return x / z; });
    }
    function draw() {
      row.innerHTML = '';
      words.forEach(function (w, i) { const c = el('span', { class: 'chip' + (i === qi ? ' hl' : ''), text: w, style: 'cursor:pointer' }, row); c.addEventListener('click', function () { qi = i; draw(); }); });
      const a = weights(qi);
      svgHost.innerHTML = '';
      const Wd = 640, H = 130, svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, svgHost);
      const x = function (i) { return 40 + i * (Wd - 80) / (words.length - 1); };
      words.forEach(function (w, j) {
        if (a[j] > 0.001) S('path', { d: 'M' + x(qi) + ',100 Q' + (x(qi) + x(j)) / 2 + ',' + (100 - 30 - Math.abs(qi - j) * 9) + ' ' + x(j) + ',100', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 0.5 + 9 * a[j], opacity: 0.25 + 0.75 * a[j] }, svg);
        S('text', { x: x(j), y: 120, 'text-anchor': 'middle', 'font-size': 13, class: 'svg-text', 'font-weight': j === qi ? 700 : 400, text: w }, svg);
        S('text', { x: x(j), y: 18, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: (a[j] * 100).toFixed(0) + '%' }, svg);
      });
      out.textContent = 'Attention of “' + words[qi] + '” (position ' + (qi + 1) + '): ' + a.map(function (v, j) { return words[j] + ' ' + (v * 100).toFixed(0) + '%'; }).join(', ') + '\nweights always sum to 100%.';
    }
    draw();
  };

  // ---------------------------------------------------------------- attention masks (also used for NCM block masks)
  W.masks = function (host) {
    const mode0 = host.getAttribute('data-mode') || 'basic';
    NCM.head(host, mode0 === 'basic' ? 'Attention masks: who may look at whom' : 'Attention over level blocks (paper Fig. 11 & Table 9)',
      mode0 === 'basic' ? 'Each row is a query position, each column a key position. A filled cell means “allowed to attend”. Masks are how one transformer architecture becomes an autoregressive model, a diffusion model, or NCM.' :
        'In NCM’s training sequence, block k holds the l<sub>k</sub> positions of level k (filled with the resized reconstruction of earlier levels). Choose a pattern. Cells are coloured by the <b>key</b> level. A prompt (prefix) can be added for conditional generation.');
    const ctr = el('div', { class: 'controls' }, host);
    const opts = mode0 === 'basic' ? [{ value: 'full', label: 'bidirectional (diffusion, encoders)' }, { value: 'causal', label: 'causal (autoregressive)' }]
      : [{ value: 'bdiag', label: 'block-diagonal (NCM)' }, { value: 'bcausal', label: 'block-causal' }, { value: 'window', label: 'sliding window (2 levels)' }, { value: 'causal', label: 'causal (reference)' }];
    let pat = opts[0].value, prefix = 0;
    NCM.select(ctr, 'Pattern', opts, pat, function (v) { pat = v; draw(); });
    if (mode0 !== 'basic') NCM.slider(ctr, 'prompt tokens', 0, 4, 1, 0, function (v) { prefix = v; draw(); });
    const svgHost = el('div', null, host), out = el('div', { class: 'note' }, host);
    const levels = mode0 === 'basic' ? [1, 1, 1, 1, 1, 1, 1, 1] : [1, 2, 3, 4];
    function draw() {
      const lv = []; if (mode0 !== 'basic') for (let i = 0; i < prefix; i++) lv.push(-1);
      levels.forEach(function (l, k) { for (let i = 0; i < l; i++) lv.push(k); });
      const n = lv.length, cs = Math.min(36, 340 / n), pad = 70, Wd = pad + n * cs + 10;
      svgHost.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + (pad + n * cs + 10), style: 'max-width:' + (Wd * 1.2) + 'px' }, svgHost);
      let allowed = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const qa = lv[i], ka = lv[j]; let ok;
        if (mode0 === 'basic') ok = pat === 'full' ? true : j <= i;
        else if (qa === -1) ok = ka === -1;              // prompt attends only to prompt
        else if (ka === -1) ok = true;                   // every target block sees the prompt
        else if (pat === 'bdiag') ok = qa === ka;
        else if (pat === 'bcausal') ok = ka <= qa;
        else if (pat === 'window') ok = ka <= qa && ka >= qa - 1;
        else ok = j <= i;
        if (ok) allowed++;
        S('rect', { x: pad + j * cs, y: pad + i * cs, width: cs - 1.5, height: cs - 1.5, rx: 2, fill: ok ? (ka === -1 ? 'var(--teal)' : mode0 === 'basic' ? 'var(--accent)' : NCM.levelColor(ka, levels.length)) : 'var(--bg-soft)' }, svg);
      }
      for (let i = 0; i < n; i++) {
        const lab = mode0 === 'basic' ? (i + 1) : (lv[i] === -1 ? 'P' : 'L' + (lv[i] + 1));
        S('text', { x: pad - 6, y: pad + i * cs + cs / 2 + 4, 'text-anchor': 'end', 'font-size': 11, class: 'svg-text', text: lab }, svg);
        S('text', { x: pad + i * cs + cs / 2, y: pad - 6, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: lab }, svg);
      }
      S('text', { x: pad, y: 16, 'font-size': 12, class: 'svg-text', text: 'key position →' }, svg);
      S('text', { x: 4, y: 40, 'font-size': 12, class: 'svg-text', text: 'query ↓' }, svg);
      const msg = {
        full: 'Every token sees every token. Used by masked diffusion models — which is why they cannot reuse a KV cache: when any token changes, every representation must be recomputed.',
        causal: 'Token i sees only tokens 1…i. This is what makes next-token prediction possible and allows KV caching.',
        bdiag: 'NCM’s choice. Each block only attends within itself (and to the prompt). It does not need to look at earlier blocks, because its input — the cumulative reconstruction ẑ(k−1) — already summarises all earlier levels. Paper Table 9 (TinyStories): Gen. PPL 15.41, fastest sampling (5.65 s).',
        bcausal: 'Each block also attends to all earlier blocks. Paper Table 9: Gen. PPL 19.63, 1.40× slower than block-diagonal.',
        window: 'Each block attends to itself and the previous block. Paper Table 9: Gen. PPL 17.84, 1.21× slower than block-diagonal.'
      };
      out.innerHTML = msg[pat] + ' <br><b>' + allowed + '</b> of ' + (n * n) + ' query–key pairs are computed.';
    }
    draw();
  };

  // ---------------------------------------------------------------- paradigms race (Figure 2)
  W.paradigms = function (host) {
    NCM.head(host, 'Three ways to generate (paper Figure 2, animated)', 'Press <b>Step</b> to advance all three generators by one forward pass of their network. Count the sequential steps and the number of positions each step processes.');
    const words = 'the quick brown fox jumps over the lazy dog .'.split(' ');
    const L = words.length;
    let T = 10, sched = [1, 2, 4, 7, 10], t = 0;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'diffusion steps T', 2, 20, 1, T, function (v) { T = v; reset(); }, null);
    NCM.button(ctr, 'Step', function () { t++; draw(); }, 'primary');
    NCM.button(ctr, 'Reset', reset);
    const grid = el('div', null, host), stats = el('div', null, host);
    const rnd = NCM.toy().T.mulberry32(4);
    let order = [];
    function reset() { t = 0; order = words.map(function (_, i) { return i; }).sort(function () { return rnd() - 0.5; }); draw(); }
    function panel(title, color) { const p = el('div', { style: 'border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0;border-left:4px solid ' + color }, grid); el('div', { html: '<b>' + title + '</b>', style: 'margin-bottom:6px' }, p); return p; }
    function draw() {
      grid.innerHTML = '';
      // AR
      const ar = panel('Next-token prediction (autoregressive)', 'var(--blue)');
      const s1 = Math.min(t, L), r1 = el('div', { class: 'tokrow' }, ar);
      words.forEach(function (w, i) { el('span', { class: 'chip' + (i < s1 ? (i === s1 - 1 ? ' hl' : '') : ' mask'), text: i < s1 ? w : '·' }, r1); });
      el('div', { class: 'small', text: (s1 >= L ? 'done after ' + L + ' steps' : 'step ' + s1 + ' / ' + L) + ' · each step adds 1 token' }, ar);
      // MDLM
      const md = panel('Masked diffusion (MDLM-style)', 'var(--green)');
      const s2 = Math.min(t, T), shown = Math.round(L * s2 / T), r2 = el('div', { class: 'tokrow' }, md);
      const vis = new Set(order.slice(0, shown)), newly = new Set(order.slice(Math.round(L * (s2 - 1) / T), shown));
      words.forEach(function (w, i) { el('span', { class: 'chip' + (vis.has(i) ? (newly.has(i) && s2 > 0 ? ' hl' : '') : ' mask'), text: vis.has(i) ? w : '?' }, r2); });
      el('div', { class: 'small', text: (s2 >= T ? 'done after ' + T + ' steps' : 'step ' + s2 + ' / ' + T) + ' · each step re-processes all ' + L + ' positions' }, md);
      // NCM
      const nc = panel('Next-Concept Modeling (NCM)', 'var(--accent)');
      const K = sched.length, s3 = Math.min(t, K);
      for (let k = 0; k < K; k++) {
        const rr = el('div', { style: 'display:flex;gap:4px;margin:3px 0;opacity:' + (k < s3 ? 1 : .25) }, nc);
        for (let i = 0; i < sched[k]; i++) el('span', { style: 'flex:1;height:18px;border-radius:5px;background:' + NCM.levelColor(k, K) + ';' + (k === s3 - 1 ? 'outline:2px solid var(--ink)' : ''), title: 'code r' + (k + 1) + ',' + (i + 1) }, rr);
      }
      if (s3 >= K) { const r3 = el('div', { class: 'tokrow' }, nc); words.forEach(function (w) { el('span', { class: 'chip ok', text: w }, r3); }); }
      el('div', { class: 'small', text: (s3 >= K ? 'done after ' + K + ' steps, then decode' : 'step ' + s3 + ' / ' + K) + ' · level k predicts its l_k codes in parallel; level sizes ' + sched.join(', ') }, nc);
      const arPos = s1, mdPos = s2 * L, ncPos = sched.slice(0, s3).reduce(function (a, b) { return a + b; }, 0);
      stats.innerHTML = '<div class="stat"><b>' + arPos + '</b><span>positions processed · AR (with KV cache)</span></div><div class="stat"><b>' + mdPos + '</b><span>positions processed · diffusion</span></div><div class="stat"><b>' + ncPos + '</b><span>positions processed · NCM</span></div>';
    }
    reset();
  };

  // ---------------------------------------------------------------- masked diffusion quality vs steps
  W.diffusion = function (host) {
    NCM.head(host, 'Why few-step diffusion struggles', 'A masked diffusion model unmasks several positions per step, choosing each one <i>independently</i> from its own probability table. Here two adjacent positions must form a city name. If both are revealed in the same step, nothing makes them agree.');
    const ctr = el('div', { class: 'controls' }, host);
    let T = 2;
    NCM.slider(ctr, 'denoising steps T (for 2 positions)', 1, 2, 1, T, function (v) { T = v; }, function (v) { return v === 1 ? '1 (both at once)' : '2 (one at a time)'; });
    NCM.button(ctr, 'Generate 30 samples', run, 'primary');
    const out = el('div', null, host), st = el('div', null, host);
    function run() {
      out.innerHTML = ''; let bad = 0; const rnd = Math.random;
      for (let n = 0; n < 30; n++) {
        let a, b;
        if (T === 1) { a = rnd() < 0.5 ? 'New' : 'Los'; b = rnd() < 0.5 ? 'York' : 'Angeles'; }
        else { a = rnd() < 0.5 ? 'New' : 'Los'; b = a === 'New' ? 'York' : 'Angeles'; }
        const okk = (a === 'New') === (b === 'York'); if (!okk) bad++;
        el('span', { class: 'chip ' + (okk ? 'ok' : 'bad'), text: 'I flew to ' + a + ' ' + b }, out);
      }
      st.innerHTML = '<div class="stat"><b>' + bad + ' / 30</b><span>incoherent samples</span></div><div class="note">With 1 step, each position’s table is 50/50 and they are sampled independently, so about half the outputs are nonsense like “New Angeles”. More steps fix it — at the cost of more sequential passes. This is the trade-off the paper calls out in its Introduction, point (i).</div>';
    }
    run();
  };
})();
