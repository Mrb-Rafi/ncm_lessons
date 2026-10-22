/* Part III widgets: the generator — factorisation, inputs, head, training, sampling, guidance, MBR, cost, ELBO. */
(function () {
  'use strict';
  const NCM = window.NCM, W = NCM.widgets, el = NCM.el, S = NCM.svg;

  // ---------------------------------------------------------------- parallel dilemma + coarse conditioning
  W.parallelDilemma = function (host) {
    NCM.head(host, 'Why conditioning on a coarse level makes parallel sampling safe', 'Two positions must agree (“New York” or “Los Angeles”). Sampling both <i>independently</i> is only safe if the information that decides <i>which city</i> is already known. In NCM that information comes from the coarse levels, through ẑ<sup>(k−1)</sup>. Slide how much the coarse level has already decided.');
    let c = 0;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'how sure the coarse “concept” is', 0, 1, 0.05, c, function (v) { c = v; draw(); }, function (v) { return Math.round(v * 100) + '%'; });
    NCM.button(ctr, 'Resample', draw, 'primary');
    const out = el('div', null, host), tb = el('div', null, host), st = el('div', null, host);
    function draw() {
      // coarse code picks a city with prob .5 each; given it, each position puts mass (0.5 + c/2) on the matching word
      const q = 0.5 + c / 2;
      out.innerHTML = ''; let bad = 0;
      for (let n = 0; n < 24; n++) {
        const city = Math.random() < 0.5 ? 0 : 1;
        const a = Math.random() < q ? city : 1 - city, b = Math.random() < q ? city : 1 - city;
        const ok = a === b; if (!ok) bad++;
        el('span', { class: 'chip ' + (ok ? 'ok' : 'bad'), text: (a ? 'Los' : 'New') + ' ' + (b ? 'Angeles' : 'York') }, out);
      }
      tb.innerHTML = '<div class="out">given the coarse concept “city = New York”:\n  p(position 1 = New)  = ' + q.toFixed(2) + '   p(position 2 = York) = ' + q.toFixed(2) + '\nindependent sampling ⇒ p(mismatch) = 2·q·(1−q) = ' + (2 * q * (1 - q)).toFixed(3) + '</div>';
      st.innerHTML = '<div class="stat"><b>' + bad + ' / 24</b><span>incoherent pairs this run</span></div>';
    }
    draw();
  };

  // ---------------------------------------------------------------- NCM training input sequence
  W.ncmInput = function (host) {
    NCM.head(host, 'What the Next-Concept Transformer actually reads', 'For level k the input is ẑ<sup>(k−1)</sup> (shape L×d) <b>resized</b> to l<sub>k</sub> positions by average pooling, projected to the hidden width, plus a learned <b>level embedding</b> (and RoPE position encoding). All blocks are concatenated into one sequence, so training needs only one forward pass.');
    const scheds = NCM.repoSchedules;
    let key = Object.keys(scheds)[0], hover = -1;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Schedule', Object.keys(scheds), key, function (v) { key = v; draw(); });
    const svgHost = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const s = scheds[key], K = s.length, blocks = s.slice(1);   // level 1 is not predicted from a reconstruction (see text)
      const tot = blocks.reduce(function (a, b) { return a + b; }, 0);
      svgHost.innerHTML = '';
      const Wd = 680, svg = S('svg', { viewBox: '0 0 ' + Wd + ' 92' }, svgHost);
      let x = 10; const sc = (Wd - 20) / tot;
      blocks.forEach(function (l, j) {
        const k = j + 1;
        const r = S('rect', { x: x, y: 20, width: Math.max(1, l * sc - 1), height: 36, rx: 3, fill: NCM.levelColor(k, K), opacity: hover === -1 || hover === k ? 1 : .35 }, svg);
        r.addEventListener('mouseenter', function () { hover = k; draw(); });
        if (l * sc > 22) S('text', { x: x + l * sc / 2, y: 43, 'text-anchor': 'middle', 'font-size': 11, fill: '#fff', 'font-weight': 700, text: 'L' + (k + 1) }, svg);
        x += l * sc;
      });
      S('text', { x: 10, y: 14, 'font-size': 11.5, class: 'svg-text', text: 'one training sequence of ' + tot + ' positions  (hover a block)' }, svg);
      S('text', { x: 10, y: 78, 'font-size': 11.5, class: 'svg-text', text: 'block k+1 input = Resize(ẑ(k), l_{k+1});  its targets = the true codes r_{k+1}' }, svg);
      const h = hover > 0 ? hover : 1;
      out.textContent = 'Block for level ' + (h + 1) + ':  input = ẑ(' + h + ') average-pooled from ' + s[K - 1] + ' to ' + s[h] + ' positions  →  predict ' + s[h] + ' codes r_' + (h + 1) + ' in parallel' +
        '\nSequence length during training: ' + tot + ' (vs. L = ' + s[K - 1] + ' tokens of text).' +
        '\nLevel 1 (' + s[0] + ' code' + (s[0] > 1 ? 's' : '') + ') has no earlier reconstruction to read. In the released TinyStories config it is drawn from the empirical frequency of level-1 codes (level0_unigram); for conditional tasks it is predicted from the prompt.';
    }
    draw();
  };

  // ---------------------------------------------------------------- tied codebook head: logits = -distance
  W.tiedHead = function (host) {
    NCM.head(host, 'The output head scores codes by distance (from the repo)', 'In <code>models/next_concept.py</code> the head projects the transformer output h to a vector p, then scores each codebook entry e<sub>j</sub> by −‖p − e<sub>j</sub>‖². Up to a constant that is 2p·e<sub>j</sub> − ‖e<sub>j</sub>‖²: the exact mirror of the tokenizer’s nearest-neighbour rule. Drag p.');
    const book = [[-1.6, 1.1], [0.4, 1.7], [1.7, 0.2], [0.9, -1.4], [-1.0, -1.1], [-0.1, 0.1]];
    let p = [0.8, 0.9], invT = 1;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'learned sharpness exp(log_inv_T)', 0.2, 5, 0.1, invT, function (v) { invT = v; draw(); }, function (v) { return v.toFixed(1); });
    const g = el('div', { class: 'grid2' }, host);
    const Wd = 300, H = 260, sc = 60, ox = Wd / 2, oy = H / 2;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, style: 'background:var(--bg-soft);border-radius:10px;touch-action:none' }, g);
    const bars = el('div', null, g);
    const X = function (v) { return ox + v * sc; }, Y = function (v) { return oy - v * sc; };
    let drag = false;
    svg.addEventListener('pointerdown', function (e) { drag = true; svg.setPointerCapture(e.pointerId); mv(e); });
    svg.addEventListener('pointermove', function (e) { if (drag) mv(e); });
    svg.addEventListener('pointerup', function () { drag = false; });
    function mv(e) { const r = svg.getBoundingClientRect(), s = Wd / r.width; p = [((e.clientX - r.left) * s - ox) / sc, (oy - (e.clientY - r.top) * s) / sc]; draw(); }
    function draw() {
      svg.innerHTML = '';
      const dsc = book.reduce(function (a, e) { return a + e[0] * e[0] + e[1] * e[1]; }, 0) / book.length;
      const logits = book.map(function (e) { return -(((p[0] - e[0]) ** 2 + (p[1] - e[1]) ** 2)) / dsc * invT; });
      const m = Math.max.apply(null, logits), ex = logits.map(function (l) { return Math.exp(l - m); }), z = ex.reduce(function (a, b) { return a + b; }), pr = ex.map(function (v) { return v / z; });
      book.forEach(function (e, j) { S('line', { x1: X(p[0]), y1: Y(p[1]), x2: X(e[0]), y2: Y(e[1]), stroke: 'var(--accent)', 'stroke-width': 0.5 + 6 * pr[j], opacity: .3 + .7 * pr[j] }, svg); S('circle', { cx: X(e[0]), cy: Y(e[1]), r: 7, fill: NCM.palette[j] }, svg); S('text', { x: X(e[0]) + 9, y: Y(e[1]) - 8, 'font-size': 12, class: 'svg-text', text: 'e' + (j + 1) }, svg); });
      S('circle', { cx: X(p[0]), cy: Y(p[1]), r: 8, fill: 'var(--ink)', stroke: 'var(--panel)', 'stroke-width': 2 }, svg);
      bars.innerHTML = '<div class="small">p(code j) = softmax(−‖p − e_j‖² / mean‖e‖² × sharpness)</div>';
      pr.forEach(function (v, j) { const r = el('div', { style: 'display:grid;grid-template-columns:30px 1fr 52px;gap:6px;align-items:center;margin:4px 0' }, bars); el('b', { text: 'e' + (j + 1), style: 'color:' + NCM.palette[j] }, r); const b = el('div', { style: 'background:var(--bg-soft);border-radius:4px' }, r); el('div', { class: 'bar-h', style: 'width:' + v * 100 + '%;background:' + NCM.palette[j] }, b); el('span', { class: 'small', text: (v * 100).toFixed(1) + '%' }, r); });
    }
    draw();
  };

  // ---------------------------------------------------------------- loss weights alpha
  W.lossWeights = function (host) {
    NCM.head(host, 'Weighting the levels in the loss (paper eq. 10)', 'w<sub>k</sub> = l<sub>k</sub><sup>−α</sup> / Σ<sub>j</sub> l<sub>j</sub><sup>1−α</sup>. A level with l<sub>k</sub> codes contributes l<sub>k</sub>·w<sub>k</sub> of the total loss. α = 0: every <i>code</i> counts equally (fine levels dominate). α = 1: every <i>level</i> counts equally.');
    let a = 0;
    const s = NCM.repoSchedules['TinyStories / LM1B (L=128, K=12)'].slice(1); // predicted levels 2..12
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'α', 0, 1.5, 0.05, a, function (v) { a = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const den = s.reduce(function (acc, l) { return acc + Math.pow(l, 1 - a); }, 0);
      const share = s.map(function (l) { return l * Math.pow(l, -a) / den; });
      NCM.chart.bars(plot, share.map(function (v, j) { return { label: 'L' + (j + 2), value: v * 100, color: NCM.levelColor(j + 1, 12), note: (v * 100).toFixed(1) }; }), { height: 220, ylabel: 'share of loss %', max: 70 });
      const coarse = share.slice(0, 5).reduce(function (x, y) { return x + y; }, 0);
      out.textContent = 'TinyStories schedule, predicted levels 2–12.  Share of loss on the coarse levels 2–6: ' + (coarse * 100).toFixed(1) + '%' +
        '\nPaper Appendix F.2.3: raising α from 0 to 1.5 moves this share from 19% to 63%; Gen. PPL after 10k steps stays 18.9–19.0 for α ≤ 1 and rises slightly to 19.4 at α = 1.5.' +
        '\n(With α = 0 and clean inputs, the NCM loss is exactly the generation term of the ELBO in Appendix A.)';
    }
    draw();
  };

  // ---------------------------------------------------------------- conditioning corruption
  W.corruption = function (host) {
    const toy = NCM.toy(), tok = toy.tok, T = toy.T;
    NCM.head(host, 'Conditioning corruption (paper §3.4, Appendix F.3)', 'During training the model reads <i>perfect</i> earlier levels (teacher forcing); at generation time it reads its <i>own</i> samples, which contain mistakes. To prepare it, some input codes are replaced by random codes from the same codebook — with a rate that decreases linearly from coarse to fine. Targets stay clean.');
    let mean = 0.5, seed = 1;
    const ids = T.ids('the old dog hid under my bed .');
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'mean corruption rate', 0, 0.75, 0.05, mean, function (v) { mean = v; draw(); }, function (v) { return v.toFixed(2); });
    NCM.button(ctr, 'Resample corruption', function () { seed++; draw(); });
    const out = el('div', null, host), dec = el('div', null, host), res = el('div', null, host);
    function draw() {
      const enc = tok.encode(ids), K = tok.levels.length, rnd = T.mulberry32(seed);
      const rates = tok.levels.map(function (_, k) { return Math.max(0, Math.min(1, mean * (1.5 - k / (K - 1)))); });
      out.innerHTML = '';
      const corrupted = enc.codes.map(function (row, k) {
        const r = el('div', { style: 'display:flex;gap:4px;align-items:center;margin:4px 0' }, out);
        el('span', { class: 'small', text: 'L' + (k + 1) + ' · rate ' + rates[k].toFixed(2), style: 'width:100px' }, r);
        return row.map(function (c) {
          const bad = rnd() < rates[k], c2 = bad ? Math.floor(rnd() * tok.books[k].length) : c;
          el('span', { text: c2, style: 'flex:1;text-align:center;font-family:var(--font-mono);font-size:12px;border-radius:5px;padding:2px 0;color:' + (bad ? 'var(--red)' : '#fff') + ';background:' + (bad ? 'var(--red-soft)' : NCM.levelColor(k, K)) + (bad ? ';outline:1.5px solid var(--red)' : '') }, r);
          return c2;
        });
      });
      dec.innerHTML = '<div class="small" style="margin-top:8px">What the decoder makes of the corrupted reconstruction (all levels):</div>';
      const zh = tok.zhatFromCodes(corrupted)[K - 1], d = tok.decode(zh), row = el('div', { class: 'tokrow' }, dec);
      d.forEach(function (w, i) { el('span', { class: 'chip ' + (w === ids[i] ? 'ok' : 'bad'), text: T.VOCAB[w] }, row); });
      res.innerHTML = '<div class="note">Paper Figure 13 (TinyStories, Gen. PPL after 10k steps, read from the figure): no corruption ≈ 49; mean rate 0.25 ≈ 23; the chosen recipe (mean 0.5) ≈ 19; 0.75 ≈ 22.5. Moderate corruption helps a lot; too much weakens the conditioning signal.</div>';
      NCM.chart.lines(el('div', null, res), [{ color: 'var(--accent)', points: [[0, 49], [0.25, 23], [0.33, 20.5], [0.5, 19], [0.66, 18.8], [0.75, 22.5]], labels: ['', '', '', 'recipe', '', ''] }, { color: 'var(--ink)', points: [[mean, 0]], marker: 'none' }], { xmin: 0, xmax: 0.8, ymin: 10, ymax: 55, xticks: [0, 0.25, 0.5, 0.75], xlabel: 'mean corruption rate', ylabel: 'Gen. PPL (↓)', height: 200 });
    }
    draw();
  };

  // ---------------------------------------------------------------- Algorithm 1 (toy)
  W.algo1 = function (host) {
    const toy = NCM.toy(), tok = toy.tok, T = toy.T, gen = toy.gen, K = tok.levels.length;
    NCM.head(host, 'Algorithm 1, one training step (toy)', 'Step through one training step on one sentence. The “model” here is the toy neighbour-based predictor, standing in for the transformer; the bookkeeping — encode, corrupt, build inputs, predict every level in one pass, score with cross-entropy — is exactly Algorithm 1.');
    const ids = T.ids('a sleepy fox ran near the park .');
    let step = 0;
    const lines = ['1  Encode x into target codes r_1:K', '2  Uniformly replace some codes → r̃_1:K (corruption)', '3–7 Build the cumulative reconstructions z̃(k−1) and resize each to l_k inputs', '8  Predict all levels in one forward pass', '9  Compute L_NCM against the clean targets r_1:K', '10 Update θ with the gradient'];
    const ctr = el('div', { class: 'controls' }, host);
    NCM.button(ctr, 'Next line ▶', function () { step = Math.min(lines.length - 1, step + 1); draw(); }, 'primary');
    NCM.button(ctr, 'Restart', function () { step = 0; draw(); });
    const code = el('pre', null, host), out = el('div', null, host);
    const enc = tok.encode(ids), rnd = T.mulberry32(5);
    const corr = enc.codes.map(function (row, k) { return row.map(function (c) { return (k > 0 && rnd() < 0.3 * (1 - k / K)) ? Math.floor(rnd() * tok.books[k].length) : c; }); });
    const zt = tok.zhatFromCodes(corr);
    function draw() {
      code.innerHTML = lines.map(function (l, i) { return i === step ? '<b style="color:var(--accent)">▶ ' + l + '</b>' : '  ' + l; }).join('\n');
      out.innerHTML = '';
      if (step === 0) { el('div', { class: 'out', text: 'x = “' + T.words(ids).join(' ') + '”\n' + enc.codes.map(function (r, k) { return 'r_' + (k + 1) + ' = [' + r.join(', ') + ']'; }).join('\n') }, out); }
      if (step === 1) { el('div', { class: 'out', text: corr.map(function (r, k) { return 'r̃_' + (k + 1) + ' = [' + r.map(function (c, i) { return c === enc.codes[k][i] ? c : '*' + c; }).join(', ') + ']'; }).join('\n') + '\n(* = replaced by a random code from that level’s codebook)' }, out); }
      if (step === 2) { el('div', { class: 'out', text: tok.levels.slice(1).map(function (l, j) { return 'block ' + (j + 2) + ': z̃(' + (j + 1) + ') resized to ' + l + ' positions → decodes as “' + T.words(tok.decode(zt[j])).join(' ') + '”'; }).join('\n') }, out); }
      if (step >= 3) {
        let tot = 0, n = 0; const rows = [];
        for (let k = 1; k < K; k++) {
          const pr = gen.levelProbs(k, zt[k - 1], 0.6);
          let lk = 0; enc.codes[k].forEach(function (c, i) { lk += -Math.log(pr[i][c]); });
          tot += lk; n += enc.codes[k].length;
          rows.push('level ' + (k + 1) + ': ' + enc.codes[k].length + ' codes predicted in parallel, mean −log p(true code) = ' + (lk / enc.codes[k].length).toFixed(3));
        }
        el('div', { class: 'out', text: rows.join('\n') + (step >= 4 ? '\n\nL_NCM (α = 0: average over all ' + n + ' codes) = ' + (tot / n).toFixed(3) : '') + (step >= 5 ? '\n\nA real transformer now back-propagates this number and nudges its ~100M weights (AdamW, peak learning rate 3·10⁻⁴ in the paper’s Table 6).' : '') }, out);
      }
    }
    draw();
  };

  // ---------------------------------------------------------------- Algorithm 2 (toy generation)
  W.algo2 = function (host) {
    const toy = NCM.toy(), tok = toy.tok, T = toy.T, gen = toy.gen, K = tok.levels.length;
    NCM.head(host, 'Algorithm 2: generate coarse-to-fine (live toy model)', 'Press <b>Next level</b>. Each press is one network evaluation (one NFE): read ẑ<sup>(k−1)</sup>, predict a probability table for every code position of level k, and sample <b>all of them at once</b>. Then update ẑ<sup>(k)</sup> = ẑ<sup>(k−1)</sup> + R<sub>k</sub>(e<sub>k</sub>(r<sub>k</sub>)) and decode.');
    let seed = 7, temp = 1, tau = 0.6, k = 0, res = null;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'temperature', 0, 2, 0.1, temp, function (v) { temp = v; reset(); }, function (v) { return v === 0 ? '0 (argmax)' : v.toFixed(1); });
    NCM.slider(ctr, 'toy model “blur” τ', 0.1, 3, 0.1, tau, function (v) { tau = v; reset(); }, function (v) { return v.toFixed(1); });
    const c2 = el('div', { class: 'controls' }, host);
    NCM.button(c2, 'Next level ▶', function () { k = Math.min(K, k + 1); draw(); }, 'primary');
    NCM.button(c2, 'Run all', function () { k = K; draw(); });
    NCM.button(c2, 'New sample', function () { seed++; reset(); });
    const box = el('div', null, host), out = el('div', { class: 'out' }, host);
    function reset() { res = gen.generate(seed, { temperature: temp, tau: tau }); k = 0; draw(); }
    function draw() {
      box.innerHTML = '';
      for (let j = 0; j < K; j++) {
        const st = res.steps[j], wrap = el('div', { style: 'display:grid;grid-template-columns:120px 1fr;gap:10px;align-items:center;margin:6px 0;opacity:' + (j < k ? 1 : .18) }, box);
        const right = el('div', null, wrap);
        const cr = el('div', { style: 'display:flex;gap:3px;margin-bottom:3px' }, right);
        st.codes.forEach(function (c) { el('span', { text: j < k ? c : '?', style: 'flex:1;text-align:center;font-family:var(--font-mono);font-size:11.5px;color:#fff;border-radius:4px;background:' + NCM.levelColor(j, K) }, cr); });
        const tr = el('div', { class: 'tokrow' }, right);
        st.words.forEach(function (w) { el('span', { class: 'chip', text: j < k ? T.VOCAB[w] : '·' }, tr); });
        wrap.insertBefore(el('div', { class: 'small', html: '<b style="color:' + NCM.levelColor(j, K) + '">step ' + (j + 1) + '</b><br>' + tok.levels[j] + ' code' + (tok.levels[j] > 1 ? 's' : '') + (j === 0 ? '<br>(from level-1 frequencies)' : '') }), right);
      }
      out.textContent = k === 0 ? 'ẑ(0) = 0. Nothing generated yet.' : 'NFE used: ' + k + ' of ' + K + '.   Output after step ' + k + ': “' + T.words(res.steps[k - 1].words).join(' ') + '”' + (k === K ? '\nDone — the final reconstruction is decoded with argmax, as in Algorithm 2 line 10.' : '');
    }
    reset();
  };

  // ---------------------------------------------------------------- translation trajectory BLEU (Table 12)
  W.trajectory = function (host) {
    NCM.head(host, 'Quality of one translation after each level (paper Table 12)', 'The paper decodes ẑ<sup>(k)</sup> after every level of one WMT14 German→English generation and scores it with BLEU against the reference. Numbers below are copied from Table 12; note that quality is not monotonic — a level can temporarily make things worse before later levels fix them.');
    const bleu = [6.0, 7.2, 6.3, 16.0, 17.5, 17.7, 24.2, 51.7, 38.9, 56.3, 77.8];
    NCM.chart.lines(el('div', null, host), [{ color: 'var(--accent)', points: bleu.map(function (b, i) { return [i + 1, b]; }), labels: bleu.map(String) }], { xmin: 1, xmax: 11, ymin: 0, ymax: 90, xticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], xlabel: 'level k (WMT14 schedule: 1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64 codes)', ylabel: 'BLEU of decoded ẑ(k)', height: 240 });
  };

  // ---------------------------------------------------------------- classifier-free guidance
  W.cfg = function (host) {
    NCM.head(host, 'Classifier-free guidance, per level (from the repo)', 'Run the model twice: once with the prompt (conditional logits c) and once with an empty prompt (unconditional logits u). Mix: logits = (1 + t)·c − t·u. In <code>_sample_level</code> the strength ramps up with depth: t = w · k/(K−1), so coarse levels are guided little and fine levels more.');
    const names = ['code A', 'code B', 'code C', 'code D'];
    const c = [2.0, 1.6, 0.2, -0.5], u = [1.0, 1.9, 0.6, -0.2];
    let w = 1, k = 6, K = 11;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'guidance weight w', 0, 3, 0.1, w, function (v) { w = v; draw(); }, function (v) { return v.toFixed(1); });
    NCM.slider(ctr, 'level index k (0…K−1)', 0, K - 1, 1, k, function (v) { k = v; draw(); }, null);
    const out = el('div', null, host), note = el('div', { class: 'note' }, host);
    const sm = function (l) { const m = Math.max.apply(null, l), e = l.map(function (x) { return Math.exp(x - m); }), z = e.reduce(function (a, b) { return a + b; }); return e.map(function (x) { return x / z; }); };
    function draw() {
      const t = w * k / (K - 1), g = c.map(function (ci, i) { return (1 + t) * ci - t * u[i]; });
      const rows = [['conditional c', sm(c)], ['unconditional u', sm(u)], ['guided (t = ' + t.toFixed(2) + ')', sm(g)]];
      out.innerHTML = '';
      const tb = el('table', null, out);
      tb.innerHTML = '<tr><th></th>' + names.map(function (n) { return '<th>' + n + '</th>'; }).join('') + '</tr>';
      rows.forEach(function (r) { const tr = el('tr', null, tb); el('td', { html: '<b>' + r[0] + '</b>' }, tr); r[1].forEach(function (p) { el('td', { html: '<div class="bar-h" style="width:' + Math.round(p * 90) + 'px"></div><span class="small">' + (p * 100).toFixed(1) + '%</span>' }, tr); }); });
      note.innerHTML = 'Guidance pushes probability toward codes the prompt makes <i>more</i> likely than the unconditional model does (here code A). Cost: two network calls per step, so the paper counts each CFG step as 2 NFE. Paper Figure 14: XSum ROUGE peaks at w = 0.5, WMT14 BLEU at w = 1.0.';
    }
    draw();
  };

  // ---------------------------------------------------------------- MBR decoding
  W.mbr = function (host) {
    const toy = NCM.toy(), T = toy.T, gen = toy.gen;
    NCM.head(host, 'Minimum Bayes Risk (MBR) decoding', 'Because NCM is cheap, we can draw several candidates and keep the one that agrees most with the others (“the consensus”). Similarity here is unigram F1 overlap; the paper uses task metrics. Generate candidates and inspect the similarity matrix.');
    let N = 6, seed = 40;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'candidates', 2, 12, 1, N, function (v) { N = v; draw(); }, null);
    NCM.button(ctr, 'New candidates', function () { seed += 50; draw(); }, 'primary');
    const out = el('div', { class: 'scroll-x' }, host), note = el('div', { class: 'note' }, host);
    const f1 = function (a, b) { const ca = {}, cb = {}; a.forEach(function (w) { ca[w] = (ca[w] || 0) + 1; }); b.forEach(function (w) { cb[w] = (cb[w] || 0) + 1; }); let ov = 0; for (const w in ca) ov += Math.min(ca[w], cb[w] || 0); return ov ? 2 * ov / (a.length + b.length) : 0; };
    function draw() {
      // condition every candidate on the same fixed first two levels (a shared "prompt-like" concept)
      const base = gen.generate(seed, { tau: 0.6 });
      const cands = []; for (let i = 0; i < N; i++) cands.push(T.words(gen.generate(seed + 1 + i, { tau: 1.2, fixedFirst: base.codes[0], fixed: { 1: base.codes[1] } }).steps[4].words));
      const sim = cands.map(function (a) { return cands.map(function (b) { return f1(a, b); }); });
      const avg = sim.map(function (r, i) { return (r.reduce(function (x, y) { return x + y; }, 0) - 1) / (N - 1); });
      const best = avg.indexOf(Math.max.apply(null, avg));
      out.innerHTML = '';
      const tb = el('table', null, out);
      tb.innerHTML = '<tr><th>#</th><th>candidate</th>' + cands.map(function (_, j) { return '<th>' + (j + 1) + '</th>'; }).join('') + '<th>avg sim</th></tr>';
      cands.forEach(function (c, i) {
        const tr = el('tr', { style: i === best ? 'background:var(--accent-soft)' : '' }, tb);
        el('td', { text: i + 1 }, tr); el('td', { html: '<code>' + c.join(' ') + '</code>' + (i === best ? ' ⭐' : '') }, tr);
        sim[i].forEach(function (s, j) { el('td', { text: i === j ? '—' : s.toFixed(2), style: 'font-family:var(--font-mono);font-size:12px;background:rgba(183,121,31,' + (i === j ? 0 : s * 0.5) + ')' }, tr); });
        el('td', { html: '<b>' + avg[i].toFixed(3) + '</b>' }, tr);
      });
      note.innerHTML = 'The starred candidate is returned. Paper (Appendix F.4): on XSum, going from 1 to 256 candidates gains +4.48 ROUGE-1; the main table uses 16 candidates on XSum (128 NFE) and 12 on WMT14 (132 NFE). Diffusion baselines spend the same NFE budget on a single sample.';
    }
    draw();
  };

  // ---------------------------------------------------------------- cost calculator
  W.cost = function (host) {
    NCM.head(host, 'Why NCM is fast: a back-of-the-envelope cost model', 'Count two things: <b>sequential steps</b> (network calls that must happen one after another) and <b>work per step</b> (positions processed, and attention pairs ≈ positions²). This simplified model ignores constant factors, but shows where the 10–31× speed-ups come from.');
    let L = 128, T = 128, sched = 'TinyStories / LM1B (L=128, K=12)';
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'NCM schedule', Object.keys(NCM.repoSchedules), sched, function (v) { sched = v; L = NCM.repoSchedules[v].slice(-1)[0]; draw(); });
    NCM.slider(ctr, 'diffusion steps T', 8, 1024, 8, T, function (v) { T = v; draw(); }, null);
    const out = el('div', null, host), plot = el('div', null, host);
    function draw() {
      const s = NCM.repoSchedules[sched]; L = s[s.length - 1];
      const ar = { steps: L, pos: L, attn: L * (L + 1) / 2 };                 // KV cache: step t attends to t keys
      const md = { steps: T, pos: T * L, attn: T * L * L };
      const nc = { steps: s.length, pos: s.reduce(function (a, b) { return a + b; }, 0), attn: s.reduce(function (a, b) { return a + b * b; }, 0) };
      out.innerHTML = '<table><tr><th>method</th><th>sequential steps</th><th>positions processed</th><th>attention pairs</th></tr>' +
        [['Autoregressive (KV cache)', ar], ['Masked diffusion, T steps', md], ['NCM', nc]].map(function (r) { return '<tr><td><b>' + r[0] + '</b></td><td>' + r[1].steps.toLocaleString() + '</td><td>' + r[1].pos.toLocaleString() + '</td><td>' + r[1].attn.toLocaleString() + '</td></tr>'; }).join('') + '</table>' +
        '<div class="note">NCM vs diffusion: ' + (md.steps / nc.steps).toFixed(1) + '× fewer steps, ' + (md.pos / nc.pos).toFixed(1) + '× fewer positions, ' + (md.attn / nc.attn).toFixed(1) + '× fewer attention pairs (block-diagonal attention: Σ l_k² instead of T·L²). Real wall-clock speed-ups measured by the paper on one H100: 21× on LM1B, 10× on OpenWebText, 31× on XSum.</div>';
      NCM.chart.bars(plot, [{ label: 'AR', value: ar.steps, color: 'var(--blue)' }, { label: 'diffusion', value: md.steps, color: 'var(--green)' }, { label: 'NCM', value: nc.steps, color: 'var(--accent)' }], { height: 180, ylabel: 'sequential steps' });
    }
    draw();
  };

  // ---------------------------------------------------------------- training budget
  W.trainBudget = function (host) {
    NCM.head(host, 'Counting training tokens (paper Appendix C.2, Figure 7)', 'Budget = updates × batch size × sequence length, summed over tokenizer and generator training. Edit the numbers to see the arithmetic; defaults are the paper’s LM1B settings.');
    const v = { tu: 25957, tb: 2048, gu: 100000, gb: 512, L: 128 };
    const ctr = el('div', { class: 'controls' }, host);
    [['tu', 'tokenizer updates'], ['tb', 'tokenizer batch'], ['gu', 'generator updates'], ['gb', 'generator batch'], ['L', 'length L']].forEach(function (p) {
      const lab = el('label', null, ctr); lab.appendChild(document.createTextNode(p[1] + ' '));
      const i = el('input', { type: 'number', value: v[p[0]], style: 'width:96px' }, lab); i.addEventListener('input', function () { v[p[0]] = +i.value || 0; draw(); });
    });
    const out = el('div', { class: 'out' }, host), plot = el('div', null, host);
    function draw() {
      const a = v.tu * v.tb * v.L, b = v.gu * v.gb * v.L, t = (a + b) / 1e9;
      out.textContent = 'tokenizer: ' + v.tu.toLocaleString() + ' × ' + v.tb.toLocaleString() + ' × ' + v.L + ' = ' + (a / 1e9).toFixed(2) + 'B\ngenerator: ' + v.gu.toLocaleString() + ' × ' + v.gb.toLocaleString() + ' × ' + v.L + ' = ' + (b / 1e9).toFixed(2) + 'B\ntotal = ' + t.toFixed(2) + 'B token positions   (baselines on LM1B: 66B → ' + (66 / t).toFixed(1) + '× more)';
      NCM.chart.bars(plot, [{ label: 'NCM', value: t, color: 'var(--accent)', note: t.toFixed(1) + 'B · PPL 93.1' }, { label: 'RDLM', value: 66, color: '#a0aec0', note: '66B · 295.4' }, { label: 'CANDI', value: 66, color: '#a0aec0', note: '66B · 124.6' }, { label: 'MDLM', value: 66, color: '#a0aec0', note: '66B · 115.5' }, { label: 'DUO', value: 66, color: '#a0aec0', note: '66B · 97.3' }, { label: 'FLM', value: 66, color: '#a0aec0', note: '66B · 112.3' }], { height: 220, ylabel: 'training tokens (B)', max: 80 });
      el('div', { class: 'note', text: 'Bar labels: training tokens · Gen. PPL on LM1B (lower is better), as in paper Figure 7.' }, plot);
    }
    draw();
  };

  // ---------------------------------------------------------------- ELBO gap
  W.elbo = function (host) {
    NCM.head(host, 'The ELBO and its gap (paper Appendix A)', 'log p(x) = log Σ<sub>r</sub> p(x | r) p(r). The tokenizer picks one hierarchy r* for x; the ELBO keeps only that term. The gap equals −log p(r* | x): how much posterior probability the model puts on <i>other</i> hierarchies that could also explain x.');
    let pr = 0.02, px = 0.9, other = 0.004;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'prior p(r*)', 0.001, 0.2, 0.001, pr, function (v) { pr = v; draw(); }, function (v) { return v.toFixed(3); });
    NCM.slider(ctr, 'decoder p(x | r*)', 0.05, 1, 0.01, px, function (v) { px = v; draw(); }, function (v) { return v.toFixed(2); });
    NCM.slider(ctr, 'Σ over other r of p(x|r)p(r)', 0, 0.05, 0.001, other, function (v) { other = v; draw(); }, function (v) { return v.toFixed(3); });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const joint = px * pr, marg = joint + other, elbo = Math.log(joint), lp = Math.log(marg), post = joint / marg;
      out.textContent = 'ELBO = log p(x | r*) + log p(r*) = ' + Math.log(px).toFixed(3) + ' + ' + Math.log(pr).toFixed(3) + ' = ' + elbo.toFixed(3) +
        '\nlog p(x) = log(' + joint.toFixed(4) + ' + ' + other.toFixed(4) + ') = ' + lp.toFixed(3) +
        '\ngap = log p(x) − ELBO = ' + (lp - elbo).toFixed(3) + '  =  −log p(r* | x) = −log(' + post.toFixed(3) + ')' +
        '\nThe bound is tight exactly when no other hierarchy explains x (posterior on r* = 1).';
    }
    draw();
  };
})();
