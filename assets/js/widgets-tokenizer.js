/* Part II widgets: coarse-to-fine images, autoencoders, VQ, residual VQ, multiscale RVQ, the toy tokenizer, schedules. */
(function () {
  'use strict';
  const NCM = window.NCM, W = NCM.widgets, el = NCM.el, S = NCM.svg;

  // ---------------------------------------------------------------- image pyramid
  W.pyramid = function (host) {
    NCM.head(host, 'Coarse-to-fine for pictures', 'An image can be described at many resolutions. Each coarse pixel is the average of a block of fine pixels. Image models such as VAR and HMAR (cited by the paper as NCM’s inspiration) generate the 1×1 version first, then 2×2, 4×4, … each step adding detail. Slide through the scales.');
    const N = 32, img = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = x - 15.5, dy = y - 15.5, r = Math.sqrt(dx * dx + dy * dy);
      let c = [120, 190, 235];                                   // sky
      if (y > 22) c = [90, 160, 80];                             // grass
      const sx = x - 24, sy = y - 6; if (sx * sx + sy * sy < 16) c = [250, 210, 60];   // sun
      if (r < 9 && y > 8) c = [240, 170, 70];                    // face
      if ((Math.abs(x - 12) < 1.5 && Math.abs(y - 15) < 1.5) || (Math.abs(x - 19) < 1.5 && Math.abs(y - 15) < 1.5)) c = [40, 30, 30];
      if (y >= 19 && y <= 20 && x >= 12 && x <= 19) c = [150, 50, 40];
      img.push(c);
    }
    const scales = [1, 2, 4, 8, 16, 32];
    let s = 2;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'scale', 0, 5, 1, s, function (v) { s = v; draw(); }, function (v) { return scales[v] + '×' + scales[v]; });
    const wrap = el('div', { style: 'display:flex;gap:16px;flex-wrap:wrap;align-items:center' }, host);
    const cv = el('canvas', { width: 256, height: 256, style: 'width:256px;height:256px;image-rendering:pixelated' }, wrap);
    const txt = el('div', { class: 'note', style: 'max-width:340px' }, wrap);
    function draw() {
      const n = scales[s], b = N / n, ctx = cv.getContext('2d');
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const c = [0, 0, 0];
        for (let y = j * b; y < (j + 1) * b; y++) for (let x = i * b; x < (i + 1) * b; x++) for (let q = 0; q < 3; q++) c[q] += img[y * N + x][q];
        ctx.fillStyle = 'rgb(' + c.map(function (v) { return Math.round(v / (b * b)); }).join(',') + ')';
        ctx.fillRect(i * 256 / n, j * 256 / n, 256 / n + 0.5, 256 / n + 0.5);
      }
      txt.innerHTML = '<b>' + n + '×' + n + ' = ' + (n * n) + ' values.</b><br>' + ['One colour: the overall “mood” of the picture.', 'Sky on top, ground below.', 'A warm blob in the middle — something is there.', 'A face shape and the sun appear.', 'Eyes and mouth become visible.', 'Full detail.'][s] + '<br><br>Text has no natural “downsampling”: you cannot average two words. NCM’s trick is to <i>learn</i> a coarse-to-fine description of text.';
    }
    draw();
  };

  // ---------------------------------------------------------------- autoencoder
  W.autoencoder = function (host) {
    const toy = NCM.toy(), T = toy.T;
    NCM.head(host, 'An autoencoder in miniature', 'The encoder maps tokens to vectors z; the decoder maps vectors back to tokens. Add noise to z and see when decoding breaks: the decoder picks, for each position, the word whose embedding is nearest.');
    const ids = T.ids('the happy fox jumped over the box .'.replace('over', 'near'));
    let noise = 0, seed = 1;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'noise added to z', 0, 2, 0.05, 0, function (v) { noise = v; draw(); }, function (v) { return v.toFixed(2); });
    NCM.button(ctr, 'New noise', function () { seed++; draw(); });
    const out = el('div', null, host);
    function draw() {
      const rnd = T.mulberry32(seed), g = function () { return Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd()); };
      const z = ids.map(function (i) { return T.EMB[i].map(function (v) { return v + noise * g(); }); });
      const dec = z.map(function (v) { return T.nearest(v, T.EMB); });
      out.innerHTML = '';
      const tb = el('table', null, out);
      tb.innerHTML = '<tr><th>input x</th><th>encoder → z</th><th>decoder → x̂</th></tr>';
      ids.forEach(function (id, i) {
        const tr = el('tr', null, tb);
        el('td', { html: '<code>' + T.VOCAB[id] + '</code>' }, tr);
        el('td', { text: '(' + z[i].map(function (v) { return v.toFixed(2); }).join(', ') + ')', style: 'font-family:var(--font-mono)' }, tr);
        el('td', { html: '<span class="chip ' + (dec[i] === id ? 'ok' : 'bad') + '">' + T.VOCAB[dec[i]] + '</span>' }, tr);
      });
      const acc = dec.filter(function (d, i) { return d === ids[i]; }).length;
      el('div', { class: 'note', html: 'Reconstruction accuracy: <b>' + acc + '/' + ids.length + '</b>. NCM’s tokenizer is an autoencoder whose middle part is not a free vector but a hierarchy of <i>discrete codes</i> — next sections.' }, out);
    }
    draw();
  };

  // ---------------------------------------------------------------- vector quantization 2-D
  W.vq2d = function (host) {
    NCM.head(host, 'Vector quantization: snapping to a codebook', 'Grey dots are data vectors. Coloured diamonds are the <b>codebook</b> entries e<sub>1</sub>…e<sub>|V|</sub>. Every point of the plane belongs to its nearest entry (coloured regions). Drag the black probe. Press <b>k-means step</b> to let the codebook learn where the data is.');
    const rnd = NCM.toy().T.mulberry32(21);
    const g = function () { return Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd()); };
    const centers = [[-2, 1.5], [1.8, 2], [0.5, -1.8], [-1.5, -1.2], [2.5, -0.5]];
    const data = []; for (let i = 0; i < 260; i++) { const c = centers[i % 5]; data.push([c[0] + 0.55 * g(), c[1] + 0.55 * g()]); }
    let V = 5, book = [], probe = [0.3, 0.4], iter = 0;
    function init() { book = []; for (let i = 0; i < V; i++) book.push([-3.5 + 7 * rnd(), -3 + 6 * rnd()]); iter = 0; }
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'codebook size |V|', 2, 16, 1, V, function (v) { V = v; init(); draw(); }, null);
    NCM.button(ctr, 'k-means step', function () { kstep(); draw(); }, 'primary');
    NCM.button(ctr, 'Re-initialise', function () { init(); draw(); });
    const Wd = 560, H = 380, sc = 62, ox = Wd / 2, oy = H / 2;
    const wrap = el('div', { style: 'position:relative;max-width:' + Wd + 'px' }, host);
    const cv = el('canvas', { width: Wd, height: H, style: 'width:100%;' }, wrap);
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, style: 'position:absolute;inset:0;width:100%;height:100%;touch-action:none' }, wrap);
    const out = el('div', { class: 'out' }, host);
    const X = function (v) { return ox + v * sc; }, Y = function (v) { return oy - v * sc; };
    const hues = function (j) { return 'hsl(' + ((j * 360) / Math.max(V, 1) + 20) + ',70%,'; };
    function nearest(p) { let b = 0, bd = Infinity; book.forEach(function (e, j) { const d = (p[0] - e[0]) ** 2 + (p[1] - e[1]) ** 2; if (d < bd) { bd = d; b = j; } }); return [b, bd]; }
    function kstep() { const s = book.map(function () { return [0, 0, 0]; }); data.forEach(function (p) { const j = nearest(p)[0]; s[j][0] += p[0]; s[j][1] += p[1]; s[j][2]++; }); book = book.map(function (e, j) { return s[j][2] ? [s[j][0] / s[j][2], s[j][1] / s[j][2]] : e; }); iter++; }
    let drag = false;
    function draw() {
      const ctx = cv.getContext('2d'), dark = document.documentElement.getAttribute('data-theme') === 'dark' || (!document.documentElement.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
      for (let py = 0; py < H; py += 4) for (let px = 0; px < Wd; px += 4) { const j = nearest([(px - ox) / sc, (oy - py) / sc])[0]; ctx.fillStyle = hues(j) + (dark ? '22%)' : '90%)'); ctx.fillRect(px, py, 4, 4); }
      svg.innerHTML = '';
      const used = new Set(); let err = 0;
      data.forEach(function (p) { const n = nearest(p); used.add(n[0]); err += n[1]; S('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.3, fill: 'var(--muted)' }, svg); });
      book.forEach(function (e, j) { S('rect', { x: X(e[0]) - 7, y: Y(e[1]) - 7, width: 14, height: 14, transform: 'rotate(45 ' + X(e[0]) + ' ' + Y(e[1]) + ')', fill: hues(j) + '45%)', stroke: 'var(--panel)', 'stroke-width': 2 }, svg); S('text', { x: X(e[0]) + 10, y: Y(e[1]) - 8, 'font-size': 11, 'font-weight': 700, class: 'svg-text', text: 'e' + (j + 1) }, svg); });
      const n = nearest(probe), e = book[n[0]];
      S('line', { x1: X(probe[0]), y1: Y(probe[1]), x2: X(e[0]), y2: Y(e[1]), stroke: 'var(--ink)', 'stroke-dasharray': '3 3' }, svg);
      const pc = S('circle', { cx: X(probe[0]), cy: Y(probe[1]), r: 8, fill: 'var(--ink)', stroke: 'var(--panel)', 'stroke-width': 2, style: 'cursor:grab' }, svg);
      pc.addEventListener('pointerdown', function (ev) { drag = true; svg.setPointerCapture(ev.pointerId); });
      out.textContent = 'probe z = (' + probe.map(function (v) { return v.toFixed(2); }).join(', ') + ')  →  code index r = ' + (n[0] + 1) + ',  quantized z_q = e' + (n[0] + 1) + ' = (' + e.map(function (v) { return v.toFixed(2); }).join(', ') + ')' +
        '\nquantization error ‖z − z_q‖² = ' + n[1].toFixed(3) +
        '\nk-means iterations: ' + iter + '   mean error over data: ' + (err / data.length).toFixed(3) + '   codebook utilization: ' + used.size + '/' + V + ' (' + Math.round(100 * used.size / V) + '%)';
    }
    svg.addEventListener('pointermove', function (ev) { if (!drag) return; const r = svg.getBoundingClientRect(), s = Wd / r.width; probe = [((ev.clientX - r.left) * s - ox) / sc, (oy - (ev.clientY - r.top) * s) / sc]; draw(); });
    svg.addEventListener('pointerup', function () { drag = false; });
    init(); draw();
  };

  // ---------------------------------------------------------------- coin change analogy
  W.coinChange = function (host) {
    NCM.head(host, 'Residual quantization is like paying with coins', 'Approximate a number using one “coin” from each codebook in turn. Each codebook is finer than the last, and each one only has to fix the <b>residual</b> — what is still missing. Drag the target.');
    const books = [[0, 0.25, 0.5, 0.75, 1.0], [-0.1, -0.05, 0, 0.05, 0.1], [-0.02, -0.01, 0, 0.01, 0.02], [-0.004, -0.002, 0, 0.002, 0.004]];
    let target = 0.687;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'target z', 0, 1, 0.001, target, function (v) { target = v; draw(); }, function (v) { return v.toFixed(3); });
    const out = el('div', null, host);
    function draw() {
      out.innerHTML = '';
      let rec = 0;
      const tb = el('table', null, out);
      tb.innerHTML = '<tr><th>level k</th><th>codebook V<sub>k</sub></th><th>residual before</th><th>chosen code</th><th>reconstruction ẑ<sup>(k)</sup></th><th>error</th></tr>';
      books.forEach(function (b, k) {
        const res = target - rec; let best = b[0]; b.forEach(function (c) { if (Math.abs(res - c) < Math.abs(res - best)) best = c; });
        rec += best;
        const tr = el('tr', null, tb);
        el('td', { html: '<b style="color:' + NCM.levelColor(k, 4) + '">' + (k + 1) + '</b>' }, tr);
        el('td', { text: '{' + b.join(', ') + '}', style: 'font-family:var(--font-mono);font-size:12px' }, tr);
        el('td', { text: res.toFixed(4), style: 'font-family:var(--font-mono)' }, tr);
        el('td', { text: (best >= 0 && k ? '+' : '') + best, style: 'font-family:var(--font-mono)' }, tr);
        el('td', { text: rec.toFixed(4), style: 'font-family:var(--font-mono)' }, tr);
        const e = Math.abs(target - rec), td = el('td', null, tr);
        el('div', { class: 'bar-h', style: 'height:10px;background:var(--red);width:' + Math.max(1, Math.min(100, e * 400)) + 'px' }, td);
        el('span', { class: 'small', text: e.toFixed(4) }, td);
      });
      el('div', { class: 'note', html: 'Each level has only 5 entries, yet four levels can express 5⁴ = 625 distinct values. Residual quantization buys precision by <i>stacking</i> small codebooks — and the first code alone already tells you the rough size of the number. That is the seed of “coarse-to-fine”.' }, out);
    }
    draw();
  };

  // ---------------------------------------------------------------- residual VQ in 2-D
  W.rvq2d = function (host) {
    NCM.head(host, 'Residual vector quantization in 2-D (paper eq. 2)', 'Drag the target vector z. Level 1 picks the nearest entry of a coarse codebook; level 2 quantizes the leftover residual with a finer codebook; and so on. The arrows add up to the reconstruction ẑ<sup>(K)</sup>.');
    const books = [], K = 4;
    for (let k = 0; k < K; k++) { const s = 1.6 / Math.pow(2.6, k), b = []; for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) b.push([i * s, j * s]); books.push(b); }
    let z = [2.37, 1.13], show = K;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'levels used', 1, K, 1, show, function (v) { show = v; draw(); }, null);
    const Wd = 560, H = 360, sc = 80, ox = 150, oy = 250;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, style: 'background:var(--bg-soft);border-radius:10px;touch-action:none' }, host);
    const out = el('div', { class: 'out' }, host);
    const X = function (v) { return ox + v * sc; }, Y = function (v) { return oy - v * sc; };
    let drag = false;
    svg.addEventListener('pointerdown', function (e) { drag = true; svg.setPointerCapture(e.pointerId); move(e); });
    svg.addEventListener('pointermove', function (e) { if (drag) move(e); });
    svg.addEventListener('pointerup', function () { drag = false; });
    function move(e) { const r = svg.getBoundingClientRect(), s = Wd / r.width; z = [((e.clientX - r.left) * s - ox) / sc, (oy - (e.clientY - r.top) * s) / sc]; draw(); }
    function draw() {
      svg.innerHTML = '';
      S('defs', { html: '<marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>' }, svg);
      books[0].forEach(function (e) { S('circle', { cx: X(e[0]), cy: Y(e[1]), r: 3, fill: NCM.levelColor(0, K), opacity: 0.45 }, svg); });
      let rec = [0, 0]; const lines = [];
      for (let k = 0; k < show; k++) {
        const res = [z[0] - rec[0], z[1] - rec[1]];
        let best = books[k][0], bd = Infinity; books[k].forEach(function (e) { const d = (res[0] - e[0]) ** 2 + (res[1] - e[1]) ** 2; if (d < bd) { bd = d; best = e; } });
        if (k > 0) books[k].forEach(function (e) { S('circle', { cx: X(rec[0] + e[0]), cy: Y(rec[1] + e[1]), r: 2, fill: NCM.levelColor(k, K), opacity: 0.5 }, svg); });
        const nr = [rec[0] + best[0], rec[1] + best[1]];
        S('line', { x1: X(rec[0]), y1: Y(rec[1]), x2: X(nr[0]), y2: Y(nr[1]), stroke: NCM.levelColor(k, K), 'stroke-width': 3, 'marker-end': 'url(#arw)' }, svg);
        rec = nr;
        lines.push('level ' + (k + 1) + ': residual (' + res.map(function (v) { return v.toFixed(3); }).join(', ') + ') → code e' + (k + 1) + ' = (' + best.map(function (v) { return v.toFixed(3); }).join(', ') + ')   error after: ' + Math.sqrt((z[0] - rec[0]) ** 2 + (z[1] - rec[1]) ** 2).toFixed(4));
      }
      S('circle', { cx: X(0), cy: Y(0), r: 4, fill: 'var(--ink)' }, svg); S('text', { x: X(0) - 14, y: Y(0) + 16, 'font-size': 12, class: 'svg-text', text: 'ẑ(0)=0' }, svg);
      S('circle', { cx: X(z[0]), cy: Y(z[1]), r: 8, fill: 'none', stroke: 'var(--red)', 'stroke-width': 2.5 }, svg);
      S('text', { x: X(z[0]) + 11, y: Y(z[1]) - 8, 'font-size': 13, 'font-weight': 700, fill: 'var(--red)', text: 'z' }, svg);
      out.textContent = lines.join('\n') + '\n\nẑ(' + show + ') = sum of chosen codes = (' + rec.map(function (v) { return v.toFixed(3); }).join(', ') + ')';
    }
    draw();
  };

  // ---------------------------------------------------------------- multiscale residual quantization on a 1-D signal
  W.multiscale1d = function (host) {
    NCM.head(host, 'Multiscale residual quantization, step by step (paper eqs. 7–8)', 'One channel of a latent sequence with L = 16 positions. Schedule l = (1, 2, 4, 8, 16). At level k: <b>(a)</b> average-pool the residual down to l<sub>k</sub> values (Down<sub>k</sub>); <b>(b)</b> snap each to the nearest codebook entry; <b>(c)</b> stretch back to 16 positions by linear interpolation (R<sub>k</sub>); <b>(d)</b> add to the reconstruction and subtract from the residual.');
    const L = 16, sched = [1, 2, 4, 8, 16], K = sched.length;
    const books = sched.map(function (_, k) { const step = 0.5 / Math.pow(1.9, k), b = []; for (let i = -6; i <= 6; i++) b.push(i * step); return b; });
    const signals = {
      'smooth wave': function (i) { return 1.2 * Math.sin(i / 2.6) + 0.4; },
      'step': function (i) { return i < 6 ? -0.8 : i < 11 ? 1.1 : 0.3; },
      'bumpy': function (i) { return 0.5 * Math.sin(i * 1.7) + 0.3 * Math.cos(i * 0.4) + (i % 5 === 0 ? 0.6 : 0); }
    };
    let sig = 'smooth wave', k = 0;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Latent z', Object.keys(signals), sig, function (v) { sig = v; k = 0; draw(); });
    NCM.button(ctr, '◀ Prev level', function () { k = Math.max(0, k - 1); draw(); });
    NCM.button(ctr, 'Next level ▶', function () { k = Math.min(K - 1, k + 1); draw(); }, 'primary');
    const T = window.ToyNCM;
    const plots = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const z = []; for (let i = 0; i < L; i++) z.push([signals[sig](i)]);
      let zhat = z.map(function () { return [0]; }), rest = z.map(function (v) { return v.slice(); }), st;
      for (let j = 0; j <= k; j++) {
        const lk = sched[j], u = lk === L ? rest : T.downArea(rest, lk);
        const codes = u.map(function (v) { let b = 0; books[j].forEach(function (c, q) { if (Math.abs(v[0] - c) < Math.abs(v[0] - books[j][b])) b = q; }); return b; });
        const q = codes.map(function (c) { return [books[j][c]]; });
        const up = lk === L ? q : T.upLinear(q, L);
        const restBefore = rest;
        zhat = zhat.map(function (v, i) { return [v[0] + up[i][0]]; });
        rest = z.map(function (v, i) { return [v[0] - zhat[i][0]]; });
        st = { u: u, q: q, up: up, codes: codes, restBefore: restBefore };
      }
      plots.innerHTML = '';
      const Wd = 640, H = 250, m = 40, svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, plots);
      const ymin = -2, ymax = 2, Y = function (v) { return 14 + (ymax - v) / (ymax - ymin) * (H - 40); }, Xi = function (i) { return m + (i + 0.5) * (Wd - m - 10) / L; };
      for (let v = -2; v <= 2; v++) { S('line', { x1: m, x2: Wd - 10, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }, svg); S('text', { x: m - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, class: 'svg-text', text: v }, svg); }
      const bw = (Wd - m - 10) / L;
      // residual before this level (bars)
      st.restBefore.forEach(function (v, i) { S('rect', { x: Xi(i) - bw * 0.42, y: Math.min(Y(0), Y(v[0])), width: bw * 0.84, height: Math.abs(Y(v[0]) - Y(0)), fill: 'var(--red)', opacity: 0.18 }, svg); });
      // pooled segments + quantized values
      const lk = sched[k];
      st.u.forEach(function (v, j) {
        const a = j * L / lk, b = (j + 1) * L / lk;
        S('line', { x1: Xi(a) - bw / 2, x2: Xi(b - 1) + bw / 2, y1: Y(v[0]), y2: Y(v[0]), stroke: 'var(--muted)', 'stroke-width': 2, 'stroke-dasharray': '4 3' }, svg);
        S('line', { x1: Xi(a) - bw / 2, x2: Xi(b - 1) + bw / 2, y1: Y(st.q[j][0]), y2: Y(st.q[j][0]), stroke: NCM.levelColor(k, K), 'stroke-width': 4 }, svg);
      });
      const poly = function (arr, color, w, dash) { S('polyline', { points: arr.map(function (v, i) { return Xi(i) + ',' + Y(v[0]); }).join(' '), fill: 'none', stroke: color, 'stroke-width': w, 'stroke-dasharray': dash || '' }, svg); };
      poly(st.up, NCM.levelColor(k, K), 1.5, '2 2');
      poly(z, 'var(--ink)', 2.5);
      poly(zhat, 'var(--accent)', 2.5);
      const lg = el('div', { class: 'legend' }, plots);
      lg.innerHTML = '<span><i style="background:var(--ink)"></i>true latent z</span><span><i style="background:var(--accent)"></i>reconstruction ẑ(' + (k + 1) + ')</span><span><i style="background:var(--red);opacity:.3"></i>residual z(' + k + ') before level ' + (k + 1) + '</span><span><i style="background:var(--muted)"></i>Down_k(residual)</span><span><i style="background:' + NCM.levelColor(k, K) + '"></i>quantized e_k(r_k) and its upsampled R_k(·)</span>';
      const err = z.reduce(function (s, v, i) { return s + (v[0] - zhat[i][0]) ** 2; }, 0) / L;
      out.textContent = 'Level ' + (k + 1) + ' of ' + K + ':  l_k = ' + lk + ' code' + (lk > 1 ? 's' : '') + '   codes r_' + (k + 1) + ' = [' + st.codes.join(', ') + ']' +
        '\ncodebook V_' + (k + 1) + ' = {' + books[k].map(function (v) { return v.toFixed(3); }).join(', ') + '}' +
        '\nmean squared error of ẑ(' + (k + 1) + ') vs z: ' + err.toFixed(5) +
        '\ntotal codes so far: ' + sched.slice(0, k + 1).reduce(function (a, b) { return a + b; }, 0) + ' (the full hierarchy has ' + sched.reduce(function (a, b) { return a + b; }, 0) + ')';
    }
    draw();
  };

  // ---------------------------------------------------------------- the toy NCM tokenizer on real (toy) sentences
  W.toyTokenizer = function (host) {
    const toy = NCM.toy(), T = toy.T, tok = toy.tok, K = tok.levels.length;
    NCM.head(host, 'A working multiscale residual VQ tokenizer for sentences', 'This is a real tokenizer trained (by k-means, greedily level by level) on 600 toy sentences. Schedule l = (' + tok.levels.join(', ') + '), codebook sizes (' + tok.sizes.join(', ') + '). Choose a sentence, then move the level slider: the decoder reads the cumulative reconstruction ẑ<sup>(k)</sup> and outputs the nearest word at each position.');
    const tmpl = ['det', 'adj', 'animal', 'verb', 'prep', 'det', 'place', 'end'];
    let ids = T.ids('the big cat sat on the mat .'), kk = 1;
    const ctr = el('div', { class: 'controls' }, host);
    const sels = [];
    tmpl.forEach(function (g, i) { if (g === 'end') return; sels.push(NCM.select(ctr, '', T.GROUPS[g].words, T.VOCAB[ids[i]], function (v) { ids[i] = T.IDX[v]; draw(); })); });
    NCM.button(ctr, 'Random sentence', function () { const s = T.makeCorpus(1, Math.floor(Math.random() * 1e6))[0]; ids = s; let j = 0; tmpl.forEach(function (g, i) { if (g !== 'end') { sels[j].value = T.VOCAB[s[i]]; j++; } }); draw(); });
    const c2 = el('div', { class: 'controls' }, host);
    NCM.slider(c2, 'decode after level k =', 1, K, 1, kk, function (v) { kk = v; draw(); }, null);
    const grid = el('div', { class: 'grid2' }, host), left = el('div', null, grid), right = el('div', null, grid);
    const stats = el('div', null, host);
    function draw() {
      const enc = tok.encode(ids);
      left.innerHTML = '';
      el('div', { class: 'small', text: 'Codes per level (what the generator will have to produce):' }, left);
      enc.trace.forEach(function (t, k) {
        const r = el('div', { style: 'display:flex;gap:4px;align-items:center;margin:4px 0;opacity:' + (k < kk ? 1 : .3) }, left);
        el('span', { class: 'small', text: 'r' + (k + 1), style: 'width:22px' }, r);
        t.codes.forEach(function (c) { el('span', { text: c, style: 'flex:1;text-align:center;font-family:var(--font-mono);font-size:12px;color:#fff;border-radius:5px;padding:2px 0;background:' + NCM.levelColor(k, K) }, r); });
      });
      el('div', { class: 'small', text: 'Decoded text after each level:', style: 'margin-top:10px' }, left);
      enc.trace.forEach(function (t, k) {
        const dec = tok.decode(t.zhat), r = el('div', { class: 'tokrow', style: 'opacity:' + (k < kk ? 1 : .3) + (k === kk - 1 ? ';outline:2px solid var(--accent);border-radius:8px;padding:2px' : '') }, left);
        dec.forEach(function (w, i) { el('span', { class: 'chip ' + (w === ids[i] ? 'ok' : (T.GROUP_OF[w] === T.GROUP_OF[ids[i]] ? 'hl' : 'bad')), text: T.VOCAB[w] }, r); });
      });
      el('div', { class: 'legend', html: '<span><i style="background:var(--green-soft);border:1px solid var(--green)"></i>exact word</span><span><i style="background:var(--accent-soft);border:1px solid var(--accent)"></i>right kind of word</span><span><i style="background:var(--red-soft);border:1px solid var(--red)"></i>wrong</span>' }, left);
      // 2-D picture of zhat^(kk)
      right.innerHTML = '';
      const Wd = 330, H = 300, sc = 40, ox = Wd / 2, oy = H / 2, svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, style: 'background:var(--bg-soft);border-radius:10px' }, right);
      const X = function (v) { return ox + v * sc; }, Y = function (v) { return oy - v * sc; };
      T.EMB.forEach(function (e, i) { S('circle', { cx: X(e[0]), cy: Y(e[1]), r: 2.5, fill: NCM.groupColor(T.GROUP_OF[i]), opacity: 0.5 }, svg); });
      const zh = enc.trace[kk - 1].zhat;
      ids.forEach(function (id, i) {
        const e = T.EMB[id];
        S('line', { x1: X(zh[i][0]), y1: Y(zh[i][1]), x2: X(e[0]), y2: Y(e[1]), stroke: 'var(--muted)', 'stroke-dasharray': '2 2' }, svg);
        S('circle', { cx: X(e[0]), cy: Y(e[1]), r: 5, fill: 'none', stroke: NCM.groupColor(T.GROUP_OF[id]), 'stroke-width': 2 }, svg);
        S('circle', { cx: X(zh[i][0]), cy: Y(zh[i][1]), r: 8, fill: NCM.levelColor(kk - 1, K) }, svg);
        S('text', { x: X(zh[i][0]), y: Y(zh[i][1]) + 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#fff', 'font-weight': 700, text: i + 1 }, svg);
      });
      el('div', { class: 'note', html: 'Filled discs: ẑ<sup>(' + kk + ')</sup> at positions 1–8. Rings: the true embeddings z. Dashed lines: the residual still to be explained. Watch the discs travel from one shared “average” point to their own words as k grows.' }, right);
      const accs = []; for (let k = 1; k <= K; k++) accs.push(tok.accuracy(toy.test.slice(0, 120), k));
      stats.innerHTML = '';
      el('div', { class: 'small', text: 'Exact-token reconstruction accuracy on 120 held-out toy sentences, decoding after each level:' }, stats);
      NCM.chart.bars(el('div', null, stats), accs.map(function (a, k) { return { label: 'k=' + (k + 1) + '\n(' + tok.levels[k] + ' codes)', value: a * 100, color: NCM.levelColor(k, K), note: (a * 100).toFixed(1) + '%' }; }), { height: 190, max: 100, ylabel: 'accuracy %' });
    }
    draw();
  };

  // ---------------------------------------------------------------- nested dropout
  W.nestedDropout = function (host) {
    NCM.head(host, 'Nested dropout: training with a random cut-off', 'At each training step the tokenizer may drop all levels finer than a random cut-off k and must reconstruct the text from ẑ<sup>(k)</sup> alone. In the released configs this happens with probability 0.5 per step (<code>fine_drop.prob: 0.5</code>), keeping at least one level (<code>min_keep: 1</code>).');
    const K = 12;
    let p = 0.5;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.slider(ctr, 'drop probability', 0, 1, 0.05, p, function (v) { p = v; }, function (v) { return v.toFixed(2); });
    NCM.button(ctr, 'Simulate 12 training steps', run, 'primary');
    const out = el('div', null, host), st = el('div', { class: 'note' }, host);
    const hist = new Array(K).fill(0);
    function run() {
      out.innerHTML = '';
      for (let s = 0; s < 12; s++) {
        let keep = K;
        if (Math.random() < p) keep = 1 + Math.floor(Math.random() * (K - 1));   // cut-off among 1..K-1 (repo: randint(kmin, kmax))
        for (let k = 0; k < keep; k++) hist[k]++;
        const r = el('div', { style: 'display:flex;gap:3px;align-items:center;margin:3px 0' }, out);
        el('span', { class: 'small', text: 'step ' + (s + 1), style: 'width:56px' }, r);
        for (let k = 0; k < K; k++) el('span', { style: 'flex:1;height:16px;border-radius:4px;background:' + (k < keep ? NCM.levelColor(k, K) : 'var(--bg-soft)') }, r);
        el('span', { class: 'small', text: keep === K ? 'all 12 levels' : 'only first ' + keep, style: 'width:90px;text-align:right' }, r);
      }
      const tot = hist[0];
      st.innerHTML = 'Across all simulated steps so far, how often each level was present: ' + hist.map(function (h, k) { return 'L' + (k + 1) + ' ' + Math.round(100 * h / tot) + '%'; }).join(' · ') + '.<br>Level 1 is <b>always</b> used, so it must carry the most important information on its own. Fine levels are often absent, so the model cannot hide essential content in them.';
    }
    run();
  };

  // ---------------------------------------------------------------- stop-gradient / straight-through
  W.stopGrad = function (host) {
    NCM.head(host, 'Where gradients flow in the tokenizer loss (paper eq. 9)', 'Pick a loss term to see which parts of the tokenizer it trains. “sg” (stop-gradient) blocks the flow; the straight-through estimator lets the decoder’s gradient jump over the non-differentiable argmin.');
    const terms = {
      rec: { label: 'reconstruction: −(1/L) Σ log p(xᵢ | ẑ(k))', path: ['dec', 'st', 'enc'], txt: 'Trains the decoder, and — thanks to the straight-through estimator ẑ ← z + sg(ẑ − z) — also the encoder, as if quantization were the identity.' },
      commit: { label: 'commitment: β · MSE(z, sg[ẑ(k)])', path: ['enc'], txt: 'Pulls the encoder output z toward the chosen codes, so the encoder “commits” to the codebook. β = 0.25 in the released configs (commitment_cost).' },
      vq: { label: 'quantization: MSE(sg[z], ẑ(k))', path: ['book'], txt: 'Pulls codebook vectors toward the encoder outputs. In the repo the codebooks are actually updated by an exponential moving average (EMA) of assigned vectors — a smoothed online k-means (decay 0.95–0.99).' }
    };
    let cur = 'rec';
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Loss term', Object.keys(terms).map(function (k) { return { value: k, label: terms[k].label }; }), cur, function (v) { cur = v; draw(); });
    const svgHost = el('div', null, host), note = el('div', { class: 'note' }, host);
    function draw() {
      svgHost.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 640 150' }, svgHost);
      const box = function (x, label, id) { const on = terms[cur].path.indexOf(id) >= 0; S('rect', { x: x, y: 45, width: 110, height: 54, rx: 10, fill: on ? 'var(--accent-soft)' : 'var(--bg-soft)', stroke: on ? 'var(--accent)' : 'var(--line)', 'stroke-width': on ? 2.5 : 1 }, svg); label.split('\n').forEach(function (l, i) { S('text', { x: x + 55, y: 68 + i * 16, 'text-anchor': 'middle', 'font-size': 13, class: 'svg-text', text: l }, svg); }); };
      box(10, 'text x', 'x'); box(150, 'encoder E\nz = E(x)', 'enc'); box(290, 'codebooks\nargmin', 'book'); box(430, 'decoder\np(x | ẑ)', 'dec');
      S('text', { x: 590, y: 76, 'font-size': 13, class: 'svg-text', text: 'loss' }, svg);
      [[120, 150], [260, 290], [400, 430], [540, 580]].forEach(function (a) { S('line', { x1: a[0], x2: a[1], y1: 72, y2: 72, stroke: 'var(--muted)', 'stroke-width': 2 }, svg); });
      if (terms[cur].path.indexOf('st') >= 0) { S('path', { d: 'M430,108 C380,140 260,140 210,104', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-dasharray': '6 4' }, svg); S('text', { x: 320, y: 146, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--accent)', text: 'straight-through: gradient skips argmin' }, svg); }
      S('text', { x: 345, y: 30, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: 'argmin has zero gradient almost everywhere' }, svg);
      note.innerHTML = terms[cur].txt;
    }
    draw();
  };

  // ---------------------------------------------------------------- level schedules
  NCM.schedules = {
    linear: function (t) { return t; }, quadratic: function (t) { return t * t; }, 'square-root': function (t) { return Math.sqrt(t); },
    logarithmic: function (t) { return Math.log(1 + 9 * t) / Math.log(10); }, exponential: function (t) { return (Math.pow(128, t) - 1) / 127; }, cosine: function (t) { return (1 - Math.cos(Math.PI * t)) / 2; }
  };
  NCM.makeSchedule = function (name, K, L) {
    const f = NCM.schedules[name], out = [];
    for (let k = 0; k < K; k++) { const t = K === 1 ? 1 : k / (K - 1); let v = Math.round(1 + (L - 1) * f(t)); if (out.length && v <= out[out.length - 1]) v = out[out.length - 1] + 1; out.push(Math.min(v, L)); }
    out[K - 1] = L; return out;
  };
  NCM.repoSchedules = {
    'TinyStories / LM1B (L=128, K=12)': [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128],
    'OpenWebText (L=1024, K=32)': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024],
    'Sudoku (L=89, K=12)': [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89],
    'WMT14 De→En (L=64, K=11)': [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64],
    'XSum (L=48, K=9)': [4, 6, 10, 14, 20, 26, 32, 40, 48]
  };
  W.ablK = function (host) {
    NCM.head(host, 'Number of levels K (Fig. 9b)', 'Gen. PPL of the final checkpoint, TinyStories, linear schedules from 1 to 128, no corruption. Values read from the figure (≈); percentages as printed in it.');
    NCM.chart.bars(el('div', null, host), [{ label: 'K=4', value: 300, note: '≈300' }, { label: 'K=8', value: 144, note: '−52%' }, { label: 'K=16', value: 98, note: '−32%' }, { label: 'K=32', value: 78, note: '−20%' }].map(function (d, i) { d.color = NCM.levelColor(i, 4); return d; }), { height: 210, ylabel: 'Gen. PPL (↓)', width: 420 });
  };
  W.ablSched = function (host) {
    NCM.head(host, 'Schedule shape (Fig. 10b)', 'Gen. PPL after 10k steps, TinyStories, K = 12 levels from 1 to 128 codes. Values printed in the figure.');
    NCM.chart.bars(el('div', null, host), [['Lin.', 88.4], ['Quad.', 59.6], ['Sqrt', 196.7], ['Log', 168.2], ['Exp.', 57.7], ['Cos.', 77.8]].map(function (d, i) { return { label: d[0], value: d[1], color: NCM.palette[i] }; }), { height: 210, ylabel: 'Gen. PPL (↓)', width: 420 });
  };

  W.schedules = function (host) {
    NCM.head(host, 'Level schedules: how fast do levels grow?', 'A schedule lists l<sub>1</sub> &lt; l<sub>2</sub> &lt; … &lt; l<sub>K</sub> = L. Compare the six shapes the paper ablates (Appendix F.1.2), or the exact schedules used by the released checkpoints (from <code>config/eval/*.yaml</code> in the GitHub repo).');
    let K = 12, L = 128, view = 'shapes', repo = Object.keys(NCM.repoSchedules)[0];
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Show', [{ value: 'shapes', label: 'six schedule shapes' }, { value: 'repo', label: 'released checkpoints' }], view, function (v) { view = v; draw(); });
    NCM.slider(ctr, 'levels K', 3, 32, 1, K, function (v) { K = v; draw(); }, null);
    NCM.slider(ctr, 'length L', 16, 1024, 16, L, function (v) { L = v; draw(); }, null);
    NCM.select(ctr, 'Checkpoint', Object.keys(NCM.repoSchedules), repo, function (v) { repo = v; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      let series = [];
      if (view === 'shapes') series = Object.keys(NCM.schedules).map(function (n, i) { const s = NCM.makeSchedule(n, K, L); return { name: n, color: NCM.palette[i], points: s.map(function (v, k) { return [k + 1, v]; }), r: 2.5 }; });
      else { const s = NCM.repoSchedules[repo]; series = [{ name: repo.split(' (')[0], color: 'var(--accent)', points: s.map(function (v, k) { return [k + 1, v]; }) }]; }
      const kx = view === 'shapes' ? K : NCM.repoSchedules[repo].length;
      const xt = []; for (let i = 1; i <= kx; i += Math.max(1, Math.round(kx / 8))) xt.push(i);
      NCM.chart.lines(plot, series, { xmin: 1, xmax: kx, ymin: 0, xticks: xt, xlabel: 'hierarchy level k', ylabel: 'codes per level l_k', height: 280, mr: 96 });
      if (view === 'shapes') out.textContent = Object.keys(NCM.schedules).map(function (n) { const s = NCM.makeSchedule(n, K, L); return (n + ':').padEnd(13) + ' [' + s.join(', ') + ']   total codes ' + s.reduce(function (a, b) { return a + b; }, 0); }).join('\n');
      else { const s = NCM.repoSchedules[repo]; out.textContent = '[' + s.join(', ') + ']\nK = ' + s.length + ' levels  ·  total codes Σl_k = ' + s.reduce(function (a, b) { return a + b; }, 0) + '  ·  L = ' + s[s.length - 1]; }
    }
    draw();
  };
})();
