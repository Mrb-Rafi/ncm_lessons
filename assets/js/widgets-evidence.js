/* Part IV widgets: metrics, results explorer, Sudoku replay, probing the hierarchy. */
(function () {
  'use strict';
  const NCM = window.NCM, W = NCM.widgets, el = NCM.el;

  // ---------------------------------------------------------------- perplexity
  W.perplexity = function (host) {
    NCM.head(host, 'Generative perplexity, by hand', 'A separate, trusted “judge” language model (GPT-2-large in the paper) reads the generated text and reports how probable each token is. Perplexity = exp(average −log p). Lower = the judge found the text more natural. Adjust the judge’s probabilities.');
    const words = ['the', 'fox', 'slept', 'under', 'the', 'tree'];
    const p = [0.3, 0.02, 0.1, 0.2, 0.6, 0.15];
    const ctr = el('div', null, host), out = el('div', { class: 'out' }, host);
    words.forEach(function (w, i) { const c = el('div', { class: 'controls', style: 'margin:2px 0' }, ctr); NCM.slider(c, 'judge p(“' + w + '” | before)', 0.001, 1, 0.001, p[i], function (v) { p[i] = v; draw(); }, function (v) { return v.toFixed(3); }); });
    function draw() {
      const nll = p.map(function (x) { return -Math.log(x); }), m = nll.reduce(function (a, b) { return a + b; }) / nll.length;
      out.textContent = 'average −log p = (' + nll.map(function (x) { return x.toFixed(2); }).join(' + ') + ') / ' + p.length + ' = ' + m.toFixed(3) + '\nGen. PPL = exp(' + m.toFixed(3) + ') = ' + Math.exp(m).toFixed(2) +
        '\nIntuition: the judge was as unsure as if choosing uniformly among ≈' + Math.exp(m).toFixed(0) + ' words at every step.';
    }
    draw();
  };

  // ---------------------------------------------------------------- unigram entropy
  W.entropy = function (host) {
    NCM.head(host, 'Unigram entropy H: catching boring, repetitive text', 'Gen. PPL can be “gamed” by repeating safe words: the judge loves “the the the”. So the paper also reports the entropy of each sample’s word-frequency distribution, H = −Σ p(w) log p(w) (natural log). Repetitive text has low H. Try the presets or type your own.');
    const presets = { natural: 'the little fox found a shiny key under the old tree and ran home to show his mother what he had found', repetitive: 'the the the cat the the cat the the the cat the the the cat the the the the cat the the', varied: 'quantum otters juggle seventeen marmalade violins beneath turquoise cathedral thunderstorms while sleepy accountants whistle' };
    const ctr = el('div', { class: 'controls' }, host);
    const ta = el('textarea', null, host); ta.value = presets.natural;
    Object.keys(presets).forEach(function (k) { NCM.button(ctr, k, function () { ta.value = presets[k]; draw(); }); });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const ws = ta.value.toLowerCase().match(/[a-z0-9']+|[^\sa-z0-9]/g) || [], c = {};
      ws.forEach(function (w) { c[w] = (c[w] || 0) + 1; });
      let H = 0; Object.values(c).forEach(function (n) { const q = n / ws.length; H -= q * Math.log(q); });
      out.textContent = ws.length + ' tokens, ' + Object.keys(c).length + ' distinct.   H = ' + H.toFixed(3) + ' nats   (maximum possible for this length: ln ' + ws.length + ' = ' + Math.log(Math.max(1, ws.length)).toFixed(3) + ')' +
        '\nReference values from the paper (Table 8, per-sample H of real data): TinyStories 3.96, LM1B 4.31, OpenWebText 5.44. A good model should be close to the data — not far below (boring) and not far above (word salad).';
    }
    ta.addEventListener('input', draw); draw();
  };

  // ---------------------------------------------------------------- BLEU and ROUGE
  W.bleuRouge = function (host) {
    NCM.head(host, 'BLEU and ROUGE: comparing with a human reference', 'Both count overlapping n-grams (runs of n consecutive words). ROUGE-1/2 use single words / pairs; ROUGE-L uses the longest common subsequence. BLEU combines 1- to 4-gram precisions with a brevity penalty. (Simplified sentence-level versions; scores ×100.)');
    const pre = [
      ['There is no magic potion that stops colds.', 'Sadly, there is no magic cure that prevents colds.'],
      ['The council approved the new bridge on Monday.', 'On Monday the new bridge was approved by the council.'],
      ['Scientists found water on a distant moon.', 'Researchers discovered ice on a faraway moon.']
    ];
    const ctr = el('div', { class: 'controls' }, host);
    pre.forEach(function (p, i) { NCM.button(ctr, 'Example ' + (i + 1), function () { ref.value = p[0]; cand.value = p[1]; draw(); }); });
    const g = el('div', { class: 'grid2' }, host);
    const l1 = el('div', null, g); el('div', { class: 'small', text: 'Reference (human)' }, l1); const ref = el('textarea', null, l1);
    const l2 = el('div', null, g); el('div', { class: 'small', text: 'Candidate (model output)' }, l2); const cand = el('textarea', null, l2);
    ref.value = pre[0][0]; cand.value = pre[0][1];
    const out = el('div', { class: 'out' }, host);
    const tk = function (s) { return s.toLowerCase().match(/[a-z0-9']+/g) || []; };
    const ngrams = function (t, n) { const c = {}; for (let i = 0; i + n <= t.length; i++) { const k = t.slice(i, i + n).join(' '); c[k] = (c[k] || 0) + 1; } return c; };
    const overlap = function (a, b) { let o = 0; for (const k in a) o += Math.min(a[k], b[k] || 0); return o; };
    const lcs = function (a, b) { const d = []; for (let i = 0; i <= a.length; i++) { d.push(new Array(b.length + 1).fill(0)); } for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = a[i - 1] === b[j - 1] ? d[i - 1][j - 1] + 1 : Math.max(d[i - 1][j], d[i][j - 1]); return d[a.length][b.length]; };
    const f = function (o, nc, nr) { if (!o) return 0; const p = o / nc, r = o / nr; return 2 * p * r / (p + r); };
    function draw() {
      const r = tk(ref.value), c = tk(cand.value);
      const r1 = f(overlap(ngrams(c, 1), ngrams(r, 1)), c.length, r.length), r2 = f(overlap(ngrams(c, 2), ngrams(r, 2)), Math.max(1, c.length - 1), Math.max(1, r.length - 1)), rl = f(lcs(c, r), c.length, r.length);
      const ps = []; for (let n = 1; n <= 4; n++) { const tot = Math.max(0, c.length - n + 1); ps.push((overlap(ngrams(c, n), ngrams(r, n)) + 1) / (tot + 1)); }   // +1 smoothing
      const bp = c.length > r.length ? 1 : Math.exp(1 - r.length / Math.max(1, c.length));
      const bleu = bp * Math.exp(ps.reduce(function (a, p) { return a + Math.log(p); }, 0) / 4);
      out.textContent = 'ROUGE-1 F = ' + (100 * r1).toFixed(1) + '   ROUGE-2 F = ' + (100 * r2).toFixed(1) + '   ROUGE-L F = ' + (100 * rl).toFixed(1) +
        '\nBLEU ≈ ' + (100 * bleu).toFixed(1) + '   (smoothed n-gram precisions ' + ps.map(function (p) { return p.toFixed(2); }).join(', ') + '; brevity penalty ' + bp.toFixed(2) + ')' +
        '\nNote how a perfectly good paraphrase can still score low: these metrics reward shared wording, not shared meaning.';
    }
    ref.addEventListener('input', draw); cand.addEventListener('input', draw); draw();
  };

  // ---------------------------------------------------------------- results explorer
  const RES = {
    uncond: {
      cols: ['TinyStories', 'LM1B', 'OpenWebText'],
      rows: [
        ['Data (reference)', '-', [null, 9.67, 3.96], [null, 53.51, 4.31], [null, 14.15, 5.44]],
        ['AR · sample', 'AR', [128, 8.89, 4.01], null, [1024, 35.45, 5.58]],
        ['AR · greedy', 'AR', [128, 5.34, 4.06], null, [1024, 1.07, 1.65]],
        ['MDLM', 'Disc.', [128, 17.89, 4.03], [128, 115.45, 4.33], [1024, 105.15, 5.63]],
        ['DUO', 'Disc.', [128, 19.63, 4.01], [128, 97.30, 4.30], [1024, 77.69, 5.55]],
        ['SEDD', 'Disc.', null, null, [1024, 101.43, 5.61]],
        ['Eso-LM', 'Disc.', null, null, [1024, 131.96, 5.65]],
        ['RDLM', 'Cont.', null, [128, 295.40, 4.35], null],
        ['CANDI', 'Cont.', [128, 22.74, 4.03], [128, 124.64, 4.34], [1024, 143.13, 5.71]],
        ['FLM', 'Cont.', [128, 57.50, 4.17], [128, 112.33, 4.33], [1024, 62.23, 5.33]],
        ['S-FLM', 'Cont.', [128, 95.25, 4.08], null, [1024, 123.87, 5.52]],
        ['FMLM+', 'Cont.', [128, 15.43, 3.90], null, [128, 66.60, 5.21]],
        ['NCM (ours)', 'Ours', [12, 15.41, 3.98], [12, 93.05, 4.31], [32, 83.12, 5.42]]
      ]
    }
  };
  W.results = function (host) {
    NCM.head(host, 'Results explorer (paper Tables 2, 3, 7, 8)', 'All numbers are copied from the paper. Pick a view. For perplexity, lower is better; for BLEU, ROUGE and Sudoku accuracy, higher is better. NFE = number of network calls per sample.');
    let view = 'uncond', ds = 0;
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'View', [{ value: 'uncond', label: 'Unconditional generation (Table 8)' }, { value: 'wmt', label: 'Translation WMT14 (Table 2)' }, { value: 'xsum', label: 'Summarization XSum (Table 2)' }, { value: 'sudoku', label: 'Sudoku (Table 3)' }, { value: 'judges', label: 'LM1B under other judges (Table 7)' }], view, function (v) { view = v; draw(); });
    const dsSel = NCM.select(ctr, 'Dataset', RES.uncond.cols.map(function (c, i) { return { value: i, label: c }; }), ds, function (v) { ds = +v; draw(); });
    const plot = el('div', null, host), tbl = el('div', { class: 'table-wrap' }, host), note = el('div', { class: 'note' }, host);
    function draw() {
      dsSel.parentNode.style.display = view === 'uncond' ? '' : 'none';
      tbl.innerHTML = ''; plot.innerHTML = '';
      if (view === 'uncond') {
        const rows = RES.uncond.rows.filter(function (r) { return r[2 + ds]; });
        const data = rows.filter(function (r) { return r[1] !== 'AR' && r[1] !== '-'; }).map(function (r) { const v = r[2 + ds]; return { label: r[0].replace(' (ours)', '') + '\nNFE ' + v[0], value: v[1], color: r[1] === 'Ours' ? 'var(--accent)' : r[1] === 'Disc.' ? 'var(--green)' : 'var(--blue)', note: v[1].toFixed(1) }; });
        NCM.chart.bars(plot, data, { height: 250, ylabel: 'Gen. PPL (↓)', mb: 52 });
        let h = '<table class="data"><tr><th>Method</th><th>Type</th><th>NFE</th><th>Gen. PPL ↓</th><th>H</th></tr>';
        rows.forEach(function (r) { const v = r[2 + ds]; h += '<tr' + (r[1] === 'Ours' ? ' class="ours"' : '') + '><td>' + r[0] + '</td><td>' + r[1] + '</td><td>' + (v[0] || '–') + '</td><td>' + v[1].toFixed(2) + '</td><td>' + v[2].toFixed(2) + '</td></tr>'; });
        tbl.innerHTML = h + '</table>';
        note.innerHTML = 'Green = discrete diffusion, blue = continuous diffusion/flow, gold = NCM. AR rows are excluded from the chart. Watch H: OpenWebText data has H = 5.44; FLM (5.33) and FMLM+ (5.21) reach low perplexity partly by being less diverse, as the paper notes. NCM’s H stays close to the data on all three datasets.';
      } else if (view === 'wmt') {
        const rows = [['AR', '99M', 25.2, 64], ['MDLM', '99M', 18.4, 64], ['DUO', '205M', 21.3, 64], ['E2D2', '99M', 24.8, 64], ['SeqDiffuSeq', '–', 21.3, 64], ['ELF', '140M', 17.2, 64], ['NCM', '120M', 22.6, 11], ['ELF + CFG', '140M', 26.4, 128], ['NCM + CFG', '120M', 23.9, 22], ['NCM + MBR', '120M', 25.2, 132]];
        NCM.chart.bars(plot, rows.map(function (r) { return { label: r[0].replace(' + ', '\n+ ') + '\nNFE ' + r[3], value: r[2], color: r[0].startsWith('NCM') ? 'var(--accent)' : r[0] === 'AR' ? '#a0aec0' : 'var(--green)' }; }), { height: 250, ylabel: 'BLEU (↑)', mb: 58 });
        tbl.innerHTML = '<table class="data"><tr><th>Method</th><th>Size</th><th>BLEU ↑</th><th>NFE ↓</th></tr>' + rows.map(function (r) { return '<tr' + (r[0].startsWith('NCM') ? ' class="ours"' : '') + '><td>' + r.join('</td><td>') + '</td></tr>'; }).join('') + '</table>';
        note.innerHTML = 'Without guidance, NCM is second among non-AR methods (behind E2D2) while using 5.8× fewer NFE. MDLM, DUO, E2D2 and SeqDiffuSeq numbers come from Hu et al. (ELF paper), as stated in the paper’s Appendix D.2.';
      } else if (view === 'xsum') {
        const rows = [['AR', '99M', 30.5, 10.2, 24.4, 64], ['MDLM', '99M', 33.4, 11.6, 25.8, 64], ['DUO', '205M', 31.4, 10.1, 25.0, 64], ['E2D2', '99M', 28.4, 8.3, 22.0, 64], ['SeqDiffuSeq', '–', 19.3, 1.7, 14.1, 64], ['ELF', '140M', 34.2, 11.1, 26.4, 64], ['AURORA-LM', '130M', 34.7, 12.2, 26.6, 128], ['NCM', '88M', 35.0, 12.8, 28.0, 8], ['ELF + CFG', '140M', 36.0, 12.2, 27.8, 128], ['AURORA-LM + CFG', '130M', 36.3, 13.4, 28.6, 184], ['NCM + CFG', '88M', 35.3, 13.0, 28.2, 16], ['NCM + MBR', '88M', 37.3, 14.0, 29.6, 128]];
        NCM.chart.bars(plot, rows.map(function (r) { return { label: r[0].replace(' + ', '\n+ ').replace('AURORA-LM', 'AURORA') + '\nNFE ' + r[5], value: r[4], color: r[0].startsWith('NCM') ? 'var(--accent)' : r[0] === 'AR' ? '#a0aec0' : 'var(--green)' }; }), { height: 260, ylabel: 'ROUGE-L (↑)', mb: 58 });
        tbl.innerHTML = '<table class="data"><tr><th>Method</th><th>Size</th><th>R-1</th><th>R-2</th><th>R-L</th><th>NFE</th></tr>' + rows.map(function (r) { return '<tr' + (r[0].startsWith('NCM') ? ' class="ours"' : '') + '><td>' + r.join('</td><td>') + '</td></tr>'; }).join('') + '</table>';
        note.innerHTML = 'Without guidance NCM has the best ROUGE-1/2/L with 8 NFE — 16× fewer than AURORA-LM. With MBR it beats every guided baseline on all three metrics.';
      } else if (view === 'sudoku') {
        const rows = [['AR sample', '25M', 128, 13.9, 5.1, 0.6], ['AR greedy', '25M', 128, 14.6, 5.1, 1.0], ['MDLM', '29M', 128, 92.0, 77.1, 30.2], ['DUO', '29M', 128, 96.3, 84.7, 58.4], ['CANDI', '–', 128, 79.3, 45.9, 16.7], ['FLM', '29M', 128, 94.2, 82.7, 44.5], ['S-FLM', '29M', 128, 94.8, 85.2, 45.0], ['FMLM+', '29M', 16, 97.7, 92.8, 81.3], ['NCM', '25M', 12, 99.5, 96.5, 86.1]];
        NCM.chart.bars(plot, rows.map(function (r) { return { label: r[0].replace(' ', '\n') + '\nNFE ' + r[2], value: r[5], color: r[0] === 'NCM' ? 'var(--accent)' : r[0].startsWith('AR') ? '#a0aec0' : 'var(--blue)' }; }), { height: 250, ylabel: 'hard boards solved % (↑)', max: 100, mb: 58 });
        tbl.innerHTML = '<table class="data"><tr><th>Method</th><th>Size</th><th>NFE</th><th>Easy</th><th>Medium</th><th>Hard</th></tr>' + rows.map(function (r) { return '<tr' + (r[0] === 'NCM' ? ' class="ours"' : '') + '><td>' + r.join('</td><td>') + '</td></tr>'; }).join('') + '</table>';
        note.innerHTML = 'Easy/medium/hard = 40/35/30 given clues. Baseline numbers from Agarwal et al., as cited in the paper.';
      } else {
        const rows = [['RDLM', '108M', '66B', 179.54, 315.40, 1293.67, 379.00, 541.90], ['CANDI', '139M', '66B', 67.75, 99.96, 208.59, 125.87, 125.54], ['MDLM', '139M', '66B', 63.63, 91.84, 186.53, 115.25, 114.31], ['DUO', '139M', '66B', 55.08, 77.24, 141.16, 96.59, 92.52], ['FLM', '139M', '66B', 68.08, 89.20, 150.11, 110.98, 104.59], ['NCM', '97M', '13.36B', 59.69, 75.23, 106.41, 91.64, 83.24]];
        NCM.chart.bars(plot, rows.filter(function (r) { return r[0] !== 'RDLM'; }).map(function (r) { return { label: r[0], value: r[7], color: r[0] === 'NCM' ? 'var(--accent)' : 'var(--blue)' }; }), { height: 220, ylabel: 'avg Gen. PPL (↓)' });
        tbl.innerHTML = '<table class="data"><tr><th>Model</th><th>Size</th><th>Tokens</th><th>Mistral-7B</th><th>Llama-3.1-8B</th><th>Gemma-2-9B</th><th>Qwen-3-14B</th><th>Avg.</th></tr>' + rows.map(function (r) { return '<tr' + (r[0] === 'NCM' ? ' class="ours"' : '') + '><td>' + r.join('</td><td>') + '</td></tr>'; }).join('') + '</table>';
        note.innerHTML = 'Does the LM1B win depend on the judge? Four other judges agree: NCM has the lowest average Gen. PPL, using 12 sampling steps (baselines 128) and ~5× fewer training tokens. RDLM is omitted from the chart (off-scale).';
      }
    }
    draw();
  };

  // ---------------------------------------------------------------- Sudoku replay (Figure 20)
  const SUD = {
    puzzle: ['.216.7.85', '76....2..', '..3..27..', '.523.....', '.........', '8.67.....', '2...84.36', '6.81..4..', '.4..9...7'].join(''),
    solution: '921647385764835291583912764452369178179458623836721549217584936698173452345296817',
    steps: [
      '967624965462694864767625964396862648696562649686869348679483486688286436654483296',
      '967624965462694864767655964396862648696562649686869348679483486676686436654483296',
      '937628965232695864967625924386862649696562322286839349679686236676986416353483256',
      '937624965262695961969626924326862649496562622626839649679686736676983454353483796',
      '937624965262695961969626924326862649496562622686839641679686786676983454353483796',
      '117624888442355561535686964656562111194952628723339649225573786686683454443783216',
      '962624355437655151557394264626267111194978626676329959275596496686673454549586176',
      '122644355444332111553592264225369177175558623636724459275584446688973454342998166',
      '222644385774835521853192764455339117171458623866721554227584436681173452445926811',
      '922644355764335591853912264455339188179458623836725544217584936661133452345996177',
      '921674355774835591553392764455369178179448623866725549217554936661173452355996817',
      '921647385764835291583912764452369178179458623836721549217584936698173452345296817'
    ],
    codes: [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]
  };
  W.sudoku = function (host) {
    NCM.head(host, 'Replay: NCM solving a hard Sudoku (paper Appendix J.7, Figure 20)', 'The boards below are the paper’s actual decoded outputs after each of the 12 levels (schedule 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89 codes). Bold cells are the 30 given clues. Green = matches the solution.');
    let k = 0;
    const ctr = el('div', { class: 'controls' }, host);
    const sl = NCM.slider(ctr, 'after step', 0, 12, 1, 0, function (v) { k = v; draw(); }, function (v) { return v === 0 ? 'puzzle' : v + ' / 12'; });
    let timer = null;
    NCM.button(ctr, '▶ Play', function () { clearInterval(timer); k = 0; timer = setInterval(function () { k++; sl.value = k; sl.dispatchEvent(new Event('input')); if (k >= 12) clearInterval(timer); }, 700); }, 'primary');
    const g = el('div', { style: 'display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start' }, host);
    const board = el('div', { class: 'sudoku' }, g);
    const side = el('div', { style: 'flex:1;min-width:240px' }, g);
    const counts = SUD.steps.map(function (s) { let c = 0; for (let i = 0; i < 81; i++) if (s[i] === SUD.solution[i]) c++; return c; });
    function draw() {
      board.innerHTML = '';
      const src = k === 0 ? SUD.puzzle : SUD.steps[k - 1];
      for (let i = 0; i < 81; i++) {
        const clue = SUD.puzzle[i] !== '.', ch = src[i] === '.' ? '' : src[i];
        const cls = (clue ? 'clue ' : '') + (k > 0 ? (ch === SUD.solution[i] ? 'ok' : 'bad') : '');
        el('div', { class: cls, text: ch }, board);
      }
      side.innerHTML = '';
      el('div', { html: k === 0 ? '<b>Puzzle:</b> 30 clues, 51 cells to fill. Tokenized (in the repo’s <code>sudoku_tokenizer.py</code>) as 81 digits plus 8 row separators = 89 tokens, vocabulary of 12 symbols.' : '<b>Step ' + k + '</b> (' + SUD.codes[k - 1] + ' codes at this level): <b>' + counts[k - 1] + '/81</b> cells correct.' }, side);
      NCM.chart.lines(el('div', null, side), [{ color: 'var(--green)', points: counts.map(function (c, i) { return [i + 1, c]; }) }, { color: 'var(--accent)', points: k ? [[k, counts[k - 1]]] : [], r: 7 }], { xmin: 1, xmax: 12, ymin: 0, ymax: 81, xticks: [1, 4, 8, 12], xlabel: 'step', ylabel: 'correct cells', height: 200, width: 420 });
      el('div', { class: 'note', text: 'Early steps produce boards that are wrong almost everywhere, even on the clues — yet they set up a global layout that later levels correct. Accuracy even dips (step 2, step 5) before climbing to a perfect 81/81 at the final level.' }, side);
    }
    draw();
  };

  // ---------------------------------------------------------------- probing the hierarchy (toy versions of Appendix H)
  W.probe = function (host) {
    const toy = NCM.toy(), T = toy.T, tok = toy.tok, gen = toy.gen, K = tok.levels.length;
    NCM.head(host, 'Probing the hierarchy (toy versions of paper Appendix H)', 'Three experiments the paper runs on TinyStories, re-created on our toy model. Each regenerates the finer levels with the sampler while keeping some coarse codes fixed.');
    let exp = 'fix', m = 2;
    const A = T.ids('the sleepy cow slept in the house .'), B = T.ids('a red bird jumped on my tree .');
    const ctr = el('div', { class: 'controls' }, host);
    NCM.select(ctr, 'Experiment', [{ value: 'fix', label: 'Shared structure: fix coarse levels, resample the rest' }, { value: 'mix', label: 'Mix coarse codes of two sentences' }, { value: 'perturb', label: 'Perturb one early code' }], exp, function (v) { exp = v; draw(); });
    NCM.slider(ctr, 'levels kept fixed', 1, 4, 1, m, function (v) { m = v; draw(); }, null);
    const out = el('div', null, host), note = el('div', { class: 'note' }, host);
    function line(lbl, ids, ref) {
      const r = el('div', { style: 'display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;margin:4px 0' }, out);
      el('span', { class: 'small', html: lbl }, r);
      const t = el('div', { class: 'tokrow' }, r);
      ids.forEach(function (w, i) { el('span', { class: 'chip' + (ref && w !== ref[i] ? ' hl' : ''), text: T.VOCAB[w] }, t); });
    }
    function draw() {
      out.innerHTML = '';
      const ea = tok.encode(A), eb = tok.encode(B);
      if (exp === 'fix') {
        line('<b>source</b>', A);
        const fixed = {}; for (let k = 1; k < m; k++) fixed[k] = ea.codes[k];
        for (let s = 0; s < 4; s++) { const g = gen.generate(100 + s, { tau: 0.8, fixedFirst: ea.codes[0], fixed: fixed }); line('seed ' + (s + 1) + ' (levels 1–' + m + ' fixed)', g.steps[K - 1].words, A); }
        note.innerHTML = 'Highlighted = differs from the source. With more coarse levels fixed, samples keep more of the source; with fewer, only the broad shape survives. The paper fixes the first 3 of 12 levels of TinyStories stories and finds the theme preserved while names, objects and wording change (Figure 17).';
      } else if (exp === 'mix') {
        line('<b>source A</b>', A); line('<b>source B</b>', B);
        const mixed = [ea.codes[1][0], eb.codes[1][1]];
        for (let s = 0; s < 3; s++) { const g = gen.generate(200 + s, { tau: 0.5, fixedFirst: ea.codes[0], fixed: { 1: mixed } }); line('mix, seed ' + (s + 1), g.steps[K - 1].words); }
        note.innerHTML = 'Level 2 has two codes: one summarising the first half of the sentence, one the second half. We take the first from A and the second from B, then generate all finer levels. The paper mixes coarse representations of two stories and gets outputs that blend characters and events from both (Figure 18).';
      } else {
        line('<b>source</b>', A);
        const clean = gen.generate(300, { tau: 0.6, fixedFirst: ea.codes[0], fixed: { 1: ea.codes[1] } });
        const pert = ea.codes[1].slice(); pert[1] = (pert[1] + 5) % tok.books[1].length;
        const bad = gen.generate(300, { tau: 0.6, fixedFirst: ea.codes[0], fixed: { 1: pert } });
        line('clean coarse codes', clean.steps[K - 1].words);
        line('one level-2 code changed', bad.steps[K - 1].words, clean.steps[K - 1].words);
        note.innerHTML = 'Same random seed, one coarse code swapped. A single coarse code controls a whole region of the output. The paper replaces two level-1 codes of TinyStories stories and sees characters, objects and events change while parts of the narrative survive (Figure 19).';
      }
    }
    draw();
  };
})();
