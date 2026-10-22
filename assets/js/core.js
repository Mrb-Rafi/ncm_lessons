/* Core of the NCM Interactive Web Book: navigation, progress, quizzes, helpers, lazy widgets. */
(function () {
  'use strict';
  const NCM = window.NCM = window.NCM || {};
  NCM.widgets = NCM.widgets || {};

  // ---------- storage (never required for correctness) ----------
  const store = {
    get(k, d) { try { const v = localStorage.getItem('ncmbook:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('ncmbook:' + k, JSON.stringify(v)); } catch (e) { /* ignore */ } }
  };
  NCM.store = store;

  // ---------- DOM / SVG helpers ----------
  const SVGNS = 'http://www.w3.org/2000/svg';
  NCM.svg = function (tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) { if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (parent) parent.appendChild(e);
    return e;
  };
  NCM.el = function (tag, attrs, parent) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'text') e.textContent = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else if (k === 'class') e.className = attrs[k];
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  NCM.css = function (name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); };
  NCM.fmt = function (x, d) { if (!isFinite(x)) return '∞'; d = d == null ? 2 : d; return Number(x).toFixed(d); };
  NCM.clamp = function (x, a, b) { return Math.max(a, Math.min(b, x)); };
  NCM.tex = function (el) {
    if (window.renderMathInElement) {
      window.renderMathInElement(el, { delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
    }
  };
  // colour for hierarchy level k (0-based) of K: violet (coarse) -> gold (fine)
  NCM.levelColor = function (k, K) {
    const t = K <= 1 ? 1 : k / (K - 1);
    const a = [124, 77, 255], b = [214, 150, 20];
    const c = a.map(function (x, i) { return Math.round(x + (b[i] - x) * t); });
    return 'rgb(' + c.join(',') + ')';
  };
  NCM.palette = ['#2b6cb0', '#c0392b', '#2f855a', '#b7791f', '#6b46c1', '#1f7a7a', '#d53f8c', '#718096'];

  // header block for a widget: title + description
  NCM.head = function (el, title, desc) {
    el.innerHTML = '';
    NCM.el('div', { class: 'wt', text: title }, el);
    if (desc) NCM.el('div', { class: 'wd', html: desc }, el);
  };
  NCM.slider = function (parent, label, min, max, step, value, oninput, fmt) {
    const lab = NCM.el('label', null, parent);
    lab.appendChild(document.createTextNode(label + ' '));
    const inp = NCM.el('input', { type: 'range', min: min, max: max, step: step, value: value }, lab);
    const v = NCM.el('span', { class: 'val' }, lab);
    const show = function () { v.textContent = fmt ? fmt(+inp.value) : inp.value; };
    inp.addEventListener('input', function () { show(); oninput(+inp.value); });
    show();
    return inp;
  };
  NCM.button = function (parent, text, onclick, cls) {
    return NCM.el('button', { text: text, class: cls || '', onclick: onclick }, parent);
  };
  NCM.select = function (parent, label, options, value, onchange) {
    const lab = NCM.el('label', null, parent);
    if (label) lab.appendChild(document.createTextNode(label + ' '));
    const s = NCM.el('select', null, lab);
    options.forEach(function (o) {
      const opt = NCM.el('option', { value: o.value != null ? o.value : o, text: o.label != null ? o.label : o }, s);
      if ((o.value != null ? o.value : o) == value) opt.selected = true;
    });
    s.addEventListener('change', function () { onchange(s.value); });
    return s;
  };

  // ---------- tiny chart helpers (SVG) ----------
  NCM.chart = {};
  // bars: data [{label, value, color, note}]
  NCM.chart.bars = function (host, data, o) {
    o = o || {};
    host.innerHTML = '';
    const W = o.width || 640, H = o.height || 240, m = { l: o.ml || 46, r: 12, t: 18, b: o.mb || 44 };
    const svg = NCM.svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, host);
    let max = o.max;
    if (!max) {
      const raw = Math.max.apply(null, data.map(function (d) { return d.value; })) * 1.12 || 1;
      const step = Math.pow(10, Math.floor(Math.log10(raw / 4)));
      const unit = [1, 2, 2.5, 5, 10].map(function (m) { return m * step; }).find(function (u) { return u * 4 >= raw; });
      max = unit * 4;
    }
    const bw = (W - m.l - m.r) / data.length;
    const y = function (v) { return H - m.b - (v / max) * (H - m.t - m.b); };
    const ticks = o.ticks || 4;
    for (let i = 0; i <= ticks; i++) {
      const v = (max * i) / ticks;
      NCM.svg('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--line)' }, svg);
      NCM.svg('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11, class: 'svg-text', text: o.tickFmt ? o.tickFmt(v) : (+v.toFixed(1)) }, svg);
    }
    data.forEach(function (d, i) {
      const x = m.l + i * bw + bw * 0.15, w = bw * 0.7;
      NCM.svg('rect', { x: x, y: y(d.value), width: w, height: Math.max(0, H - m.b - y(d.value)), rx: 3, fill: d.color || 'var(--accent)' }, svg);
      NCM.svg('text', { x: x + w / 2, y: y(d.value) - 5, 'text-anchor': 'middle', 'font-size': 11.5, 'font-weight': 600, class: 'svg-text', text: d.note != null ? d.note : (o.valFmt ? o.valFmt(d.value) : d.value) }, svg);
      const lines = String(d.label).split('\n');
      lines.forEach(function (ln, j) {
        NCM.svg('text', { x: x + w / 2, y: H - m.b + 15 + j * 13, 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: ln }, svg);
      });
    });
    if (o.ylabel) NCM.svg('text', { x: 12, y: (H - m.b + m.t) / 2, transform: 'rotate(-90 12 ' + (H - m.b + m.t) / 2 + ')', 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: o.ylabel }, svg);
    return svg;
  };
  // lines: series [{name, color, points:[[x,y],...], dash}]
  NCM.chart.lines = function (host, series, o) {
    o = o || {};
    host.innerHTML = '';
    const W = o.width || 640, H = o.height || 260, m = { l: o.ml || 50, r: o.mr || 16, t: 16, b: 42 };
    const svg = NCM.svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, host);
    let xs = [], ys = [];
    series.forEach(function (s) { s.points.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
    const xmin = o.xmin != null ? o.xmin : Math.min.apply(null, xs), xmax = o.xmax != null ? o.xmax : Math.max.apply(null, xs);
    const ymin = o.ymin != null ? o.ymin : Math.min(0, Math.min.apply(null, ys)), ymax = o.ymax != null ? o.ymax : Math.max.apply(null, ys) * 1.08;
    const lx = o.logx, f = function (v) { return lx ? Math.log2(v) : v; };
    const X = function (v) { return m.l + ((f(v) - f(xmin)) / (f(xmax) - f(xmin) || 1)) * (W - m.l - m.r); };
    const Y = function (v) { return H - m.b - ((v - ymin) / (ymax - ymin || 1)) * (H - m.t - m.b); };
    const yt = o.yticks || 4;
    for (let i = 0; i <= yt; i++) {
      const v = ymin + ((ymax - ymin) * i) / yt;
      NCM.svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: 'var(--line)' }, svg);
      NCM.svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, class: 'svg-text', text: o.yfmt ? o.yfmt(v) : (+v.toFixed(o.ydec != null ? o.ydec : 1)) }, svg);
    }
    (o.xticks || []).forEach(function (v) {
      NCM.svg('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: o.xfmt ? o.xfmt(v) : v }, svg);
      NCM.svg('line', { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4, stroke: 'var(--muted)' }, svg);
    });
    NCM.svg('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: 'var(--muted)' }, svg);
    if (o.xlabel) NCM.svg('text', { x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: o.xlabel }, svg);
    if (o.ylabel) NCM.svg('text', { x: 12, y: (H - m.b + m.t) / 2, transform: 'rotate(-90 12 ' + (H - m.b + m.t) / 2 + ')', 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: o.ylabel }, svg);
    series.forEach(function (s) {
      if (s.points.length > 1 && !s.noline) {
        NCM.svg('polyline', { points: s.points.map(function (p) { return X(p[0]) + ',' + Y(p[1]); }).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2.2, 'stroke-dasharray': s.dash || '' }, svg);
      }
      s.points.forEach(function (p, i) {
        if (s.marker === 'star') {
          const cx = X(p[0]), cy = Y(p[1]), pts = [];
          for (let j = 0; j < 10; j++) { const r = j % 2 ? 5 : 11, a = -Math.PI / 2 + (j * Math.PI) / 5; pts.push((cx + r * Math.cos(a)) + ',' + (cy + r * Math.sin(a))); }
          NCM.svg('polygon', { points: pts.join(' '), fill: s.color, stroke: 'var(--panel)' }, svg);
        } else if (s.marker !== 'none') {
          NCM.svg('circle', { cx: X(p[0]), cy: Y(p[1]), r: s.r || 3.6, fill: s.color, stroke: 'var(--panel)', 'stroke-width': 1.2 }, svg);
        }
        if (s.labels && s.labels[i]) NCM.svg('text', { x: X(p[0]) + 8, y: Y(p[1]) - 6, 'font-size': 11, class: 'svg-text', text: s.labels[i] }, svg);
      });
      if (s.name && s.points.length) {
        const last = s.labelAt != null ? s.points[s.labelAt] : s.points[s.points.length - 1];
        NCM.svg('text', { x: X(last[0]) + (s.ldx || 8), y: Y(last[1]) + (s.ldy || 4), 'font-size': 12, 'font-weight': 700, fill: s.color, text: s.name, style: 'font-family:var(--font-ui)' }, svg);
      }
    });
    return { svg: svg, X: X, Y: Y };
  };

  // ---------- quizzes ----------
  function initQuiz(q) {
    const src = q.querySelector('script[type="application/json"]');
    if (!src) return;
    let items;
    try { items = JSON.parse(src.textContent); } catch (e) { return; }
    NCM.el('div', { class: 'qh', text: 'Check yourself' }, q);
    items.forEach(function (it) {
      const qq = NCM.el('div', { class: 'q', html: it.q }, q);
      const opts = NCM.el('div', { class: 'opts' }, q);
      const fb = NCM.el('div', { class: 'fb' }, q);
      it.o.forEach(function (o, i) {
        const b = NCM.el('button', { class: 'opt', html: o }, opts);
        b.addEventListener('click', function () {
          opts.querySelectorAll('.opt').forEach(function (x) { x.classList.remove('right', 'wrong'); });
          if (i === it.a) { b.classList.add('right'); fb.innerHTML = '✅ ' + (it.e || 'Correct!'); }
          else { b.classList.add('wrong'); fb.innerHTML = '❌ Not quite. ' + (it.h || 'Try again.'); }
          NCM.tex(fb);
        });
      });
      NCM.tex(qq); NCM.tex(opts);
    });
  }

  // ---------- table of contents, progress ----------
  function buildToc() {
    const nav = document.getElementById('toc');
    const chapters = Array.from(document.querySelectorAll('section.chapter'));
    let lastPart = null, n = 0;
    const done = store.get('done', {});
    chapters.forEach(function (ch) {
      const part = ch.getAttribute('data-part');
      if (part && part !== lastPart) { NCM.el('div', { class: 'part', text: part }, nav); lastPart = part; }
      const isCh = !ch.hasAttribute('data-nonum');
      if (isCh) n++;
      const a = NCM.el('a', { href: '#' + ch.id }, nav);
      NCM.el('span', { class: 'num', text: isCh ? n : '•' }, a);
      NCM.el('span', { text: ch.getAttribute('data-title') }, a);
      NCM.el('span', { class: 'tick', text: '✓' }, a);
      if (done[ch.id]) a.classList.add('done');
      a.addEventListener('click', function () { document.body.classList.remove('menu-open'); });
      ch._tocLink = a;
      const k = ch.querySelector('.kicker');
      if (k && isCh) k.textContent = 'Chapter ' + n + (k.textContent ? ' · ' + k.textContent : '');
      // chapter footer with "mark as read"
      if (isCh) {
        const foot = NCM.el('div', { class: 'chapter-foot' }, ch);
        const btn = NCM.el('button', { class: 'mark-read' + (done[ch.id] ? ' done' : ''), text: done[ch.id] ? '✓ Read' : 'Mark chapter as read' }, foot);
        btn.addEventListener('click', function () {
          const d = store.get('done', {});
          d[ch.id] = !d[ch.id]; store.set('done', d);
          btn.classList.toggle('done', d[ch.id]); btn.textContent = d[ch.id] ? '✓ Read' : 'Mark chapter as read';
          a.classList.toggle('done', d[ch.id]); updateProgress();
        });
        const next = chapters[chapters.indexOf(ch) + 1];
        if (next) NCM.el('a', { href: '#' + next.id, text: 'Next: ' + next.getAttribute('data-title') + ' →' }, foot);
      }
    });
    const obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          nav.querySelectorAll('a.active').forEach(function (x) { x.classList.remove('active'); });
          e.target._tocLink.classList.add('active');
          store.set('last', e.target.id);
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    chapters.forEach(function (c) { obs.observe(c); });
    updateProgress();
  }
  function updateProgress() {
    const chapters = Array.from(document.querySelectorAll('section.chapter:not([data-nonum])'));
    const done = store.get('done', {});
    const k = chapters.filter(function (c) { return done[c.id]; }).length;
    const bar = document.getElementById('progressBar'), lab = document.getElementById('progressLabel');
    if (bar) bar.style.width = (100 * k / chapters.length) + '%';
    if (lab) lab.textContent = k + ' / ' + chapters.length + ' chapters read';
  }

  // ---------- widgets: lazy initialisation ----------
  function initWidgets() {
    const els = Array.from(document.querySelectorAll('[data-widget]'));
    const start = function (el) {
      if (el._started) return; el._started = true;
      const name = el.getAttribute('data-widget'), fn = NCM.widgets[name];
      if (!fn) { el.innerHTML = '<div class="note">Widget “' + name + '” failed to load.</div>'; return; }
      try { fn(el); NCM.tex(el); } catch (err) { console.error(name, err); el.innerHTML = '<div class="note">Widget “' + name + '” hit an error: ' + err.message + '</div>'; }
    };
    if (!('IntersectionObserver' in window)) { els.forEach(start); return; }
    const obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { start(e.target); obs.unobserve(e.target); } });
    }, { rootMargin: '600px 0px' });
    els.forEach(function (el) { obs.observe(el); });
  }

  // ---------- theme & menu ----------
  function initChrome() {
    const saved = store.get('theme', null);
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    const tb = document.getElementById('themeBtn');
    if (tb) tb.addEventListener('click', function () {
      const cur = document.documentElement.getAttribute('data-theme') ||
        (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const nx = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', nx); store.set('theme', nx);
    });
    const mb = document.getElementById('menuBtn');
    if (mb) mb.addEventListener('click', function () { document.body.classList.toggle('menu-open'); });
    const resume = document.getElementById('resumeBtn');
    const last = store.get('last', null);
    if (resume) {
      if (last && document.getElementById(last)) { resume.href = '#' + last; resume.style.display = ''; }
      else resume.style.display = 'none';
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    initChrome();
    buildToc();
    document.querySelectorAll('.quiz').forEach(initQuiz);
    NCM.tex(document.body);
    initWidgets();
  });
})();
