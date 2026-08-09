/* viewer.js — shared rollout-rendering engine for the Zymera coverage gallery.
 *
 * One canvas renderer + play/slider/speed controls + sidebar list, factored out
 * of the original single-page index.html so every category page reuses the SAME
 * code. No build step, no dependencies — loaded after data.js via <script>.
 *
 * Visual semantics (kept identical across all pages):
 *   covered cell = soft green (#cfe3c8) · wall = dark slate (#3a4150)
 *   dashed teal (#1b7f76) = delivered comm link (in-range this step)
 *   faint coloured square = per-agent SENSE window · ring = per-agent COMM range
 *   dot colour = skill / role / agent-id, per the run's `kind` (Okabe-Ito palette)
 *
 * Public API:
 *   Viewer.COL, Viewer.KINDS                  — the colourblind-safe palette + legends
 *   Viewer.keysFor(filter)                    — TRAJ keys passing the predicate, sorted by coverage desc
 *   Viewer.legendHTML()                       — the shared "covered · wall · …" strip markup
 *   Viewer.init({ filter, mount, ... })       — wire up a viewer into a page; returns a small controller
 *
 * `filter` is `(key, value) => bool`; each page passes the SAME key-prefix rule
 * its category uses (mirrors groupOf in the original viewer), so a page only ever
 * shows its own runs.
 */
(function (global) {
  "use strict";

  var TRAJ = global.TRAJ || (global.window && global.window.TRAJ) || {};

  /* ---- self-contained animated-GIF encoder (median-cut quantizer + GIF89a/LZW) --------
   * No dependencies/workers; used by the "⬇ GIF" button. The LZW code width grows one
   * step LATE (next === 2^codeSize + 1) — the encoder builds its dictionary one entry
   * ahead of the decoder, so growing at 2^codeSize desyncs. Verified against Python PIL. */
  var GIFEnc = (function () {
    function medianCut(pix, maxColors) {
      function mk(a) {
        var rmn = 255, rmx = 0, gmn = 255, gmx = 0, bmn = 255, bmx = 0, i, p;
        for (i = 0; i < a.length; i++) {
          p = a[i];
          if (p[0] < rmn) rmn = p[0]; if (p[0] > rmx) rmx = p[0];
          if (p[1] < gmn) gmn = p[1]; if (p[1] > gmx) gmx = p[1];
          if (p[2] < bmn) bmn = p[2]; if (p[2] > bmx) bmx = p[2];
        }
        return { a: a, dr: rmx - rmn, dg: gmx - gmn, db: bmx - bmn };
      }
      var boxes = [mk(pix)];
      while (boxes.length < maxColors) {
        var bi = -1, best = -1, i;
        for (i = 0; i < boxes.length; i++) {
          if (boxes[i].a.length < 2) continue;
          var rng = Math.max(boxes[i].dr, boxes[i].dg, boxes[i].db);
          if (rng > best) { best = rng; bi = i; }
        }
        if (bi < 0) break;
        var bx = boxes[bi];
        var ch = (bx.dr >= bx.dg && bx.dr >= bx.db) ? 0 : (bx.dg >= bx.db ? 1 : 2);
        bx.a.sort(function (p, q) { return p[ch] - q[ch]; });
        var mid = bx.a.length >> 1;
        boxes.splice(bi, 1, mk(bx.a.slice(0, mid)), mk(bx.a.slice(mid)));
      }
      return boxes.map(function (bx) {
        var r = 0, g = 0, b = 0, n = bx.a.length, i;
        for (i = 0; i < n; i++) { r += bx.a[i][0]; g += bx.a[i][1]; b += bx.a[i][2]; }
        return [Math.round(r / n), Math.round(g / n), Math.round(b / n)];
      });
    }
    function makeMapper(pal) {
      var cache = new Map();
      return function (r, g, b) {
        var key = (r << 16) | (g << 8) | b, v = cache.get(key);
        if (v !== undefined) return v;
        var best = 0, bd = 1e9, i, p, dr, dg, db, d;
        for (i = 0; i < pal.length; i++) {
          p = pal[i]; dr = r - p[0]; dg = g - p[1]; db = b - p[2]; d = dr * dr + dg * dg + db * db;
          if (d < bd) { bd = d; best = i; if (d === 0) break; }
        }
        cache.set(key, best); return best;
      };
    }
    function lzw(indices, minCode) {
      var clear = 1 << minCode, eoi = clear + 1, codeSize = minCode + 1, next = eoi + 1;
      var dict = new Map(), out = [], buf = 0, bits = 0;
      function emit(code) { buf |= code << bits; bits += codeSize; while (bits >= 8) { out.push(buf & 0xff); buf >>= 8; bits -= 8; } }
      emit(clear);
      var prefix = indices[0], i, k, key, nc;
      for (i = 1; i < indices.length; i++) {
        k = indices[i]; key = (prefix << 8) | k; nc = dict.get(key);
        if (nc !== undefined) { prefix = nc; }
        else {
          emit(prefix); dict.set(key, next++);
          if (next === (1 << codeSize) + 1 && codeSize < 12) codeSize++;
          if (next === 4096) { emit(clear); dict.clear(); codeSize = minCode + 1; next = eoi + 1; }
          prefix = k;
        }
      }
      emit(prefix); emit(eoi);
      if (bits > 0) out.push(buf & 0xff);
      return out;
    }
    function encode(nFrames, W, H, delayMs, getRGBA) {
      var sample = [], t, i, d, npix = W * H, sampleFrames = [];
      var step = Math.max(1, Math.floor(nFrames / 6));
      for (t = 0; t < nFrames; t += step) sampleFrames.push(t);
      if (sampleFrames[sampleFrames.length - 1] !== nFrames - 1) sampleFrames.push(nFrames - 1);
      var stride = Math.max(1, Math.floor(npix / 4000)) * 4;
      for (var s = 0; s < sampleFrames.length; s++) {
        d = getRGBA(sampleFrames[s]);
        for (i = 0; i < d.length; i += stride) sample.push([d[i], d[i + 1], d[i + 2]]);
      }
      var pal = medianCut(sample, 255), map = makeMapper(pal);
      var gctBits = 2; while ((1 << gctBits) < pal.length) gctBits++;
      var tableSize = 1 << gctBits, minCode = gctBits, out = [];
      function b(x) { out.push(x & 0xff); }
      function b16(x) { out.push(x & 0xff, (x >> 8) & 0xff); }
      function str(sx) { for (var j = 0; j < sx.length; j++) out.push(sx.charCodeAt(j)); }
      function blocks(by) { var o = 0; while (o < by.length) { var n = Math.min(255, by.length - o); b(n); for (var j = 0; j < n; j++) b(by[o + j]); o += n; } b(0); }
      str("GIF89a"); b16(W); b16(H); b(0x80 | ((gctBits - 1) << 4) | (gctBits - 1)); b(0); b(0);
      for (i = 0; i < tableSize; i++) { if (i < pal.length) { b(pal[i][0]); b(pal[i][1]); b(pal[i][2]); } else { b(0); b(0); b(0); } }
      str("\x21\xff\x0bNETSCAPE2.0\x03\x01"); b16(0); b(0);
      var delayCs = Math.max(2, Math.round(delayMs / 10));
      for (t = 0; t < nFrames; t++) {
        d = getRGBA(t);
        var idx = new Uint8Array(npix);
        for (i = 0; i < npix; i++) idx[i] = map(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]);
        b(0x21); b(0xf9); b(0x04); b(0x04); b16(delayCs); b(0); b(0);
        b(0x2c); b16(0); b16(0); b16(W); b16(H); b(0); b(minCode);
        blocks(lzw(idx, minCode));
      }
      b(0x3b);
      return Uint8Array.from(out);
    }
    return { encode: encode };
  })();

  // Okabe-Ito colourblind-safe qualitative palette (agent dots / skills / roles).
  var COL = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
             '#56B4E9', '#999999', '#F0E442', '#000000', '#882255'];
  // What each `kind`'s dot colours mean.
  var KINDS = { skill: ['disperse', 'flock', 'hold'], role: ['explorer', 'relay'] };

  function keysFor(filter) {
    var ks = [];
    for (var k in TRAJ) {
      if (!Object.prototype.hasOwnProperty.call(TRAJ, k)) continue;
      if (!filter || filter(k, TRAJ[k])) ks.push(k);
    }
    ks.sort(function (a, b) { return TRAJ[b].cov - TRAJ[a].cov; });
    return ks;
  }

  function sizeStr(v) {
    return v.grid[0] + '×' + v.grid[1] + ' · ' + v.frames[0].pos.length + ' agents';
  }

  // The shared "what the colours mean" strip (identical wording everywhere).
  function legendHTML() {
    return '' +
      '<span style="background:#cfe3c8;color:#2c4327;padding:1px 7px;border-radius:3px">covered</span> · ' +
      '<span style="background:#3a4150;color:#fff;padding:1px 7px;border-radius:3px">wall</span> · ' +
      'agents = dots (skill / role / id) · faint square = <b>sense</b> · big square = <b>comm range</b> (Chebyshev) · ' +
      '<span style="color:#1b7f76">– – –</span> = <b>delivered link</b> · ' +
      '<span style="color:#9aa1ad">· · ·</span> = <b>in range, not delivered</b>';
  }

  // Resolve an element from either an id string or a node.
  function el(ref) {
    if (!ref) return null;
    return typeof ref === 'string' ? document.getElementById(ref) : ref;
  }

  /* Wire a viewer into the page.
   * opts:
   *   filter   (key,val)=>bool   which runs this page shows (required)
   *   mount    id|node           a sidebar/list container that gets the run items (required)
   *   canvas   id|node           the <canvas> (default id "cv")
   *   slider, play, tlabel, speed, info, desc, clegend, count, search
   *                              ids|nodes for the controls (each optional; absent → skipped)
   *   accent   css colour        used for the selected-item highlight (default "#2c6cb0")
   *   group    (key,val)=>string optional grouping label → renders collapsible sections
   *   onLoad   (key,val)=>void   optional hook after a run is loaded
   */
  function init(opts) {
    opts = opts || {};
    var filter = opts.filter || function () { return true; };
    var accent = opts.accent || '#2c6cb0';

    var side = el(opts.mount);
    var cv = el(opts.canvas || 'cv');
    var ctx = cv.getContext('2d');
    var slider = el(opts.slider || 'slider');
    var playBtn = el(opts.play || 'play');
    var tlabel = el(opts.tlabel || 'tlabel');
    var speed = el(opts.speed || 'speed');
    var info = el(opts.info || 'info');
    var desc = el(opts.desc || 'desc');
    var clegend = el(opts.clegend || 'clegend');
    var count = el(opts.count);
    var search = el(opts.search);

    var cur = null, curKey = null, t = 0, playing = false, cell = 18;
    var q = '';
    var collapsed = {};

    function xy(idx) { var W = cur.grid[1]; return [idx % W, Math.floor(idx / W)]; }
    function ctr(c) { return (c + 0.5) * cell; }

    function colorLegend() {
      if (!clegend) return;
      if (cur.kind in KINDS) {
        var nm = KINDS[cur.kind];
        clegend.innerHTML = 'dot colour = ' + (cur.kind === 'skill' ? 'skill' : 'role') + ': ' +
          nm.map(function (n, i) {
            return '<span class="sw" style="background:' + COL[i] + '"></span>' + n;
          }).join(' &nbsp; ');
      } else {
        clegend.innerHTML = 'dot colour = individual agent id (homogeneous policy — no shared roles)';
      }
    }

    function draw() {
      if (!cur) return;
      var H = cur.grid[0], W = cur.grid[1], f = cur.frames[t];
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, cv.width, cv.height);
      var idx, x, y;
      for (var ci = 0; ci < f.cov.length; ci++) {
        idx = f.cov[ci]; x = idx % W; y = Math.floor(idx / W);
        ctx.fillStyle = '#cfe3c8'; ctx.fillRect(x * cell, y * cell, cell, cell);
      }
      for (var wi = 0; wi < cur.walls.length; wi++) {
        idx = cur.walls[wi]; x = idx % W; y = Math.floor(idx / W);
        ctx.fillStyle = '#3a4150'; ctx.fillRect(x * cell, y * cell, cell, cell);
      }
      ctx.strokeStyle = '#eceef2'; ctx.lineWidth = .5;
      for (var i = 0; i <= W; i++) { ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, H * cell); ctx.stroke(); }
      for (var j = 0; j <= H; j++) { ctx.beginPath(); ctx.moveTo(0, j * cell); ctx.lineTo(W * cell, j * cell); ctx.stroke(); }
      // per-agent sense window + comm-range ring
      f.pos.forEach(function (p, k) {
        var r = p[0], c = p[1], col = COL[(f.tags[k] || 0) % COL.length], s = cur.sense_r;
        ctx.fillStyle = col + '22'; ctx.fillRect((c - s) * cell, (r - s) * cell, (2 * s + 1) * cell, (2 * s + 1) * cell);
        // comm range = Chebyshev ball -> a SQUARE (matches the DiskTopology metric;
        // a circle understated the corners and made overlapping rings read as a link)
        var cr = cur.comm_r;
        ctx.strokeStyle = col + '55'; ctx.lineWidth = 1;
        ctx.strokeRect((c - cr) * cell, (r - cr) * cell, (2 * cr + 1) * cell, (2 * cr + 1) * cell);
      });
      // potential links: within Chebyshev comm range but NOT delivered this step
      // (gossip delay / dropout) — faint dotted grey, so "in range" reads distinct from a
      // delivered link (dashed teal) AND from out-of-range (no line at all). The truth of
      // connectivity is the teal line; the grey is "could talk, didn't this step".
      ctx.setLineDash([2, 3]); ctx.lineWidth = 1; ctx.strokeStyle = '#9aa1ad';
      var crp = cur.comm_r, NP = f.pos.length;
      for (var pi = 0; pi < NP; pi++) for (var pj = pi + 1; pj < NP; pj++) {
        var dch = Math.max(Math.abs(f.pos[pi][0] - f.pos[pj][0]), Math.abs(f.pos[pi][1] - f.pos[pj][1]));
        if (dch > crp) continue;
        var deliv = false;
        for (var ej = 0; ej < f.edges.length; ej++) {
          var ee = f.edges[ej];
          if ((ee[0] === pi && ee[1] === pj) || (ee[0] === pj && ee[1] === pi)) { deliv = true; break; }
        }
        if (!deliv) {
          ctx.beginPath(); ctx.moveTo(ctr(f.pos[pi][1]), ctr(f.pos[pi][0]));
          ctx.lineTo(ctr(f.pos[pj][1]), ctr(f.pos[pj][0])); ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      // dashed teal delivered comm links
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6; ctx.strokeStyle = '#1b7f76';
      for (var ei = 0; ei < f.edges.length; ei++) {
        var e = f.edges[ei], a = f.pos[e[0]], b = f.pos[e[1]];
        ctx.beginPath(); ctx.moveTo(ctr(a[1]), ctr(a[0])); ctx.lineTo(ctr(b[1]), ctr(b[0])); ctx.stroke();
      }
      ctx.setLineDash([]);
      // agent dots
      f.pos.forEach(function (p, k) {
        var col = COL[(f.tags[k] || 0) % COL.length];
        ctx.beginPath(); ctx.arc(ctr(p[1]), ctr(p[0]), cell * .33, 0, 7);
        ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = '#2a2f3a'; ctx.lineWidth = 1; ctx.stroke();
      });
      if (tlabel) tlabel.textContent = 't = ' + t;
    }

    function load(k) {
      if (!(k in TRAJ)) return;
      curKey = k; cur = TRAJ[k];
      var H = cur.grid[0], W = cur.grid[1];
      cell = Math.max(7, Math.min(22, Math.floor(620 / Math.max(H, W))));
      cv.width = W * cell; cv.height = H * cell;
      if (slider) { slider.max = cur.frames.length - 1; slider.value = 0; }
      t = 0;
      if (info) info.textContent = H + '×' + W + ' · ' + cur.frames[0].pos.length +
        ' agents · comm_r ' + cur.comm_r + ' · sense_r ' + cur.sense_r;
      if (desc) desc.innerHTML = cur.desc || '';
      var items = side ? side.querySelectorAll('.item') : [];
      for (var n = 0; n < items.length; n++) items[n].classList.toggle('on', items[n].dataset.k === k);
      colorLegend();
      draw();
      if (opts.onLoad) opts.onLoad(k, cur);
    }

    // ---- sidebar / list -------------------------------------------------
    function passesSearch(v) {
      if (!q) return true;
      return (v.label + ' ' + (v.cat || '')).toLowerCase().indexOf(q) !== -1;
    }

    function buildFlat() {
      side.innerHTML = '';
      var ks = keysFor(filter).filter(function (k) { return passesSearch(TRAJ[k]); });
      ks.forEach(function (k) {
        var v = TRAJ[k];
        var it = document.createElement('div');
        it.className = 'item' + (k === curKey ? ' on' : '');
        it.dataset.k = k;
        it.style.setProperty('--accent', accent);
        it.innerHTML = '<div class="nm">' + v.label + '</div>' +
          '<div class="meta">' + sizeStr(v) + ' &nbsp;·&nbsp; ' + v.cov + '% cov</div>';
        it.onclick = function () { load(k); };
        side.appendChild(it);
      });
      if (count) count.textContent = ks.length + ' runs';
      return ks;
    }

    function buildGrouped() {
      side.innerHTML = '';
      var ks = keysFor(filter).filter(function (k) { return passesSearch(TRAJ[k]); });
      var groups = {}, order = [];
      ks.forEach(function (k) {
        var g = opts.group(k, TRAJ[k]);
        if (!(g in groups)) { groups[g] = []; order.push(g); }
        groups[g].push(k);
      });
      order.sort();
      order.forEach(function (g) {
        var open = !collapsed[g];
        var gh = document.createElement('div'); gh.className = 'grph';
        gh.innerHTML = '<span><span class="tw">' + (open ? '▾' : '▸') + '</span>' +
          g.replace(/^\d+ · /, '') + '</span><span class="ct">' + groups[g].length + '</span>';
        gh.onclick = function () { collapsed[g] = !collapsed[g]; rebuild(); };
        side.appendChild(gh);
        var body = document.createElement('div'); body.style.display = open ? 'block' : 'none';
        groups[g].forEach(function (k) {
          var v = TRAJ[k];
          var it = document.createElement('div');
          it.className = 'item' + (k === curKey ? ' on' : '');
          it.dataset.k = k; it.style.setProperty('--accent', accent);
          it.innerHTML = '<div class="nm">' + v.label + '</div>' +
            '<div class="meta">' + sizeStr(v) + ' &nbsp;·&nbsp; ' + v.cov + '% cov</div>';
          it.onclick = function () { load(k); };
          body.appendChild(it);
        });
        side.appendChild(body);
      });
      if (count) count.textContent = ks.length + ' runs';
      return ks;
    }

    function rebuild() { return opts.group ? buildGrouped() : buildFlat(); }

    // ---- controls -------------------------------------------------------
    if (slider) slider.oninput = function () { t = +slider.value; draw(); };
    if (playBtn) playBtn.onclick = function () {
      playing = !playing; playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
    };
    if (search) search.oninput = function () { q = search.value.toLowerCase().trim(); rebuild(); };

    // ---- download the current rollout as an animated GIF ----------------
    function exportGif(btn) {
      if (!cur || typeof GIFEnc === 'undefined') return;
      var old = btn ? btn.textContent : null;
      if (btn) { btn.disabled = true; btn.textContent = '⏳ GIF…'; }
      var wasPlaying = playing; playing = false;
      var savedT = t, W = cv.width, H = cv.height, n = cur.frames.length;
      var delay = Math.round(1000 / ((speed && +speed.value) || 8));
      setTimeout(function () {           // yield so the button text repaints first
        try {
          var bytes = GIFEnc.encode(n, W, H, delay, function (i) {
            t = i; draw(); return ctx.getImageData(0, 0, W, H).data;
          });
          t = savedT; draw();
          var a = document.createElement('a');
          a.href = URL.createObjectURL(new Blob([bytes], { type: 'image/gif' }));
          a.download = (curKey || 'rollout') + '.gif';
          document.body.appendChild(a); a.click(); a.remove();
          setTimeout(function () { URL.revokeObjectURL(a.href); }, 3000);
        } finally {
          playing = wasPlaying;
          if (btn) { btn.disabled = false; btn.textContent = old; }
        }
      }, 30);
    }
    // auto-inject a "⬇ GIF" button next to Play — so every viewer page gets it for free
    if (playBtn && playBtn.parentNode) {
      var gifBtn = document.createElement('button');
      gifBtn.textContent = '⬇ GIF';
      gifBtn.title = 'Download this rollout as an animated GIF';
      gifBtn.className = playBtn.className;
      gifBtn.onclick = function () { exportGif(gifBtn); };
      playBtn.parentNode.appendChild(gifBtn);
    }

    function tick() {
      if (playing && cur) { t = (t + 1) % cur.frames.length; if (slider) slider.value = t; draw(); }
      setTimeout(tick, 1000 / ((speed && +speed.value) || 8));
    }

    var initial = rebuild();
    if (initial.length) load(initial[0]);
    tick();

    return { load: load, rebuild: rebuild, current: function () { return curKey; } };
  }

  global.Viewer = { COL: COL, KINDS: KINDS, keysFor: keysFor, legendHTML: legendHTML, init: init };
})(typeof window !== 'undefined' ? window : this);
