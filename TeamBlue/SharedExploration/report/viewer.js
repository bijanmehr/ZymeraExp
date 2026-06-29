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
      'agents = dots (skill / role / id) · faint square = <b>sense</b> · ring = <b>comm range</b> · ' +
      '<span style="color:#1b7f76">– – –</span> = <b>comm link</b>';
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
        ctx.beginPath(); ctx.arc(ctr(c), ctr(r), cur.comm_r * cell, 0, 7);
        ctx.strokeStyle = col + '66'; ctx.lineWidth = 1; ctx.stroke();
      });
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
