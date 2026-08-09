/* nav.js — single source of truth for the gallery top-nav.
   Every page includes this (root pages: "nav.js"; pages/ pages: "../nav.js").
   It rewrites <nav class="top"> identically everywhere, fixes the relative paths for the
   page's location, and marks the active link — so switching pages never drops an option.
   Styling comes from each page's existing nav.top CSS (page.css / inline). */
(function () {
  // hrefs are written relative to the report/ ROOT; nav.js rewrites them per location.
  var LINKS = [
    ['index.html', 'Overview'],
    ['arenas.html', 'Arenas'],
    ['mapmaker.html', 'Map maker'],
    ['pages/architecture.html', 'Architecture'],
    ['pages/best-arch.html', 'Best-arch × maps'],
    ['pages/recipe-ablation.html', 'Recipe ablation'],
    ['pages/frontier.html', 'Cov↔conn frontier'],
    ['pages/occupancy-ab.html', 'Occupancy A/B'],
    ['pages/warm-bigger.html', 'Warm-start → bigger'],
    ['__spacer__', ''],
    ['pages/open-floor.html', 'Open floor'],
    ['pages/team-size.html', 'Team size'],
    ['pages/obstacles.html', 'Obstacles'],
    ['pages/crowded.html', 'Crowded'],
    ['pages/es-ladder.html', 'ES ladder'],
    ['pages/connectivity-shootout.html', 'Connectivity'],
    ['pages/warm-start.html', 'Warm-start']
  ];
  function apply() {
    var nav = document.querySelector('nav.top');
    if (!nav) return;
    var path = location.pathname;
    var inPages = /\/pages\//.test(path);
    var prefix = inPages ? '../' : '';
    var file = path.substring(path.lastIndexOf('/') + 1) || 'index.html';
    var curKey = (inPages ? 'pages/' : '') + file;
    var html = '<a class="home" href="' + prefix + 'index.html">ZYMERA · coverage gallery</a>';
    for (var i = 0; i < LINKS.length; i++) {
      var href = LINKS[i][0], label = LINKS[i][1];
      if (href === '__spacer__') { html += '<span style="flex:1"></span>'; continue; }
      var active = (href === curKey) ? ' active' : '';
      html += '<a class="lnk' + active + '" href="' + prefix + href + '">' + label + '</a>';
    }
    nav.innerHTML = html;
  }
  apply();  // the script is included after <nav>, so this rewrites it with no flash…
  // …and again once the DOM is fully parsed, in case a page includes nav.js from <head>.
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply);
})();
