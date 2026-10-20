// Shared left sidebar (the same four sections as the Watchlist page's rail), so every page has a way back.
// On a phone-width window it becomes a bar along the bottom. Also the shared search matcher (window.tmSearch).
(() => {
  fetch('/api/refresh?ifNew=1', { method: 'POST' }).catch(() => { });  // pull a newly finished close (any page, at most every 15 min)
  fetch('/api/stock/queue-all', { method: 'POST' }).catch(() => { });   // evaluate every stock in the background while the app is open
  if (document.getElementById('rail')) return;                  // the Watchlist page has its own rail
  const icon = {
    watch: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    signals: '<path d="M4 18l5-6 4 3 7-9"/><path d="M15 6h5v5"/>',
    markets: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    sim: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
    auto: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/><path d="M8.5 3.8L7 2M15.5 3.8L17 2"/>',
  };
  const items = [['/', 'Watchlist', 'watch', 'Watchlist and charts'], ['/forward', 'AI Signals', 'signals', 'AI buy/sell indicators and forward test'],
    ['/markets', 'Markets', 'markets', 'NASDAQ and SGX rankings, 5-day forecasts and AI analysis'], ['/autotrade', 'Auto-trade', 'auto', 'Automatic trading on the Alpaca paper account (simulated money)'],
    ['/sim', 'Simulation', 'sim', 'Engine simulation terminal']];
  const here = location.pathname.replace(/\/$/, '') || '/';
  const css = document.createElement('style');
  css.textContent = `#tmrail{position:fixed;left:0;top:0;bottom:0;width:60px;background:#121317;border-right:1px solid #24272e;display:flex;flex-direction:column;align-items:center;padding-top:8px;gap:4px;z-index:50}
#tmrail a{width:52px;padding:8px 0 6px;border-radius:8px;color:#8a8f99;text-decoration:none;display:flex;flex-direction:column;align-items:center;gap:3px;font:10px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#tmrail a:hover{background:#20232a;color:#e9eaee;text-decoration:none}#tmrail a.on{color:#3b82f6}#tmrail svg{width:20px;height:20px}
body{padding-left:60px}
@media (max-width:760px){#tmrail{top:auto;right:0;width:auto;height:54px;flex-direction:row;justify-content:space-around;padding:0;border-right:0;border-top:1px solid #24272e}body{padding-left:0;padding-bottom:54px}}`;
  document.head.appendChild(css);
  const nav = document.createElement('nav');
  nav.id = 'tmrail';
  nav.innerHTML = items.map(([href, label, ic, title]) =>
    `<a href="${href}" title="${title}" class="${here === href ? 'on' : ''}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">${icon[ic]}</svg>${label}</a>`).join('');
  document.body.prepend(nav);
})();

// Search over /api/symbols: index stand-ins, sector groups and ranked stocks. Matches the symbol, the words people use
// for an index ("nasdaq composite", "ndx"), and company names by word start ("micr" -> Micron, "nasdaq comp").
window.tmSearch = (() => {
  let list = null;
  const norm = t => String(t || '').toLowerCase().replace(/&/g, ' and ').replace(/[^a-z0-9.^ -]/g, ' ').replace(/\s+/g, ' ').trim();
  const clean = n => String(n || '').replace(/\s+(Class [A-Z] )?(Common Stock|Ordinary Shares|American Depositary Shares|Common Shares|Units?)\b.*$/i, '');
  async function load() {
    if (list) return list;
    try { list = await (await fetch('/api/symbols')).json(); } catch (e) { list = []; }
    for (const x of list) { x.dn = clean(x.n); x.nw = norm(x.dn).split(' '); x.al = norm(x.a); }
    return list;
  }
  function score(x, q, qw) {
    const s = x.s.toLowerCase(), base = s.replace(/\.si$/, '');
    if (s === q || base === q || (x.al && (' ' + x.al + ' ').includes(' ' + q + ' '))) return 0;
    if (s.startsWith(q) || base.startsWith(q) || (x.al && (' ' + x.al).includes(' ' + q))) return 1;
    if (qw.every(w => x.nw.some(n => n.startsWith(w)))) return x.nw[0].startsWith(qw[0]) ? 2 : 2.5;   // name starts with it first
    if (norm(x.n).includes(q)) return 3;
    return 9;
  }
  const kindRank = { index: 0, sector: 1, stock: 2 };
  return {
    clean,
    async find(raw, n = 8) {
      const q = norm(raw); if (!q) return [];
      const qw = q.split(' '), L = await load(), out = [];
      for (const x of L) { const sc = score(x, q, qw); if (sc < 9) out.push([sc, x]); }
      out.sort((a, b) => a[0] - b[0] || kindRank[a[1].k] - kindRank[b[1].k] || a[1].s.localeCompare(b[1].s));
      return out.slice(0, n).map(o => o[1]);
    },
    href: x => x.k === 'index' ? `/markets#/nasdaq/index/${x.s}` : x.k === 'sector' ? `/markets#/nasdaq/sector/${x.s}` : `/markets#/${x.m}/stock/${encodeURIComponent(x.s)}`,
    chip: x => x.k === 'index' ? 'INDEX' : x.k === 'sector' ? 'SECTOR' : x.m === 'sgx' ? 'SGX' : 'NASDAQ',
  };
})();
