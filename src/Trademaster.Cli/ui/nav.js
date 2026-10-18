// Shared left sidebar (the same four sections as the Watchlist page's rail), so every page has a way back.
// On a phone-width window it becomes a bar along the bottom.
(() => {
  const icon = {
    watch: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    signals: '<path d="M4 18l5-6 4 3 7-9"/><path d="M15 6h5v5"/>',
    markets: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    sim: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  };
  const items = [['/', 'Watchlist', 'watch', 'Watchlist and charts'], ['/forward', 'AI Signals', 'signals', 'AI buy/sell indicators and forward test'],
    ['/markets', 'Markets', 'markets', 'NASDAQ and SGX rankings, 5-day forecasts and AI analysis'], ['/sim', 'Simulation', 'sim', 'Engine simulation terminal']];
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
