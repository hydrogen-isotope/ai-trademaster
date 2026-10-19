"""
NASDAQ indexes and sector groups for the Markets hub (research only; CPU; no orders).

The real Nasdaq Composite, Nasdaq-100 and Nasdaq futures are not in any data source we have (Alpaca stocks/ETFs,
Eulerpool equities), so the indexes are shown through ETFs that track them: ONEQ (Nasdaq Composite) and QQQ
(Nasdaq-100), labelled as stand-ins. Sector groups are built from our own NASDAQ universe (SEC industry codes, equal
weight), each next to a sector ETF for comparison.

5-day forecasts for the ETFs use the v7 market model: QQQ's drift and GJR-GARCH variance; another ETF takes its beta to
QQQ plus its own residual volatility. These are not recalibrated (no forward record yet) and are labelled so.

Writes private/reports/v6/indexes.json. Usage: python -m trademaster_ml.indexes
"""
import json, math, os, time
from datetime import datetime

import numpy as np

from . import alpaca, llm_news, stock_eval, v6

OUT = os.path.join("private", "reports", "v6", "indexes.json")
DAYS = 260
INDEX_ETFS = {"ONEQ": "Nasdaq Composite", "QQQ": "Nasdaq-100", "SPY": "S&P 500"}
# key, label, SIC 2-digit codes, sector ETF shown alongside
SECTORS = [
    ("semis", "Semiconductors & electronics", {36}, "SMH"),
    ("software", "Software & IT services", {73}, "IGV"),
    ("hardware", "Computers & machinery", {35}, "XLK"),
    ("biotech", "Biotech & pharma", {28}, "IBB"),
    ("medical", "Medical devices & health services", {38, 80}, "XLV"),
    ("financials", "Banks & financials", {60, 61, 62, 63, 64, 67}, "XLF"),
    ("realestate", "Real estate", {65}, "XLRE"),
    ("comms", "Communications & media", {27, 48, 78}, "XLC"),
    ("consumer", "Consumer & retail", {20, 21, 22, 23, 39, 52, 53, 54, 55, 56, 57, 58, 59, 70, 72, 79, 82, 83}, "XLY"),
    ("industrials", "Industrials & transport", {15, 16, 17, 24, 25, 30, 32, 34, 37, 40, 42, 44, 45, 47, 50, 51, 87}, "XLI"),
    ("energy", "Energy & materials", {10, 12, 13, 14, 26, 29, 33}, "XLE"),
    ("utilities", "Utilities", {49}, "XLU"),
]
SECTOR_OF = {c: k for k, _, codes, _ in SECTORS for c in codes}


def stats(dates, closes):
    """Simple returns over the usual look-backs, ending at the last close."""
    c = np.asarray(closes, float)
    back = lambda k: float(c[-1] / c[-1 - k] - 1) if len(c) > k and c[-1 - k] > 0 else None
    y = dates[-1][:4]
    first = next((i for i, d in enumerate(dates) if d[:4] == y), None)
    ytd = float(c[-1] / c[first - 1] - 1) if first and first > 0 else None
    return {"d1": back(1), "w1": back(5), "m1": back(21), "m3": back(63), "ytd": ytd, "y1": back(252)}


def forecast(close, asof, mu, var, beta=1.0, idio=0.0, h=5):
    """h-day Student-t forecast from the market model (uncalibrated): mean beta * drift, var beta^2 V + h idio^2."""
    m, sd = beta * mu, math.sqrt(beta * beta * var + h * idio * idio)
    R = stock_eval.v7.GRID.copy()
    return {**stock_eval.dist(close, m, sd, R), "cone": stock_eval.cone(close, asof, m, sd, R, days=h), "beta": beta, "h": h,
            "note": "market model (QQQ drift + GJR-GARCH), not recalibrated"}


def market_models(asof):
    """{h: {mu5, v5}} for each horizon whose v7 forecast is for `asof` (keys keep the 5-day names; values are h-day)."""
    out = {}
    for h in (1, 5, 20):
        fp = os.path.join(v6.V7_OUT, "forecast.json") if h == 5 else os.path.join(v6.V7_OUT, f"h{h}", "forecast.json")
        gp = os.path.join(v6.V7_OUT, f"h{h}", "gate.json")
        if h != 5 and not (os.path.exists(gp) and json.load(open(gp)).get("pass")):
            continue                                                     # v7 amendment A: a horizon that failed its gate is not shown
        if os.path.exists(fp):
            fc = json.load(open(fp))
            if fc.get("asof") == asof and fc.get("market_model"):
                out[h] = fc["market_model"]
    return out


def update(log=print):
    t0 = time.time()
    v6.use_market("nasdaq")
    d = v6.GPU_DIR
    meta = json.load(open(os.path.join(d, "meta.json")))
    T, N, dates, syms = meta["T"], meta["N"], meta["dates"], meta["symbols"]
    asof = dates[-1]
    # ---- ETFs (completed daily bars up to the panel's last close only)
    etfs = sorted(set(INDEX_ETFS) | {e for *_, e in SECTORS})
    start = time.mktime(datetime.fromisoformat(dates[-DAYS - 10]).timetuple())
    creds = alpaca.credentials()
    bars = alpaca.get_bars(etfs, "1Day", start, feed=creds[2], creds=creds)
    ser = {}
    for s, rows in bars.items():
        pts = [(datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"), c) for t, o, h, l, c, v in rows]
        pts = [(dd, c) for dd, c in pts if dd <= asof]
        if len(pts) > 30:
            ser[s] = pts[-DAYS:]
    mms = market_models(asof)
    mm = mms.get(5)
    q = ser.get("QQQ")
    rq = {dd: c for dd, c in q} if q else {}
    out_etf = {}
    for s, pts in ser.items():
        dd, cc = [p[0] for p in pts], [p[1] for p in pts]
        f, fh = None, {}
        if mms and q and pts[-1][0] == asof:
            common = [x for x in dd if x in rq]
            a = np.diff(np.log([dict(pts)[x] for x in common]))
            b = np.diff(np.log([rq[x] for x in common]))
            beta = float(np.dot(a, b) / np.dot(b, b)) if s != "QQQ" and len(b) > 60 else 1.0
            idio = float(np.std(a - beta * b)) if s != "QQQ" else 0.0
            fh = {str(h): forecast(cc[-1], asof, x["mu5"], x["v5"], beta, idio, h) for h, x in mms.items()}
            f = fh.get("5")
        out_etf[s] = {"label": INDEX_ETFS.get(s) or next((lab for _, lab, _, e in SECTORS if e == s), s), "close": cc[-1],
                      "last": dd[-1], "series": [[x, round(y, 4)] for x, y in pts], "stats": stats(dd, cc), "forecast": f, "forecasts": fh}
    # ---- sector groups from our universe (equal weight, members = that day's universe)
    lc = np.fromfile(os.path.join(d, "logclose.f32"), np.float32).reshape(T, N)
    mask = np.fromfile(os.path.join(d, "mask.f32"), np.float32).reshape(T, N) > 0.5
    ind = np.fromfile(os.path.join(d, "industry.f32"), np.float32)
    sec = np.array([SECTOR_OF.get(int(c), "other") if np.isfinite(c) and c > 0 else "other" for c in ind])
    rk = json.load(open(os.path.join(v6.OUT, "rankings.json")))
    rows = {r["symbol"]: r for r in rk["rankings"]} if rk.get("asof") == asof else {}
    news = json.load(open(llm_news.TODAY)) if os.path.exists(llm_news.TODAY) else None
    news = news["stocks"] if news and news.get("asof") == asof else {}
    ip = os.path.join(stock_eval.EVALS, "nasdaq", f"index_{asof}.json")
    ai = json.load(open(ip)) if os.path.exists(ip) else {}
    t_from = T - DAYS
    r = np.clip(np.diff(lc[t_from - 1:].astype(np.float64), axis=0), -0.5, 0.5)       # log returns, days t_from..T-1
    prev_in = mask[t_from - 1:T - 1]                                                       # universe at the previous close
    members = {}
    out_sec = []
    for key, label, _, etf in SECTORS + [("other", "Other / unclassified", set(), None)]:
        cols = sec == key
        ok = prev_in & cols[None, :] & np.isfinite(r)
        n_day = ok.sum(1)
        eq = np.where(n_day > 0, np.where(ok, np.expm1(r), 0).sum(1) / np.maximum(n_day, 1), 0.0)
        level = 100 * np.cumprod(1 + eq)
        today = [syms[j] for j in np.nonzero(mask[T - 1] & cols)[0]]
        mem = [rows[s] for s in today if s in rows]
        for s in today:
            members[s] = key
        last_ok = ok[-1]
        pu = [m["fc"]["p_up"] for m in mem if m.get("fc")]
        nd = [news[s]["news_dir"] for s in today if s in news]
        sig = [ai[s]["signal"] for s in today if s in ai and ai[s].get("signal")]
        dd = dates[t_from:]
        out_sec.append({"key": key, "label": label, "etf": etf, "n": len(today),
                        "series": [[x, round(float(y), 3)] for x, y in zip(dd, level)], "stats": stats(dd, level),
                        "breadth": float(np.mean(r[-1][last_ok] > 0)) if last_ok.any() else None,
                        "avg_p_up": float(np.mean(pu)) if pu else None, "avg_score": float(np.mean([m["score_pct"] for m in mem])) if mem else None,
                        "news": {"mean": float(np.mean(nd)), "n_stocks": len(nd)} if nd else None,
                        "signals": {"buy": sig.count("buy"), "sell": sig.count("sell"), "evaluated": sum(1 for s in today if s in ai)},
                        "top": [m["symbol"] for m in sorted(mem, key=lambda m: -m["score_pct"])[:3]],
                        "bottom": [m["symbol"] for m in sorted(mem, key=lambda m: m["score_pct"])[:3]]})
    out_sec = [x for x in out_sec if x["n"] > 0]
    out = {"asof": asof, "generated": datetime.now().isoformat(timespec="seconds"), "etfs": out_etf, "sectors": out_sec,
           "members": members, "market_model": mm, "market_models": {str(h): x for h, x in mms.items()},
           "note": "ONEQ and QQQ are ETFs that track the Nasdaq Composite and Nasdaq-100 (the indexes themselves and Nasdaq "
                   "futures are not in our data sources). Sectors: our NASDAQ universe grouped by SEC industry code, equal weight. "
                   "Sector ETFs cover S&P or broad US stocks, shown for comparison."}
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, OUT)
    log(f"indexes {asof}: {len(out_etf)} ETFs, {len(out_sec)} sectors, {len(members)} stocks; {time.time() - t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    update()
