"""
Download ~30 days of 1-minute bars (Yahoo keeps 1-minute history for about 30 days, served in 7-day chunks).
Unofficial source, personal research only. Saves compact {ts, close, volume} per symbol, merged with what is already on disk.

Usage: python -m trademaster_ml.fetch1m --out private/data/hist_1m_30d --days 29
"""
import argparse, json, os, sys, time, urllib.request

from .fetch import STOCKS, ETFS, SECTOR_ETF


def chunk(sym, p1, p2):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1m&period1={p1}&period2={p2}"
           f"&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    return res.get("timestamp") or [], q.get("close") or [], q.get("volume") or []


def fetch_all(out="private/data/hist_1m_30d", days=29, log=print):
    """Pull the last `days` of 1-minute bars and MERGE them into existing files, so history keeps growing
    beyond Yahoo's ~30-day window. Returns (symbols saved, errors)."""
    if not os.path.abspath(out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    os.makedirs(out, exist_ok=True)
    end = int(time.time())
    start = end - days * 86400
    saved, errors = [], []
    for s in STOCKS + ETFS:
        path = os.path.join(out, f"{s}.json")
        rows = {}
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            rows = {t: (c, v) for t, c, v in zip(old["ts"], old["close"], old["volume"])}
        n0 = len(rows)
        try:
            p1 = start
            while p1 < end:
                p2 = min(p1 + 7 * 86400, end)
                for t, c, v in zip(*chunk(s, p1, p2)):
                    if c is not None:
                        rows[t] = (c, v or 0)
                p1 = p2
                time.sleep(0.25)
        except Exception as e:                       # keep what we had for this symbol
            errors.append(f"{s}: {e}")
        if rows:
            ts = sorted(rows)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"ts": ts, "close": [rows[t][0] for t in ts], "volume": [rows[t][1] for t in ts]}, f)
            os.replace(tmp, path)
            saved.append(s)
            log(f"{s:5s} {len(ts):6d} bars (+{len(ts) - n0} new)")
    with open(os.path.join(out, "universe.json"), "w", encoding="utf-8") as f:
        json.dump({"stocks": [s for s in STOCKS if s in saved], "etfs": [e for e in ETFS if e in saved],
                   "reference_etf": {s: e for s, e in SECTOR_ETF.items() if s in saved}, "interval": "1m",
                   "source": "Yahoo Finance chart API (unofficial, personal research)"}, f, indent=1)
    log(f"{len(saved)}/{len(STOCKS) + len(ETFS)} symbols in {out}" + (f", {len(errors)} fetch errors" if errors else ""))
    return saved, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/data/hist_1m_30d")
    ap.add_argument("--days", type=int, default=29)
    a = ap.parse_args()
    fetch_all(a.out, a.days)


if __name__ == "__main__":
    main()
