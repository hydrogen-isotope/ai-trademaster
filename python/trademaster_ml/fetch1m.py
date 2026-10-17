"""
Download ~30 days of 1-minute bars (Yahoo keeps 1-minute history for about 30 days, served in 7-day chunks).
Unofficial source, personal research only. Saves compact {ts, close, volume} per symbol.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/data/hist_1m_30d")
    ap.add_argument("--days", type=int, default=29)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    os.makedirs(a.out, exist_ok=True)
    end = int(time.time())
    start = end - a.days * 86400
    saved = []
    for s in STOCKS + ETFS:
        rows = {}
        try:
            p1 = start
            while p1 < end:
                p2 = min(p1 + 7 * 86400, end)
                for t, c, v in zip(*chunk(s, p1, p2)):
                    if c is not None:
                        rows[t] = (c, v or 0)
                p1 = p2
                time.sleep(0.25)
            ts = sorted(rows)
            if not ts:
                raise ValueError("no bars")
            with open(os.path.join(a.out, f"{s}.json"), "w", encoding="utf-8") as f:
                json.dump({"ts": ts, "close": [rows[t][0] for t in ts], "volume": [rows[t][1] for t in ts]}, f)
            saved.append(s)
            print(f"{s:5s} {len(ts):6d} bars")
        except Exception as e:
            print(f"{s:5s} FAILED {e}", file=sys.stderr)
    with open(os.path.join(a.out, "universe.json"), "w", encoding="utf-8") as f:
        json.dump({"stocks": [s for s in STOCKS if s in saved], "etfs": [e for e in ETFS if e in saved],
                   "reference_etf": {s: e for s, e in SECTOR_ETF.items() if s in saved}, "interval": "1m",
                   "source": "Yahoo Finance chart API (unofficial, personal research)"}, f, indent=1)
    print(f"{len(saved)}/{len(STOCKS) + len(ETFS)} symbols saved to {a.out}")


if __name__ == "__main__":
    main()
