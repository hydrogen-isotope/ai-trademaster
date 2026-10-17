"""
Download 1-minute bars from the Alpaca Market Data API (all-exchange SIP bars; free plan: history up to 15 minutes
ago) and MERGE them into the per-symbol files already on disk, so the history keeps growing. Regular session only.
Saves compact {ts, close, volume} per symbol (bar start times), plus universe.json.

Usage: python -m trademaster_ml.fetch1m --out private/data/hist_1m_30d --days 29
"""
import argparse, json, os, time

from . import alpaca
from .fetch import STOCKS, ETFS, SECTOR_ETF


def fetch_all(out="private/data/hist_1m_30d", days=29, log=print):
    """Pull the last `days` of 1-minute bars and merge them into existing files. Returns (symbols saved, errors)."""
    if not os.path.abspath(out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    os.makedirs(out, exist_ok=True)
    end = int(time.time()) - 16 * 60
    start = end - days * 86400
    symbols = STOCKS + ETFS
    errors = []
    try:
        fresh = alpaca.get_bars(symbols, "1Min", start, end, feed="sip")
    except alpaca.AlpacaConfigError:
        raise
    except Exception as e:                               # keep what is already on disk
        fresh, errors = {}, [f"download failed: {e}"]
    saved = []
    for s in symbols:
        path = os.path.join(out, f"{s}.json")
        rows = {}
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            rows = {t: (c, v) for t, c, v in zip(old["ts"], old["close"], old["volume"])}
        n0 = len(rows)
        for t, o, h, l, c, v in alpaca.regular_session(fresh.get(s, [])):
            rows[t] = (c, v)
        if not rows:
            errors.append(f"{s}: no bars")
            continue
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
                   "source": "Alpaca Market Data API v2 (SIP, split-adjusted); older rows may come from earlier downloads"}, f, indent=1)
    log(f"{len(saved)}/{len(symbols)} symbols in {out}" + (f", {len(errors)} problems" if errors else ""))
    return saved, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/data/hist_1m_30d")
    ap.add_argument("--days", type=int, default=29)
    a = ap.parse_args()
    try:
        fetch_all(a.out, a.days)
    except alpaca.AlpacaConfigError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
