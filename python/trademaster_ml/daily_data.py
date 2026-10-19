"""
Daily closing prices for the 5-day forecaster, one file per instrument: private/data/daily/{SYMBOL}.json
{dates, close, source}.

  stocks   Alpaca daily bars (SIP, split-adjusted) when a key is configured, otherwise the regular-session close
           (16:00 bar) from the hourly history already on disk
  futures  the roll-adjusted settlement series built by databento_client (private/data/futures/{ROOT}.json)

Usage: python -m trademaster_ml.daily_data --stocks MU NVDA AMD AAPL --futures GC SI
"""
import argparse, json, os, time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import alpaca

NY = ZoneInfo("America/New_York")
OUT = os.path.join("private", "data", "daily")


def from_hourly(sym, hist="private/data/hist_1h"):
    r = json.load(open(os.path.join(hist, f"{sym}-1h.json"), encoding="utf-8"))["chart"]["result"][0]
    closes = {}
    for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"]):
        if c is None:
            continue
        d = datetime.fromtimestamp(t, NY)
        closes[d.date()] = c                       # bars are time-ordered: the last one of the day wins (16:00 print)
    days = sorted(closes)
    return {"dates": [str(d) for d in days], "close": [closes[d] for d in days],
            "source": f"{hist} (regular-session last print of each day)"}


def from_alpaca(sym, years=6):
    end = int(time.time()) - 16 * 60
    now_ny = datetime.now(NY)
    if now_ny.hour * 60 + now_ny.minute < 16 * 60 + 20:              # today's bar is still forming: leave it out (bars are
        end = min(end, int(now_ny.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) - 60)   # stamped at midnight NY)
    rows = alpaca.get_bars([sym], "1Day", end - years * 365 * 86400, end, feed="sip").get(sym, [])
    days = [datetime.fromtimestamp(t, NY).date() for t, *_ in rows]
    return {"dates": [str(d) for d in days], "close": [r[4] for r in rows], "source": "Alpaca 1Day bars (SIP, split-adjusted)"}


def from_futures(root, src="private/data/futures"):
    d = json.load(open(os.path.join(src, f"{root}.json"), encoding="utf-8"))
    return {"dates": d["dates"], "close": d["close"], "source": d["source"] + ", " + d["adjustment"],
            "contract": d["contract"], "settle": d["settle"]}


def save(sym, doc, out=OUT):
    os.makedirs(out, exist_ok=True)
    doc = doc | {"symbol": sym}
    json.dump(doc, open(os.path.join(out, f"{sym}.json"), "w"))
    return doc


def load(sym, out=OUT):
    return json.load(open(os.path.join(out, f"{sym}.json"), encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=["MU", "NVDA", "AMD", "AAPL"])
    ap.add_argument("--futures", nargs="*", default=["GC", "SI"])
    ap.add_argument("--offline", action="store_true", help="stocks from the hourly history on disk, even if a key exists")
    a = ap.parse_args()
    use_alpaca = not a.offline
    if use_alpaca:
        try:
            alpaca.credentials()
        except alpaca.AlpacaConfigError:
            use_alpaca = False
            print("no Alpaca key: stocks from the hourly history on disk")
    for s in a.stocks:
        doc = save(s, from_alpaca(s) if use_alpaca else from_hourly(s))
        print(f"{s:5s} {len(doc['dates'])} days {doc['dates'][0]}..{doc['dates'][-1]}  ({doc['source']})")
    for r in a.futures:
        p = os.path.join("private", "data", "futures", f"{r}.json")
        if not os.path.exists(p):
            print(f"{r:5s} no futures data yet: run python -m trademaster_ml.databento_client --download (needs a key)")
            continue
        doc = save(r, from_futures(r))
        print(f"{r:5s} {len(doc['dates'])} days {doc['dates'][0]}..{doc['dates'][-1]}  ({doc['source']})")


if __name__ == "__main__":
    main()
