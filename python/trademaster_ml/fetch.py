"""
Download intraday bar history from the Alpaca Market Data API (official API, all-exchange SIP bars; free plan:
history up to 15 minutes ago). Regular session only. Writes {symbol}-{interval}.json in the chart format that
dataset.build_panel() reads, plus universe.json.

Usage: python -m trademaster_ml.fetch --out private/data/hist --interval 5m --range 60d
       intervals: 1m, 5m, 1h (session-aligned hours 09:30, 10:30, ...); range: Nd or Ny (e.g. 60d, 3y)
"""
import argparse, json, os, sys, time

from . import alpaca

SEMIS = ["MU", "NVDA", "AMD", "AVGO", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ADI", "NXPI", "ON",
         "MCHP", "ASML", "ARM", "SMCI"]
OTHERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "ADBE", "CSCO", "COST", "PEP", "AMGN", "HON", "INTU",
          "ISRG", "BKNG", "SBUX", "GILD", "MDLZ", "ADP", "VRTX", "REGN", "PANW", "SNPS", "CDNS", "MELI", "PYPL", "MAR",
          "ORLY", "CTAS", "MNST", "CSX", "ABNB", "FTNT", "ADSK", "CHTR", "PCAR", "ROST", "PAYX", "KDP", "AEP", "ODFL",
          "CPRT", "KHC", "IDXX", "FAST", "EXC", "DXCM", "BIIB", "CTSH", "EA", "VRSK", "XEL", "DDOG", "TEAM", "ZS",
          "CRWD", "WDAY", "TTD", "LULU", "PDD", "PLTR", "APP", "CEG", "COIN", "MSTR", "WBD", "TMUS", "CMCSA", "AXON"]
STOCKS = SEMIS + OTHERS
ETFS = ["SMH", "SOXX", "QQQ", "SPY", "XLK"]
# Public sector label per stock -> reference ETF (no holdings used).
SECTOR_ETF = {s: "SMH" for s in SEMIS} | {s: "QQQ" for s in OTHERS}


def range_seconds(rng):
    n, unit = int(rng[:-1]), rng[-1].lower()
    return n * {"d": 86400, "y": 365 * 86400}[unit]


def download(symbols, interval, rng, log=print):
    """{symbol: [(t, o, h, l, c, v)]} of regular-session bars."""
    end = int(time.time()) - 16 * 60                      # free plan: SIP history ends 15 minutes ago
    start = end - range_seconds(rng)
    frame = {"1m": "1Min", "5m": "5Min", "1h": "30Min"}[interval]
    raw = alpaca.get_bars(symbols, frame, start, end, feed="sip", log=log)
    if interval == "1h":
        return {s: alpaca.session_hours(rows) for s, rows in raw.items()}
    return {s: alpaca.regular_session(rows) for s, rows in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/data/hist")
    ap.add_argument("--interval", default="5m", choices=["1m", "5m", "1h"])
    ap.add_argument("--range", default="60d")
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    try:
        alpaca.credentials()
    except alpaca.AlpacaConfigError as e:
        raise SystemExit(str(e))
    os.makedirs(a.out, exist_ok=True)
    data = download(STOCKS + ETFS, a.interval, a.range)
    saved = []
    for s in STOCKS + ETFS:
        rows = data.get(s) or []
        if not rows:
            print(f"{s:5s} FAILED no bars", file=sys.stderr)
            continue
        with open(os.path.join(a.out, f"{s}-{a.interval}.json"), "w", encoding="utf-8") as f:
            json.dump(alpaca.chart_json(rows), f)
        print(f"{s:5s} {len(rows):6d} bars")
        saved.append(s)
    with open(os.path.join(a.out, "universe.json"), "w", encoding="utf-8") as f:
        json.dump({"stocks": [s for s in STOCKS if s in saved], "etfs": [e for e in ETFS if e in saved],
                   "reference_etf": {s: e for s, e in SECTOR_ETF.items() if s in saved}, "interval": a.interval,
                   "source": "Alpaca Market Data API v2 (SIP, split-adjusted)"}, f, indent=1)
    print(f"{len(saved)}/{len(STOCKS) + len(ETFS)} symbols saved to {a.out}")


if __name__ == "__main__":
    main()
