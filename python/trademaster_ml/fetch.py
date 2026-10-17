"""
Download intraday bar history from Yahoo Finance's public chart endpoint (unofficial; personal research only,
not a licensed feed). Regular session only.

Usage: python -m trademaster_ml.fetch --out data/hist --interval 5m --range 60d
"""
import argparse, json, os, sys, time, urllib.request

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


def fetch(sym, interval, rng):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={rng}&includePrePost=false"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="private/data/hist")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--range", default="60d")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok, saved = 0, []
    for s in STOCKS + ETFS:
        try:
            j = fetch(s, a.interval, a.range)
            res = j["chart"]["result"][0]
            n = len(res.get("timestamp") or [])
            if n == 0:
                raise ValueError("no bars")
            with open(os.path.join(a.out, f"{s}-{a.interval}.json"), "w", encoding="utf-8") as f:
                json.dump(j, f)
            print(f"{s:5s} {n:6d} bars")
            ok += 1
            saved.append(s)
        except Exception as e:  # keep going; report at the end
            print(f"{s:5s} FAILED {e}", file=sys.stderr)
        time.sleep(0.3)
    with open(os.path.join(a.out, "universe.json"), "w", encoding="utf-8") as f:
        json.dump({"stocks": [s for s in STOCKS if s in saved], "etfs": [e for e in ETFS if e in saved],
                   "reference_etf": {s: e for s, e in SECTOR_ETF.items() if s in saved}, "interval": a.interval,
                   "source": "Yahoo Finance chart API (unofficial, personal research)"}, f, indent=1)
    print(f"{ok}/{len(STOCKS) + len(ETFS)} symbols saved to {a.out}")


if __name__ == "__main__":
    main()
