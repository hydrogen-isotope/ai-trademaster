"""
EODHD client for SGX (Singapore Exchange) end-of-day data. Research only, read-only market data.

The API key lives in private/config/eodhd.json  {"api_token": "..."}  (created by the user; never printed or committed).

  coverage   SGX coverage report before paying for a plan: number of active and delisted SGX symbols, and history depth
             and split adjustment for a few test tickers -> private/reports/sgx/coverage.json
             (the free tier allows ~20 calls/day and about 1 year of history, so depth beyond that needs the paid plan)

Usage: python -m trademaster_ml.eodhd coverage
"""
import argparse, json, os, time, urllib.error, urllib.parse, urllib.request

BASE = "https://eodhd.com/api"
CONF = os.path.join("private", "config", "eodhd.json")
OUT = os.path.join("private", "reports", "sgx")
TEST = ["D05.SG", "Z74.SG", "ES3.SG", "O39.SG", "C6L.SG"]       # DBS, Singtel, STI ETF, OCBC, Singapore Airlines


def token():
    if not os.path.exists(CONF):
        raise SystemExit(f"missing {CONF}: create it with {{\"api_token\": \"<your EODHD key>\"}}")
    t = json.load(open(CONF)).get("api_token", "")
    if not t or t.startswith("PASTE"):
        raise SystemExit(f"{CONF} has no api_token yet")
    return t


def get(path, params=None, retries=4):
    """GET an EODHD endpoint as JSON. The key is sent as a query parameter (EODHD's documented method) and never logged."""
    q = {"api_token": token(), "fmt": "json", **(params or {})}
    url = f"{BASE}/{path}?{urllib.parse.urlencode(q)}"
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AI-Trademaster research"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"EODHD refused {path} (HTTP {e.code}): this endpoint or market may need a paid plan") from None
            if e.code == 429 or e.code >= 500:
                time.sleep(5 + 10 * a)
                continue
            raise SystemExit(f"EODHD error on {path}: HTTP {e.code}") from None
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError):
            time.sleep(5 + 10 * a)
    raise SystemExit(f"EODHD unreachable for {path}")


def coverage(args):
    os.makedirs(OUT, exist_ok=True)
    report = {"checked": time.strftime("%Y-%m-%d %H:%M")}
    active = get("exchange-symbol-list/SG")
    delisted = get("exchange-symbol-list/SG", {"delisted": 1})
    kinds = {}
    for s in active:
        kinds[s.get("Type", "?")] = kinds.get(s.get("Type", "?"), 0) + 1
    report["symbols"] = {"active": len(active), "delisted": len(delisted) if isinstance(delisted, list) else None, "active_by_type": kinds}
    report["tickers"] = {}
    for t in TEST:
        rows = get(f"eod/{t}", {"from": "2000-01-01"})
        if isinstance(rows, list) and rows:
            report["tickers"][t] = {"first": rows[0]["date"], "last": rows[-1]["date"], "days": len(rows),
                                    "close_last": rows[-1]["close"], "adjusted_last": rows[-1].get("adjusted_close")}
        else:
            report["tickers"][t] = {"error": str(rows)[:200]}
    json.dump(report, open(os.path.join(OUT, "coverage.json"), "w"), indent=1)
    print(f"SGX symbols: {report['symbols']['active']} active ({kinds}), {report['symbols']['delisted']} delisted", flush=True)
    for t, x in report["tickers"].items():
        print(f"  {t}: " + (f"{x['first']} .. {x['last']} ({x['days']} days), last close {x['close_last']} / adjusted {x['adjusted_last']}"
                            if "first" in x else x["error"]), flush=True)
    print(f"wrote {OUT}/coverage.json (calls used: {2 + len(TEST)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage"])
    a = ap.parse_args()
    {"coverage": coverage}[a.cmd](a)


if __name__ == "__main__":
    main()
