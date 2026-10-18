"""
Twelve Data client for SGX (Singapore Exchange, MIC XSES) end-of-day data. Research only, read-only market data.

The API key lives in private/config/twelvedata.json  {"apikey": "..."}  (created by the user; never printed or committed).

  coverage   what the current key can do for SGX before paying for anything: plan and credits, the SGX symbol list,
             and daily history (earliest date, number of days, split check) for a few test tickers
             -> private/reports/sgx/twelvedata_coverage.json

Usage: python -m trademaster_ml.twelvedata coverage
"""
import argparse, json, os, time, urllib.error, urllib.parse, urllib.request

BASE = "https://api.twelvedata.com"
CONF = os.path.join("private", "config", "twelvedata.json")
OUT = os.path.join("private", "reports", "sgx")
TEST = ["D05", "Z74", "ES3", "O39", "C6L"]       # DBS, Singtel, STI ETF, OCBC, Singapore Airlines


def apikey():
    if not os.path.exists(CONF):
        raise SystemExit(f"missing {CONF}: create it with {{\"apikey\": \"<your Twelve Data key>\"}}")
    k = json.load(open(CONF)).get("apikey", "")
    if not k or k.startswith("PASTE"):
        raise SystemExit(f"{CONF} has no apikey yet")
    return k


def get(path, params=None):
    """GET a Twelve Data endpoint. Waits politely between calls (free tier: 8 per minute). The key is never logged."""
    q = {**(params or {}), "apikey": apikey()}
    url = f"{BASE}/{path}?{urllib.parse.urlencode(q)}"
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AI-Trademaster research"}), timeout=60) as r:
                d = json.load(r)
            time.sleep(8)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(15 + 15 * a)
                continue
            return {"status": "error", "code": e.code, "message": f"HTTP {e.code}"}
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError):
            time.sleep(10 + 10 * a)
    return {"status": "error", "message": "unreachable"}


def coverage(args):
    os.makedirs(OUT, exist_ok=True)
    rep = {"checked": time.strftime("%Y-%m-%d %H:%M")}
    u = get("api_usage")
    rep["plan"] = {k: u.get(k) for k in ("plan_category", "plan_limit", "plan_daily_limit", "current_usage", "daily_usage") if k in u} or u
    print(f"plan: {rep['plan']}", flush=True)
    st = get("stocks", {"mic_code": "XSES"})
    rep["symbols"] = len(st.get("data", [])) if isinstance(st, dict) else None
    print(f"SGX symbols visible: {rep['symbols']}", flush=True)
    rep["tickers"] = {}
    for t in TEST:
        e = get("earliest_timestamp", {"symbol": t, "mic_code": "XSES", "interval": "1day"})
        ts = get("time_series", {"symbol": t, "mic_code": "XSES", "interval": "1day", "outputsize": 5000, "order": "asc"})
        if ts.get("status") == "ok":
            v = ts["values"]
            rep["tickers"][t] = {"earliest": e.get("datetime"), "first_returned": v[0]["datetime"], "last": v[-1]["datetime"],
                                 "days": len(v), "currency": ts.get("meta", {}).get("currency"), "last_close": v[-1]["close"]}
            print(f"  {t}: earliest {e.get('datetime')}, returned {v[0]['datetime']} .. {v[-1]['datetime']} ({len(v)} days), "
                  f"last close {v[-1]['close']} {ts.get('meta', {}).get('currency')}", flush=True)
        else:
            rep["tickers"][t] = {"error": ts.get("message", str(ts))[:300], "code": ts.get("code")}
            print(f"  {t}: {rep['tickers'][t]['error']}", flush=True)
    json.dump(rep, open(os.path.join(OUT, "twelvedata_coverage.json"), "w"), indent=1)
    print(f"wrote {OUT}/twelvedata_coverage.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage"])
    a = ap.parse_args()
    {"coverage": coverage}[a.cmd](a)


if __name__ == "__main__":
    main()
