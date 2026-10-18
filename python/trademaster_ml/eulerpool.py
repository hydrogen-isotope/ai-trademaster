"""
Eulerpool client for SGX (Singapore) equities. Research only, read-only market data.

The API key lives in private/config/eulerpool.json  {"api_key": "..."}  (the user's own key; never printed or committed),
sent only to api.eulerpool.com as a bearer token.

  coverage   what the key can do for SGX before building anything: Singapore equities in the screener, and daily
             history depth for DBS, Singtel, OCBC, Singapore Airlines -> private/reports/sgx/eulerpool_coverage.json

Usage: python -m trademaster_ml.eulerpool coverage
"""
import argparse, json, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

BASE = "https://api.eulerpool.com/api/1"
CONF = os.path.join("private", "config", "eulerpool.json")
OUT = os.path.join("private", "reports", "sgx")
TEST = {"DBS": "DBS Group", "Singtel": "Singapore Telecommunications", "OCBC": "Oversea-Chinese Banking", "SIA": "Singapore Airlines"}


def key():
    if not os.path.exists(CONF):
        raise SystemExit(f"missing {CONF}")
    k = json.load(open(CONF)).get("api_key", "")
    if not k:
        raise SystemExit(f"{CONF} has no api_key")
    return k


def call(method, path, params=None, body=None):
    """Returns (status, parsed JSON or text). Never logs the key."""
    url = f"{BASE}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key()}", "Accept": "application/json", "Content-Type": "application/json",
        "User-Agent": "AI-Trademaster research"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode()
                return r.status, (json.loads(txt) if txt.strip()[:1] in "[{" else txt)
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            if e.code == 429 or e.code >= 500:
                time.sleep(5 + 10 * a)
                continue
            return e.code, txt[:400]
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError) as e:
            last = str(e)
            time.sleep(5 + 5 * a)
    return 0, f"unreachable: {last if 'last' in dir() else ''}"


def coverage(args):
    os.makedirs(OUT, exist_ok=True)
    rep = {"checked": time.strftime("%Y-%m-%d %H:%M")}
    st, meta = call("GET", "screener/metadata")
    rep["metadata_status"] = st
    countries = []
    if st == 200:
        blob = json.dumps(meta)
        countries = sorted({c for c in ("SG", "SGP", "Singapore") if f'"{c}"' in blob})
    print(f"screener metadata: HTTP {st}; Singapore codes seen: {countries}", flush=True)
    sg = []
    for code in countries or ["SG", "Singapore"]:
        page = 0
        while page < 20:
            # note: adding an "order" field makes the screener return nothing, so results are unsorted
            st, rows = call("POST", "screener/screen", body={"filters": [{"id": "country", "value": [code], "equality": "IN"}],
                                                           "page": page, "size": 100})
            if st != 200 or not isinstance(rows, list) or not rows:
                if page == 0:
                    print(f"  screen country={code}: HTTP {st} {str(rows)[:200]}", flush=True)
                break
            sg += rows
            page += 1
            if len(rows) < 100:
                break
        if sg:
            break
    rep["singapore_equities"] = len(sg)
    rep["sgx_tickers"] = sum(str(r.get("ticker", "")).endswith(".SI") for r in sg)
    rep["sample"] = [{k: r.get(k) for k in ("ticker", "isin", "name", "marketCap")} for r in sg[:10]]
    print(f"Singapore equities in the screener: {len(sg)}", flush=True)
    for r in sg[:10]:
        print(f"  {r.get('ticker')}  {r.get('isin')}  {str(r.get('name'))[:40]}", flush=True)
    rep["history"] = {}
    for label, needle in TEST.items():
        hit = next((r for r in sg if needle.lower() in str(r.get("name", "")).lower()), None)
        ids = [x for x in ([hit.get("isin"), hit.get("ticker")] if hit else []) if x]
        res = {"ids_tried": ids}
        for ident in ids:
            st, c = call("GET", f"charting/ohlcv/{urllib.parse.quote(ident)}", {"resolution": "D", "from": 946684800,
                                                                                 "to": int(time.time())})
            if st == 200 and isinstance(c, dict) and c.get("t"):
                t = c["t"]
                day = lambda s: datetime.fromtimestamp(s / 1000 if s > 1e11 else s, timezone.utc).strftime("%Y-%m-%d")
                res.update({"id": ident, "first": day(t[0]), "last": day(t[-1]), "days": len(t), "last_close": c["c"][-1]})
                break
            res[f"error_{ident}"] = f"HTTP {st} {str(c)[:150]}"
        rep["history"][label] = res
        print(f"  {label}: " + (f"{res['id']}: {res['first']} .. {res['last']} ({res['days']} days), last close {res['last_close']}"
                                 if "first" in res else json.dumps(res)[:300]), flush=True)
    json.dump(rep, open(os.path.join(OUT, "eulerpool_coverage.json"), "w"), indent=1)
    print(f"wrote {OUT}/eulerpool_coverage.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage"])
    a = ap.parse_args()
    {"coverage": coverage}[a.cmd](a)


if __name__ == "__main__":
    main()
