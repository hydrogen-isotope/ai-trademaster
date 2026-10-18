"""
Options data for v5 (research only; read-only market data; never places orders).

  fetch     for each stock: list monthly option contracts (Alpaca trading API, active + expired, expiries from Feb 2024),
            keep strikes within +-30% of the stock price over each expiry's life, then download their daily bars
            (Alpaca market-data API) -> private/data/options/{SYMBOL}.json   (resumable, one file per stock)
  features  implied volatility (Black-Scholes), skew, put/call volume -> private/data/options/features.npz

Usage: python -m trademaster_ml.options fetch | features
"""
import argparse, json, math, os, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from . import alpaca

DIR = os.path.join("private", "data", "options")
TRADING = "https://paper-api.alpaca.markets"
STANDARD = re.compile(r"^[A-Z]{1,5}\d{6}[CP]\d{8}$")          # skips adjusted contracts like 1QCOM... after corporate actions
START = date(2024, 2, 1)


def _trading_get(path, params, creds, retries=5):
    kid, sec, _ = creds
    url = TRADING + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if a == retries - 1:
                raise RuntimeError(f"trading API failed: {e}") from None
            time.sleep(3 + 5 * a)


def is_monthly(d):
    """Third-Friday expiries (or the Thursday before when that Friday is a holiday)."""
    return (d.weekday() == 4 and 15 <= d.day <= 21) or (d.weekday() == 3 and 15 <= d.day + 1 <= 21)


def list_contracts(sym, creds, end):
    out, token = [], None
    for status in ("inactive", "active"):
        token = None
        while True:
            p = {"underlying_symbols": sym, "status": status, "expiration_date_gte": str(START), "expiration_date_lte": str(end), "limit": 10000}
            if token:
                p["page_token"] = token
            d = _trading_get("/v2/options/contracts", p, creds)
            out += d.get("option_contracts", [])
            token = d.get("next_page_token")
            time.sleep(0.35)
            if not token:
                break
    return out


def fetch_one(sym, closes, creds, log):
    path = os.path.join(DIR, f"{sym}.json")
    if os.path.exists(path):
        return 0
    end = date.today() + timedelta(days=70)
    cons = [c for c in list_contracts(sym, creds, end) if is_monthly(date.fromisoformat(c["expiration_date"]))]
    by_exp = {}
    for c in cons:
        by_exp.setdefault(c["expiration_date"], []).append(c)
    keep = []
    for exp, cs in by_exp.items():
        e = date.fromisoformat(exp)
        win = [v for d, v in closes.items() if e - timedelta(days=60) <= date.fromisoformat(d) <= e]
        if not win:
            continue
        lo, hi = 0.7 * min(win), 1.3 * max(win)
        keep += [c["symbol"] for c in cs if lo <= float(c["strike_price"]) <= hi and STANDARD.match(c["symbol"])]
    rows = []
    for i in range(0, len(keep), 100):
        group = keep[i:i + 100]
        exps = [date(2000 + int(s[-15:-13]), int(s[-13:-11]), int(s[-11:-9])) for s in group]
        start, stop = min(exps) - timedelta(days=62), max(exps)
        token = None
        while True:
            p = {"symbols": ",".join(group), "timeframe": "1Day", "start": str(max(start, START)), "end": str(min(stop, date.today() - timedelta(days=1))), "limit": 10000}
            if token:
                p["page_token"] = token
            d = alpaca._get("/v1beta1/options/bars", p, creds)
            for s, bars in (d.get("bars") or {}).items():
                rows += [(b["t"][:10], s, b["c"], b["v"]) for b in bars]
            token = d.get("next_page_token")
            time.sleep(0.35)
            if not token:
                break
    tmp = path + ".tmp"
    json.dump({"symbol": sym, "contracts": len(keep), "bars": rows}, open(tmp, "w"))
    os.replace(tmp, path)
    log(f"{sym}: {len(by_exp)} monthly expiries, {len(keep)} contracts kept, {len(rows)} daily bars")
    return len(rows)


def fetch():
    os.makedirs(DIR, exist_ok=True)
    creds = alpaca.credentials()
    u = json.load(open(os.path.join("private", "data", "daily_universe", "universe.json")))
    stocks = u["stocks"]
    def closes(s):
        d = json.load(open(os.path.join("private", "data", "daily_universe", f"{s}.json")))
        return {x: c for x, c in zip(d["dates"], d["close"]) if x >= str(START - timedelta(days=60))}
    todo = [s for s in stocks if not os.path.exists(os.path.join(DIR, f"{s}.json"))]
    print(f"options fetch: {len(stocks) - len(todo)} done, {len(todo)} to go", flush=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    failed = []
    def safe(s):
        try:
            return fetch_one(s, closes(s), creds, log)
        except Exception as e:                                    # one bad stock must not stop the rest
            failed.append(s); log(f"{s}: FAILED ({str(e)[:160]}); skipped, will retry next run")
            return 0
    with ThreadPoolExecutor(max_workers=2) as ex:                 # 2 workers x ~3 req/s stays under 200/min per API
        list(ex.map(safe, todo))
    if failed:
        print(f"options fetch finished with {len(failed)} skipped: {failed}", flush=True)
    print("options fetch complete", flush=True)


# ---------------------------------------------------------------- implied volatility
def _ncdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def bs_price(S, K, T, r, sig, call):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    c = S * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2)
    return np.where(call, c, c - S + K * np.exp(-r * T))


def implied_vol(price, S, K, T, r, call):
    """Vectorised bisection on [0.01, 5]; NaN when the price is outside no-arbitrage bounds."""
    price, S, K, T, r = (np.asarray(x, float) for x in (price, S, K, T, r))
    call = np.asarray(call, bool)
    intrinsic = np.where(call, np.maximum(S - K * np.exp(-r * T), 0), np.maximum(K * np.exp(-r * T) - S, 0))
    ok = (price > intrinsic + 1e-6) & (price < np.where(call, S, K)) & (T > 0)
    lo, hi = np.full(price.shape, 0.01), np.full(price.shape, 5.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_high = bs_price(S, K, T, r, mid, call) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def features():
    """Per stock and date: iv_atm, skew, pc_vol from the monthly expiry nearest 30 days (>= 7)."""
    from .daily_v2 import load_panel
    dates, syms, is_stock, C, V, ref, spy = load_panel()
    y = json.load(open(os.path.join("private", "data", "daily_extra", "yields.json")))
    ymap = dict(zip(y["dates"], y["dgs2"]))
    r_of = {}
    last = 4.0
    for d in dates:
        last = ymap.get(d, last); r_of[d] = last / 100
    di = {d: i for i, d in enumerate(dates)}
    T_, N_ = C.shape
    iv_atm, skew, pc = (np.full((T_, N_), np.nan) for _ in range(3))
    for j, s in enumerate(syms):
        p = os.path.join(DIR, f"{s}.json")
        if not is_stock[j] or not os.path.exists(p):
            continue
        rows = json.load(open(p))["bars"]
        by_day = {}
        for d, sym, c, v in rows:
            if v > 0 and d in di:
                by_day.setdefault(d, []).append((sym, c, v))
        for d, lst in by_day.items():
            t = di[d]; S = C[t, j]
            if not np.isfinite(S):
                continue
            today = date.fromisoformat(d)
            exps = sorted({sym[-15:-9] for sym, _, _ in lst})
            def days_to(e):
                return (date(2000 + int(e[:2]), int(e[2:4]), int(e[4:6])) - today).days
            cands = [e for e in exps if days_to(e) >= 7]
            if not cands:
                continue
            e = min(cands, key=lambda e: abs(days_to(e) - 30))
            sel = [(sym, c, v) for sym, c, v in lst if sym[-15:-9] == e]
            K = np.array([int(sym[-8:]) / 1000 for sym, _, _ in sel]); call = np.array([sym[-9] == "C" for sym, _, _ in sel])
            price = np.array([c for _, c, _ in sel]); vol = np.array([v for _, _, v in sel])
            Tyr = days_to(e) / 365.0
            iv = implied_vol(price, np.full(len(sel), S), K, np.full(len(sel), Tyr), np.full(len(sel), r_of[d]), call)
            def nearest(target, want_call):
                m = (call == want_call) & np.isfinite(iv)
                if not m.any():
                    return np.nan
                k = np.argmin(np.where(m, np.abs(K - target), np.inf))
                return iv[k] if abs(K[k] / target - 1) < 0.06 else np.nan
            a, b = nearest(S, True), nearest(S, False)
            iv_atm[t, j] = np.nanmean([a, b]) if np.isfinite([a, b]).any() else np.nan
            skew[t, j] = nearest(0.9 * S, False) - nearest(1.1 * S, True)
            pc[t, j] = math.log((vol[~call].sum() + 1) / (vol[call].sum() + 1))
        print(f"{s}: {np.isfinite(iv_atm[:, j]).sum()} days with ATM IV", flush=True)
    np.savez_compressed(os.path.join(DIR, "features.npz"), dates=np.array(dates), syms=np.array(syms), iv_atm=iv_atm, skew=skew, pc_vol=pc)
    print("options features saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "features"])
    a = ap.parse_args()
    fetch() if a.cmd == "fetch" else features()


if __name__ == "__main__":
    main()
