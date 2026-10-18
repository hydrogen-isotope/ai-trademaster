"""
SGX (Singapore Exchange) daily data from Eulerpool, for the SGX ranker and price forecasts. Research only; protocol:
docs/FORECAST-SGX-PROTOCOL.md (survivor-only universe; synthetic STI proxy as the market series).

  assets   SGX tickers (.SI) of Singapore-domiciled companies in Eulerpool's screener -> private/data/sgx/assets.json
  bars     one-off adjusted daily history from 2000 for every ticker (resumable) -> private/data/sgx/bars/*.json
  panel    one float32 panel: private/data/sgx/panel.npz (dates x symbols: open, high, low, close, volume)
  update   stream only recent days (re-fetches a ticker's full history when its adjusted past changed, e.g. a split)
  export   raw float32 files for `trademaster gpu-features --in private/data/sgx/gpu --top 150 --min-price 0.2`,
           including the synthetic STI proxy (top 30 priced >= 1 by lagged one-year traded value, capped weights)

Usage: python -m trademaster_ml.sgx assets | bars | panel | update | export
"""
import argparse, json, os, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import eulerpool

DIR = os.path.join("private", "data", "sgx")
BARS = os.path.join(DIR, "bars")
GPU = os.path.join(DIR, "gpu")
START = 946684800                                   # 2000-01-01
PROXY_N = 30


def _day(t):
    return datetime.fromtimestamp(t / 1000 if t > 1e11 else t, timezone.utc).strftime("%Y-%m-%d")


def assets():
    os.makedirs(DIR, exist_ok=True)
    rows, page = [], 0
    while page < 30:
        st, r = eulerpool.call("POST", "screener/screen", body={"filters": [{"id": "country", "value": ["SG"], "equality": "IN"}],
                                                                "page": page, "size": 100})
        if st != 200 or not r:
            break
        rows += r
        page += 1
        if len(r) < 100:
            break
    out = [{"symbol": r["ticker"], "isin": r.get("isin"), "name": r.get("name") or "", "currency": r.get("currency")}
           for r in rows if str(r.get("ticker", "")).endswith(".SI")]
    json.dump(out, open(os.path.join(DIR, "assets.json"), "w"), indent=0)
    print(f"{len(out)} SGX tickers (survivor-only: current Singapore-domiciled companies)", flush=True)
    return out


def _fetch(sym, start=START):
    st, c = eulerpool.call("GET", f"charting/ohlcv/{urllib.parse.quote(sym)}", {"resolution": "D", "from": int(start), "to": int(time.time())})
    if st == 200 and isinstance(c, dict) and c.get("t"):
        return [[_day(t), o, h, l, cl, v] for t, o, h, l, cl, v in zip(c["t"], c["o"], c["h"], c["l"], c["c"], c["v"])]
    return None


def bars():
    os.makedirs(BARS, exist_ok=True)
    syms = [a["symbol"] for a in json.load(open(os.path.join(DIR, "assets.json")))]
    todo = [s for s in syms if not os.path.exists(os.path.join(BARS, s + ".json"))]
    t0, done, missing = time.time(), [0], []

    def one(s):
        rows = _fetch(s)
        if rows:
            json.dump(rows, open(os.path.join(BARS, s + ".json"), "w"))
        else:
            missing.append(s)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  {done[0]}/{len(todo)} tickers, {(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(0.25)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, todo))
    print(f"bars complete: {len(syms) - len(missing)} tickers with history, {len(missing)} without ({', '.join(missing[:10])})", flush=True)


def panel():
    files = sorted(f for f in os.listdir(BARS) if f.endswith(".json"))
    data = {f[:-5]: json.load(open(os.path.join(BARS, f))) for f in files}
    days = sorted({r[0] for rows in data.values() for r in rows})
    # keep real trading days only: a day counts when at least 20% of the tickers trading that year have a bar
    cnt = {}
    for rows in data.values():
        for r in rows:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
    by_year = {}
    for d in days:
        by_year[d[:4]] = max(by_year.get(d[:4], 0), cnt[d])
    days = [d for d in days if cnt[d] >= 0.2 * by_year[d[:4]]]
    di = {d: i for i, d in enumerate(days)}
    syms = sorted(data)
    T, N = len(days), len(syms)
    arr = {k: np.full((T, N), np.nan, np.float32) for k in ("open", "high", "low", "close", "volume")}
    for j, s in enumerate(syms):
        for d, o, h, l, c, v in data[s]:
            i = di.get(d)
            if i is not None and c and c > 0:
                arr["open"][i, j], arr["high"][i, j], arr["low"][i, j], arr["close"][i, j], arr["volume"][i, j] = o, h, l, c, v
    # protocol amendment A: drop one-day spikes (bad ticks): a close >50% away from both neighbours, in opposite directions
    c = arr["close"]
    lc = np.log(np.where(c > 0, c, np.nan))
    nxt = np.full_like(lc, np.nan)
    prv = np.full_like(lc, np.nan)
    for j in range(N):                                                    # neighbours = previous/next day with a close
        ok = np.nonzero(np.isfinite(lc[:, j]))[0]
        if len(ok) > 2:
            prv[ok[1:], j] = lc[ok[:-1], j]
            nxt[ok[:-1], j] = lc[ok[1:], j]
    up, down = lc - prv, nxt - lc
    spike = (np.abs(up) > np.log(1.5)) & (np.abs(down) > np.log(1.5)) & (np.sign(up) != np.sign(down))
    for k in arr:
        arr[k][spike] = np.nan
    # protocol amendment A: drop tickers whose adjusted history exceeds 100 (quote currency). These are big share
    # consolidations: prices are scaled back but volumes are not, so their past traded value is wildly inflated
    keep = np.nanmax(np.where(np.isfinite(arr["close"]), arr["close"], 0), axis=0) <= 100
    dropped = [s for s, k in zip(syms, keep) if not k]
    syms = [s for s, k in zip(syms, keep) if k]
    for k in arr:
        arr[k] = arr[k][:, keep]
    N = len(syms)
    meta = {a["symbol"]: a for a in json.load(open(os.path.join(DIR, "assets.json")))}
    tmp = os.path.join(DIR, "panel.tmp.npz")
    np.savez(tmp, dates=np.array(days), symbols=np.array(syms), names=np.array([meta.get(s, {}).get("name", "") for s in syms]),
             currency=np.array([str(meta.get(s, {}).get("currency")) for s in syms]), **arr)
    os.replace(tmp, os.path.join(DIR, "panel.npz"))
    print(f"SGX panel: {T} days x {N} tickers ({days[0]} .. {days[-1]}); {int(spike.sum())} one-day spikes removed; {len(dropped)} consolidation-distorted tickers dropped ({', '.join(dropped)})", flush=True)
    return days[-1]


def update():
    """Re-fetch the last ~3 weeks for every ticker; a ticker whose adjusted past changed (split/bonus) is re-fetched in
    full. Then rebuild the panel."""
    syms = [a["symbol"] for a in json.load(open(os.path.join(DIR, "assets.json")))]
    since = time.time() - 21 * 86400
    stats = {"updated": 0, "refetched": 0, "no_recent_bars": 0}

    def one(s):
        path = os.path.join(BARS, s + ".json")
        old = json.load(open(path)) if os.path.exists(path) else []
        recent = _fetch(s, since)
        if not recent:
            stats["no_recent_bars"] += 1                                  # illiquid or never traded: nothing new
            return
        known = {r[0]: r[4] for r in old}
        overlap = [(known[r[0]], r[4]) for r in recent if r[0] in known and known[r[0]]]
        if not old or any(abs(n / o - 1) > 0.01 for o, n in overlap):
            full = _fetch(s)
            if full:
                json.dump(full, open(path, "w"))
                stats["refetched"] += 1
            return
        merged = {r[0]: r for r in old}
        merged.update({r[0]: r for r in recent})
        json.dump([merged[d] for d in sorted(merged)], open(path, "w"))
        stats["updated"] += 1
        time.sleep(0.2)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, syms))
    print(f"SGX update: {stats}", flush=True)
    return panel()


def sti_proxy(close, volume):
    """Log level of a synthetic STI (protocol amendment A): each day the return of the PROXY_N stocks priced >= 1 with the
    largest one-year average traded value (as of the previous day), weighted by that traded value with each weight capped at 15%;
    single-day moves beyond 30% are left out as likely bad ticks."""
    T, N = close.shape
    lc = np.log(np.where(close > 0, close, np.nan))
    r = np.vstack([np.full((1, N), np.nan), np.diff(lc, axis=0)])
    tv = np.where(np.isfinite(close * volume), close * volume, 0.0)
    cs = np.cumsum(tv, axis=0)
    tv250 = (cs - np.vstack([np.zeros((250, N)), cs[:-250]])) / 250     # one-year average: ignores short bursts
    level = np.zeros(T)
    for t in range(1, T):
        w = tv250[t - 1].copy()
        w[~(np.isfinite(r[t]) & (np.abs(np.nan_to_num(r[t])) <= 0.3) & (np.nan_to_num(close[t - 1]) >= 1.0))] = 0
        top = np.argsort(-w)[:PROXY_N]
        ww = w[top]
        if ww.sum() <= 0:
            level[t] = level[t - 1]
            continue
        ww = ww / ww.sum()
        for _ in range(10):                                               # cap at 15%, redistribute the excess
            over = ww > 0.15
            if not over.any():
                break
            extra = (ww[over] - 0.15).sum()
            ww[over] = 0.15
            under = ~over & (ww > 0)
            ww[under] += extra * ww[under] / ww[under].sum()
        # an index averages simple returns; averaging log returns would add a downward drag of ~variance/2 per day
        level[t] = level[t - 1] + float(np.log1p(np.dot(ww, np.expm1(np.nan_to_num(r[t, top])))))
    return level.astype(np.float32)


def export():
    p = np.load(os.path.join(DIR, "panel.npz"))
    os.makedirs(GPU, exist_ok=True)
    c, v = p["close"].astype(np.float64), p["volume"].astype(np.float64)
    T, N = c.shape
    lc = np.where(c > 0, np.log(np.where(c > 0, c, 1)), np.nan).astype(np.float32)
    lc.tofile(os.path.join(GPU, "logclose.f32"))
    p["volume"].astype(np.float32).tofile(os.path.join(GPU, "volume.f32"))
    sti_proxy(c, v).tofile(os.path.join(GPU, "market.f32"))
    np.ones(N, np.float32).tofile(os.path.join(GPU, "eligible.f32"))
    for f in ("industry.f32",):
        if os.path.exists(os.path.join(GPU, f)):
            os.remove(os.path.join(GPU, f))
    json.dump({"T": T, "N": N, "dates": [str(d) for d in p["dates"]], "symbols": [str(s) for s in p["symbols"]],
               "market": "synthetic STI proxy (top 30 priced >= 1 by lagged one-year traded value, weights capped at 15%)", "extras": []},
              open(os.path.join(GPU, "meta.json"), "w"))
    print(f"exported SGX {T} x {N} to {GPU}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assets", "bars", "panel", "update", "export"])
    a = ap.parse_args()
    globals()[a.cmd]()


if __name__ == "__main__":
    main()
