"""
Forward-only test of SGX long-only 12-1 momentum, under docs/FORECAST-SGX-PROTOCOL.md amendment B. Research only.

Each month-end close (first formation 2026-10-30) the top 10% of the SGX universe by 12-1 momentum are logged to
private/reports/sgx/momentum_shadow.jsonl, before their outcome exists. Matured months are scored against the STI
proxy after 30 bps per side. Summary + provisional picks -> private/reports/sgx/momentum.json (shown on the SGX page).

Usage: python -m trademaster_ml.sgx_momentum update
"""
import argparse, json, math, os
from datetime import date, datetime, timedelta

import numpy as np

from . import sgx

OUT = os.path.join("private", "reports", "sgx")
LOG = os.path.join(OUT, "momentum_shadow.jsonl")
SUMMARY = os.path.join(OUT, "momentum.json")
FIRST_FORMATION = "2026-10-30"
TOP_UNIVERSE, MIN_PRICE, DECILE, MIN_PICKS, COST = 150, 0.20, 0.10, 10, 0.0030
MIN_MONTHS, MAX_MONTHS = 12, 36


def load():
    p = np.load(os.path.join(sgx.DIR, "panel.npz"))
    close, volume = p["close"].astype(np.float64), p["volume"].astype(np.float64)
    return [str(d) for d in p["dates"]], [str(s) for s in p["symbols"]], [str(n) for n in p["names"]], close, volume


def is_month_end(dates, i):
    """The last trading day of its month: the next panel date is in another month, or (for the latest close) the next
    weekday falls in another month."""
    if i + 1 < len(dates):
        return dates[i + 1][:7] != dates[i][:7]
    d = date.fromisoformat(dates[i]) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m") != dates[i][:7]


def picks_at(t, close, volume):
    """Indices of the momentum portfolio formed at close t (data up to t only) and the universe size."""
    T, N = close.shape
    if t < 252:
        return [], 0
    tv = np.where(np.isfinite(close[t - 19:t + 1] * volume[t - 19:t + 1]), close[t - 19:t + 1] * volume[t - 19:t + 1], 0).mean(0)
    ok_px = np.nan_to_num(close[t]) >= MIN_PRICE
    tv = np.where(ok_px, tv, -1)
    uni = [j for j in np.argsort(-tv)[:TOP_UNIVERSE] if tv[j] > 0]
    lc = np.log(np.where(close > 0, close, np.nan))
    sig = {}
    for j in uni:
        a, b = lc[t - 21, j], lc[t - 252, j]
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(lc[t - 252:t - 20, j]).sum() >= 200:
            sig[j] = a - b
    if not sig:
        return [], len(uni)
    k = max(MIN_PICKS, int(math.ceil(DECILE * len(sig))))
    return sorted(sig, key=lambda j: -sig[j])[:k], len(uni)


def _last_close(close, j, t0, t1):
    """Close at t1, or the last close in (t0, t1] if the stock stopped trading."""
    seg = close[t0 + 1:t1 + 1, j]
    ok = np.nonzero(np.isfinite(seg) & (seg > 0))[0]
    return seg[ok[-1]] if len(ok) else close[t0, j]


def update(args=None):
    os.makedirs(OUT, exist_ok=True)
    dates, syms, names, close, volume = load()
    di = {d: i for i, d in enumerate(dates)}
    market = sgx.sti_proxy(close, volume).astype(np.float64)
    logged = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    have = {r["formation"] for r in logged}
    # 1. log every month-end formation from FIRST_FORMATION that is in the panel and not yet logged
    for i, d in enumerate(dates):
        if d >= FIRST_FORMATION and d not in have and is_month_end(dates, i):
            pk, nu = picks_at(i, close, volume)
            rec = {"formation": d, "written": datetime.now().isoformat(timespec="seconds"), "universe": nu,
                   "picks": [syms[j] for j in pk]}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logged.append(rec)
            print(f"logged momentum picks formed {d}: {len(pk)} stocks", flush=True)
    logged.sort(key=lambda r: r["formation"])
    # 2. score matured months (a month matures at the next formation close)
    months, prev = [], set()
    for a, b in zip(logged, logged[1:]):
        t0, t1 = di.get(a["formation"]), di.get(b["formation"])
        if t0 is None or t1 is None:
            continue
        cur = set(a["picks"])
        rets = [_last_close(close, syms.index(s), t0, t1) / close[t0, syms.index(s)] - 1 for s in a["picks"] if s in syms]
        gross = float(np.mean(rets)) if rets else 0.0
        changed = 1.0 if not prev else (len(cur - prev) + len(prev - cur)) / max(len(cur), 1)   # sum of |weight change|
        net = gross - COST * changed
        bench = float(math.expm1(market[t1] - market[t0]))
        months.append({"formation": a["formation"], "end": b["formation"], "gross": gross, "net": net, "benchmark": bench,
                       "excess": net - bench, "turnover": changed})
        prev = cur
    ex = np.array([m["excess"] for m in months])
    n = len(ex)
    mean = float(ex.mean()) if n else None
    t = float(ex.mean() / (ex.std(ddof=1) / math.sqrt(n))) if n >= 3 and ex.std(ddof=1) > 0 else None
    cum = float(np.prod(1 + np.array([m["net"] for m in months])) / np.prod(1 + np.array([m["benchmark"] for m in months])) - 1) if n else None
    status = "forward test, unproven"
    if n >= MIN_MONTHS:
        if t is not None and mean > 0 and t > 2 and cum > 0:
            status = "promoted"
        elif mean <= 0 or n >= MAX_MONTHS:
            status = "dropped"
    # 3. provisional picks as of the latest close (shown only; never counted)
    last = len(dates) - 1
    pk, nu = picks_at(last, close, volume)
    lc = np.log(np.where(close > 0, close, np.nan))
    provisional = [{"symbol": syms[j], "name": names[j], "close": float(close[last, j]),
                    "mom12_1": float(math.expm1(lc[last - 21, j] - lc[last - 252, j]))} for j in pk]
    out = {"protocol": "docs/FORECAST-SGX-PROTOCOL.md amendment B", "first_formation": FIRST_FORMATION, "status": status,
           "asof": dates[last], "provisional_universe": nu, "provisional_picks": provisional,
           "formations_logged": len(logged), "months": months, "matured_months": n, "mean_monthly_excess": mean, "t": t,
           "cumulative_excess": cum, "needed_months": MIN_MONTHS,
           "next_formation": next((d for d in [FIRST_FORMATION] if d > dates[last]), None)}
    tmp = SUMMARY + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, SUMMARY)
    print(f"SGX momentum: {len(logged)} formations logged, {n} matured months; status {status}; "
          f"provisional picks ({dates[last]}): {', '.join(p['symbol'] for p in provisional[:12])}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update"])
    ap.parse_args()
    update()


if __name__ == "__main__":
    main()
