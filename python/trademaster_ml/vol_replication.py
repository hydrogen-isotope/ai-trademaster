"""
Volatility-targeting replication under docs/VOL-TARGET-REPLICATION-PROTOCOL.md: one fixed rule on 15 index ETFs
(QQQ excluded), judged by the average hedged return. Research only; no orders.

Usage: python -m trademaster_ml.vol_replication fetch | run
"""
import json, math, os, sys, time

import numpy as np

from . import qqq_timing as Q

SYMS = ("SPY", "DIA", "IWM", "MDY", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
WINDOW, LONG, SCALE = 60, 1260, 0.9
PERIODS = {"final": ("2019-01-02", "9999"), "history": ("2000-01-03", "2018-12-31")}
OUT = os.path.join("private", "reports", "qqq_timing", "replication.json")
CHUNKS = [("1999-01-01", "2005-12-31"), ("2006-01-01", "2012-12-31"), ("2013-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]


def fetch(log=print):
    from . import twelvedata as T
    for s in SYMS:
        path = os.path.join(Q.DIR, f"{s}_td_all.json")
        if os.path.exists(path):
            continue
        rows = []
        for a, b in CHUNKS:
            r = T.get("time_series", {"symbol": s, "interval": "1day", "start_date": a, "end_date": b, "outputsize": 5000,
                                      "order": "ASC", "adjust": "all"})
            if r.get("status") == "ok":
                rows += r["values"]
            time.sleep(8)                                              # free plan: 8 requests a minute
        json.dump(rows, open(path, "w"))
        log(f"{s}: {len(rows)} days, {rows[0]['datetime'] if rows else '-'} -> {rows[-1]['datetime'] if rows else '-'}", flush=True)


def exposure(D):
    T = len(D.C)
    sd, sl = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(LONG, T):
        sd[t] = D.lr[t - WINDOW + 1:t + 1].std(ddof=1) * math.sqrt(252)
        sl[t] = D.lr[t - LONG + 1:t + 1].std(ddof=1) * math.sqrt(252)
    return np.minimum(1.0, SCALE * sl / sd)


def nw_t(x, lag=5):
    e = x - x.mean()
    v = (e @ e) / len(e)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / len(e)
    return float(x.mean() / math.sqrt(v / len(x)))


def run(log=print):
    res = {"rule": {"window": WINDOW, "long": LONG, "scale": SCALE, "cap": 1.0}, "periods": {}}
    data = {s: Q.load(f"{s}_td_all.json") for s in SYMS}
    for name, (a, b) in PERIODS.items():
        per, hedged = {}, {}
        for s, D in data.items():
            tg = exposure(D)
            first = int(np.argmax(np.isfinite(tg))) + 2
            if not np.isfinite(tg).any() or D.dates[-1] < a:
                continue
            d0 = max(D.idx(a), first)
            d1 = D.last_idx(b)
            if d1 - d0 < 250:
                continue
            r, tr, c = Q.simulate(D, None, d0, d1, tg=tg)
            ra, rf = D.R[d0:d1 + 1], D.rf[d0:d1 + 1]
            m = Q.metrics(r, rf, ra, tr, c)
            h = Q.metrics(ra, rf, ra)
            per[s] = {"from": D.dates[d0], "to": D.dates[d1], "design": m, "hold": h}
            hs = (r - rf) - m["beta"] * (ra - rf)
            for d, x in zip(D.dates[d0:d1 + 1], hs):
                hedged.setdefault(d, []).append(x)
            log(f"  {name} {s} {D.dates[d0]}->{D.dates[d1]}: alpha {m['alpha_ann']:+.2%}/yr (t {m['alpha_t']:+.2f}), "
                f"CAGR {m['cagr']:+.1%} vs {h['cagr']:+.1%}, worst drawdown {m['max_dd']:.0%} vs {h['max_dd']:.0%}, "
                f"Sharpe {m['sharpe']:.2f} vs {h['sharpe']:.2f}", flush=True)
        days = sorted(hedged)
        avg = np.array([np.mean(hedged[d]) for d in days])
        t = nw_t(avg)
        res["periods"][name] = {"assets": per, "days": len(days), "mean_hedged_ann": float(avg.mean() * 252), "nw_t": t,
                                "positive_alpha": sum(v["design"]["alpha_ann"] > 0 for v in per.values()),
                                "n_assets": len(per)}
        log(f"{name}: {len(per)} assets, average hedged return {avg.mean() * 252:+.2%}/yr (t {t:+.2f}), "
            f"{res['periods'][name]['positive_alpha']}/{len(per)} assets with positive alpha", flush=True)
    f = res["periods"]["final"]
    res["pass"] = bool(f["mean_hedged_ann"] > 0 and f["nw_t"] >= 2)
    log(f"primary test (2019 onwards): {'PASS' if res['pass'] else 'fails'}")
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
