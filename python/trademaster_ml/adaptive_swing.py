"""
Adaptive swing back-test under docs/ADAPTIVE-SWING-PROTOCOL.md: buy the LightGBM ranking's top 3 at the open, sell at a
take-profit, a -5% stop, or after 5 sessions; compared with QQQ over the same holding window. Research only.

Usage: python -m trademaster_ml.adaptive_swing
"""
import json, math, os

import numpy as np

from . import v6

OUT = os.path.join("private", "reports", "adaptive_swing.json")
US, STOP, MAX_DAYS, TOP, COST = (0.01, 0.015, 0.02, 0.03), 0.05, 5, 3, 0.0010
PERIODS = {"dev": ("2021-01-04", "2024-12-31"), "test": ("2025-01-02", "2026-10-09")}


def load():
    P = v6.Panel()
    raw = np.load(os.path.join("private", "data", "us", "panel.npz"))
    rd, rs = [str(x) for x in raw["dates"]], [str(x) for x in raw["symbols"]]
    di, si = {d: i for i, d in enumerate(rd)}, {s: j for j, s in enumerate(rs)}
    ti = np.array([di.get(d, -1) for d in P.dates])
    ni = np.array([si.get(s, -1) for s in P.symbols])
    O, H, L, C = (raw[k].astype(float) for k in ("open", "high", "low", "close"))
    path = os.path.join(v6.OUT, "pred_G_adaptive.npy")          # walk-forward picks recomputed on the current panel
    if os.path.exists(path) and len(np.load(path)) == len(P.t):
        g = np.load(path)
    else:
        X = np.hstack([P.X, v6.regime(P)[P.t]]).astype(np.float32)
        d0, d1 = P.idx(PERIODS["dev"][0]), v6.period_bounds(P, "test")[1]
        g = v6.lgb_walk_forward(P, X, d0 - 1, d1, log=None)
        np.save(path, g)
    return P, g, ti, ni, O, H, L, C, si["QQQ"]


def trade(t0, j, O, H, L, C, u):
    """Entry at the open of raw day t0 for raw stock j. Returns (log return, exit day offset, how) or None."""
    e = O[t0, j]
    if not np.isfinite(e) or e <= 0:
        return None
    tp, sl = e * (1 + u), e * (1 - STOP)
    for k in range(MAX_DAYS):
        t = t0 + k
        if t >= O.shape[0] or not np.isfinite(C[t, j]):
            return None
        if L[t, j] <= sl:                                          # the stop first when both happen that day
            return math.log(sl / e), k, "stop"
        if H[t, j] >= tp:
            return math.log(tp / e), k, "target"
    return math.log(C[t0 + MAX_DAYS - 1, j] / e), MAX_DAYS - 1, "time"


def run():
    P, g, ti, ni, O, H, L, C, q = load()
    res = {}
    for name, (d0, d1) in PERIODS.items():
        res[name] = {}
        days = [i for i in range(1, P.T) if d0 <= P.dates[i] <= d1]
        for u in US:
            ex, how, held, per_day = [], {"target": 0, "stop": 0, "time": 0}, [], {}
            for i in days:
                s, e = P.day_start[i - 1], P.day_start[i]              # scores known at close i-1
                sc = g[s:e]
                if not np.isfinite(sc).any():
                    continue
                top = np.argsort(np.where(np.isfinite(sc), -sc, np.inf))[:TOP]
                t0 = ti[i]
                for k in top:
                    j = ni[P.n[s + k]]
                    if j < 0 or t0 < 0:
                        continue
                    r = trade(t0, j, O, H, L, C, u)
                    if r is None:
                        continue
                    lr, off, h = r
                    qe, qx = O[t0, q], C[t0 + off, q]
                    if not (np.isfinite(qe) and np.isfinite(qx)):
                        continue
                    x = (math.exp(lr) - 1 - COST) - (qx / qe - 1)
                    ex.append(x); how[h] += 1; held.append(off + 1)
                    per_day.setdefault(i, []).append(x)
            dm = np.array([np.mean(v) for v in per_day.values()])
            t = dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm))) if len(dm) > 2 else float("nan")
            ex = np.array(ex)
            res[name][str(u)] = {"trades": int(len(ex)), "mean_excess_pct": float(ex.mean() * 100), "t_by_day": float(t),
                                 "hit_vs_qqq": float(np.mean(ex > 0)), "avg_days_held": float(np.mean(held)),
                                 "exits": {k: v / len(ex) for k, v in how.items()}}
            x = res[name][str(u)]
            print(f"{name} u={u:.1%}: {x['trades']} trades, excess vs QQQ {x['mean_excess_pct']:+.3f}% per trade (t {t:+.2f}), "
                  f"beat QQQ {x['hit_vs_qqq']:.0%}, held {x['avg_days_held']:.1f} days, exits {', '.join(f'{k} {v:.0%}' for k, v in x['exits'].items())}",
                  flush=True)
    best = max(US, key=lambda u: res["dev"][str(u)]["mean_excess_pct"])
    f = res["test"][str(best)]
    res["chosen_u"], res["pass"] = best, bool(f["mean_excess_pct"] > 0 and f["t_by_day"] >= 2)
    print(f"chosen on development: u={best:.1%}; final check: excess {f['mean_excess_pct']:+.3f}% per trade (t {f['t_by_day']:+.2f}) -> "
          f"{'PASS' if res['pass'] else 'fails'}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    v6.use_market("nasdaq")
    run()
