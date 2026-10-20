"""
QQQ timing study under docs/QQQ-TIMING-PROTOCOL.md: volatility targeting and trend filtering on QQQ, with optional
leverage, judged by alpha over holding QQQ. Research only; no orders.

Usage: python -m trademaster_ml.qqq_timing search | final
"""
import csv, itertools, json, math, os, sys, time

import numpy as np
from scipy import stats

from .engineered import dsr

DIR = os.path.join("private", "data", "timing")
OUT = os.path.join("private", "reports", "qqq_timing")
DEV = ("2000-01-03", "2018-12-31")
FINAL_FROM = "2019-01-02"
COST, BAND, BORROW_SPREAD = 1e-4, 0.10, 0.015


def designs():
    out = [dict(target=s, window=w, cap=L, trend=tr) for s, w, L, tr in
           itertools.product((0.15, 0.20, 0.25, 0.30), (20, 60), (1.0, 1.5, 2.0), (0, 100, 200))]
    out += [dict(target=None, window=None, cap=L, trend=tr) for tr, L in itertools.product((100, 200), (1.0, 1.5, 2.0))]
    return out


class Data:
    def __init__(self, dates, close, rf_annual_pct):
        self.dates = list(dates)
        self.C = np.asarray(close, float)
        self.R = np.r_[0.0, self.C[1:] / self.C[:-1] - 1]
        self.lr = np.r_[0.0, np.log(self.C[1:] / self.C[:-1])]
        self.rf = np.asarray(rf_annual_pct, float) / 100 / 252

    def idx(self, d):
        """First index with date >= d."""
        return next(i for i, x in enumerate(self.dates) if x >= d)

    def last_idx(self, d):
        return max(i for i, x in enumerate(self.dates) if x <= d)


def load(name="qqq_td_all.json"):
    rows = json.load(open(os.path.join(DIR, name)))
    rows = sorted({r["datetime"]: r for r in rows}.values(), key=lambda r: r["datetime"])
    vol = np.array([float(r["volume"]) for r in rows])
    keep = [i for i in range(len(rows)) if i < 20 or vol[i] >= 0.1 * np.median(vol[i - 20:i])]
    rows = [rows[i] for i in keep]
    dates = [r["datetime"] for r in rows]
    rates, last = {}, None
    for r in csv.DictReader(open(os.path.join(DIR, "DTB3.csv"))):
        v = r.get("DTB3", "")
        if v not in ("", "."):
            rates[r["observation_date"]] = float(v)
    rk = sorted(rates)
    rf, j = [], 0
    for d in dates:
        while j < len(rk) and rk[j] <= d:
            last = rates[rk[j]]
            j += 1
        rf.append(last if last is not None else 0.0)
    return Data(dates, [float(r["close"]) for r in rows], rf)


def target_series(D, p):
    """Exposure decided from data through close t (index t)."""
    T = len(D.C)
    tg = np.full(T, np.nan)
    if p["target"] is None:
        base = np.full(T, p["cap"])
    else:
        w = p["window"]
        sd = np.full(T, np.nan)
        for t in range(w, T):
            sd[t] = D.lr[t - w + 1:t + 1].std(ddof=1) * math.sqrt(252)
        base = np.minimum(p["cap"], p["target"] / sd)
    if p["trend"]:
        n = p["trend"]
        cs = np.r_[0.0, np.cumsum(D.C)]
        sma = np.full(T, np.nan)
        sma[n - 1:] = (cs[n:] - cs[:-n]) / n
        up = D.C > sma
        base = np.where(np.isnan(sma), np.nan, np.where(up, base, 0.0))
    tg[:] = base
    return tg


def simulate(D, p, d0, d1, tg=None):
    """Daily net returns over days d0..d1 (return of day s = close s-1 -> close s). Target from close t is traded at close t+1."""
    tg = target_series(D, p) if tg is None else tg
    x = tg[d0 - 2]
    if not np.isfinite(x):
        raise ValueError("not enough history before the start")
    r_p, traded, costs = [], [], []
    for s in range(d0, d1 + 1):
        cash = 1 - x
        fin = D.rf[s] if cash >= 0 else D.rf[s] + BORROW_SPREAD / 252
        r = x * D.R[s] + cash * fin
        xd = x * (1 + D.R[s]) / (1 + r)
        new = tg[s - 1]
        tr = 0.0
        if np.isfinite(new) and (abs(new - xd) > BAND or (new == 0) != (xd == 0)):
            tr = abs(new - xd)
            xd = new
        c = tr * COST
        r_p.append(r - c); traded.append(tr); costs.append(c)
        x = xd
    return np.array(r_p), np.array(traded), np.array(costs)


def _hac_alpha(y, xq, lag=5):
    X = np.column_stack([np.ones_like(xq), xq])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    g = X * e[:, None]
    S = g.T @ g
    for k in range(1, lag + 1):
        G = g[k:].T @ g[:-k]
        S += (1 - k / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(beta[0]), float(beta[1]), float(beta[0] / math.sqrt(V[0, 0])), e


def metrics(r, rf, rq, traded=None, costs=None, exposure=None):
    ex, exq = r - rf, rq - rf
    a, b, t, e = _hac_alpha(ex, exq)
    wealth = np.cumprod(1 + r)
    years = len(r) / 252
    ar_d = a / e.std(ddof=1)
    return {"days": int(len(r)), "cagr": float(wealth[-1] ** (1 / years) - 1), "vol": float(r.std(ddof=1) * math.sqrt(252)),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)),
            "max_dd": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
            "alpha_ann": a * 252, "beta": b, "alpha_t": t, "appraisal": float(ar_d * math.sqrt(252)), "ar_daily": float(ar_d),
            "skew": float(stats.skew(e)), "kurt": float(stats.kurtosis(e, fisher=False)),
            "turnover_ann": float(traded.mean() * 252) if traded is not None else 0.0,
            "cost_ann": float(costs.mean() * 252) if costs is not None else 0.0}


def qqq_metrics(D, d0, d1):
    r = D.R[d0:d1 + 1]
    return metrics(r, D.rf[d0:d1 + 1], r)


def _fmt(m):
    return (f"CAGR {m['cagr']:+.1%}, vol {m['vol']:.1%}, Sharpe {m['sharpe']:.2f}, worst drawdown {m['max_dd']:.1%}, "
            f"alpha {m['alpha_ann']:+.2%}/yr (t {m['alpha_t']:+.2f}), beta {m['beta']:.2f}")


def search(log=print):
    os.makedirs(OUT, exist_ok=True)
    D = load()
    d0, d1 = D.idx(DEV[0]), D.last_idx(DEV[1])
    rf, rq = D.rf[d0:d1 + 1], D.R[d0:d1 + 1]
    rows = []
    for p in designs():
        r, tr, c = simulate(D, p, d0, d1)
        rows.append({"design": p, **metrics(r, rf, rq, tr, c)})
    best = max(rows, key=lambda m: m["alpha_t"])
    g, sr0 = dsr(best["ar_daily"], best["days"], best["skew"], best["kurt"], np.array([m["ar_daily"] for m in rows]))
    res = {"period": [D.dates[d0], D.dates[d1]], "trials": len(rows), "qqq": qqq_metrics(D, d0, d1), "best": best,
           "dsr": g, "sr0_daily": sr0, "gate_open": bool(g >= 0.95), "rows": rows}
    json.dump(res, open(os.path.join(OUT, "search.json"), "w"), indent=1)
    log(f"development {D.dates[d0]} -> {D.dates[d1]}, {len(rows)} designs")
    log(f"  QQQ held:  {_fmt(res['qqq'])}")
    log(f"  selected {best['design']}: {_fmt(best)}")
    log(f"  deflated Sharpe {g:.3f} (gate 0.95) -> {'final check may be opened' if res['gate_open'] else 'gate closed, study stops'}")
    return res


def final(log=print):
    res = json.load(open(os.path.join(OUT, "search.json")))
    if not res["gate_open"]:
        raise SystemExit("the development gate is closed; the final period stays unopened (protocol section 5)")
    D = load()
    d0, d1 = D.idx(FINAL_FROM), len(D.C) - 1
    p = res["best"]["design"]
    r, tr, c = simulate(D, p, d0, d1)
    m = metrics(r, D.rf[d0:d1 + 1], D.R[d0:d1 + 1], tr, c)
    q = qqq_metrics(D, d0, d1)
    out = {"period": [D.dates[d0], D.dates[d1]], "design": p, "design_metrics": m, "qqq": q,
           "pass": bool(m["alpha_ann"] > 0 and m["alpha_t"] >= 2),
           "beat_qqq_cagr": bool(m["cagr"] > q["cagr"]), "smaller_drawdown": bool(m["max_dd"] > q["max_dd"])}
    json.dump(out, open(os.path.join(OUT, "final.json"), "w"), indent=1)
    log(f"final {D.dates[d0]} -> {D.dates[d1]}")
    log(f"  QQQ held: {_fmt(q)}")
    log(f"  design:   {_fmt(m)}")
    log(f"  -> {'PASS' if out['pass'] else 'fails'}")
    return out


if __name__ == "__main__":
    {"search": search, "final": final}[sys.argv[1]]()
