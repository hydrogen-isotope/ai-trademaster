"""
Engineered strategy under docs/ENGINEERED-STRATEGY-PROTOCOL.md: signal arrays, the shared daily back-test engine, the
288-design search on development with a Deflated Sharpe Ratio gate, and the one final check. Research only, CPU.

  python -m trademaster_ml.engineered search     development grid + DSR (writes search.json; does not open the final)
  python -m trademaster_ml.engineered final      the final check of the selected design (refuses if already run)
"""
import itertools, json, math, os, sys, time

import numpy as np
from scipy import stats

from . import earnings, v6

DIR = os.path.join("private", "reports", "engineered")
DEV = ("2019-01-02", "2024-12-31")
FINAL_FROM = "2025-01-02"
GRID = dict(alpha=(0.0, 0.5), q=(0.0, 0.1, 0.2, 0.3), gamma=(0.0, 0.5, 1.0), lam=(0.0, 0.5, 1.0), e=("keep", "halve"),
            mode=("day", "overnight"), base=("equal", "size"))          # base: amendment A
COST_STOCK, COST_QQQ, BAND = 5e-4, 1e-4, 0.20
SIZE_CAP = 0.10                                                     # amendment A: no stock above 10% of the sleeve


# ---------------------------------------------------------------- 1-2. data and signals (all known at close t)

def _raw(P, name):
    meta = json.load(open(os.path.join(v6.GPU_DIR, "features.json")))
    f = meta["features"].index(name)
    raw = np.memmap(os.path.join(v6.GPU_DIR, "raw.f32"), np.float32, "r", shape=(len(meta["features"]), P.T, P.N))
    return np.array(raw[f], dtype=np.float64)


def _pct(M, U):
    """Cross-sectional percentile (0..1) per day among universe stocks with a value; NaN elsewhere."""
    out = np.full(M.shape, np.nan)
    for t in range(M.shape[0]):
        ok = U[t] & np.isfinite(M[t])
        if ok.sum() > 1:
            x = M[t, ok]
            out[t, ok] = np.argsort(np.argsort(x)) / (ok.sum() - 1)
    return out


class Data:
    def __init__(self, log=print):
        P = v6.Panel()
        self.P, T, N = P, P.T, P.N
        self.dates = P.dates
        U = np.zeros((T, N), bool)
        U[P.t, P.n] = True
        self.U = U
        g = np.load(os.path.join(DIR, "pred_G_2019.npy"))
        G = np.full((T, N), np.nan)
        G[P.t, P.n] = g
        self.rG = _pct(G, U)
        self.r_att = _pct(_raw(P, "volume_surprise"), U)
        ldv = _raw(P, "log_dollar_vol20")
        self.dv = np.where(np.isfinite(ldv), np.exp(np.clip(ldv, -50, 50)), 0.0)
        vol = _raw(P, "vol20")
        med = np.nanmedian(np.where(U, vol, np.nan), axis=1, keepdims=True)
        self.vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.where(np.isfinite(med), med, 0.02))
        ev, n_days = earnings.reaction_events(P.dates, P.symbols)
        tt, nn = np.nonzero(U)
        self.earn = np.zeros((T, N), bool)
        self.earn[tt, nn] = earnings.window_counts(ev, n_days, N, tt, nn, 1) > 0      # can be priced in at t+1
        lc = P.lc
        r = np.full((T, N), np.nan)
        r[1:] = np.expm1(lc[1:] - lc[:-1])
        self.r_cc = np.nan_to_num(r, nan=0.0)                       # return of day t (close t-1 -> close t)
        raw = np.load(os.path.join("private", "data", "us", "panel.npz"))
        rd, rs = {str(d): i for i, d in enumerate(raw["dates"])}, {str(s): j for j, s in enumerate(raw["symbols"])}
        ti = np.array([rd.get(d, -1) for d in P.dates])
        nj = np.array([rs.get(s, -1) for s in P.symbols])
        O = raw["open"].astype(np.float64)[np.maximum(ti, 0)][:, np.maximum(nj, 0)]
        O[:, nj < 0] = np.nan
        O[ti < 0] = np.nan
        C = np.exp(lc)
        co = np.full((T, N), np.nan)
        co[1:] = O[1:] / C[:-1] - 1                                  # close t-1 -> open t
        self.r_co = np.nan_to_num(np.where(np.abs(co) < 1, co, np.nan), nan=0.0)
        m = P.market
        self.q_cc = np.zeros(T)
        self.q_cc[1:] = np.nan_to_num(np.expm1(m[1:] - m[:-1]))
        if log:
            log(f"data: {T} days x {N} stocks; G coverage from {P.dates[np.isfinite(self.rG).any(1).argmax()]}", flush=True)

    def idx(self, d):
        return self.P.idx(d)


# ---------------------------------------------------------------- 3-4. portfolio, execution, costs

def target(D, t, p):
    ok = D.U[t] & np.isfinite(D.rG[t])
    w = np.zeros(D.P.N)
    if p["q"] > 0:
        ok &= D.rG[t] >= p["q"]
    if not ok.any():
        return w
    x = D.vol[t, ok] ** (-p["gamma"]) if p["gamma"] else np.ones(ok.sum())
    if p.get("base") == "size":
        x = x * D.dv[t, ok]
    if p["lam"]:
        x = x * (1 + p["lam"] * (2 * np.nan_to_num(D.r_att[t, ok], nan=0.5) - 1))
    if p["e"] == "halve":
        x = x * np.where(D.earn[t, ok], 0.5, 1.0)
    if not x.sum() > 0:
        return w
    x = x / x.sum()
    if p.get("base") == "size":                                       # cap at SIZE_CAP, spreading the excess pro rata
        cap = max(SIZE_CAP, 1.0 / len(x))                             # (an equal share when fewer than 10 stocks)
        for _ in range(50):
            over = x > cap + 1e-12
            if not over.any():
                break
            spare = (x[over] - cap).sum()
            x[over] = cap
            x[~over] += spare * x[~over] / x[~over].sum()
    w[ok] = x * (1 - p["alpha"])
    return w


def simulate(D, p, d0, d1):
    """Daily excess over QQQ for decisions at closes d0..d1-1 (returns realised on days d0+1..d1)."""
    N = D.P.N
    w = np.zeros(N)
    ex, traded, costs = [], [], []
    for t in range(d0, d1):
        tg = target(D, t, p)
        if p["mode"] == "day":
            enter_exit = (tg > 0) != (w > 0)
            far = np.abs(tg - w) > BAND * tg
            new = np.where(enter_exit | far, tg, w)
            tr = np.abs(new - w).sum()
            c = COST_STOCK * tr
            r_s = (new * D.r_cc[t + 1]).sum()
            gross = new * (1 + D.r_cc[t + 1])
            w = gross / (1 + r_s + p["alpha"] * D.q_cc[t + 1]) if (1 + r_s) > 0 else gross
            port = r_s + p["alpha"] * D.q_cc[t + 1] - c
        else:
            tr = 2 * tg.sum()
            c = COST_STOCK * tr
            port = (tg * D.r_co[t + 1]).sum() + p["alpha"] * D.q_cc[t + 1] - c
        ex.append(port - D.q_cc[t + 1])
        traded.append(tr)
        costs.append(c)
    return np.array(ex), np.array(traded), np.array(costs)


def metrics(ex, traded, costs):
    sd = ex.std(ddof=1)
    ir_d = ex.mean() / sd if sd > 0 else 0.0
    lag = 5
    e = ex - ex.mean()
    v = (e @ e) / len(e)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / len(e)
    nw_t = ex.mean() / math.sqrt(v / len(e)) if v > 0 else 0.0
    cum = np.cumsum(ex)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {"days": int(len(ex)), "excess_ann": float(ex.mean() * 252), "ir": float(ir_d * math.sqrt(252)), "sr_daily": float(ir_d),
            "nw_t": float(nw_t), "tracking_error": float(sd * math.sqrt(252)), "worst_excess_dd": dd,
            "turnover_ann": float(traded.mean() * 252), "cost_drag_ann": float(costs.mean() * 252),
            "skew": float(stats.skew(ex)), "kurt": float(stats.kurtosis(ex, fisher=False))}


def dsr(sr, n_obs, skew, kurt, trial_srs):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), per-period Sharpe ratios."""
    n = len(trial_srs)
    v = np.var(trial_srs, ddof=1)
    em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / n) + em * stats.norm.ppf(1 - 1 / (n * math.e)))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf(z)), float(sr0)


def designs():
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def search(log=print):
    t0 = time.time()
    D = Data(log)
    d0, d1 = D.idx(DEV[0]), D.idx(DEV[1] + "~") - 1
    rows = []
    for k, p in enumerate(designs()):
        ex, tr, c = simulate(D, p, d0, d1)
        m = metrics(ex, tr, c)
        rows.append({"design": p, **m})
        if k % 24 == 0:
            log(f"  {k + 1}/{len(designs())} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    best = max(rows, key=lambda r: r["ir"])
    d, sr0 = dsr(best["sr_daily"], best["days"], best["skew"], best["kurt"], np.array([r["sr_daily"] for r in rows]))
    qqq_only = next(r for r in rows if r["design"]["alpha"] == 0.5 and r["design"]["q"] == 0 and r["design"]["gamma"] == 0
                    and r["design"]["lam"] == 0 and r["design"]["e"] == "keep" and r["design"]["mode"] == "day"
                    and r["design"]["base"] == "equal")
    out = {"period": [D.dates[d0], D.dates[d1]], "trials": len(rows), "selected": best, "dsr": d, "sr0_daily": sr0,
           "gate_open_final": bool(d >= 0.95), "reference_half_qqq_half_equal_weight": qqq_only, "all": rows,
           "minutes": (time.time() - t0) / 60}
    json.dump(out, open(os.path.join(DIR, "search.json"), "w"), indent=1)
    top = sorted(rows, key=lambda r: -r["ir"])[:8]
    for r in top:
        log(f"  IR {r['ir']:+.2f}  excess {r['excess_ann']:+.1%}/yr  t {r['nw_t']:+.1f}  DD {r['worst_excess_dd']:+.1%}  "
            f"turnover {r['turnover_ann']:.0f}x  cost {r['cost_drag_ann']:.1%}  {r['design']}", flush=True)
    log(f"selected {best['design']}: IR {best['ir']:+.2f}, excess {best['excess_ann']:+.1%}/yr; DSR {d:.3f} "
        f"-> final {'OPEN' if d >= 0.95 else 'stays closed'}", flush=True)
    return out


def final(log=print):
    path = os.path.join(DIR, "final.json")
    if os.path.exists(path):
        raise SystemExit("the final check runs once (protocol): final.json exists")
    s = json.load(open(os.path.join(DIR, "search.json")))
    if not s["gate_open_final"]:
        raise SystemExit("the DSR gate is closed: the final check is not run")
    D = Data(log)
    p = s["selected"]["design"]
    d0, d1 = D.idx(FINAL_FROM), D.P.T - 1
    ex, tr, c = simulate(D, p, d0, d1)
    m = metrics(ex, tr, c)
    ok = bool(m["excess_ann"] > 0 and m["nw_t"] >= 2 and m["worst_excess_dd"] > -0.25)
    out = {"design": p, "period": [D.dates[d0], D.dates[d1]], **m, "pass": ok}
    json.dump(out, open(path, "w"), indent=1)
    log(f"FINAL {D.dates[d0]}..{D.dates[d1]}: excess {m['excess_ann']:+.1%}/yr, IR {m['ir']:+.2f}, NW t {m['nw_t']:+.2f}, "
        f"worst excess DD {m['worst_excess_dd']:+.1%} -> {'PASS' if ok else 'fails'}", flush=True)
    return out


if __name__ == "__main__":
    v6.use_market("nasdaq")
    os.makedirs(DIR, exist_ok=True)
    {"search": search, "final": final}[sys.argv[1]]()
