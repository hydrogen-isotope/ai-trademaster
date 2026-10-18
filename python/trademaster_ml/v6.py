"""
v6 cross-stock NASDAQ ranker: walk-forward evaluation under docs/FORECAST-V6-CROSS-SECTION-PROTOCOL.md.

Inputs (from `universe_us export` + `trademaster gpu-features`): private/data/us/gpu/{meta.json, features.json,
logclose.f32, market.f32, ranks.f32, mask.f32}.

  dev    baselines B1 (momentum), B2 (reversal) and R (ridge) on development, placebo + timing checks, stop gate
  test   (later, once) final test

Usage: python -m trademaster_ml.v6 dev
"""
import argparse, json, os, sys, time

import numpy as np

GPU_DIR = os.path.join("private", "data", "us", "gpu")
OUT = os.path.join("private", "reports", "v6")
H = 5
TRAIN_START, DEV_START, DEV_END, TEST_START = "2017-01-03", "2021-01-04", "2024-12-31", "2025-01-02"
RIDGE_ALPHA = 10.0
COST, BORROW = 0.0010, 0.01        # per side on traded weight; per year on the short side


class Panel:
    """Universe stock-days as flat rows (day index, stock index), with features and targets."""

    def __init__(self, d=GPU_DIR):
        meta = json.load(open(os.path.join(d, "meta.json")))
        fmeta = json.load(open(os.path.join(d, "features.json")))
        self.dates = meta["dates"]
        self.symbols = meta["symbols"]
        T, N = meta["T"], meta["N"]
        self.T, self.N = T, N
        self.names = fmeta["features"]
        mask = np.fromfile(os.path.join(d, "mask.f32"), np.float32).reshape(T, N) > 0.5
        lc = np.fromfile(os.path.join(d, "logclose.f32"), np.float32).reshape(T, N).astype(np.float64)
        self.lc = lc
        self.market = np.fromfile(os.path.join(d, "market.f32"), np.float32).astype(np.float64)
        self.t, self.n = np.nonzero(mask)                                  # row order: by day, then stock
        self.day_start = np.searchsorted(self.t, np.arange(T + 1))         # rows of day i: day_start[i]:day_start[i+1]
        F = len(self.names)
        ranks = np.memmap(os.path.join(d, "ranks.f32"), np.float32, "r", shape=(F, T, N))
        X = np.empty((len(self.t), F), np.float32)
        for f in range(F):
            X[:, f] = ranks[f][self.t, self.n]
        self.X = np.where(np.isfinite(X), X - 0.5, 0.0).astype(np.float32)   # missing rank -> neutral
        self.fwd = self.forward(H)                                         # [rows] raw 5-day log return
        self.y = self.demean(self.fwd)                                     # minus universe median that day
        self.yrank = self.xs_rank(self.y)                                  # training target

    def idx(self, date):
        return int(np.searchsorted(np.array(self.dates), date))

    def forward(self, h):
        """Log return close t -> close t+h; a stock that stops trading inside the window is scored to its last close."""
        lc, T = self.lc, self.T
        best = np.full(len(self.t), np.nan)                                # last finite close within t+1..t+h
        for k in range(1, h + 1):
            v = lc[np.minimum(self.t + k, T - 1), self.n]
            best = np.where(np.isfinite(v) & (self.t + k <= T - 1), v, best)
        return np.where(self.t + h <= T - 1, best - lc[self.t, self.n], np.nan)

    def per_day(self, a, fn):
        out = np.full(len(a), np.nan)
        for i in range(self.T):
            s, e = self.day_start[i], self.day_start[i + 1]
            if e > s:
                out[s:e] = fn(a[s:e])
        return out

    def demean(self, a):
        def f(x):
            ok = np.isfinite(x)
            return x - np.median(x[ok]) if ok.sum() >= 10 else x * np.nan
        return self.per_day(a, f)

    def xs_rank(self, a):
        def f(x):
            ok = np.isfinite(x)
            r = np.full(len(x), np.nan)
            if ok.sum() >= 10:
                r[ok] = (np.argsort(np.argsort(x[ok])) / (ok.sum() - 1)) - 0.5
            return r
        return self.per_day(a, f)


def nw_t(x, lag=4):
    """Mean / Newey-West standard error."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n))


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(P, pred, d0, d1, label):
    """IC, decile spread and the weekly long-short / long-only portfolios over days d0..d1 (inclusive)."""
    ics, spreads = [], []
    for i in range(d0, d1 + 1):
        s, e = P.day_start[i], P.day_start[i + 1]
        if e - s < 50:
            continue
        p, y = pred[s:e], P.y[s:e]
        ic = spearman(p, y)
        if np.isfinite(ic):
            ics.append(ic)
            ok = np.isfinite(p) & np.isfinite(y)
            q = np.quantile(p[ok], [0.1, 0.9])
            spreads.append(y[ok][p[ok] >= q[1]].mean() - y[ok][p[ok] <= q[0]].mean())
    # weekly portfolio
    ls, lo, hits, turns = [], [], [], []
    prev_l, prev_s = set(), set()
    for i in range(d0, d1 + 1 - H + 1, H):
        s, e = P.day_start[i], P.day_start[i + 1]
        p, r = pred[s:e], P.fwd[s:e]
        ok = np.isfinite(p) & np.isfinite(r)
        if ok.sum() < 50:
            continue
        q = np.quantile(p[ok], [0.1, 0.9])
        L = set(P.n[s:e][ok & (p >= q[1])])
        S = set(P.n[s:e][ok & (p <= q[0])])
        rl = np.expm1(r[ok & (p >= q[1])]).mean()
        rs = np.expm1(r[ok & (p <= q[0])]).mean()
        turn_l = 1.0 if not prev_l else 1 - len(L & prev_l) / max(len(L), 1)
        turn_s = 1.0 if not prev_s else 1 - len(S & prev_s) / max(len(S), 1)
        cost = COST * 2 * (turn_l + turn_s)                                 # sell old + buy new, each side
        ls.append(rl - rs - cost - BORROW / 52)
        mret = np.expm1(P.market[min(i + H, P.T - 1)] - P.market[i])
        lo.append(rl - mret - COST * 2 * turn_l)
        hits.append(rl > rs)
        turns.append((turn_l + turn_s) / 2)
        prev_l, prev_s = L, S
    ls, lo = np.array(ls), np.array(lo)
    eq = np.cumsum(ls)
    res = {
        "model": label, "days": len(ics),
        "ic": float(np.mean(ics)), "ic_t_nw": nw_t(ics), "ic_pos_share": float(np.mean(np.array(ics) > 0)),
        "decile_spread_5d": float(np.mean(spreads)),
        "ls_net_ann": float(ls.mean() * 52), "ls_sharpe": float(ls.mean() / ls.std() * np.sqrt(52)) if ls.std() > 0 else float("nan"),
        "ls_maxdd": float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else float("nan"),
        "weeks": len(ls), "top_beats_bottom": float(np.mean(hits)), "turnover": float(np.mean(turns)),
        "longonly_vs_qqq_ann": float(lo.mean() * 52), "longonly_ir": float(lo.mean() / lo.std() * np.sqrt(52)) if lo.std() > 0 else float("nan"),
    }
    return res


def month_starts(P, d0, d1):
    out, last = [], None
    for i in range(d0, d1 + 1):
        m = P.dates[i][:7]
        if m != last:
            out.append(i)
            last = m
    return out


def walk_forward(P, d0, d1, fit, X=None, log=print):
    """Refit on the first trading day of each month on rows whose 5-day target ends before that day."""
    X = P.X if X is None else X
    pred = np.full(len(P.t), np.nan)
    t_train0 = P.idx(TRAIN_START)
    refits = month_starts(P, d0, d1)
    for k, r in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else d1 + 1
        tr = slice(P.day_start[t_train0], P.day_start[max(r - H, t_train0)])
        Xtr, ytr = X[tr], P.yrank[tr]
        ok = np.isfinite(ytr)
        model = fit(Xtr[ok], ytr[ok])
        te = slice(P.day_start[r], P.day_start[nxt])
        pred[te] = model(X[te])
        if log and k % 12 == 0:
            log(f"    refit {P.dates[r]}: {ok.sum():,} training rows")
    return pred


def ridge(alpha=RIDGE_ALPHA):
    def fit(X, y):
        X = X.astype(np.float64)
        A = X.T @ X + alpha * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        return lambda Z: Z.astype(np.float64) @ w
    return fit


def dev(args):
    t0 = time.time()
    P = Panel()
    d0, d1 = P.idx(DEV_START), P.idx(DEV_END + "~")  - 1
    print(f"panel: {P.T} days x {P.N} symbols, {len(P.t):,} universe stock-days, {len(P.names)} features; "
          f"development {P.dates[d0]} .. {P.dates[d1]}", flush=True)
    fi = {n: i for i, n in enumerate(P.names)}
    res = []
    b1 = P.X[:, fi["mom12_1"]].astype(float)
    res.append(evaluate(P, b1, d0, d1, "B1 momentum 12-1"))
    b2 = -P.X[:, fi["mom5"]].astype(float)
    res.append(evaluate(P, b2, d0, d1, "B2 reversal 1w"))
    print("  ridge walk-forward", flush=True)
    r = walk_forward(P, d0, d1, ridge())
    res.append(evaluate(P, r, d0, d1, "R ridge"))
    # checks
    rng = np.random.default_rng(0)
    y_saved = P.y
    P.y = P.per_day(P.y, lambda x: rng.permutation(x))
    placebo = evaluate(P, r, d0, d1, "placebo (shuffled targets)")
    P.y = y_saved
    lagged = np.zeros_like(P.X)
    # features of the same stock one trading day earlier (0 = neutral when not in the universe that day)
    key = P.t.astype(np.int64) * P.N + P.n
    pos = {k: i for i, k in enumerate(key)}
    prev = np.array([pos.get(k - P.N, -1) for k in key])
    lagged[prev >= 0] = P.X[prev[prev >= 0]]
    r_lag = walk_forward(P, d0, d1, ridge(), X=lagged, log=None)
    timing = evaluate(P, r_lag, d0, d1, "R with features lagged 1 extra day")
    best = max(res, key=lambda x: x["ic"])
    gate = best["ic"] >= 0.015
    out = {"period": [P.dates[d0], P.dates[d1]], "universe_rows": int(len(P.t)), "features": P.names, "results": res,
           "placebo": placebo, "timing_check": timing, "stop_gate": {"best": best["model"], "best_ic": best["ic"], "pass": gate},
           "minutes": (time.time() - t0) / 60}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "dev.json"), "w"), indent=1)
    for x in res + [placebo, timing]:
        print(f"  {x['model']:<38} IC {x['ic']:+.4f} (t {x['ic_t_nw']:+.1f}, positive {x['ic_pos_share']:.0%})  "
              f"L/S net {x['ls_net_ann']:+.1%}/yr Sharpe {x['ls_sharpe']:+.2f} maxDD {x['ls_maxdd']:.1%} "
              f"top>bottom {x['top_beats_bottom']:.0%} turnover {x['turnover']:.0%}  long-only vs QQQ {x['longonly_vs_qqq_ann']:+.1%}/yr",
              flush=True)
    print(f"stop gate: best {best['model']} IC {best['ic']:+.4f} -> {'PASS, continue to G and M' if gate else 'STOP (IC < 0.015)'}", flush=True)
    print(f"wrote {OUT}/dev.json ({out['minutes']:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev"])
    a = ap.parse_args()
    {"dev": dev}[a.cmd](a)


if __name__ == "__main__":
    main()
