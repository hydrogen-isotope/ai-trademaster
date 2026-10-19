"""
v6 cross-stock NASDAQ ranker: walk-forward evaluation under docs/FORECAST-V6-CROSS-SECTION-PROTOCOL.md.

Inputs (from `universe_us export` + `trademaster gpu-features`): private/data/us/gpu/{meta.json, features.json,
logclose.f32, market.f32, ranks.f32, mask.f32}.

  dev         baselines B1 (momentum), B2 (reversal) and R (ridge) on development, placebo + timing checks, stop gate
  models      all protocol models (B1, B2, R, G, M, E) on --period dev|test; the test runs once
  mlp-export  inputs for the GPU MLP trainer (trademaster gpu-mlp); --period latest = one fit on all data
  rank        today's rankings for the dashboard with the adopted model (ridge; CPU only)

Usage: python -m trademaster_ml.v6 dev | models --period dev | mlp-export --period dev
"""
import argparse, hashlib, json, os, sys, time
from datetime import datetime

import numpy as np

GPU_DIR = os.path.join("private", "data", "us", "gpu")
OUT = os.path.join("private", "reports", "v6")
H = 5
TRAIN_START, DEV_START, DEV_END, TEST_START = "2017-01-03", "2021-01-04", "2024-12-31", "2025-01-02"
RIDGE_ALPHA = 10.0
COST, BORROW = 0.0010, 0.01        # per side on traded weight; per year on the short side
_NASDAQ_PERIODS, _NASDAQ_COST = (TRAIN_START, DEV_START, DEV_END, TEST_START), COST
CANDIDATE = "R ridge"                # amendment C (before the final test): best development IC
MARKET = "nasdaq"
NAMES_FILE = os.path.join("private", "data", "us", "assets.json")
V7_OUT = os.path.join("private", "reports", "v7")


def use_market(market):
    """Point every path, period and cost at one market. NASDAQ is the default; SGX follows docs/FORECAST-SGX-PROTOCOL.md."""
    global MARKET, GPU_DIR, OUT, TRAIN_START, DEV_START, DEV_END, TEST_START, COST, NAMES_FILE, V7_OUT, MLP_DIR, MODELS
    MARKET = market
    if market == "sgx":
        GPU_DIR = os.path.join("private", "data", "sgx", "gpu")
        OUT = os.path.join("private", "reports", "sgx", "v6")
        TRAIN_START, DEV_START, DEV_END, TEST_START = "2003-01-02", "2016-01-04", "2022-12-30", "2023-01-03"
        COST = 0.0030
        NAMES_FILE = os.path.join("private", "data", "sgx", "assets.json")
        V7_OUT = os.path.join("private", "reports", "sgx", "v7")
        MODELS = os.path.join("private", "models", "sgx")
    else:                                                    # back to the NASDAQ defaults (also after an SGX step)
        GPU_DIR = os.path.join("private", "data", "us", "gpu")
        OUT = os.path.join("private", "reports", "v6")
        TRAIN_START, DEV_START, DEV_END, TEST_START = _NASDAQ_PERIODS
        COST = _NASDAQ_COST
        NAMES_FILE = os.path.join("private", "data", "us", "assets.json")
        V7_OUT = os.path.join("private", "reports", "v7")
        MODELS = os.path.join("private", "models", "v6")
    MLP_DIR = os.path.join(OUT, "mlp")
BORROW_HARD = 0.10                 # amendment A: realistic borrow for the speculative short leg


class Panel:
    """Universe stock-days as flat rows (day index, stock index), with features and targets."""

    def __init__(self, d=None):
        d = d or GPU_DIR
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


def evaluate(P, pred, d0, d1, label, series=False):
    """IC, decile spread and the weekly long-short / long-only portfolios over days d0..d1 (inclusive).
    series=True also returns the daily IC and weekly returns (for the dashboard)."""
    ics, spreads, ic_days, wk_days = [], [], [], []
    for i in range(d0, d1 + 1):
        s, e = P.day_start[i], P.day_start[i + 1]
        if e - s < 50:
            continue
        p, y = pred[s:e], P.y[s:e]
        ic = spearman(p, y)
        if np.isfinite(ic):
            ics.append(ic)
            ic_days.append(P.dates[i])
            ok = np.isfinite(p) & np.isfinite(y)
            q = np.quantile(p[ok], [0.1, 0.9])
            spreads.append(y[ok][p[ok] >= q[1]].mean() - y[ok][p[ok] <= q[0]].mean())
    # weekly portfolio
    ls, ls_hard, lo, hits, turns, legs_l, legs_s = [], [], [], [], [], [], []
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
        ls_hard.append(rl - rs - cost - BORROW_HARD / 52)
        legs_l.append(rl)
        legs_s.append(rs)
        mret = np.expm1(P.market[min(i + H, P.T - 1)] - P.market[i])
        lo.append(rl - mret - COST * 2 * turn_l)
        hits.append(rl > rs)
        turns.append((turn_l + turn_s) / 2)
        prev_l, prev_s = L, S
        wk_days.append(P.dates[i])
    ls, lo, ls_hard = np.array(ls), np.array(lo), np.array(ls_hard)
    eq = np.cumsum(ls)
    res = {
        "model": label, "days": len(ics),
        "ic": float(np.mean(ics)), "ic_t_nw": nw_t(ics), "ic_pos_share": float(np.mean(np.array(ics) > 0)),
        "decile_spread_5d": float(np.mean(spreads)),
        "ls_net_ann": float(ls.mean() * 52), "ls_sharpe": float(ls.mean() / ls.std() * np.sqrt(52)) if ls.std() > 0 else float("nan"),
        "ls_maxdd": float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else float("nan"),
        "weeks": len(ls), "top_beats_bottom": float(np.mean(hits)), "turnover": float(np.mean(turns)),
        "ls_net_ann_borrow10": float(ls_hard.mean() * 52),
        "ls_sharpe_borrow10": float(ls_hard.mean() / ls_hard.std() * np.sqrt(52)) if ls_hard.std() > 0 else float("nan"),
        "long_leg_ann": float(np.mean(legs_l) * 52), "short_leg_ann": float(np.mean(legs_s) * 52),
        "longonly_vs_qqq_ann": float(lo.mean() * 52), "longonly_ir": float(lo.mean() / lo.std() * np.sqrt(52)) if lo.std() > 0 else float("nan"),
    }
    if series:
        res["series"] = {"ic": [[d, float(x)] for d, x in zip(ic_days, ics)],
                         "weeks": [[d, float(a), float(b), float(c)] for d, a, b, c in zip(wk_days, ls, ls_hard, lo)]}
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


# ---------------------------------------------------------------- nonlinear models (G, M) and ensemble (E)

LGB_PARAMS = dict(objective="regression", learning_rate=0.2, max_depth=8, num_leaves=210, bagging_fraction=0.8789, bagging_freq=1,
                  feature_fraction=0.8879, lambda_l1=205.6999, lambda_l2=580.9768, n_estimators=1000, verbose=-1)
VAL_DAYS = 252
MLP_DIR = os.path.join(OUT, "mlp")


def expanding_z(x, minp=250):
    """z-score of x[t] against x[..t] only (no look-ahead)."""
    out = np.full(len(x), np.nan)
    s = s2 = 0.0
    c = 0
    for t, v in enumerate(x):
        if np.isfinite(v):
            s += v
            s2 += v * v
            c += 1
        if c >= minp and np.isfinite(v):
            m = s / c
            out[t] = (v - m) / np.sqrt(max(s2 / c - m * m, 1e-12))
    return out


def regime(P):
    """Market-wide inputs given to the nonlinear models: QQQ 21-day return, QQQ 20-day vol, 10-year yield 21-day change."""
    m = P.market
    r1 = np.diff(m, prepend=np.nan)
    ret21 = m - np.concatenate([np.full(21, np.nan), m[:-21]])
    vol20 = np.array([np.nanstd(r1[max(0, t - 19):t + 1]) if t >= 20 else np.nan for t in range(P.T)])
    y = json.load(open(os.path.join("private", "data", "daily_extra", "yields.json")))
    ym = dict(zip(y["dates"], y["dgs10"]))
    y10 = np.array([ym.get(d, np.nan) for d in P.dates], float)
    for t in range(1, P.T):                                                # carry the last known yield
        if not np.isfinite(y10[t]):
            y10[t] = y10[t - 1]
    d21 = y10 - np.concatenate([np.full(21, np.nan), y10[:-21]])
    R = np.stack([expanding_z(ret21), expanding_z(vol20), expanding_z(d21)], 1)
    return np.nan_to_num(R).astype(np.float32)


def schedule(P, d0, d1):
    """Per monthly refit: refit day, train rows [a, b), validation rows (the last 12 months) and prediction rows."""
    t0 = P.idx(TRAIN_START)
    refits = month_starts(P, d0, d1)
    out = []
    for k, r in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else d1 + 1
        end = max(r - H, t0)                                               # labels end before the refit day
        vstart = max(end - VAL_DAYS, t0)
        out.append({"refit": int(r), "train": [int(P.day_start[t0]), int(P.day_start[max(vstart - H, t0)])],
                    "val": [int(P.day_start[vstart]), int(P.day_start[end])], "pred": [int(P.day_start[r]), int(P.day_start[nxt])]})
    return out


def lgb_walk_forward(P, X, d0, d1, log=print):
    import lightgbm as lgb
    pred = np.full(len(P.t), np.nan)
    for s in schedule(P, d0, d1):
        tr, va = slice(*s["train"]), slice(*s["val"])
        ok_tr, ok_va = np.isfinite(P.yrank[tr]), np.isfinite(P.yrank[va])
        dtr = lgb.Dataset(X[tr][ok_tr], P.yrank[tr][ok_tr], free_raw_data=True)
        dva = lgb.Dataset(X[va][ok_va], P.yrank[va][ok_va], reference=dtr)
        params = {**LGB_PARAMS, "num_threads": max(1, (os.cpu_count() or 4) - 2), "seed": 0}
        rounds = params.pop("n_estimators")
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred[slice(*s["pred"])] = m.predict(X[slice(*s["pred"])], num_iteration=m.best_iteration)
        if log:
            log(f"    G refit {P.dates[s['refit']]}: {ok_tr.sum():,} train / {ok_va.sum():,} val rows, {m.best_iteration} trees", flush=True)
    return pred


def xs_rank_pred(P, pred):
    def f(x):
        ok = np.isfinite(x)
        r = np.full(len(x), np.nan)
        if ok.sum() >= 2:
            r[ok] = np.argsort(np.argsort(x[ok])) / (ok.sum() - 1) - 0.5
        return r
    return P.per_day(pred, f)


def period_bounds(P, period):
    if period == "dev":
        return P.idx(DEV_START), P.idx(DEV_END + "~") - 1
    return P.idx(TEST_START), P.T - 1 - H


def mlp_export(args):
    """Inputs for the GPU MLP trainer (trademaster gpu-mlp): X [R, F] float32, y [R] float32, schedule_<period>.json."""
    P = Panel()
    d0, d1 = period_bounds(P, args.period)
    X = np.hstack([P.X, regime(P)[P.t]]).astype(np.float32)
    os.makedirs(MLP_DIR, exist_ok=True)
    X.tofile(os.path.join(MLP_DIR, "X.f32"))
    P.yrank.astype(np.float32).tofile(os.path.join(MLP_DIR, "y.f32"))
    sch = schedule(P, d0, d1)
    json.dump({"rows": int(X.shape[0]), "features": int(X.shape[1]), "period": args.period, "refits": sch},
              open(os.path.join(MLP_DIR, f"schedule_{args.period}.json"), "w"))
    print(f"exported {X.shape[0]:,} x {X.shape[1]} for the {args.period} MLP ({len(sch)} refits) to {MLP_DIR}", flush=True)


def models(args):
    """All protocol models on a period. The final test (--period test) runs once: it refuses if results exist."""
    t0 = time.time()
    path = os.path.join(OUT, f"{args.period}_models.json")
    if args.period == "test" and os.path.exists(path):
        raise SystemExit(f"{path} exists: the final test runs once (protocol)")
    P = Panel()
    d0, d1 = period_bounds(P, args.period)
    print(f"{args.period}: {P.dates[d0]} .. {P.dates[d1]}, {len(P.t):,} universe stock-days", flush=True)
    fi = {n: i for i, n in enumerate(P.names)}
    X = np.hstack([P.X, regime(P)[P.t]]).astype(np.float32)
    preds = {"B1 momentum 12-1": P.X[:, fi["mom12_1"]].astype(float), "B2 reversal 1w": -P.X[:, fi["mom5"]].astype(float)}
    print("  R ridge", flush=True)
    preds["R ridge"] = walk_forward(P, d0, d1, ridge(), log=None)
    gfile = os.path.join(OUT, f"pred_G_{args.period}.npy")
    if os.path.exists(gfile) and not args.refit:
        preds["G LightGBM"] = np.load(gfile)
    else:
        print("  G LightGBM", flush=True)
        preds["G LightGBM"] = lgb_walk_forward(P, X, d0, d1)
        os.makedirs(OUT, exist_ok=True)
        np.save(gfile, preds["G LightGBM"])
    mfile = os.path.join(MLP_DIR, f"pred_M_{args.period}.f32")
    if os.path.exists(mfile):
        preds["M MLP (GPU)"] = np.fromfile(mfile, np.float32).astype(float)
        preds["E ensemble G+M"] = xs_rank_pred(P, preds["G LightGBM"]) + xs_rank_pred(P, preds["M MLP (GPU)"])
    else:
        print(f"  (no {mfile}: M and E skipped; run trademaster gpu-mlp first)", flush=True)
    res = [evaluate(P, p, d0, d1, k) for k, p in preds.items()]
    out = {"period": [P.dates[d0], P.dates[d1]], "results": res, "minutes": (time.time() - t0) / 60}
    if args.period == "test":
        e = next(x for x in res if x["model"].startswith(CANDIDATE))         # amendment C: candidate = R
        base = max(x["ic"] for x in res if x["model"][:2] in ("B1", "B2"))
        out["adoption"] = {"candidate": e["model"], "ic_gt_0.02_t_gt_2": bool(e["ic"] > 0.02 and e["ic_t_nw"] > 2), "beats_best_baseline": bool(e["ic"] > base),
                           "net_sharpe_gt_0.5": bool(e["ls_sharpe"] > 0.5)}
        out["adoption"]["adopted"] = all(v for k, v in out["adoption"].items() if k != "candidate")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    for x in res:
        print(f"  {x['model']:<20} IC {x['ic']:+.4f} (t {x['ic_t_nw']:+.1f})  L/S Sharpe {x['ls_sharpe']:+.2f} "
              f"[10% borrow {x['ls_sharpe_borrow10']:+.2f}]  legs {x['long_leg_ann']:+.0%}/{x['short_leg_ann']:+.0%}  "
              f"top>bottom {x['top_beats_bottom']:.0%}  long-only vs QQQ {x['longonly_vs_qqq_ann']:+.1%}/yr (IR {x['longonly_ir']:+.2f})",
              flush=True)
    if "adoption" in out:
        print(f"adoption: {out['adoption']}", flush=True)
    print(f"wrote {path} ({out['minutes']:.1f} min)", flush=True)


# ---------------------------------------------------------------- daily rankings (dashboard)

MODELS = os.path.join("private", "models", "v6")


def latest_schedule(P):
    """One refit 'today': training labels end on the last panel day; prediction rows = the last day."""
    t0, r = P.idx(TRAIN_START), P.T
    end = r - H
    vstart = max(end - VAL_DAYS, t0)
    return {"refit": r - 1, "train": [int(P.day_start[t0]), int(P.day_start[max(vstart - H, t0)])],
            "val": [int(P.day_start[vstart]), int(P.day_start[end])], "pred": [int(P.day_start[r - 1]), int(P.day_start[r])]}


def mlp_predict_cpu(X, path):
    """Seed-averaged MLP output from weights saved by `trademaster gpu-mlp --period latest` (same layout as mlp.hlsl)."""
    meta = json.load(open(path + ".json"))
    F, Hd, S = meta["F"], meta["H"], meta["S"]
    W = np.fromfile(path, np.float32).reshape(S, -1).astype(np.float64)
    out = np.zeros(len(X))
    for s in range(S):
        w = W[s]
        o = 0
        W1 = w[o:o + F * Hd].reshape(F, Hd); o += F * Hd
        b1 = w[o:o + Hd]; o += Hd
        W2 = w[o:o + Hd * Hd].reshape(Hd, Hd); o += Hd * Hd
        b2 = w[o:o + Hd]; o += Hd
        W3 = w[o:o + Hd]; o += Hd
        b3 = w[o]
        h = np.maximum(X @ W1 + b1, 0)
        h = np.maximum(h @ W2 + b2, 0)
        out += h @ W3 + b3
    return out / S


def calibration(P, score, periods):
    """Out-of-sample decile table: mean relative 5-day return and share beating the median, by score decile."""
    ys = [[] for _ in range(10)]
    for d0, d1 in periods:
        for i in range(d0, d1 + 1):
            s, e = P.day_start[i], P.day_start[i + 1]
            p, y = score[s:e], P.y[s:e]
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() < 50:
                continue
            dec = np.minimum((np.argsort(np.argsort(p[ok])) * 10) // ok.sum(), 9)
            for k in range(10):
                ys[k].append(y[ok][dec == k])
    out = []
    for k in range(10):
        y = np.concatenate(ys[k]) if ys[k] else np.array([np.nan])
        out.append({"decile": k + 1, "mean_rel_5d": float(np.expm1(np.nanmean(y))), "p_beat_median": float(np.nanmean(y > 0)), "n": int(len(y))})
    return out


def rank(args):
    """Today's rankings with the adopted model (R, ridge: final test 2026-10-18), refit on all labelled rows."""
    t0 = time.time()
    P = Panel()
    last = P.T - 1
    sch = latest_schedule(P)
    tr = slice(sch["train"][0], sch["val"][1])                          # ridge has no early stopping: train + val
    ok = np.isfinite(P.yrank[tr])
    model = ridge()(P.X[tr][ok], P.yrank[tr][ok])
    rows = slice(*sch["pred"])
    score, label = model(P.X[rows]), "R: ridge"
    # out-of-sample walk-forward history of the same model (2021 -> latest), for calibration, IC and portfolio curves
    periods = [period_bounds(P, "dev"), period_bounds(P, "test")]
    hist = walk_forward(P, periods[0][0], periods[1][1], ridge(), log=None)
    cal = calibration(P, hist, periods)
    ev = evaluate(P, hist, periods[1][0], periods[1][1], label, series=True)
    # today's table
    names = {a["symbol"]: a.get("name") or "" for a in json.load(open(NAMES_FILE))}
    n_today = P.n[rows]
    pct = np.argsort(np.argsort(score)) / max(len(score) - 1, 1)
    dec = np.minimum((pct * 10).astype(int), 9)
    fi = {n: i for i, n in enumerate(P.names)}
    why = [k for k in ("mom12_1", "resid_mom12_1", "dist_252_high", "vol60", "max_ret21", "earnings_yield", "gross_profitability",
                       "sue", "ind_mom126") if k in fi]
    table = []
    for k, j in enumerate(n_today):
        x = P.X[rows][k]
        table.append({"symbol": P.symbols[j], "name": names.get(P.symbols[j], ""), "close": float(np.exp(P.lc[last, j])),
                      "score_pct": float(pct[k] * 100), "decile": int(dec[k]) + 1,
                      "exp_rel_5d": cal[dec[k]]["mean_rel_5d"] if cal else None, "p_beat": cal[dec[k]]["p_beat_median"] if cal else None,
                      "why": {w: float(x[fi[w]] + 0.5) for w in why}})
    table.sort(key=lambda r: -r["score_pct"])
    # v7 price forecasts (shadow), when written for the same close
    v7 = None
    fp = os.path.join(V7_OUT, "forecast.json")
    if os.path.exists(fp):
        f = json.load(open(fp))
        if f.get("asof") == P.dates[last]:
            for r in table:
                r["fc"] = f["rows"].get(r["symbol"])
            sp = os.path.join(V7_OUT, "shadow_score.json")
            v7 = {"model": f["model"], "status": f["status"], "shadow": json.load(open(sp)) if os.path.exists(sp) else None,
                  "news": f.get("news"), "horizons": {}}
            # v7 amendment A: the same model at 1 and 20 trading days, shown only if its pre-registered gate passed
            for h in (1, 20):
                fh, gh = os.path.join(V7_OUT, f"h{h}", "forecast.json"), os.path.join(V7_OUT, f"h{h}", "gate.json")
                if os.path.exists(fh) and os.path.exists(gh):
                    fj, gj = json.load(open(fh)), json.load(open(gh))
                    if fj.get("asof") == P.dates[last] and gj.get("pass"):            # a horizon that failed its gate is not shown
                        for r in table:
                            r[f"fc{h}"] = fj["rows"].get(r["symbol"])
                        v7["horizons"][str(h)] = {"model": fj["model"], "gate": gj}
            if v7["shadow"]:
                v7["shadow"].pop("days", None)
                if isinstance(v7["shadow"].get("news"), dict):
                    v7["shadow"]["news"].pop("days", None)
    status = {"adopted": None, "note": "final test not run yet"}
    tp = os.path.join(OUT, "test_models.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        status = {"adopted": t.get("adoption", {}).get("adopted"), "note": "per the pre-registered final test", "adoption": t.get("adoption")}
    out = {"asof": P.dates[last], "model": label, "status": status, "universe": len(table), "calibration": cal, "rankings": table,
           "history": ev, "market": MARKET,
           "dev": (json.load(open(os.path.join(OUT, "dev_models.json")))["results"] if os.path.exists(os.path.join(OUT, "dev_models.json"))
                   else json.load(open(os.path.join(OUT, "dev.json")))["results"] if os.path.exists(os.path.join(OUT, "dev.json")) else None),
           "test": json.load(open(tp))["results"] if os.path.exists(tp) else None, "features": P.names, "v7": v7,
           "generated": datetime.now().isoformat(timespec="seconds"), "seconds": time.time() - t0}
    tmp = os.path.join(OUT, "rankings.json.tmp")
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, os.path.join(OUT, "rankings.json"))
    print(f"rankings {P.dates[last]}: {len(table)} stocks, model {label}; top: {', '.join(r['symbol'] for r in table[:10])}", flush=True)


def mlp_export_latest():
    P = Panel()
    X = np.hstack([P.X, regime(P)[P.t]]).astype(np.float32)
    os.makedirs(MLP_DIR, exist_ok=True)
    X.tofile(os.path.join(MLP_DIR, "X.f32"))
    P.yrank.astype(np.float32).tofile(os.path.join(MLP_DIR, "y.f32"))
    json.dump({"rows": int(X.shape[0]), "features": int(X.shape[1]), "period": "latest", "asof": P.dates[-1],
               "refits": [latest_schedule(P)]}, open(os.path.join(MLP_DIR, "schedule_latest.json"), "w"))
    print(f"exported the latest MLP fit ({P.dates[-1]}) to {MLP_DIR}", flush=True)


def test_r(args):
    """Final test of the baselines and the ridge candidate only (the SGX protocol's candidate). Runs once."""
    t0 = time.time()
    path = os.path.join(OUT, "test_models.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: the final test runs once (protocol)")
    P = Panel()
    d0, d1 = period_bounds(P, "test")
    fi = {n: i for i, n in enumerate(P.names)}
    preds = {"B1 momentum 12-1": P.X[:, fi["mom12_1"]].astype(float), "B2 reversal 1w": -P.X[:, fi["mom5"]].astype(float),
             "R ridge": walk_forward(P, d0, d1, ridge(), log=None)}
    res = [evaluate(P, p, d0, d1, k) for k, p in preds.items()]
    e = res[2]
    base = max(res[0]["ic"], res[1]["ic"])
    adoption = {"candidate": e["model"], "ic_gt_0.02_t_gt_2": bool(e["ic"] > 0.02 and e["ic_t_nw"] > 2),
                "beats_best_baseline": bool(e["ic"] > base), "net_sharpe_gt_0.5": bool(e["ls_sharpe"] > 0.5)}
    adoption["adopted"] = all(v for k, v in adoption.items() if k != "candidate")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"market": MARKET, "period": [P.dates[d0], P.dates[d1]], "results": res, "adoption": adoption,
               "minutes": (time.time() - t0) / 60}, open(path, "w"), indent=1)
    for x in res:
        print(f"  {x['model']:<20} IC {x['ic']:+.4f} (t {x['ic_t_nw']:+.1f})  top>bottom {x['top_beats_bottom']:.0%}  L/S Sharpe {x['ls_sharpe']:+.2f} "
              f"[10% borrow {x['ls_sharpe_borrow10']:+.2f}]  long-only vs market {x['longonly_vs_qqq_ann']:+.1%}/yr (IR {x['longonly_ir']:+.2f})", flush=True)
    print(f"adoption: {adoption}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "models", "mlp-export", "rank", "test-r"])
    ap.add_argument("--period", choices=["dev", "test", "latest"], default="dev")
    ap.add_argument("--refit", action="store_true", help="retrain G even if saved predictions exist")
    ap.add_argument("--market", choices=["nasdaq", "sgx"], default="nasdaq")
    a = ap.parse_args()
    use_market(a.market)
    if a.cmd == "mlp-export" and a.period == "latest":
        return mlp_export_latest()
    if a.cmd == "models" and a.period == "latest":
        raise SystemExit("models runs on dev or test")
    {"dev": dev, "models": models, "mlp-export": mlp_export, "rank": rank, "test-r": test_r}[a.cmd](a)


if __name__ == "__main__":
    main()
