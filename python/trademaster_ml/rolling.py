"""
Rolling week test over every week in the 1-minute history, at several prediction horizons, with costs.

For each consecutive pair of weeks: fit on week k (betas + ridge weights), freeze, predict week k+1.
For each horizon h (minutes): target = log return over the next h minutes (same session), and the paper portfolio
rebalances every h minutes: long the 10 highest predictions, short the 10 lowest, flat at the close.
Costs: turnover x assumed one-way cost (bps of traded notional). We have no bid/ask quotes, so costs are scenarios.

Usage: python -m trademaster_ml.rolling --out private/reports/rolling
"""
import argparse, json, os
from collections import OrderedDict

import numpy as np

from .weektest import load, session, fit_beta, demean
from .handweights import spearman

FEATS = ["lead1", "lead5", "rev1", "rev2_5", "rev6_15", "rev16_30", "volrev", "open"]
HORIZONS = [1, 5, 10, 15, 30]
COSTS = [0.5, 1.0, 2.0]          # one-way bps scenarios
SKIP, TOP, RIDGE = 5, 10, 1e-4


def features(c, v, beta, ref):
    """[T, N, F] in bps; row t uses rows <= t of the same session only."""
    T, N = c.shape
    r = np.zeros((T, N))
    r[1:] = np.nan_to_num(np.log(c[1:] / c[:-1]))
    re = r[:, ref]
    e = r - beta * re

    def lagsum(x, a, b):          # sum of x[t-a] .. x[t-b]  (a <= b)
        out = np.zeros_like(x)
        for k in range(a, b + 1):
            out[k:] += x[:-k] if k else x
        return out

    cs = np.cumsum(v, axis=0)
    prev = np.zeros((T, N))
    for t in range(1, T):
        lo = max(0, t - 30)
        prev[t] = (cs[t - 1] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo)
    vs = np.clip(np.log((v + 1) / (prev + 1)), -3, 3)
    opening = (np.arange(T) < 30).astype(float)[:, None]
    X = np.stack([beta * re, beta * lagsum(re, 0, 4), e, lagsum(e, 1, 4), lagsum(e, 5, 14), lagsum(e, 15, 29),
                  e * vs, e * opening], axis=2)
    return X * 1e4


def target(c, h):
    y = np.full(c.shape, np.nan)
    y[:-h] = np.log(c[h:] / c[:-h]) * 1e4
    return y


def weeks_of(days):
    out = OrderedDict()
    for d in days:
        out.setdefault(d.isocalendar()[:2], []).append(d)
    return [w for w in out.values() if len(w) >= 4]          # skip partial weeks


def fit(sess, train_days, ref, stocks, h):
    rets = np.vstack([np.log(sess[d][0][1:] / sess[d][0][:-1]) for d in train_days])
    beta = fit_beta(rets, ref)
    Xs, ys = [], []
    for d in train_days:
        c, v = sess[d]
        X, y = features(c, v, beta, ref), target(c, h)
        Xd = np.stack([demean(X[:, :, f], stocks) for f in range(len(FEATS))], axis=2)[SKIP:-h]
        Xs.append(Xd.reshape(-1, len(FEATS))); ys.append(demean(y, stocks)[SKIP:-h].reshape(-1))
    X, y = np.vstack(Xs), np.concatenate(ys)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    y = np.clip(y, -5 * y.std(), 5 * y.std())
    sd = X.std(0)
    Z = X / sd
    w = np.linalg.solve(Z.T @ Z + RIDGE * len(Z) * np.eye(len(FEATS)), Z.T @ y) / sd
    return beta, w


def trade_day(mu, act_h, h):
    """Rebalance every h minutes. Returns per-rebalance gross bps, total turnover, ICs."""
    T, N = mu.shape
    wprev = np.zeros(N)
    gross, turn, ics = [], 0.0, []
    for t in range(SKIP, T - h, h):
        k = np.isfinite(mu[t]) & np.isfinite(act_h[t])
        if k.sum() < 4 * TOP:
            continue
        idx = np.where(k)[0]
        o = idx[np.argsort(mu[t, idx])]
        w = np.zeros(N)
        w[o[-TOP:]] = 1.0 / TOP
        w[o[:TOP]] = -1.0 / TOP
        turn += np.abs(w - wprev).sum()
        wprev = w
        gross.append(float(w @ np.nan_to_num(act_h[t])))
        ics.append(spearman(mu[t, idx], act_h[t, idx]))
    turn += np.abs(wprev).sum()                               # flat at the close
    return gross, turn, ics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist_1m_30d")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    syms, ts, days, close, vol, ref, stocks = load(a.hist)
    wk = weeks_of(sorted(set(days)))
    sess = {d: session(close, vol, days, d) for w in wk for d in w}
    rng = np.random.default_rng(2)
    folds, tamper, tamper_fail = [], 0, 0
    for k in range(len(wk) - 1):
        train, test = wk[k], wk[k + 1]
        assert max(train) < min(test)
        fold = dict(train=[str(train[0]), str(train[-1])], test=[str(test[0]), str(test[-1])], horizons={})
        for h in HORIZONS:
            beta, w = fit(sess, train, ref, stocks, h)
            days_out = []
            for d in test:
                c, v = sess[d]
                F = features(c, v, beta, ref)
                if h == 1:                                       # causality check, once per test day
                    for K in rng.choice(np.arange(SKIP, len(c) - 2), 4, replace=False):
                        c2, v2 = c.copy(), v.copy()
                        c2[K + 1:] *= np.exp(rng.normal(0, 0.05, c2[K + 1:].shape))
                        v2[K + 1:] = rng.integers(0, 10 ** 6, v2[K + 1:].shape)
                        tamper += 1
                        tamper_fail += not np.array_equal(features(c2, v2, beta, ref)[:K + 1], F[:K + 1])
                mu = (F @ w)[:, stocks]
                gross, turn, ics = trade_day(mu, target(c, h)[:, stocks], h)
                days_out.append(dict(date=str(d), gross_bps=float(np.sum(gross)), turnover=float(turn),
                                     ic=float(np.nanmean(ics)), rebalances=len(gross)))
            g = sum(x["gross_bps"] for x in days_out); tv = sum(x["turnover"] for x in days_out)
            fold["horizons"][str(h)] = dict(
                weights=dict(zip(FEATS, map(float, w))), days=days_out,
                ic=float(np.mean([x["ic"] for x in days_out])), gross_bps_per_day=g / len(test),
                turnover_per_day=tv / len(test), breakeven_one_way_bps=g / tv,
                net_bps_per_day={str(cst): (g - cst * tv) / len(test) for cst in COSTS})
            hh = fold["horizons"][str(h)]
            print(f"{fold['test'][0]} h={h:2d}  IC {hh['ic']:.3f}  gross/day {hh['gross_bps_per_day']:8.1f}  "
                  f"turnover/day {hh['turnover_per_day']:7.1f}  break-even {hh['breakeven_one_way_bps']:.2f} bps")
        folds.append(fold)
    out = dict(horizons=HORIZONS, costs_one_way_bps=COSTS, top=TOP, features=FEATS, folds=folds,
               tamper_checks=tamper, tamper_failures=int(tamper_fail))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "rolling.json"), "w"), indent=1)
    print(f"tamper checks {tamper}, failures {tamper_fail}")
    if tamper_fail:
        raise SystemExit("LEAK CHECK FAILED")


if __name__ == "__main__":
    main()
