"""
Fit on one week of 1-minute bars, then predict the next 5 trading days out of sample (weights frozen).

Features per stock at minute t (bars <= t of the same session only; first 5 minutes skipped):
  lead   = beta * r_etf(t)                 sector move the stock may still catch up to
  rev1   = e(t)                             last minute's stock-specific move (e = r - beta * r_etf)
  rev5   = e(t-1) + ... + e(t-4)            older stock-specific moves
  volrev = e(t) * volume surprise(t)        heavy-volume moves may continue rather than reverse
  open   = e(t) * [first 30 minutes]        the open behaves differently
Target: next-minute return minus the cross-sectional mean (bps). Fit: ridge regression, closed form, CPU.
The hand-weighted model (handweights.py) is scored on the same days for comparison.

Usage: python -m trademaster_ml.weektest --train 2026-10-05:2026-10-09 --test 2026-10-12:2026-10-16 --out private/reports/weektest-1012
"""
import argparse, json, os
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np

from .handweights import W_LEADLAG, W_REV1, W_REV5, BETA_SHRINK, spearman

NY = ZoneInfo("America/New_York")
FEATS = ["lead", "rev1", "rev5", "volrev", "open"]
HAND = np.array([W_LEADLAG, -W_REV1, -W_REV5, 0.0, 0.0])
SKIP = 5
RIDGE = 1.0


def load(hist):
    u = json.load(open(os.path.join(hist, "universe.json"), encoding="utf-8"))
    syms = u["stocks"] + u["etfs"]
    data = {s: json.load(open(os.path.join(hist, f"{s}.json"), encoding="utf-8")) for s in syms}
    ts = sorted(set().union(*[set(d["ts"]) for d in data.values()]))
    pos = {t: k for k, t in enumerate(ts)}
    close = np.full((len(ts), len(syms)), np.nan)
    vol = np.zeros((len(ts), len(syms)))
    for j, s in enumerate(syms):
        idx = [pos[t] for t in data[s]["ts"]]
        close[idx, j] = data[s]["close"]
        vol[idx, j] = data[s]["volume"]
    days = np.array([datetime.fromtimestamp(t, NY).date() for t in ts])
    ref = np.array([syms.index(u["reference_etf"][s]) if s in u["reference_etf"] else j for j, s in enumerate(syms)])
    stocks = np.array([s not in u["etfs"] for s in syms])
    return syms, np.array(ts), days, close, vol, ref, stocks


def session(close, vol, days, d):
    c, v = close[days == d].copy(), vol[days == d].copy()
    for t in range(1, len(c)):                      # forward-fill from earlier bars only
        m = ~np.isfinite(c[t])
        c[t, m] = c[t - 1, m]
    return c, v


def fit_beta(rets, ref):
    beta = np.ones(rets.shape[1])
    for i, j in enumerate(ref):
        if i != j:
            ok = np.isfinite(rets[:, i]) & np.isfinite(rets[:, j])
            x, y = rets[ok, j], rets[ok, i]
            beta[i] = BETA_SHRINK * (x @ y / (x @ x)) + (1 - BETA_SHRINK)
    return beta


def features(c, v, beta, ref):
    """[T, N, F]; row t uses rows <= t only."""
    T, N = c.shape
    r = np.zeros((T, N))
    r[1:] = np.nan_to_num(np.log(c[1:] / c[:-1]))
    e = r - beta * r[:, ref]
    rev5 = np.zeros((T, N))
    for k in range(1, 5):
        rev5[k:] += e[:-k]
    # volume surprise: log(volume / average volume of the previous up-to-30 minutes), clipped
    cs = np.cumsum(v, axis=0)
    prev = np.zeros((T, N))
    for t in range(1, T):
        lo = max(0, t - 30)
        prev[t] = (cs[t - 1] - (cs[lo - 1] if lo > 0 else 0)) / (t - lo)
    vs = np.clip(np.log((v + 1) / (prev + 1)), -3, 3)
    opening = (np.arange(T) < 30).astype(float)[:, None]
    X = np.stack([beta * r[:, ref], e, rev5, e * vs, e * opening], axis=2)
    return X * 1e4


def target(c):
    y = np.full(c.shape, np.nan)
    y[:-1] = np.log(c[1:] / c[:-1]) * 1e4
    return y


def demean(a, stocks):
    a = a[:, stocks]
    return a - np.nanmean(a, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist_1m_30d")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    syms, ts, days, close, vol, ref, stocks = load(a.hist)
    alld = sorted(set(days))
    rng_ = lambda s: [d for d in alld if date.fromisoformat(s.split(":")[0]) <= d <= date.fromisoformat(s.split(":")[1])]
    train_days, test_days = rng_(a.train), rng_(a.test)
    assert max(train_days) < min(test_days), "train must end before test starts"

    # --- fit on the training week only
    sess = {d: session(close, vol, days, d) for d in train_days + test_days}
    rets = np.vstack([np.log(sess[d][0][1:] / sess[d][0][:-1]) for d in train_days])
    beta = fit_beta(rets, ref)
    Xs, ys = [], []
    for d in train_days:
        c, v = sess[d]
        X, y = features(c, v, beta, ref), target(c)
        Xd = np.stack([demean(X[:, :, f], stocks) for f in range(len(FEATS))], axis=2)[SKIP:-1]
        yd = demean(y, stocks)[SKIP:-1]
        Xs.append(Xd.reshape(-1, len(FEATS))); ys.append(yd.reshape(-1))
    X, y = np.vstack(Xs), np.concatenate(ys)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    lim = 5 * y.std()
    y = np.clip(y, -lim, lim)                          # news spikes shouldn't drive the weights
    sd = X.std(0)
    Z = X / sd
    w = np.linalg.solve(Z.T @ Z + RIDGE * len(Z) * 1e-4 * np.eye(len(FEATS)), Z.T @ y) / sd

    # --- predict the next 5 days, frozen weights, and check causality while we're at it
    rng = np.random.default_rng(1)
    per_day, tamper_fail, ls_curve = [], 0, {"fitted": [], "hand": []}
    deciles = {}
    for d in test_days:
        c, v = sess[d]
        F = features(c, v, beta, ref)
        for K in rng.choice(np.arange(SKIP, len(c) - 2), 5, replace=False):     # tamper the future
            c2, v2 = c.copy(), v.copy()
            c2[K + 1:] *= np.exp(rng.normal(0, 0.05, c2[K + 1:].shape)); v2[K + 1:] = rng.integers(0, 10 ** 6, v2[K + 1:].shape)
            if not np.array_equal(features(c2, v2, beta, ref)[:K + 1], F[:K + 1]):
                tamper_fail += 1
        y = target(c)
        row = {"date": str(d)}
        for name, wv in [("fitted", w), ("hand", HAND)]:
            mu = (F @ wv)[:, stocks]
            act = y[:, stocks]
            ics, ls = [], []
            for t in range(SKIP, len(c) - 1):
                k = np.isfinite(act[t]) & np.isfinite(mu[t])
                if k.sum() >= 40:
                    pt, at = mu[t, k], act[t, k]
                    ics.append(spearman(pt, at))
                    o = np.argsort(pt)
                    ls.append(at[o[-10:]].mean() - at[o[:10]].mean())   # top-10 minus bottom-10, gross bps
            ics = np.array(ics)
            m = (np.arange(len(c))[:, None] >= SKIP) & np.isfinite(act) & np.isfinite(mu) & (act != 0)
            row[name] = dict(ic=float(np.nanmean(ics)), ic_t=float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics))),
                             hit=float(np.mean(np.sign(mu[m]) == np.sign(act[m]))), ls_bps_per_min=float(np.mean(ls)))
            ls_curve[name] += ls
            if name == "fitted":
                pm, pa = mu[m], act[m]
                q = np.percentile(pm, np.linspace(0, 100, 11))
                idx = np.clip(np.digitize(pm, q[1:-1]), 0, 9)
                deciles[str(d)] = dict(pred=[float(pm[idx == k].mean()) for k in range(10)],
                                       actual=[float(pa[idx == k].mean()) for k in range(10)],
                                       se=[float(pa[idx == k].std() / np.sqrt((idx == k).sum())) for k in range(10)])
        per_day.append(row)
        print(row)
    out = dict(train=[str(x) for x in train_days], test=[str(x) for x in test_days], features=FEATS,
               weights_fitted=dict(zip(FEATS, map(float, w))), weights_hand=dict(zip(FEATS, map(float, HAND))),
               tamper_checks=5 * len(test_days), tamper_failures=tamper_fail, per_day=per_day, deciles=deciles,
               ls_curve={k: list(map(float, v)) for k, v in ls_curve.items()})
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "weektest.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["weights_fitted", "tamper_checks", "tamper_failures"]}, indent=1))
    if tamper_fail:
        raise SystemExit("LEAK CHECK FAILED")


if __name__ == "__main__":
    main()
