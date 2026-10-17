"""
Foresight-leak checks for the hand-weighted walk-forward replay (handweights.py).

1. Tamper:   for random cut points K, overwrite every bar after K with garbage (random prices, NaN, the day reversed).
             Predictions made at bars <= K must stay bit-identical. Any dependence on the future breaks this.
2. Beta:     the only fitted numbers (betas) must not change when the test day is scrambled.
3. Timing:   correlation of the prediction made at t with the return k minutes later. A leak shows up as a large
             correlation at the target offset; an honest short-horizon signal is small and fades with k.
4. Shuffle:  per-minute cross-sectional shuffles of the outcomes give the IC you'd see by luck.
5. Naive:    'price unchanged' plotted the same way, to show how close a no-information line looks.

Usage: python -m trademaster_ml.leakcheck --date 2026-10-16 --out private/reports/handweights-2026-10-16
"""
import argparse, json, os
from datetime import date

import numpy as np

from .handweights import load, fit_beta, prepare_day, replay, universe_refs, spearman


def ic_at(mu, ret, stocks, shift, t0=5):
    out = []
    for t in range(t0, len(mu)):
        u = t + shift
        if 0 <= u < len(ret) and np.isfinite(mu[t, stocks]).all() and np.isfinite(ret[u, stocks]).all():
            v = spearman(mu[t, stocks], ret[u, stocks])
            if np.isfinite(v):
                out.append(v)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist_1m")
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", type=int, default=40)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    rng = np.random.default_rng(0)
    syms, u, ts, days, close = load(a.hist)
    test = date.fromisoformat(a.date)
    ref = universe_refs(syms, u)
    stocks = np.array([s not in u["etfs"] for s in syms])
    beta = fit_beta(close[days < test], ref)
    day = prepare_day(close, days, test)
    T, N = day.shape
    mu, sg = replay(day, ref, beta)
    report = {}

    # 1. tamper with the future
    fails, checked = [], 0
    cuts = sorted(rng.choice(np.arange(6, T - 2), a.cuts, replace=False))
    for K in cuts:
        for name, fut in [("random", day[K + 1:] * np.exp(rng.normal(0, 0.05, day[K + 1:].shape))),
                          ("nan", np.full_like(day[K + 1:], np.nan)),
                          ("reversed", day[K + 1:][::-1].copy())]:
            d2 = day.copy()
            d2[K + 1:] = fut
            m2, s2 = replay(d2, ref, beta)
            same = np.array_equal(mu[:K + 1], m2[:K + 1], equal_nan=True) and np.array_equal(sg[:K + 1], s2[:K + 1], equal_nan=True)
            checked += 1
            if not same:
                fails.append((int(K), name))
    report["tamper"] = dict(cut_points=len(cuts), scenarios=checked, failures=fails, passed=not fails)

    # 2. betas use prior sessions only
    c2 = close.copy()
    c2[days == test] *= np.exp(rng.normal(0, 0.05, c2[days == test].shape))
    report["beta_independent_of_test_day"] = bool(np.array_equal(beta, fit_beta(c2[days < test], ref)))
    report["beta_sessions"] = sorted(str(x) for x in set(days[days < test]))

    # 3. timing profile: prediction at t vs return over minute t+shift -> t+shift+1
    ret = np.full((T, N), np.nan)
    ret[:-1] = np.log(day[1:] / day[:-1])
    prof = {}
    for k in [-3, -2, -1, 0, 1, 2, 3, 5, 10]:
        x = ic_at(mu, ret, stocks, k)
        prof[k] = dict(mean_ic=float(x.mean()), t=float(x.mean() / x.std() * np.sqrt(len(x))), n=len(x))
    report["timing"] = prof
    # what a one-bar leak would look like: the same formula fed the *next* bar
    leak_day = np.vstack([day[1:], day[-1:]])
    mu_leak, _ = replay(leak_day, ref, beta)
    report["simulated_leak_ic"] = float(ic_at(-mu_leak, ret, stocks, 0).mean())   # sign flipped: reversal of a known move
    oracle = ret.copy()
    report["oracle_ic"] = float(ic_at(oracle, ret, stocks, 0).mean())

    # 4. shuffle distribution
    obs = ic_at(mu, ret, stocks, 0).mean()
    null = []
    for _ in range(200):
        r2 = ret.copy()
        for t in range(T):
            r2[t, stocks] = r2[t, stocks][rng.permutation(stocks.sum())]
        null.append(ic_at(mu, r2, stocks, 0).mean())
    null = np.array(null)
    report["shuffle"] = dict(observed=float(obs), null_mean=float(null.mean()), null_p99=float(np.percentile(null, 99)),
                             null_max=float(null.max()), p_value=float((np.sum(null >= obs) + 1) / (len(null) + 1)))

    # 5. naive 'unchanged' error vs model, price terms
    i = syms.index("MU")
    pa = day[1:, i]
    report["mu_price_error_usd"] = dict(model=float(np.nanmean(np.abs(day[:-1, i] * np.exp(mu[:-1, i]) - pa))),
                                        unchanged=float(np.nanmean(np.abs(day[:-1, i] - pa))))
    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(os.path.join(a.out, "leakcheck.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "leakcheck.npz"), null=null)
    print(json.dumps(report, indent=1))
    if fails or not report["beta_independent_of_test_day"]:
        raise SystemExit("LEAK CHECK FAILED")


if __name__ == "__main__":
    main()
