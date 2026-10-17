"""
Hand-weighted cross-stock model: no training, weights set by hand from market-microstructure priors.
Walk-forward replay on one session of 1-minute bars, revealing one bar at a time (no foresight).

    mu_i(t) = W_LEADLAG * beta_i * r_etf(t)            # sector move not yet fully priced into the stock
            - W_REV1   * e_i(t)                          # last-minute idiosyncratic move partly reverses
            - W_REV5   * sum_{k=1..4} e_i(t-k)          # older idiosyncratic moves reverse more weakly
    e_i(t)  = r_i(t) - beta_i * r_etf(t)                 # residual vs the stock's sector ETF (public label, no holdings)
    sigma_i = EWMA std of the stock's own 1-min returns (half-life 20 bars), from bars <= t only

beta_i is the only fitted number: ordinary least squares on the sessions BEFORE the test day, shrunk halfway
towards 1. The test day never touches it.

Usage: python -m trademaster_ml.handweights --hist private/data/hist_1m --date 2026-10-16 --out private/reports/handweights-2026-10-16
"""
import argparse, json, math, os
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
W_LEADLAG, W_REV1, W_REV5 = 0.05, 0.10, 0.03
BETA_SHRINK = 0.5
VOL_HALFLIFE = 20


def load(hist):
    u = json.load(open(os.path.join(hist, "universe.json"), encoding="utf-8"))
    syms = u["stocks"] + u["etfs"]
    series = {}
    for s in syms:
        r = json.load(open(os.path.join(hist, f"{s}-1m.json"), encoding="utf-8"))["chart"]["result"][0]
        series[s] = dict(zip(r["timestamp"], r["indicators"]["quote"][0]["close"]))
    ts = sorted(set().union(*[set(v) for v in series.values()]))
    close = np.array([[series[s].get(t) if series[s].get(t) is not None else np.nan for s in syms] for t in ts], float)
    days = np.array([datetime.fromtimestamp(t, NY).date() for t in ts])
    return syms, u, np.array(ts), days, close


class BarFeed:
    """Reveals bars strictly one at a time. The model can only read what has been revealed."""

    def __init__(self, close):
        self._close, self.n = close, 0

    def reveal(self):
        self.n += 1
        return self._close[self.n - 1]

    def history(self):
        return self._close[: self.n]


class HandModel:
    def __init__(self, syms, ref, beta):
        self.ref, self.beta = ref, beta
        self.var = None

    def predict(self, hist):
        """hist: [t+1, N] closes revealed so far. Returns (mu, sigma) of the next 1-min log return per stock."""
        n = hist.shape[1]
        if len(hist) < 2:
            return np.full(n, np.nan), np.full(n, np.nan)
        lr = np.log(hist[-6:])
        r = np.diff(lr, axis=0)                         # up to 5 most recent 1-min returns
        r = np.nan_to_num(r)
        r_etf = r[:, self.ref]
        e = r - self.beta * r_etf
        mu = W_LEADLAG * self.beta * r_etf[-1] - W_REV1 * e[-1] - W_REV5 * e[:-1].sum(0)
        a = 1 - 0.5 ** (1 / VOL_HALFLIFE)
        self.var = r[-1] ** 2 if self.var is None else (1 - a) * self.var + a * r[-1] ** 2
        return mu, np.sqrt(np.maximum(self.var, 1e-10))


def fit_beta(close_prior, ref):
    r = np.diff(np.log(close_prior), axis=0)
    beta = np.ones(close_prior.shape[1])
    for i, j in enumerate(ref):
        if i == j:
            continue
        ok = np.isfinite(r[:, i]) & np.isfinite(r[:, j])
        x, y = r[ok, j], r[ok, i]
        beta[i] = BETA_SHRINK * (x @ y / (x @ x)) + (1 - BETA_SHRINK) * 1.0
    return beta


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def prepare_day(close, days, test):
    """Rows of the test session, with missing prints forward-filled from earlier bars only."""
    day = close[days == test].copy()
    for t in range(1, len(day)):
        m = ~np.isfinite(day[t])
        day[t, m] = day[t - 1, m]
    return day


def replay(day, ref, beta):
    """Walk forward bar by bar. Returns mu, sigma [T, N]: row t is predicted from bars 0..t only."""
    T, N = day.shape
    feed, model = BarFeed(day), HandModel(None, ref, beta)
    mu_all, sg_all = np.full((T, N), np.nan), np.full((T, N), np.nan)
    for t in range(T - 1):
        feed.reveal()
        mu_all[t], sg_all[t] = model.predict(feed.history())   # sees bars 0..t only
    return mu_all, sg_all


def universe_refs(syms, u):
    return np.array([syms.index(u["reference_etf"][s]) if s in u["reference_etf"] else syms.index(s) for s in syms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist_1m")
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    syms, u, ts, days, close = load(a.hist)
    test = date.fromisoformat(a.date)
    ref = universe_refs(syms, u)
    beta = fit_beta(close[days < test], ref)                     # prior sessions only
    day = prepare_day(close, days, test)
    T, N = day.shape
    mu_all, sg_all = replay(day, ref, beta)
    act_all = np.full((T, N), np.nan)
    act_all[:-1] = np.log(day[1:] / day[:-1])                    # scoring only; never passed to the model
    stocks = np.array([s not in u["etfs"] for s in syms])
    ok = np.isfinite(mu_all) & np.isfinite(act_all)
    ics = [spearman(mu_all[t, stocks], act_all[t, stocks]) for t in range(T - 1) if ok[t, stocks].all() and t >= 5]
    ics = np.array([x for x in ics if np.isfinite(x)])
    m = ok & stocks[None, :] & (np.arange(T)[:, None] >= 5)
    pm, pa = mu_all[m] * 1e4, act_all[m] * 1e4
    moved = pa != 0                                            # a flat minute has no direction to call
    pm, pa = pm[moved], pa[moved]
    big = np.abs(pm) > np.percentile(np.abs(pm), 80)
    rmse_model = float(np.sqrt(np.mean((pa - pm) ** 2)))
    rmse_zero = float(np.sqrt(np.mean(pa ** 2)))
    z = (act_all[m] - mu_all[m]) / sg_all[m]
    summary = dict(date=a.date, bars=int(T), stocks=int(stocks.sum()), predictions=int(m.sum()),
                   mean_ic=float(ics.mean()), ic_tstat=float(ics.mean() / ics.std() * math.sqrt(len(ics))),
                   ic_positive_share=float((ics > 0).mean()),
                   hit_rate_all=float(np.mean(np.sign(pm) == np.sign(pa))),
                   hit_rate_top20pct_conviction=float(np.mean(np.sign(pm[big]) == np.sign(pa[big]))),
                   rmse_bps_model=rmse_model, rmse_bps_zero=rmse_zero,
                   sigma_calibration_std_z=float(np.std(z)),
                   weights=dict(leadlag=W_LEADLAG, rev1=W_REV1, rev5=W_REV5, beta_shrink=BETA_SHRINK,
                                vol_halflife=VOL_HALFLIFE))
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(a.out, "preds.npz"), ts=ts[days == test], syms=np.array(syms), close=day, mu=mu_all,
             sigma=sg_all, actual=act_all, ics=ics, beta=beta)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
