"""
v7: 5-day price forecast for every ranked NASDAQ stock, under docs/FORECAST-V7-PRICE-PROTOCOL.md (research only, CPU).

  5-day log return of stock i ~ Student-t(5):  mean = beta_i * market drift + shrink * v6 tilt
                                                 sd = sqrt(beta_i^2 * QQQ GARCH 5-day variance + 5 * idio_vol_i^2)
  then symmetric recalibration pooled over stocks (the v3 method). Compared with RW (mean 0) and MKT (no tilt).

  dev       development 2021-2024; picks the tilt shrink by CRPS
  report    the 2025-26 period (already seen by v6's final test; labelled as such)
  forecast  today's forecasts for the dashboard + the forward shadow log
  score     scores matured shadow forecasts

Usage: python -m trademaster_ml.v7 dev | report | forecast | score
"""
import argparse, json, math, os, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from . import v6
from .daily_v2 import GRID, recal_levels_sym

OUT = os.path.join("private", "reports", "v7")
NU = 5
TSCALE = math.sqrt((NU - 2) / NU)                 # unit-variance Student-t
SHRINKS = (0.5, 0.75, 1.0)
CAL_DAYS = 250
DRIFT_SHRINK = 0.5
NEWS_MKT_W, NEWS_STOCK_W = 0.010, 0.0025     # v8 amendment B priors: 5-day log return per unit of market index S / news_dir
I10, I50, I90 = (int(np.argmin(np.abs(GRID - x))) for x in (0.10, 0.50, 0.90))
I025, I975 = (int(np.argmin(np.abs(GRID - x))) for x in (0.025, 0.975))


# ---------------------------------------------------------------- distribution helpers

def t_ppf(levels):
    return stats.t.ppf(levels, NU) * TSCALE


def t_cdf(z):
    return stats.t.cdf(z / TSCALE, NU)


def quantiles(m, sd, R):
    """[rows, 199] quantiles at the nominal GRID levels, given recalibrated raw levels R (199,)."""
    return m[:, None] + sd[:, None] * t_ppf(R)[None, :]


def p_le(x, m, sd, R):
    """Calibrated P(Y <= x): raw CDF mapped through the inverse of the recalibration."""
    return np.interp(t_cdf((x - m) / sd), R, GRID, left=0.0, right=1.0)


def crps_rows(q, y):
    """Pinball approximation of CRPS on the GRID, per row."""
    return 2 * np.mean(np.where(y[:, None] < q, 1 - GRID[None, :], GRID[None, :]) * np.abs(y[:, None] - q), axis=1)


# ---------------------------------------------------------------- market: drift and GJR-GARCH-t variance

def _fit_garch(args):
    key, r_pct = args
    import torch
    torch.set_num_threads(1)
    from .daily_forecast import Garch
    g = Garch().fit(np.asarray(r_pct))
    return key, {k: float(v) for k, v in g.p.items() if k in ("omega", "alpha", "gamma", "beta", "nu")}


def garch_params(P, refits):
    """{refit day index: params}, fitted on QQQ daily returns up to the refit day; cached by refit date."""
    os.makedirs(v6.V7_OUT, exist_ok=True)
    path = os.path.join(v6.V7_OUT, "garch_cache.json")                 # daily GARCH fits do not depend on the horizon
    cache = json.load(open(path)) if os.path.exists(path) else {}
    r = np.diff(P.market) * 100
    jobs = []
    for d in refits:
        key = P.dates[d]
        if key not in cache:
            x = r[:d]
            jobs.append((key, x[np.isfinite(x)][-2000:]))
    if jobs:
        print(f"  fitting QQQ GJR-GARCH-t for {len(jobs)} monthly refits (parallel)", flush=True)
        with ProcessPoolExecutor(max(1, min(len(jobs), (os.cpu_count() or 4) - 2))) as ex:
            for key, p in ex.map(_fit_garch, jobs):
                cache[key] = p
        json.dump(cache, open(path, "w"))
    return {d: cache[P.dates[d]] for d in refits}


def market_parts(P, refits, d1):
    """Per day: 5-day market drift and 5-day market variance (fractions), each day using that month's GARCH fit."""
    T = P.T
    r = np.diff(P.market, prepend=np.nan)
    ok = np.isfinite(r)
    csum, cnt = np.cumsum(np.where(ok, r, 0)), np.cumsum(ok)
    drift = DRIFT_SHRINK * H * csum / np.maximum(cnt, 1)
    v5 = np.full(T, np.nan)
    params = garch_params(P, refits)
    rp = np.where(ok, r * 100, 0.0)
    for k, d in enumerate(refits):
        end = refits[k + 1] if k + 1 < len(refits) else d1 + 1
        p = params[d]
        phi = p["alpha"] + p["gamma"] / 2 + p["beta"]
        s2 = np.nanvar(rp[1:d])
        for t in range(1, end):
            e = rp[t]
            s2 = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e * e + p["beta"] * s2   # = s2 for day t+1
            if t >= d:
                tot, ek = 0.0, s2
                for _ in range(H):
                    tot += ek
                    ek = p["omega"] + phi * ek
                v5[t] = tot / 1e4
    return drift, v5


H = v6.H


# ---------------------------------------------------------------- the forecasting engine

def raw_feature(P, name):
    meta = json.load(open(os.path.join(v6.GPU_DIR, "features.json")))
    f = meta["features"].index(name)
    raw = np.memmap(os.path.join(v6.GPU_DIR, "raw.f32"), np.float32, "r", shape=(len(meta["features"]), P.T, P.N))
    return raw[f][P.t, P.n].astype(np.float64)


def day_pct(P, a, rows_from, rows_to):
    """Cross-sectional percentile (0..1) within each day, for rows rows_from:rows_to (whole days)."""
    out = np.full(rows_to - rows_from, np.nan)
    t0, t1 = P.t[rows_from], P.t[rows_to - 1]
    for i in range(t0, t1 + 1):
        s, e = P.day_start[i], P.day_start[i + 1]
        x = a[s - rows_from:e - rows_from]
        if len(x) > 1:
            out[s - rows_from:e - rows_from] = np.argsort(np.argsort(x)) / (len(x) - 1)
    return out


class Engine:
    """Walk-forward forecasts for every universe stock-day in [c0, d1]: means for RW / MKT / v7(s), sd, decile,
    and per refit month the recalibrated levels of each model (from PITs known before the refit day)."""

    def __init__(self, P, c0, d1, shrinks=SHRINKS, log=print):
        self.P, self.c0, self.d1 = P, c0, d1
        self.refits = v6.month_starts(P, c0, d1)
        a, b = P.day_start[c0], P.day_start[d1 + 1]
        self.a, self.b = a, b
        n = b - a
        drift, v5 = market_parts(P, self.refits, d1)
        beta = raw_feature(P, "beta252")[a:b]
        idio = raw_feature(P, "idio_vol60")[a:b]
        beta = np.where(np.isfinite(beta), beta, 1.0)
        tt = P.t[a:b]
        med = np.zeros(n)
        for i in range(c0, d1 + 1):                                           # missing idio -> day's median
            s, e = P.day_start[i] - a, P.day_start[i + 1] - a
            x = idio[s:e]
            med[s:e] = np.nanmedian(x) if np.isfinite(x).any() else 0.02
        idio = np.where(np.isfinite(idio), idio, med)
        self.sd = np.sqrt(beta ** 2 * v5[tt] + H * idio ** 2)
        self.m_mkt = beta * drift[tt]
        self.mu_m, self.v_m, self.beta, self.idio = drift[tt], v5[tt], beta, idio
        self.tilt = np.zeros(n)
        self.decile = np.zeros(n, int)
        self.score = np.zeros(n)
        t_train0 = P.idx(v6.TRAIN_START)
        fit = v6.ridge()
        for k, r in enumerate(self.refits):
            nxt = self.refits[k + 1] if k + 1 < len(self.refits) else d1 + 1
            tr = slice(P.day_start[t_train0], P.day_start[max(r - H, t_train0)])
            ok = np.isfinite(P.yrank[tr])
            model = fit(P.X[tr][ok], P.yrank[tr][ok])
            # tilt table from the model's own predictions on (past) training rows
            ptr = model(P.X[tr])
            pct_tr = day_pct(P, ptr, tr.start, tr.stop)
            dec_tr = np.minimum((pct_tr * 10).astype(int), 9)
            y_tr = P.y[tr]
            table = np.array([np.nanmean(y_tr[(dec_tr == q) & np.isfinite(y_tr)]) for q in range(10)])
            s, e = P.day_start[r], P.day_start[nxt]
            pr = model(P.X[s:e])
            pct = day_pct(P, pr, s, e)
            dec = np.minimum((pct * 10).astype(int), 9)
            self.score[s - a:e - a] = pr
            self.decile[s - a:e - a] = dec
            self.tilt[s - a:e - a] = table[dec]
            if log and k % 12 == 0:
                log(f"    refit {P.dates[r]}: tilt by decile {np.round(table * 100, 2).tolist()} (% per 5 days)", flush=True)
        self.y = P.fwd[a:b]                                                  # realised 5-day log return
        self.means = {"RW": np.zeros(n), "MKT": self.m_mkt}
        for s_ in shrinks:
            self.means[f"v7 s={s_}"] = self.m_mkt + s_ * self.tilt
        # recalibration per refit month and model, from PITs of rows whose outcome was known before the refit day
        self.R = {}
        known_t = tt + H                                                      # outcome known at close t+H
        for name, m in self.means.items():
            pit = t_cdf((self.y - m) / self.sd)
            for r in self.refits:
                sel = (known_t < r) & (tt >= r - H - CAL_DAYS) & np.isfinite(pit)
                p = pit[sel]
                if len(p) > 200_000:
                    p = np.random.default_rng(r).choice(p, 200_000, replace=False)
                self.R[(name, r)] = recal_levels_sym(p, GRID) if len(p) >= 20 else GRID.copy()
        self.month_of = np.searchsorted(np.array(self.refits), tt, side="right") - 1

    def rows(self, d0, d1):
        return np.arange(self.P.day_start[d0] - self.a, self.P.day_start[d1 + 1] - self.a)


def evaluate(E, d0, d1, label_prefix="", return_daily=False):
    """Protocol metrics for every model over days d0..d1 (and the per-day mean CRPS of each model if asked)."""
    P = E.P
    idx = E.rows(d0, d1)
    tt = P.t[E.a + idx]
    y = E.y[idx]
    res = {}
    daily = {}
    for name, mall in E.means.items():
        crps = np.full(len(idx), np.nan)
        cov80 = np.full(len(idx), np.nan)
        cov95 = np.full(len(idx), np.nan)
        pup = np.full(len(idx), np.nan)
        for k, r in enumerate(E.refits):
            sel = np.nonzero(E.month_of[idx] == k)[0]
            sel = sel[np.isfinite(y[sel])]
            if len(sel) == 0:
                continue
            R = E.R[(name, r)]
            m, sd, yy = mall[idx][sel], E.sd[idx][sel], y[sel]
            for c0 in range(0, len(sel), 50_000):                           # chunks keep memory small
                cs = slice(c0, c0 + 50_000)
                q = quantiles(m[cs], sd[cs], R)
                crps[sel[cs]] = crps_rows(q, yy[cs])
                cov80[sel[cs]] = (yy[cs] >= q[:, I10]) & (yy[cs] <= q[:, I90])
                cov95[sel[cs]] = (yy[cs] >= q[:, I025]) & (yy[cs] <= q[:, I975])
            pup[sel] = 1 - p_le(0.0, m, sd, R)
        ok = np.isfinite(crps)
        up = (y > 0).astype(float)
        m = mall[idx]
        dir_ok = ok & (m != 0) & (y != 0)
        hits = (np.sign(m[dir_ok]) == np.sign(y[dir_ok])).mean() if dir_ok.any() else float("nan")
        nh = int(dir_ok.sum())
        brier = np.mean((pup[ok] - up[ok]) ** 2)
        clim = np.mean((np.mean(up[ok]) - up[ok]) ** 2)
        by_dec = []
        for q in range(10):
            s = ok & (E.decile[idx] == q)
            ds = s & (m != 0) & (y != 0)
            by_dec.append({"decile": q + 1, "hit": float((np.sign(m[ds]) == np.sign(y[ds])).mean()) if ds.any() else None,
                           "p_up_mean": float(np.mean(pup[s])), "up_rate": float(np.mean(up[s])), "cov80": float(np.mean(cov80[s]))})
        res[name] = {
            "rows": int(ok.sum()), "crps": float(np.mean(crps[ok])), "cov80": float(np.mean(cov80[ok])), "cov95": float(np.mean(cov95[ok])),
            "hit": float(hits), "hit_n": nh,
            "hit_p": float(stats.binomtest(int(round(hits * nh)), nh, 0.5).pvalue) if nh else None,
            "up_rate": float(np.mean(up[ok])), "brier": float(brier), "brier_skill_vs_constant": float(1 - brier / clim),
            "by_decile": by_dec,
        }
        dd = {}
        for i, c in zip(tt[ok], crps[ok]):
            dd.setdefault(int(i), []).append(c)
        daily[name] = {i: float(np.mean(v)) for i, v in dd.items()}
    from .daily_v4 import block_bootstrap_diff
    days = sorted(daily["RW"])
    for name in res:
        a = [daily[name][i] for i in days]
        b = [daily["RW"][i] for i in days]
        res[name]["crps_vs_rw"] = block_bootstrap_diff(a, b)
    return (res, daily) if return_daily else res


def print_table(res, title):
    print(title, flush=True)
    print(f"  {'model':<12}{'CRPS':>10}{'vs RW (95% CI)':>34}{'cov80':>8}{'cov95':>8}{'up/down hit':>13}{'Brier skill':>13}", flush=True)
    for name, x in res.items():
        d, lo, hi = x["crps_vs_rw"]
        hit = f"{x['hit']:.1%}" if np.isfinite(x["hit"]) else "—"
        print(f"  {name:<12}{x['crps']:>10.5f}{f'{d:+.2e} [{lo:+.2e}, {hi:+.2e}]':>34}{x['cov80']:>8.1%}{x['cov95']:>8.1%}{hit:>13}"
              f"{x['brier_skill_vs_constant']:>+13.4f}", flush=True)


# ---------------------------------------------------------------- commands

def dev(args):
    t0 = time.time()
    P = v6.Panel()
    d0, d1 = v6.period_bounds(P, "dev")
    E = Engine(P, d0 - CAL_DAYS - 30, d1)
    res = evaluate(E, d0, d1)
    best = min((k for k in res if k.startswith("v7")), key=lambda k: res[k]["crps"])
    os.makedirs(OUT, exist_ok=True)
    json.dump({"period": [P.dates[d0], P.dates[d1]], "results": res, "chosen": best, "shrink": float(best.split("=")[1]),
               "minutes": (time.time() - t0) / 60}, open(os.path.join(OUT, "dev.json"), "w"), indent=1)
    print_table(res, f"development {P.dates[d0]} .. {P.dates[d1]}")
    print(f"chosen: {best}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def chosen_shrink():
    p = os.path.join(OUT, "dev.json")
    if not os.path.exists(p):
        raise SystemExit("run `v7 dev` first (it picks the shrink)")
    return json.load(open(p))["shrink"]


def report(args):
    t0 = time.time()
    s = chosen_shrink()
    P = v6.Panel()
    d0, d1 = v6.period_bounds(P, "test")
    E = Engine(P, d0 - CAL_DAYS - 30, d1, shrinks=(s,))
    res = evaluate(E, d0, d1)
    json.dump({"period": [P.dates[d0], P.dates[d1]], "label": "seen (used by v6's final test)", "shrink": s, "results": res,
               "minutes": (time.time() - t0) / 60}, open(os.path.join(OUT, "seen.json"), "w"), indent=1)
    print_table(res, f"SEEN period {P.dates[d0]} .. {P.dates[d1]} (already used by v6's final test)")


def forecast(args):
    s = chosen_shrink()
    P = v6.Panel()
    last = P.T - 1
    E = Engine(P, last - CAL_DAYS - 40, last, shrinks=(s,), log=None)
    idx = E.rows(last, last)
    r = E.refits[-1]
    name = f"v7 s={s}"
    from . import v7r                                                         # amendment B: event-aware ranges
    adopted, w = v7r.adopted(H)
    E_v7 = E
    if adopted and P.dates[last] >= v7r.FIRST_LIVE:
        parts = v7r.EventParts(E, log=None)
        E = v7r.with_sd(E, name, v7r.blend(v7r.sd_event(E, parts), v7r.iv_rows(E), w))
        ranges = "v7r"
    else:
        ranges = "v7"
    m, sd = E.means[name][idx], E.sd[idx]
    R, Rrw = E.R[(name, r)], E.R[("RW", r)]
    q = quantiles(m, sd, R)
    close = np.exp(P.lc[last, P.n[E.a + idx]])
    pup = 1 - p_le(0.0, m, sd, R)
    # P(stock beats QQQ over 5 days): difference has mean m - mu_m and variance (beta-1)^2 V_m + 5 idio^2
    rel_sd = np.sqrt(np.maximum((E.beta[idx] - 1) ** 2 * E.v_m[idx] + sd ** 2 - E.beta[idx] ** 2 * E.v_m[idx], 1e-12))
    pbeat = 1 - t_cdf((0 - (m - E.mu_m[idx])) / rel_sd)
    rows = {}
    for k, i in enumerate(idx):
        sym = P.symbols[P.n[E.a + i]]
        px = lambda j: float(close[k] * math.exp(q[k, j]))
        rows[sym] = {"close": float(close[k]), "median": px(I50), "lo80": px(I10), "hi80": px(I90), "lo95": px(I025), "hi95": px(I975),
                     "p_up": float(pup[k]), "p_beat_qqq": float(pbeat[k]), "exp_ret": float(math.expm1(q[k, I50]))}
    os.makedirs(OUT, exist_ok=True)
    out = {"asof": P.dates[last], "model": name, "status": "shadow", "horizon_days": H, "rows": rows, "ranges": ranges,
           # the market model itself (benchmark ETF): 5-day drift and GJR-GARCH variance, for the Indexes view
           "market_model": {"mu5": float(E.mu_m[idx[0]]), "v5": float(E.v_m[idx[0]])}}
    # v8 amendment B: news-adjusted variant from the rolling LLM news (fixed prior weights; forward shadow test only)
    news_path = os.path.join("private", "data", "news", "news_today.json")
    news = json.load(open(news_path)) if os.path.exists(news_path) and v6.MARKET == "nasdaq" and H == 5 else None   # US news, 5-day only
    if news and news.get("asof") == P.dates[last]:
        S = news["market"]["S"]
        syms = [P.symbols[P.n[E.a + i]] for i in idx]
        nd = np.array([news["stocks"].get(sy, {}).get("news_dir", 0.0) for sy in syms])
        m_news = m + E.beta[idx] * NEWS_MKT_W * S + NEWS_STOCK_W * nd
        qn = quantiles(m_news, sd, R)
        pup_n = 1 - p_le(0.0, m_news, sd, R)
        for k, sy in enumerate(syms):
            st = news["stocks"].get(sy)
            rows[sy]["news"] = {"median": float(close[k] * math.exp(qn[k, I50])), "lo80": float(close[k] * math.exp(qn[k, I10])),
                                "hi80": float(close[k] * math.exp(qn[k, I90])), "p_up": float(pup_n[k]), "news_dir": float(nd[k]),
                                "n": st["n"] if st else 0, "top": st["top"][:2] if st else []}
        out["news"] = {"status": "unproven (forward shadow test)", "market_index": S, "market_n": news["market"]["n"],
                       "ai_anxiety": news["market"]["ai_anxiety"], "window": news["window"],
                       "most_negative": news["market"]["most_negative"], "most_positive": news["market"]["most_positive"],
                       "weights": {"market_per_unit_S": NEWS_MKT_W, "stock_per_unit_news_dir": NEWS_STOCK_W}}
        nlog = os.path.join(OUT, "shadow_news.jsonl")
        ndone = {json.loads(l)["asof"] for l in open(nlog)} if os.path.exists(nlog) else set()
        if P.dates[last] not in ndone:
            with open(nlog, "a") as f:
                f.write(json.dumps({"asof": P.dates[last], "news_generated": news["generated"], "S": S, "R_v7": R.tolist(),
                                    "rows": [[sy, float(m_news[k]), float(m[k]), float(sd[k])] for k, sy in enumerate(syms)]}) + "\n")
    json.dump(out, open(os.path.join(OUT, "forecast.json"), "w"))
    # forward shadow logs: written before the outcome exists, once per date. v7 keeps its own; v7r (amendment B) has its own
    def shadow(path, EE):
        done = {json.loads(l)["asof"] for l in open(path)} if os.path.exists(path) else set()
        if P.dates[last] in done:
            return False
        with open(path, "a") as f:
            f.write(json.dumps({"asof": P.dates[last], "model": name, "ranges": "v7r" if EE is not E_v7 else "v7",
                                "R_v7": EE.R[(name, r)].tolist(), "R_rw": EE.R[("RW", r)].tolist(),
                                "rows": [[P.symbols[P.n[EE.a + i]], float(EE.means[name][i]), float(EE.sd[i]), float(close[k])]
                                         for k, i in enumerate(idx)]}) + "\n")
        return True
    log = os.path.join(OUT, "shadow.jsonl")
    done = {P.dates[last]} if not shadow(log, E_v7) else set()
    if ranges == "v7r":
        shadow(os.path.join(OUT, "shadow_r.jsonl"), E)
    print(f"v7 forecast {P.dates[last]}: {len(rows)} stocks, {name}; mean P(up) {np.mean(pup):.1%}; "
          f"shadow log {'appended' if P.dates[last] not in done else 'already has this date'}", flush=True)


def score(args):
    log = os.path.join(OUT, "shadow.jsonl")
    if not os.path.exists(log):
        print("no shadow forecasts yet", flush=True)
        return
    P = v6.Panel()
    di = {d: i for i, d in enumerate(P.dates)}
    si = {s: j for j, s in enumerate(P.symbols)}
    per_day = []
    for line in open(log):
        rec = json.loads(line)
        t = di.get(rec["asof"])
        if t is None or t + H > P.T - 1:
            continue
        R7, Rrw = np.array(rec["R_v7"]), np.array(rec["R_rw"])
        sym, m, sd = zip(*[(r[0], r[1], r[2]) for r in rec["rows"]])
        j = np.array([si.get(s, -1) for s in sym])
        y = np.full(len(j), np.nan)
        for k in range(1, H + 1):                                            # last close within t+1..t+H
            v = P.lc[t + k, np.maximum(j, 0)]
            y = np.where(np.isfinite(v) & (j >= 0), v - P.lc[t, np.maximum(j, 0)], y)
        ok = np.isfinite(y)
        m, sd, y = np.array(m)[ok], np.array(sd)[ok], y[ok]
        c7 = crps_rows(quantiles(m, sd, R7), y).mean()
        crw = crps_rows(quantiles(np.zeros_like(m), sd, Rrw), y).mean()
        q = quantiles(m, sd, R7)
        cov = np.mean((y >= q[:, I10]) & (y <= q[:, I90]))
        hit = np.mean(np.sign(m[(m != 0) & (y != 0)]) == np.sign(y[(m != 0) & (y != 0)]))
        per_day.append({"asof": rec["asof"], "n": int(ok.sum()), "crps_v7": float(c7), "crps_rw": float(crw), "cov80": float(cov), "hit": float(hit)})
    out = {"matured_dates": len(per_day), "needed": 60, "days": per_day}
    if len(per_day) >= 5:
        from .daily_v4 import block_bootstrap_diff
        out["crps_v7_minus_rw"] = block_bootstrap_diff([d["crps_v7"] for d in per_day], [d["crps_rw"] for d in per_day], block=max(5, H))
        out["cov80"] = float(np.mean([d["cov80"] for d in per_day]))
        out["hit"] = float(np.mean([d["hit"] for d in per_day]))
    if len(per_day) >= 60:
        d, lo, hi = out["crps_v7_minus_rw"]
        out["promoted"] = bool(hi < 0 and 0.75 <= out["cov80"] <= 0.85)
    # news-adjusted variant vs v7 (v8 amendment B)
    nlog = os.path.join(OUT, "shadow_news.jsonl")
    nd_days = []
    if os.path.exists(nlog):
        for line in open(nlog):
            rec = json.loads(line)
            t = di.get(rec["asof"])
            if t is None or t + H > P.T - 1:
                continue
            R7 = np.array(rec["R_v7"])
            sym, mn, m7, sd = zip(*rec["rows"])
            j = np.array([si.get(s, -1) for s in sym])
            y = np.full(len(j), np.nan)
            for k in range(1, H + 1):
                v = P.lc[t + k, np.maximum(j, 0)]
                y = np.where(np.isfinite(v) & (j >= 0), v - P.lc[t, np.maximum(j, 0)], y)
            ok = np.isfinite(y)
            mn, m7, sd, y = np.array(mn)[ok], np.array(m7)[ok], np.array(sd)[ok], y[ok]
            hit = lambda m: float(np.mean(np.sign(m[(m != 0) & (y != 0)]) == np.sign(y[(m != 0) & (y != 0)])))
            nd_days.append({"asof": rec["asof"], "crps_news": float(crps_rows(quantiles(mn, sd, R7), y).mean()),
                            "crps_v7": float(crps_rows(quantiles(m7, sd, R7), y).mean()), "hit_news": hit(mn), "hit_v7": hit(m7)})
    out["news"] = {"matured_dates": len(nd_days), "needed": 60, "days": nd_days}
    if len(nd_days) >= 5:
        from .daily_v4 import block_bootstrap_diff
        out["news"]["crps_news_minus_v7"] = block_bootstrap_diff([d["crps_news"] for d in nd_days], [d["crps_v7"] for d in nd_days])
        out["news"]["hit_news"] = float(np.mean([d["hit_news"] for d in nd_days]))
        out["news"]["hit_v7"] = float(np.mean([d["hit_v7"] for d in nd_days]))
    if len(nd_days) >= 60:
        out["news"]["promoted"] = bool(out["news"]["crps_news_minus_v7"][2] < 0 and out["news"]["hit_news"] >= out["news"]["hit_v7"])
    json.dump(out, open(os.path.join(OUT, "shadow_score.json"), "w"), indent=1)
    print(f"news-adjusted shadow test: {len(nd_days)} of 60 forecast dates matured", flush=True)
    print(f"shadow test: {len(per_day)} of 60 forecast dates matured" +
          (f"; CRPS v7 - RW {out['crps_v7_minus_rw'][0]:+.2e}, 80% coverage {out['cov80']:.1%}, hit {out['hit']:.1%}" if len(per_day) >= 5 else ""),
          flush=True)


def _out():
    return v6.V7_OUT if H == 5 else os.path.join(v6.V7_OUT, f"h{H}")


def use_market(market):
    global OUT
    v6.use_market(market)
    OUT = _out()


HORIZONS = (1, 5, 20)                                                  # amendment A


def set_horizon(h):
    """Amendment A: the same model at horizon h trading days (also sets v6.H, which the panel's targets use; set it
    back to 5 before ranking)."""
    global H, OUT
    H = v6.H = h
    OUT = _out()


def gate(args=None):
    """Amendment A gate for showing a horizon: development CRPS below RW's, and 80% coverage within 0.75-0.85 on both
    development and the seen period."""
    dv, sn = json.load(open(os.path.join(OUT, "dev.json"))), json.load(open(os.path.join(OUT, "seen.json")))
    name = dv["chosen"]
    d, s_ = dv["results"][name], sn["results"][name]
    checks = {"dev_crps_below_rw": d["crps"] < dv["results"]["RW"]["crps"], "dev_cov80_ok": 0.75 <= d["cov80"] <= 0.85,
              "seen_cov80_ok": 0.75 <= s_["cov80"] <= 0.85}
    out = {"horizon": H, "model": name, "pass": all(checks.values()), "checks": checks,
           "dev": {k: d[k] for k in ("crps", "cov80", "cov95", "hit")}, "seen": {k: s_[k] for k in ("crps", "cov80", "cov95", "hit")},
           "rw_dev_crps": dv["results"]["RW"]["crps"]}
    json.dump(out, open(os.path.join(OUT, "gate.json"), "w"), indent=1)
    print(f"gate h={H}: {'PASS' if out['pass'] else 'FAIL'} {checks}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "report", "forecast", "score", "gate"])
    ap.add_argument("--market", choices=["nasdaq", "sgx"], default="nasdaq")
    ap.add_argument("--horizon", type=int, choices=HORIZONS, default=5)
    a = ap.parse_args()
    use_market(a.market)
    set_horizon(a.horizon)
    {"dev": dev, "report": report, "forecast": forecast, "score": score, "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    main()
