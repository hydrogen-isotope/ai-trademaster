"""
v7 amendment B: event-aware ranges ("v7r"), under docs/FORECAST-V7-PRICE-PROTOCOL.md. Research only, CPU.

Same mean as v7; only the spread changes:
  earnings:  sd^2 = beta^2 * V_m + idio^2 * (h + n_e * (k_i - 1)),  n_e = earnings reaction days in (t, t+h]
  options:   log sd^2 = (1 - w) * log sd_e^2 + w * log(iv^2 * h / 252)   (stocks with an implied volatility that day)

  dev     picks w on 5-day development rows with an IV (2024-04-01 .. 2024-12-31), then applies the adoption rule for
          each horizon (1, 5, 20) over development 2021-2024 and reports the seen period 2025-26
          -> private/reports/v7[/h{h}]/ranges_b.json
Usage: python -m trademaster_ml.v7r dev
"""
import argparse, copy, json, os, time

import numpy as np

from . import earnings, v6, v7
from .daily_v2 import GRID, recal_levels_sym
from .daily_v4 import block_bootstrap_diff

WS = (0.0, 0.25, 0.5, 0.75)
K_CAP, K_PRIOR, K_LAST, K_YEARS = 100.0, 4.0, 8, 3
IV_FROM, IV_TO = "2024-04-01", "2024-12-31"
OPTIONS = os.path.join("private", "data", "options", "features.npz")
LIVE_IV = os.path.join("private", "data", "options", "iv_live.json")
FIRST_LIVE = "2026-10-20"        # adopted after the 2026-10-19 forecasts were made: v7r forecasts start from the next close


def full_raw(P, name):
    meta = json.load(open(os.path.join(v6.GPU_DIR, "features.json")))
    f = meta["features"].index(name)
    raw = np.memmap(os.path.join(v6.GPU_DIR, "raw.f32"), np.float32, "r", shape=(len(meta["features"]), P.T, P.N))
    return np.array(raw[f], dtype=np.float64)


class EventParts:
    """Per Engine row: n_e (earnings reaction days in the window) and k (the stock's earnings-day variance multiple,
    refitted monthly from events before the refit day)."""

    def __init__(self, E, log=print):
        P, H = E.P, v7.H
        T, N = P.T, P.N
        ev, n_days = earnings.reaction_events(P.dates, P.symbols, future=H)
        tt, nn = P.t[E.a:E.b], P.n[E.a:E.b]
        self.ne = earnings.window_counts(ev, n_days, N, tt, nn, H)
        # earnings-day variance ratio q for every past event
        r = np.vstack([np.full((1, N), np.nan), np.diff(P.lc, axis=0)])
        rm = np.concatenate([[np.nan], np.diff(P.market)])
        beta, idio = full_raw(P, "beta252"), full_raw(P, "idio_vol60")
        ev = ev[(ev[:, 0] >= 1) & (ev[:, 1] <= T - 1)]
        et, en = ev[:, 1], ev[:, 2]                                              # dated by the last possible day
        b = np.where(np.isfinite(beta[ev[:, 0] - 1, en]), beta[ev[:, 0] - 1, en], 1.0)
        iv = idio[ev[:, 0] - 1, en]
        q0 = (r[ev[:, 0], en] - b * rm[ev[:, 0]]) ** 2 / iv ** 2
        q1 = (r[et, en] - b * rm[et]) ** 2 / iv ** 2
        q = np.minimum(K_CAP, np.fmax(q0, q1))                                  # unknown time: the larger of the two days
        ok = np.isfinite(q) & (iv > 0)
        et, en, q = et[ok], en[ok], q[ok]
        order = np.argsort(et, kind="stable")
        et, en, q = et[order], en[order], q[order]
        self.k = np.ones(len(tt))
        self.kbar = {}
        for m, rf in enumerate(E.refits):
            sel = (et < rf) & (et >= rf - 252 * K_YEARS)
            kbar = float(np.median(q[sel])) if sel.sum() >= 50 else 1.0
            self.kbar[P.dates[rf]] = kbar
            kn = np.full(N, kbar)
            past = np.nonzero(et < rf)[0]
            by = {}
            for i in past:                                                         # last K_LAST events per stock
                by.setdefault(en[i], []).append(q[i])
            for n_, lst in by.items():
                x = lst[-K_LAST:]
                kn[n_] = (len(x) * np.mean(x) + K_PRIOR * kbar) / (len(x) + K_PRIOR)
            rows = np.nonzero(E.month_of == m)[0]
            self.k[rows] = kn[nn[rows]]
        if log:
            ks = list(self.kbar.items())
            log(f"    earnings: {len(et)} past events; pooled k from {ks[0]} to {ks[-1]}; rows with an event in the window "
                f"{np.mean(self.ne > 0):.1%}", flush=True)


def sd_event(E, parts):
    return np.sqrt(E.beta ** 2 * E.v_m + E.idio ** 2 * (v7.H + parts.ne * (parts.k - 1)))


def iv_rows(E):
    """Annualised ATM implied volatility per Engine row (nan where none)."""
    P = E.P
    out = np.full(E.b - E.a, np.nan)
    if not os.path.exists(OPTIONS):
        return out
    z = np.load(OPTIONS)
    di = {str(d): i for i, d in enumerate(z["dates"])}
    sj = {str(s): j for j, s in enumerate(z["syms"])}
    iv = z["iv_atm"]
    tt, nn = P.t[E.a:E.b], P.n[E.a:E.b]
    ti = np.array([di.get(P.dates[t], -1) for t in range(P.T)])
    nj = np.array([sj.get(s, -1) for s in P.symbols])
    a, b = ti[tt], nj[nn]
    ok = (a >= 0) & (b >= 0)
    out[ok] = iv[a[ok], b[ok]]
    if os.path.exists(LIVE_IV):                                                # today's IV (refreshed after each close)
        live = json.load(open(LIVE_IV))
        if live.get("asof") in P.dates:
            t_live = P.dates.index(live["asof"])
            sel = np.nonzero(tt == t_live)[0]
            for k in sel:
                v = live["iv"].get(P.symbols[nn[k]])
                if v is not None:
                    out[k] = v
    return np.where((out > 0.03) & (out < 5), out, np.nan)


def blend(sd_e, iv, w):
    if w == 0:
        return sd_e
    v_iv = iv ** 2 * v7.H / 252
    return np.where(np.isfinite(iv), np.exp(0.5 * ((1 - w) * np.log(sd_e ** 2) + w * np.log(v_iv))), sd_e)


def with_sd(E, name, sd, y_mask=None):
    """A copy of the engine with one model, a new spread and its own recalibration (same PIT rule as v7)."""
    E2 = copy.copy(E)
    E2.sd = sd
    E2.means = {name: E.means[name], "RW": E.means["RW"]}                    # evaluate() scores every model vs RW
    if y_mask is not None:
        E2.y = np.where(y_mask, E.y, np.nan)
    tt = E.P.t[E.a:E.b]
    known_t = tt + v7.H
    E2.R = {}
    for nm, m in E2.means.items():
        pit = v7.t_cdf((E.y - m) / sd)
        for r in E.refits:
            sel = (known_t < r) & (tt >= r - v7.H - v7.CAL_DAYS) & np.isfinite(pit)
            p = pit[sel]
            if len(p) > 200_000:
                p = np.random.default_rng(r).choice(p, 200_000, replace=False)
            E2.R[(nm, r)] = recal_levels_sym(p, GRID) if len(p) >= 20 else GRID.copy()
    return E2


def compare(E, name, sd_old, sd_new, d0, d1):
    ra, da = v7.evaluate(with_sd(E, name, sd_old), d0, d1, return_daily=True)
    rb, db = v7.evaluate(with_sd(E, name, sd_new), d0, d1, return_daily=True)
    days = sorted(set(da[name]) & set(db[name]))
    diff = block_bootstrap_diff([db[name][i] for i in days], [da[name][i] for i in days], block=max(5, v7.H))
    pick = lambda x: {k: x[name][k] for k in ("rows", "crps", "cov80", "cov95")}
    return {"v7": pick(ra), "v7r": pick(rb), "v7r_minus_v7": list(diff)}


def choose_w(log=print):
    v7.set_horizon(5)
    P = v6.Panel()
    s = v7.chosen_shrink()
    name = f"v7 s={s}"
    d0, d1 = P.idx(IV_FROM), P.idx(IV_TO + "~") - 1
    E = v7.Engine(P, d0 - v7.CAL_DAYS - 30, d1, shrinks=(s,), log=None)
    parts = EventParts(E, log)
    sd_e, iv = sd_event(E, parts), iv_rows(E)
    mask = np.isfinite(iv)
    log(f"  choosing w on {P.dates[d0]} .. {P.dates[d1]}: rows with an IV {int(mask[E.rows(d0, d1)].sum())}", flush=True)
    res = {}
    for w in WS:
        r = v7.evaluate(with_sd(E, name, blend(sd_e, iv, w), y_mask=mask), d0, d1)[name]
        res[str(w)] = {"crps": r["crps"], "cov80": r["cov80"], "rows": r["rows"]}
        log(f"    w={w}: CRPS {r['crps']:.6f}, 80% coverage {r['cov80']:.1%} ({r['rows']} rows)", flush=True)
    w = float(min(res, key=lambda k: res[k]["crps"]))
    return w, res


def dev(args, log=print):
    t0 = time.time()
    w, wres = choose_w(log)
    log(f"chosen w = {w}", flush=True)
    summary = {}
    for h in v7.HORIZONS:
        v7.set_horizon(h)
        P = v6.Panel()
        s = v7.chosen_shrink()
        name = f"v7 s={s}"
        out = {"horizon": h, "w": w, "w_choice": wres, "model": name}
        for period in ("dev", "test"):
            d0, d1 = v6.period_bounds(P, period)
            E = v7.Engine(P, d0 - v7.CAL_DAYS - 30, d1, shrinks=(s,), log=None)
            parts = EventParts(E, log)
            sd_new = blend(sd_event(E, parts), iv_rows(E), w)
            out["seen" if period == "test" else "dev"] = c = compare(E, name, E.sd, sd_new, d0, d1)
            out[("seen" if period == "test" else "dev") + "_period"] = [P.dates[d0], P.dates[d1]]
            log(f"  h={h} {period}: CRPS v7 {c['v7']['crps']:.6f} -> v7r {c['v7r']['crps']:.6f} "
                f"(diff {c['v7r_minus_v7'][0]:+.2e} [{c['v7r_minus_v7'][1]:+.2e}, {c['v7r_minus_v7'][2]:+.2e}]); "
                f"80% coverage {c['v7']['cov80']:.1%} -> {c['v7r']['cov80']:.1%}", flush=True)
        d = out["dev"]
        out["adopted"] = bool(d["v7r_minus_v7"][2] < 0 and 0.75 <= d["v7r"]["cov80"] <= 0.85)
        out["minutes"] = (time.time() - t0) / 60
        json.dump(out, open(os.path.join(v7.OUT, "ranges_b.json"), "w"), indent=1)
        summary[h] = out["adopted"]
        log(f"  h={h}: {'ADOPTED' if out['adopted'] else 'not adopted'}", flush=True)
    v7.set_horizon(5)
    log(f"amendment B: {summary} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return summary


def adopted(h=None):
    """(adopted, w) for the current or given horizon."""
    h = h or v7.H
    p = os.path.join(v6.V7_OUT if h == 5 else os.path.join(v6.V7_OUT, f"h{h}"), "ranges_b.json")
    if not os.path.exists(p):
        return False, 0.0
    d = json.load(open(p))
    return bool(d.get("adopted")), float(d.get("w", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev"])
    a = ap.parse_args()
    v7.use_market("nasdaq")
    dev(a)


if __name__ == "__main__":
    main()
