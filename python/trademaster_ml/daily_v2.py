"""
5-day forecast v2 (research only; CPU only). Protocol, fixed before any result: docs/FORECAST-V2-PROTOCOL.md

  mean        pooled cross-stock ridge on ranked features -> each stock's 1..5-day return relative to the average stock
              (in units of its 20-day volatility), plus a shrunk long-run SPY drift
  volatility  v1 GJR-GARCH(1,1)-t (daily_forecast.Garch)
  ranges      quantile recalibration from the model's own past forecasts whose outcomes are known

Usage:
  python -m trademaster_ml.daily_v2 select     # development period only: pick the ridge penalty
  python -m trademaster_ml.daily_v2 run        # development + final test for MU, NVDA, AMD, AAPL (slow, CPU)
  python -m trademaster_ml.daily_v2 refresh    # re-download daily bars for the universe (Alpaca, needs a key)
  python -m trademaster_ml.daily_v2 forecast   # next 5 trading days from the latest close
"""
import argparse, bisect, json, math, os
from datetime import date, datetime, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .daily_forecast import Garch, crps_samples, crps_normal, dm_test, next_trading_days, H, QS, OUT, FC_DIR

UNI = os.path.join("private", "data", "daily_universe")
TARGETS = ["MU", "NVDA", "AMD", "AAPL"]
DEV_START, DEV_END = "2020-01-02", "2025-10-10"
TEST_N = 250
RETRAIN = 21
MIN_TRAIN = 756
LAMBDAS = [1, 10, 100, 1000]          # ridge: minimise mean squared error + (lambda / 1000) * |w|^2
CAL_WINDOW, CAL_SHRINK = 250, 100
N_PATHS = 10000
FEATS = ["rev1", "rev5", "mom1m", "mom3m", "mom6m", "mom12m", "vol20", "volratio", "hi52", "vsurp",
         "rel5", "rel20", "spy5", "spy20"]
GRID = np.linspace(0.005, 0.995, 199)  # quantile levels used to represent a forecast distribution


# ---------------------------------------------------------------- data
def load_panel(d=UNI):
    u = json.load(open(os.path.join(d, "universe.json")))
    stocks, etfs = u["stocks"], u["etfs"]
    syms = stocks + etfs
    data = {s: json.load(open(os.path.join(d, f"{s}.json"))) for s in syms}
    dates = data["SPY"]["dates"]
    pos = {x: i for i, x in enumerate(dates)}
    T, N = len(dates), len(syms)
    C, V = np.full((T, N), np.nan), np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        for x, c, v in zip(data[s]["dates"], data[s]["close"], data[s]["volume"]):
            if x in pos:
                C[pos[x], j], V[pos[x], j] = c, v
    is_stock = np.array([s in stocks for s in syms])
    ref = np.array([syms.index(u["reference_etf"].get(s, "QQQ")) if s in stocks else j for j, s in enumerate(syms)])
    return dates, syms, is_stock, C, V, ref, syms.index("SPY")


def lag(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def roll(a, w, fn):
    out = np.full_like(a, np.nan)
    if len(a) >= w:
        out[w - 1:] = fn(sliding_window_view(a, w, axis=0), axis=-1)
    return out


def rank_signed(x, mask):
    """Cross-sectional rank among `mask` columns each day, scaled to [-1, 1]; NaN where missing."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        cols = np.where(mask & np.isfinite(x[t]))[0]
        if len(cols) < 10:
            continue
        r = np.argsort(np.argsort(x[t, cols]))
        out[t, cols] = 2 * (r + 0.5) / len(cols) - 1
    return out


def build(C, V, is_stock, ref, spy):
    L = np.log(C)
    r1 = L - lag(L, 1)
    vol20 = roll(r1, 20, np.nanstd)
    vol120 = roll(r1, 120, np.nanstd)
    raw = {
        "rev1": r1, "rev5": L - lag(L, 5),
        "mom1m": lag(L, 5) - lag(L, 21), "mom3m": lag(L, 21) - lag(L, 63),
        "mom6m": lag(L, 63) - lag(L, 126), "mom12m": lag(L, 126) - lag(L, 252),
        "vol20": vol20, "volratio": np.log(vol20 / vol120),
        "hi52": L - roll(L, 252, np.nanmax),
        "vsurp": np.log(roll(V, 5, np.nanmean) / roll(V, 60, np.nanmean)),
        "rel5": (L - lag(L, 5)) - (L - lag(L, 5))[:, ref], "rel20": (L - lag(L, 20)) - (L - lag(L, 20))[:, ref],
    }
    F = np.stack([rank_signed(raw[k], is_stock) for k in FEATS[:12]], axis=2)
    s5 = (L[:, spy] - lag(L, 5)[:, spy]) / (vol20[:, spy] * math.sqrt(5))
    s20 = (L[:, spy] - lag(L, 20)[:, spy]) / (vol20[:, spy] * math.sqrt(20))
    mk = np.stack([np.repeat(s5[:, None], C.shape[1], 1), np.repeat(s20[:, None], C.shape[1], 1)], axis=2)
    F = np.concatenate([F, np.clip(mk, -4, 4)], axis=2)
    F[:, ~is_stock, :] = np.nan
    return L, r1, vol20, F


def rel_targets(L, vol20, is_stock):
    """Y[h][t, i] = (stock return t -> t+h minus the average stock's) / (vol20 * sqrt h); NaN when unknown."""
    Y = {}
    for h in range(1, H + 1):
        fwd = np.full_like(L, np.nan)
        fwd[:-h] = L[h:] - L[:-h]
        fwd[:, ~is_stock] = np.nan
        rel = fwd - np.nanmean(fwd, axis=1, keepdims=True)
        Y[h] = rel / (vol20 * math.sqrt(h))
    return Y


# ---------------------------------------------------------------- mean model
def fit_ridge(F, Y, t_last, lam):
    """Train on rows t <= t_last (caller passes origin - h so every target window ends by the origin)."""
    X = F[:t_last + 1].reshape(-1, F.shape[2])
    y = Y[:t_last + 1].reshape(-1)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], np.clip(y[ok], -6, 6)
    n = len(y)
    A = X.T @ X / n + lam / 1000 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y / n)


def walk_forward_mean(F, Y, first, last, lam, log=None):
    """P[h][t, i]: normalized relative-return prediction made at origin t, for origins first..last."""
    P = {h: np.full(F.shape[:2], np.nan) for h in range(1, H + 1)}
    W, trained = {}, None
    for t in range(first, last + 1):
        if trained is None or t - trained >= RETRAIN:
            W = {h: fit_ridge(F, Y[h], t - h, lam) for h in range(1, H + 1)}
            trained = t
        x = F[t]
        ok = np.isfinite(x).all(1)
        for h in range(1, H + 1):
            P[h][t, ok] = x[ok] @ W[h]
        if log and (t - first) % 250 == 0:
            log(f"  mean model: origin {t - first + 1}/{last - first + 1}")
    return P, W


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def mean_metrics(P, Y, first, last, is_stock):
    """Pooled cross-stock IC and direction hit at h=5, on non-overlapping origins (every 5th day)."""
    ics, hits = [], []
    for t in range(first, last + 1, H):
        p, y = P[H][t], Y[H][t]
        ok = np.isfinite(p) & np.isfinite(y) & is_stock
        if ok.sum() < 20:
            continue
        ics.append(spearman(p[ok], y[ok]))
        hits.append(np.mean(np.sign(p[ok]) == np.sign(y[ok])))
    ics, hits = np.array(ics), np.array(hits)
    t_hit = (hits.mean() - 0.5) / (hits.std(ddof=1) / math.sqrt(len(hits)))
    p_hit = 0.5 * math.erfc(t_hit / math.sqrt(2))                   # one-sided, normal approximation
    return dict(windows=int(len(ics)), ic=float(ics.mean()), ic_t=float(ics.mean() / ics.std(ddof=1) * math.sqrt(len(ics))),
                hit=float(hits.mean()), hit_t=float(t_hit), hit_p_one_sided=float(p_hit))


def market_drift(L, spy, t):
    r = np.diff(L[max(0, t - 1260):t + 1, spy])
    return 0.5 * float(np.nanmean(r))


# ---------------------------------------------------------------- distributions and recalibration
def quantiles_from_samples(s, levels):
    return np.quantile(s, levels)


def recal_levels(pits, levels):
    """Map nominal quantile levels through the empirical distribution of past PIT values (shrunk to identity)."""
    n = len(pits)
    if n < 20:
        return levels
    w = n / (n + CAL_SHRINK)
    return np.clip((1 - w) * levels + w * np.quantile(pits, levels), 0.001, 0.999)


def recal_levels_sym(pits, levels):
    """v3: symmetric recalibration. Widen or narrow each central interval to the coverage past forecasts actually
    achieved, keeping it centred on the model's median (the median is never moved)."""
    n = len(pits)
    if n < 20:
        return levels
    w = n / (n + CAL_SHRINK)
    dev = np.abs(np.asarray(pits) - 0.5)
    c = np.abs(2 * levels - 1)                                  # central coverage whose edge is this level
    c_new = np.clip(2 * np.quantile(dev, c), 0, 0.998)
    sym = 0.5 + np.sign(levels - 0.5) * c_new / 2
    return np.clip((1 - w) * levels + w * sym, 0.001, 0.999)


def recal_for(variant):
    return recal_levels_sym if variant == "v3sym" else recal_levels


def crps_from_quantiles(q, y):
    """CRPS approximated by the pinball loss averaged over the quantile grid (x2)."""
    return float(2 * np.mean(np.where(y < q, 1 - GRID, GRID) * np.abs(y - q)))


def eval_symbol(sym, j, L, r1, vol20, P, spy, first, last, dev_last, log=print):
    """Walk-forward distribution forecasts for one stock; returns per-origin records for 4 variants."""
    rng = np.random.default_rng(13)
    r = np.diff(L[:, j])                                          # r[t-1] = return into day t
    recs, model = [], None
    pits = {v: {h: [] for h in range(H)} for v in ["v1", "v1cal", "v2", "v2cal", "v3sym"]}   # PIT values, in origin order
    pit_t = {v: {h: [] for h in range(H)} for v in pits}                                # their origins
    for k, t in enumerate(range(first, last + 1)):
        hist = r[:t][np.isfinite(r[:t])]
        refit = 10 if t <= dev_last else 5
        if model is None or k % refit == 0:
            model = Garch().fit(hist[-1000:] * 100)
        else:
            p = model.p
            e = hist[-1] * 100
            p["next_var"] = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e * e + p["beta"] * p["next_var"]
        sims = model.simulate(0.0, n=N_PATHS, rng=rng)
        drift = market_drift(L, spy, t)
        mu = np.array([(P[h][t, j] if np.isfinite(P[h][t, j]) else 0.0) * vol20[t, j] * math.sqrt(h) + drift * h for h in range(1, H + 1)])
        y = np.array([L[t + h, j] - L[t, j] if t + h < len(L) else np.nan for h in range(1, H + 1)])
        rec = {"t": t, "y": y.tolist(), "v": {}}
        for v in pits:
            rows = []
            for h in range(H):
                s = sims[:, h] + (mu[h] if v.startswith("v2") else 0.0)
                k_known = bisect.bisect_right(pit_t[v][h], t - h - 1)          # outcome day origin+h+1 <= t
                past = pits[v][h][max(0, k_known - CAL_WINDOW):k_known]
                lv = recal_for(v)(np.array(past), GRID) if v in ("v1cal", "v2cal", "v3sym") else GRID
                q = np.quantile(s, lv)
                pit = float(np.mean(s <= y[h])) if np.isfinite(y[h]) else None
                if pit is not None:
                    pits[v][h].append(pit); pit_t[v][h].append(t)
                if np.isfinite(y[h]):
                    q10, q90 = np.interp([0.10, 0.90], GRID, q)
                    q025, q975 = np.interp([0.025, 0.975], GRID, q)
                    med = np.interp(0.5, GRID, q)
                    rows.append(dict(qs=np.interp(QS, GRID, q).tolist(), crps=crps_from_quantiles(q, y[h]), in80=bool(q10 <= y[h] <= q90), in95=bool(q025 <= y[h] <= q975),
                                     ape=float(abs(math.exp(med) - math.exp(y[h])) / math.exp(y[h])), dir=float(np.sign(med) == np.sign(y[h]))))
                else:
                    rows.append(None)
            rec["v"][v] = rows
        sd60 = np.std(hist[-60:])
        rec["base"] = [crps_normal(0, sd60 * math.sqrt(h + 1), y[h]) if np.isfinite(y[h]) else None for h in range(H)]
        rec["ape_rw"] = [float(abs(1 - math.exp(y[h])) / math.exp(y[h])) if np.isfinite(y[h]) else None for h in range(H)]
        recs.append(rec)
        if log and (k + 1) % 250 == 0:
            log(f"  {sym}: {k + 1}/{last - first + 1} origins")
    return recs, {v: {h: pits[v][h][-CAL_WINDOW:] for h in range(H)} for v in pits}


def summarize(recs, lo, hi, variant):
    rows = [r for r in recs if lo <= r["t"] <= hi]
    out = []
    for h in range(H):
        rr = [(r["v"][variant][h], r["base"][h], r["ape_rw"][h]) for r in rows if r["v"][variant][h] is not None]
        m = np.array([x[0]["crps"] for x in rr]); b = np.array([x[1] for x in rr])
        out.append(dict(h=h + 1, n=len(rr), coverage=dict(in80=float(np.mean([x[0]["in80"] for x in rr])), in95=float(np.mean([x[0]["in95"] for x in rr]))),
                        crps=float(m.mean()), crps_base=float(b.mean()), crps_skill_vs_baseline=float(1 - m.mean() / b.mean()),
                        mape_model=float(np.mean([x[0]["ape"] for x in rr])), mape_rw=float(np.mean([x[2] for x in rr])),
                        dir_hit=float(np.mean([x[0]["dir"] for x in rr]))))
    return out


def passes_calibration(hs):
    return all(abs(x["coverage"]["in80"] - 0.80) <= 0.06 and abs(x["coverage"]["in95"] - 0.95) <= 0.04 for x in hs)


def refresh(out=UNI, years=10.2):
    """Re-download daily bars (SIP, split-adjusted) for every symbol in the universe file."""
    import time
    from zoneinfo import ZoneInfo
    from . import alpaca
    ny = ZoneInfo("America/New_York")
    u = json.load(open(os.path.join(out, "universe.json")))
    syms = u["stocks"] + u["etfs"]
    end = int(time.time()) - 16 * 60
    data = alpaca.get_bars(syms, "1Day", end - int(years * 365 * 86400), end, feed="sip")
    for s in syms:
        rows = data.get(s) or []
        if len(rows) < 300:
            print(f"{s}: only {len(rows)} bars, keeping the old file")
            continue
        days = [str(datetime.fromtimestamp(r[0], ny).date()) for r in rows]
        tmp = os.path.join(out, f"{s}.json.tmp")
        json.dump({"symbol": s, "dates": days, "open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
                   "close": [r[4] for r in rows], "volume": [r[5] for r in rows], "source": "Alpaca 1Day bars (SIP, split-adjusted)"}, open(tmp, "w"))
        os.replace(tmp, os.path.join(out, f"{s}.json"))
    last = datetime.fromtimestamp(max(r[0] for r in data["SPY"]), ny).date() if data.get("SPY") else "?"
    print(f"refreshed {len(syms)} symbols through {last}")


# ---------------------------------------------------------------- commands
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "forecast", "refresh", "holdout"])
    ap.add_argument("--asof", help="holdout: last close the model may see (e.g. 2026-10-09)")
    a = ap.parse_args()
    if a.cmd == "refresh":
        refresh()
        return
    os.makedirs(OUT, exist_ok=True)
    dates, syms, is_stock, C, V, ref, spy = load_panel()
    L, r1, vol20, F = build(C, V, is_stock, ref, spy)
    Y = rel_targets(L, vol20, is_stock)
    T = len(dates)
    first_dev = next(i for i, d in enumerate(dates) if d >= DEV_START)
    dev_last = max(i for i, d in enumerate(dates) if d <= DEV_END)
    test_last = T - 1 - H
    test_first = test_last - TEST_N + 1
    assert first_dev >= MIN_TRAIN + H and test_first > dev_last
    sel_path = os.path.join(OUT, "v2_selection.json")

    if a.cmd == "select":
        res = {}
        for lam in LAMBDAS:
            P, _ = walk_forward_mean(F, Y, first_dev, dev_last, lam)
            res[lam] = mean_metrics(P, Y, first_dev, dev_last, is_stock)
            print(f"lambda {lam:5d}: dev IC {res[lam]['ic']:+.4f} (t {res[lam]['ic_t']:+.2f})  cross-stock direction hit {res[lam]['hit'] * 100:.2f}%")
        best = max(res, key=lambda k: res[k]["ic"])
        json.dump({"chosen_lambda": best, "dev": {str(k): v for k, v in res.items()}, "dev_period": [dates[first_dev], dates[dev_last]]},
                  open(sel_path, "w"), indent=1)
        print(f"chosen lambda (highest development IC): {best}")
        return

    lam = json.load(open(sel_path))["chosen_lambda"]
    if a.cmd == "run":
        P, _ = walk_forward_mean(F, Y, first_dev, test_last, lam, log=print)
        mm = {"dev": mean_metrics(P, Y, first_dev, dev_last, is_stock), "test": mean_metrics(P, Y, test_first, test_last, is_stock)}
        print(f"cross-stock mean model  dev: IC {mm['dev']['ic']:+.4f} hit {mm['dev']['hit'] * 100:.2f}%   "
              f"TEST: IC {mm['test']['ic']:+.4f} hit {mm['test']['hit'] * 100:.2f}% (p={mm['test']['hit_p_one_sided']:.3f})")
        results, pit_store = {}, {}
        for sym in TARGETS:
            j = syms.index(sym)
            recs, pits = eval_symbol(sym, j, L, r1, vol20, P, spy, first_dev, test_last, dev_last)
            results[sym] = {v: {"dev": summarize(recs, first_dev, dev_last, v), "test": summarize(recs, test_first, test_last, v)}
                            for v in ["v1", "v1cal", "v2", "v2cal", "v3sym"]}
            last_rec = recs[-1]                                     # origin = last test origin (information only)
            for v in ["v1", "v1cal", "v3sym"]:
                ins = sum(1 for h in range(H) if last_rec["v"][v][h] and last_rec["v"][v][h]["in80"])
                print(f"  {sym} hold-out from {dates[last_rec['t']]}: {v} {ins}/5 inside the 80% range")
            pit_store[sym] = pits
        # protocol: adopt the mean model if, over DEVELOPMENT, recalibrated v2 beats recalibrated zero-mean on pooled CRPS
        pooled = lambda v, per: np.mean([np.mean([x["crps"] for x in results[s][v][per]]) for s in TARGETS])
        adopt_mean = pooled("v2cal", "dev") < pooled("v1cal", "dev")
        chosen = "v2cal" if adopt_mean else "v1cal"
        # v3 amendment: between the shifting and the symmetric recalibration, keep the lower DEVELOPMENT pooled CRPS
        dev_crps = {v: float(pooled(v, "dev")) for v in ["v1", "v1cal", "v2cal", "v3sym"]}
        print("development pooled CRPS:", {k: round(v * 1e4, 3) for k, v in dev_crps.items()})
        if not adopt_mean and dev_crps["v3sym"] < dev_crps["v1cal"]:
            chosen = "v3sym"
        test_skill = 1 - pooled(chosen, "test") / np.mean([np.mean([x["crps_base"] for x in results[s][chosen]["test"]]) for s in TARGETS])
        rules = {
            "calibration": {s: passes_calibration(results[s][chosen]["test"]) for s in TARGETS},
            "crps_skill_vs_random_walk": float(test_skill), "crps_rule_pass": bool(test_skill > 0),
            "direction": mm["test"], "direction_rule_pass": bool(mm["test"]["hit"] > 0.5 and mm["test"]["hit_p_one_sided"] < 0.05),
        }
        doc = {"protocol": "docs/FORECAST-V2-PROTOCOL.md (incl. v3 amendment)", "dev_pooled_crps": dev_crps, "lambda": lam, "adopted_mean_model": bool(adopt_mean), "chosen_variant": chosen,
               "dev_period": [dates[first_dev], dates[dev_last]], "test_period": [dates[test_first], dates[test_last]],
               "mean_model": mm, "results": results, "rules": rules, "created_at": datetime.now(timezone.utc).isoformat()}
        json.dump(doc, open(os.path.join(OUT, "v2_results.json"), "w"), indent=1)
        json.dump({"pits": pit_store, "variant": chosen, "asof": dates[T - 1]}, open(os.path.join(OUT, "v2_pits.json"), "w"))
        # dashboard-compatible validation summary for the chosen variant
        summ = {s: {"symbol": s, "model": "v2", "variant": chosen, "origins": TEST_N, "h": results[s][chosen]["test"],
                    "passed": bool(rules["calibration"][s] and rules["crps_rule_pass"]),
                    "calibrated": bool(rules["calibration"][s]), "beats_baseline": bool(rules["crps_rule_pass"]),
                    "pooled_skill": float(rules["crps_skill_vs_random_walk"]),
                    "avg_skill": float(np.mean([x["crps_skill_vs_baseline"] for x in results[s][chosen]["test"]]))} for s in TARGETS}
        json.dump({"validated_at": doc["created_at"], "model": "v2", "symbols": summ}, open(os.path.join(OUT, "validation_summary.json"), "w"), indent=1)
        print(f"adopted mean model: {adopt_mean} -> variant {chosen}")
        for s in TARGETS:
            for v in ["v1", "v1cal", "v2", "v2cal", "v3sym"]:
                t_ = results[s][v]["test"]
                print(f"{s:5s} {v:6s} TEST 80%: " + " ".join(f"{x['coverage']['in80']:.2f}" for x in t_) + "  95%: " + " ".join(f"{x['coverage']['in95']:.2f}" for x in t_)
                      + f"  skill {np.mean([x['crps_skill_vs_baseline'] for x in t_]) * 100:+.1f}%  dir {np.mean([x['dir_hit'] for x in t_]) * 100:.1f}%")
        print("RULES:", json.dumps(rules["calibration"]), f"crps skill {test_skill * 100:+.2f}% pass={rules['crps_rule_pass']}",
              f"direction hit {mm['test']['hit'] * 100:.2f}% p={mm['test']['hit_p_one_sided']:.3f} pass={rules['direction_rule_pass']}")
        return

    if a.cmd == "holdout":
        # Re-run the shipped variant over the 260 origins up to --asof, so the self-correcting ranges at --asof only
        # use forecasts whose outcomes were known by then; then show the forecast made at --asof against what happened.
        chosen = json.load(open(os.path.join(OUT, "v2_results.json")))["chosen_variant"]
        t_as = dates.index(a.asof)
        P = {h: np.full(L.shape, np.nan) for h in range(1, H + 1)}
        if chosen == "v2cal":
            P, _ = walk_forward_mean(F, Y, t_as - 260, t_as, lam)
        res = []
        for sym in TARGETS:
            j = syms.index(sym)
            recs, _ = eval_symbol(sym, j, L, r1, vol20, P, spy, t_as - 260, t_as, t_as - 1, log=None)
            rec = recs[-1]
            last = float(C[t_as, j])
            rows = []
            for h in range(H):
                x = rec["v"][chosen][h]
                q = dict(zip(["p2_5", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p97_5"], (last * math.exp(v) for v in x["qs"])))
                act = float(C[t_as + h + 1, j])
                rows.append({"date": dates[t_as + h + 1], **q, "actual": act, "in80": x["in80"], "in95": x["in95"],
                             "err_pct": (q["p50"] / act - 1) * 100})
            res.append({"symbol": sym, "asof": a.asof, "last_close": last, "model": "v2", "variant": chosen, "days": rows})
            print(f"{sym:5s} fit to {a.asof} close {last:.2f}: " + "  ".join(
                f"{r_['date'][5:]} mid {r_['p50']:.2f} [80% {r_['p10']:.0f}-{r_['p90']:.0f}] act {r_['actual']:.2f} ({'in' if r_['in80'] else 'OUT'})" for r_ in rows))
        json.dump({"asof": a.asof, "model": "v2", "created_at": datetime.now(timezone.utc).isoformat(), "forecasts": res},
                  open(os.path.join(OUT, f"holdout_{a.asof}.json"), "w"), indent=1)
        return

    # forecast: next 5 trading days for each target from the latest close
    res = json.load(open(os.path.join(OUT, "v2_results.json")))
    pits = json.load(open(os.path.join(OUT, "v2_pits.json")))["pits"]
    chosen = res["chosen_variant"]
    t = T - 1
    W = {h: fit_ridge(F, Y[h], t - h, lam) for h in range(1, H + 1)}
    days = next_trading_days(date.fromisoformat(dates[t]))
    fcs, rng = [], np.random.default_rng(17)
    for sym in TARGETS:
        j = syms.index(sym)
        r = np.diff(L[:, j]); r = r[np.isfinite(r)]
        g = Garch().fit(r[-1000:] * 100)
        sims = g.simulate(0.0, n=20000, rng=rng)
        drift = market_drift(L, spy, t)
        x = F[t, j]
        rows = []
        for h in range(H):
            mu = (float(x @ W[h + 1]) * vol20[t, j] * math.sqrt(h + 1) if np.isfinite(x).all() and chosen == "v2cal" else 0.0) + (drift * (h + 1) if chosen == "v2cal" else 0.0)
            lv = recal_for(chosen)(np.array(pits[sym][chosen][str(h)]), np.array(QS))
            q = np.quantile(sims[:, h] + mu, lv)
            last = float(C[t, j])
            rows.append({"date": str(days[h]), **{f"p{int(x_ * 1000) / 10:g}".replace(".", "_"): last * math.exp(v) for x_, v in zip(QS, q)},
                         "mu_bps": mu * 1e4})
        val = res["rules"]["calibration"][sym] and res["rules"]["crps_rule_pass"]
        fcs.append({"symbol": sym, "asof": dates[t], "last_close": float(C[t, j]), "kind": "stock", "price_basis": "official close",
                    "validated": bool(val), "garch": g.p, "mean_model": "cross-stock ridge" if chosen == "v2cal" else "random walk",
                    "recalibration": {"v3sym": "symmetric", "v1cal": "shifting", "v2cal": "shifting"}.get(chosen, "none"),
                    "model": "v2", "days": rows, "source": "Alpaca 1Day bars (SIP)"})
        print(f"{sym:5s} last {C[t, j]:.2f}  " + "  ".join(f"{r_['date'][5:]}: {r_['p50']:.2f} [{r_['p10']:.2f}-{r_['p90']:.2f}]" for r_ in rows))
    os.makedirs(FC_DIR, exist_ok=True)
    json.dump({"created_at": datetime.now(timezone.utc).isoformat(), "asof": dates[t], "horizon_days": H, "model": "v2", "forecasts": fcs,
               "note": "Research forecasts, not advice. Bands are model probabilities, not guarantees."},
              open(os.path.join(FC_DIR, f"{dates[t]}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
