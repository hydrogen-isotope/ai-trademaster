"""
Five-trading-day price forecasts with honest validation (research only; not trading advice). CPU only.

Model (per instrument, on daily log returns of closes; futures use roll-adjusted settlements):
  volatility  GJR-GARCH(1,1) with Student-t shocks, fitted by maximum likelihood (PyTorch L-BFGS on the CPU)
  mean        zero (random walk). A challenger mean (ridge on the instrument's own 1/5/20-day returns) is scored in the
              same validation but is only used if it beats the random walk significantly (pre-registered rule).
  forecast    20,000 simulated 5-day paths -> price quantiles for each of the next 5 trading-day closes

Validation (walk-forward, no foresight): for each of the last 250 origins, fit on closes up to the origin only,
forecast the next 5 closes, then compare with what happened. Baseline: random walk with normal shocks and 60-day
volatility. Pre-registered pass rule per instrument, for every horizon h=1..5:
  |coverage(80% band) - 0.80| <= 0.06  and  |coverage(95% band) - 0.95| <= 0.04
and the average CRPS skill vs the baseline over h=1..5 must be >= -1%.
Instruments that fail are still shown, labelled NOT VALIDATED.

Usage:
  python -m trademaster_ml.daily_forecast validate --symbols MU NVDA AMD AAPL GC SI
  python -m trademaster_ml.daily_forecast forecast --symbols MU NVDA AMD AAPL GC SI
  python -m trademaster_ml.daily_forecast score
"""
import argparse, json, math, os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import torch

from . import daily_data
from .futures import SPECS

torch.set_num_threads(4)
H = 5
N_PATHS = 20000
ORIGINS = 250
MIN_HISTORY = 400
REFIT_EVERY = 5
QS = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
GATE = dict(cov80_tol=0.06, cov95_tol=0.04, min_skill=-0.01)
OUT = os.path.join("private", "reports", "daily_forecast")
FC_DIR = os.path.join("private", "data", "forecasts")

# NYSE full-day closures (verify yearly against NYSE's official calendar). COMEX settlements follow the same
# business days closely enough for choosing the next 5 trading dates; history uses the dates actually observed.
NYSE_HOLIDAYS = {date(2026, 11, 26), date(2026, 12, 25), date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
                 date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
                 date(2027, 11, 25), date(2027, 12, 24)}


def next_trading_days(d, n=H):
    out, x = [], d
    while len(out) < n:
        x += timedelta(days=1)
        if x.weekday() < 5 and x not in NYSE_HOLIDAYS:
            out.append(x)
    return out


# ---------------------------------------------------------------- GJR-GARCH(1,1)-t
class Garch:
    """Returns in percent. Parameters are transformed so that every fit is stationary and positive."""

    def __init__(self):
        self.p = None

    @staticmethod
    def _unpack(th, var0):
        omega = torch.nn.functional.softplus(th[0]) * var0 * 0.05 + 1e-6
        persist = torch.sigmoid(th[1]) * 0.998                    # alpha + gamma/2 + beta < 1
        w = torch.softmax(th[2:5], 0)                              # shares of persistence
        alpha, gamma2, beta = persist * w[0], persist * w[1], persist * w[2]
        gamma = 2 * gamma2
        nu = 2.1 + torch.nn.functional.softplus(th[5])
        return omega, alpha, gamma, beta, nu

    def _nll(self, th, r, var0):
        omega, alpha, gamma, beta, nu = self._unpack(th, var0)
        s2 = var0
        ll = torch.zeros((), dtype=torch.float64)
        c = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(math.pi * (nu - 2))
        for t in range(len(r)):
            e = r[t]
            ll = ll + c - 0.5 * torch.log(s2) - (nu + 1) / 2 * torch.log1p(e * e / (s2 * (nu - 2)))
            s2 = omega + (alpha + gamma * (e < 0).double()) * e * e + beta * s2
        return -ll / len(r)

    def fit(self, r_pct):
        r = torch.tensor(r_pct, dtype=torch.float64)
        var0 = r.var()
        th = torch.tensor([0.0, 3.0, -1.5, -2.0, 2.0, 1.5], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = self._nll(th, r, var0)
            loss.backward()
            return loss
        try:
            opt.step(closure)
            ok = torch.isfinite(th).all().item()
        except Exception:
            ok = False
        if not ok:                                                 # fall back to EWMA-like fixed parameters
            th = torch.tensor([0.0, 4.0, -2.0, -3.0, 2.0, 1.5], dtype=torch.float64)
        with torch.no_grad():
            omega, alpha, gamma, beta, nu = (float(x) for x in self._unpack(th.detach(), var0))
            s2 = float(var0)
            for e in r_pct:
                s2 = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2
        self.p = dict(omega=omega, alpha=alpha, gamma=gamma, beta=beta, nu=nu, next_var=s2, fitted=bool(ok))
        return self

    def simulate(self, mu_pct, n=N_PATHS, h=H, rng=None):
        """Cumulative log returns (fraction) for horizons 1..h, shape [n, h]."""
        rng = rng or np.random.default_rng(0)
        p = self.p
        nu = p["nu"]
        s2 = np.full(n, p["next_var"])
        cum = np.zeros((n, h))
        acc = np.zeros(n)
        for k in range(h):
            z = rng.standard_t(nu, n) * math.sqrt((nu - 2) / nu)
            e = np.sqrt(s2) * z
            acc = acc + (mu_pct[k] if np.ndim(mu_pct) else mu_pct) + e
            cum[:, k] = acc
            s2 = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e * e + p["beta"] * s2
        return cum / 100.0


# ---------------------------------------------------------------- challenger mean
def ridge_mean(r, lam=50.0):
    """Ridge on the instrument's own past: next-day return ~ r1, r5, r20 (all strictly before the target day)."""
    X, y = [], []
    for t in range(20, len(r) - 1):
        X.append([r[t], r[t - 4:t + 1].sum(), r[t - 19:t + 1].sum()]); y.append(r[t + 1])
    X, y = np.array(X), np.array(y)
    sd = X.std(0) + 1e-12
    Z = X / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(3), Z.T @ y) / sd
    x_now = np.array([r[-1], r[-5:].sum(), r[-20:].sum()])
    return float(x_now @ w)


# ---------------------------------------------------------------- scoring helpers
def crps_samples(samples, y):
    s = np.sort(samples[:: max(1, len(samples) // 2000)])
    n = len(s)
    term1 = np.mean(np.abs(s - y))
    i = np.arange(1, n + 1)
    term2 = np.sum((2 * i - n - 1) * s) / (n * n)               # = 0.5 E|X - X'| for sorted samples
    return term1 - term2


def crps_normal(mu, sd, y):
    z = (y - mu) / sd
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def dm_test(d, lag):
    d = np.asarray(d)
    n = len(d)
    m = d.mean()
    x = d - m
    v = x @ x / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    if v <= 0:
        return 0.0, 1.0
    z = m / math.sqrt(v / n)
    return z, math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- validation
def validate_symbol(sym, origins=ORIGINS, log=print):
    d = daily_data.load(sym)
    closes = np.array(d["close"], float)
    dates = d["dates"]
    r = np.diff(np.log(closes))                                   # r[i] = return from day i to day i+1
    T = len(closes)
    last_origin = T - 1 - H
    first_origin = max(MIN_HISTORY, last_origin - origins + 1)
    rng = np.random.default_rng(7)
    recs, model, chall_mu = [], None, 0.0
    for k, o in enumerate(range(first_origin, last_origin + 1)):
        hist = r[:o]                                              # returns up to the origin close only
        if model is None or k % REFIT_EVERY == 0:
            model = Garch().fit(hist[-1000:] * 100)
            chall_mu = ridge_mean(hist[-1000:])
        else:                                                     # roll the variance forward with known returns only
            p = model.p
            e = hist[-1] * 100
            p["next_var"] = p["omega"] + (p["alpha"] + p["gamma"] * (e < 0)) * e * e + p["beta"] * p["next_var"]
        sims = model.simulate(0.0, rng=rng)
        sims_c = sims + chall_mu * np.arange(1, H + 1)
        sd60 = hist[-60:].std()
        actual = np.log(closes[o + 1:o + 1 + H] / closes[o])
        rec = {"origin": dates[o], "close": float(closes[o]), "targets": dates[o + 1:o + 1 + H],
               "actual_price": closes[o + 1:o + 1 + H].tolist(), "h": []}
        for h in range(H):
            y = actual[h]
            s, sc = sims[:, h], sims_c[:, h]
            q = np.quantile(s, QS)
            base_sd = sd60 * math.sqrt(h + 1)
            rec["h"].append(dict(
                y=float(y), q=q.tolist(),
                in50=bool(q[3] <= y <= q[5]), in80=bool(q[2] <= y <= q[6]), in95=bool(q[0] <= y <= q[8]),
                base_in80=bool(abs(y) <= 1.2816 * base_sd), base_in95=bool(abs(y) <= 1.96 * base_sd),
                crps=float(crps_samples(s, y)), crps_chall=float(crps_samples(sc, y)), crps_base=float(crps_normal(0, base_sd, y)),
                ape=float(abs(math.exp(q[4]) - math.exp(y)) / math.exp(y)), ape_rw=float(abs(1 - math.exp(y)) / math.exp(y))))
        recs.append(rec)
        if log and (k + 1) % 50 == 0:
            log(f"  {sym}: {k + 1}/{last_origin - first_origin + 1} origins")
    return summarize(sym, recs), recs


def summarize(sym, recs):
    out = {"symbol": sym, "origins": len(recs), "first_origin": recs[0]["origin"], "last_origin": recs[-1]["origin"], "h": []}
    passed = True
    skills = []
    for h in range(H):
        rows = [r["h"][h] for r in recs]
        cov = {k: float(np.mean([x[k] for x in rows])) for k in ["in50", "in80", "in95", "base_in80", "base_in95"]}
        c, cc, cb = (np.array([x[k] for x in rows]) for k in ["crps", "crps_chall", "crps_base"])
        skill = 1 - c.mean() / cb.mean()
        skill_ch = 1 - cc.mean() / c.mean()
        z, p = dm_test(cb - c, h)
        zc, pc = dm_test(c - cc, h)
        ok = abs(cov["in80"] - 0.80) <= GATE["cov80_tol"] and abs(cov["in95"] - 0.95) <= GATE["cov95_tol"]
        passed &= ok
        skills.append(skill)
        out["h"].append(dict(h=h + 1, coverage=cov, crps_skill_vs_baseline=float(skill), dm_p=float(p),
                             challenger_skill_vs_model=float(skill_ch), challenger_dm_p=float(pc),
                             mape_model=float(np.mean([x["ape"] for x in rows])), mape_rw=float(np.mean([x["ape_rw"] for x in rows])),
                             calibrated=bool(ok)))
    out["avg_skill"] = float(np.mean(skills))
    out["passed"] = bool(passed and out["avg_skill"] >= GATE["min_skill"])
    # pre-registered: use the challenger mean only if it beats the random walk at every horizon with p < 0.05
    out["use_challenger_mean"] = all(x["challenger_skill_vs_model"] > 0 and x["challenger_dm_p"] < 0.05 for x in out["h"])
    out["gate"] = GATE
    return out


# ---------------------------------------------------------------- live forecasts
def forecast_symbol(sym, validation=None, rng=None, asof=None):
    """Forecast the 5 trading days after the last close (or after `asof`: only closes up to that date are used)."""
    d = daily_data.load(sym)
    future = None
    if asof:
        k = max(i for i, x in enumerate(d["dates"]) if x <= asof)
        future = {"dates": d["dates"][k + 1:k + 1 + H], "close": d["close"][k + 1:k + 1 + H]}
        d = {kk: (v[:k + 1] if isinstance(v, list) else v) for kk, v in d.items()}
    closes = np.array(d["close"], float)
    r = np.diff(np.log(closes))
    m = Garch().fit(r[-1000:] * 100)
    mu = ridge_mean(r[-1000:]) * 100 if validation and validation.get("use_challenger_mean") else 0.0
    sims = m.simulate(mu, rng=rng or np.random.default_rng(11))
    last = closes[-1]
    asof_d = date.fromisoformat(d["dates"][-1])
    asof = asof_d
    days = [date.fromisoformat(x) for x in future["dates"]] if future and len(future["dates"]) == H else next_trading_days(asof)
    rows = []
    for h in range(H):
        q = np.quantile(sims[:, h], QS)
        rows.append({"date": str(days[h]), **{f"p{int(x * 1000) / 10:g}".replace(".", "_"): float(last * math.exp(v)) for x, v in zip(QS, q)}})
    is_future = sym in SPECS
    return {"symbol": sym, "asof": str(asof), "last_close": float(last), "kind": "future" if is_future else "stock",
            "price_basis": "roll-adjusted settlement (front contract by open interest)" if is_future else "official close",
            "contract": d.get("contract", [None])[-1] if is_future else None,
            "validated": bool(validation and validation.get("passed")), "garch": m.p, "mean_model": "ridge" if mu else "random walk",
            "days": rows, "source": d["source"],
            "actual": future}


def score_forecasts():
    """Fill in actual closes for past forecast files as they become available."""
    if not os.path.isdir(FC_DIR):
        return []
    res = []
    for f in sorted(os.listdir(FC_DIR)):
        if not f.endswith(".json"):
            continue
        p = os.path.join(FC_DIR, f)
        doc = json.load(open(p))
        for fc in doc["forecasts"]:
            try:
                d = daily_data.load(fc["symbol"])
            except FileNotFoundError:
                continue
            px = dict(zip(d["dates"], d["close"]))
            for row in fc["days"]:
                if row["date"] in px:
                    row["actual"] = px[row["date"]]
                    row["in80"] = row["p10"] <= row["actual"] <= row["p90"]
                    row["in95"] = row["p2_5"] <= row["actual"] <= row["p97_5"]
                    row["abs_pct_error_median"] = abs(row["p50"] / row["actual"] - 1) * 100
        json.dump(doc, open(p, "w"), indent=1)
        res.append(doc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "forecast", "score", "holdout"])
    ap.add_argument("--asof", help="holdout: last date the model may see (e.g. 2026-10-09)")
    ap.add_argument("--symbols", nargs="+", default=["MU", "NVDA", "AMD", "AAPL", "GC", "SI"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    avail = [s for s in a.symbols if os.path.exists(os.path.join(daily_data.OUT, f"{s}.json"))]
    missing = [s for s in a.symbols if s not in avail]
    if missing:
        print("no daily data yet for:", " ".join(missing), "(run python -m trademaster_ml.daily_data)")
    if a.cmd == "validate":
        allsum = {}
        for s in avail:
            summ, recs = validate_symbol(s)
            allsum[s] = summ
            json.dump(recs, open(os.path.join(OUT, f"validation_{s}.json"), "w"))
            h5 = summ["h"][-1]
            print(f"{s:5s} {'PASS' if summ['passed'] else 'FAIL'}  80%-band hit h1..5: "
                  + " ".join(f"{x['coverage']['in80']:.2f}" for x in summ["h"])
                  + f"  95%: " + " ".join(f"{x['coverage']['in95']:.2f}" for x in summ["h"])
                  + f"  CRPS skill avg {summ['avg_skill'] * 100:+.1f}%  median error h5 {h5['mape_model'] * 100:.2f}% (RW {h5['mape_rw'] * 100:.2f}%)"
                  + ("  challenger mean adopted" if summ["use_challenger_mean"] else ""))
        json.dump({"validated_at": datetime.now(timezone.utc).isoformat(), "symbols": allsum},
                  open(os.path.join(OUT, "validation_summary.json"), "w"), indent=1)
    elif a.cmd == "forecast":
        vpath = os.path.join(OUT, "validation_summary.json")
        val = json.load(open(vpath))["symbols"] if os.path.exists(vpath) else {}
        fcs = [forecast_symbol(s, val.get(s)) for s in avail]
        asof = max(f["asof"] for f in fcs)
        os.makedirs(FC_DIR, exist_ok=True)
        doc = {"created_at": datetime.now(timezone.utc).isoformat(), "asof": asof, "horizon_days": H, "forecasts": fcs,
               "note": "Research forecasts, not advice. Bands are model probabilities, not guarantees."}
        json.dump(doc, open(os.path.join(FC_DIR, f"{asof}.json"), "w"), indent=1)
        for f in fcs:
            tag = "validated" if f["validated"] else "NOT VALIDATED"
            print(f"{f['symbol']:5s} last {f['last_close']:.2f} ({f['asof']}) [{tag}]  "
                  + "  ".join(f"{r['date'][5:]}: {r['p50']:.2f} [{r['p10']:.2f}-{r['p90']:.2f}]" for r in f["days"]))
    elif a.cmd == "holdout":
        fcs = [forecast_symbol(s, asof=a.asof) for s in avail]
        res = []
        for f in fcs:
            act = f["actual"]
            rows = []
            for r, px in zip(f["days"], act["close"]):
                rows.append({**r, "actual": px, "in80": r["p10"] <= px <= r["p90"], "in95": r["p2_5"] <= px <= r["p97_5"],
                             "err_pct": (r["p50"] / px - 1) * 100, "err_nochange_pct": (f["last_close"] / px - 1) * 100})
            f["days"] = rows
            res.append(f)
            print(f"{f['symbol']:5s} fit to {f['asof']} close {f['last_close']:.2f}: "
                  + "  ".join(f"{r['date'][5:]} pred {r['p50']:.2f} act {r['actual']:.2f} ({'in' if r['in80'] else 'OUT'})" for r in rows))
        json.dump({"asof": a.asof, "created_at": datetime.now(timezone.utc).isoformat(), "forecasts": res},
                  open(os.path.join(OUT, f"holdout_{a.asof}.json"), "w"), indent=1)
    else:
        docs = score_forecasts()
        print(f"scored {len(docs)} forecast files")


if __name__ == "__main__":
    main()
