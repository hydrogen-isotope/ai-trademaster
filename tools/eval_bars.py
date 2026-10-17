"""
Prediction-vs-actual check of the B0 factor-residual baseline on real 1-minute bars.

Point-in-time: the prediction at bar t uses closes up to and including t only. Beta is estimated with an
exponentially weighted covariance up to t-1. Overnight gaps are excluded (returns reset each session).
Parameters are the engine defaults (config/default.json), NOT fitted to this data.

Usage: py -I tools/eval_bars.py data/live MU SMH --lookback 30 --horizon 10 --kappa 0.5 --halflife 300
Output: JSON summary on stdout and <dir>/<SYM>-eval.json with the per-bar series.
"""
import argparse, json, math, os, sys
from datetime import datetime, timezone, timedelta


def load(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    out = {}
    for ts, c in zip(r["timestamp"], q["close"]):
        if c is not None and c > 0:
            out[ts] = float(c)
    return out, r["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("symbol")
    ap.add_argument("ref")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--halflife", type=int, default=300)
    a = ap.parse_args()

    px, meta = load(os.path.join(a.dir, f"{a.symbol}-1m.json"))
    ref, _ = load(os.path.join(a.dir, f"{a.ref}-1m.json"))
    now = datetime.now(timezone.utc).timestamp()
    # Drop the still-forming current minute.
    ts = sorted(t for t in px if t in ref and t + 60 <= now)
    ny = timezone(timedelta(hours=-4))  # EDT in October
    day = [datetime.fromtimestamp(t, ny).date() for t in ts]

    alpha = 1 - 0.5 ** (1 / a.halflife)
    cov = var = 0.0
    beta = 1.0
    res_hist = []
    rows = []
    for i, t in enumerate(ts):
        if i == 0 or day[i] != day[i - 1]:
            res_hist = []          # new session: no overnight returns
            rows.append(None)
            continue
        r = math.log(px[t] / px[ts[i - 1]])
        rr = math.log(ref[t] / ref[ts[i - 1]])
        res = r - beta * rr          # beta known before this bar
        cov = (1 - alpha) * cov + alpha * r * rr
        var = (1 - alpha) * var + alpha * rr * rr
        if var > 1e-14:
            beta = max(-3.0, min(3.0, cov / var))
        res_hist.append(res)
        if len(res_hist) < a.lookback:
            rows.append(None)
            continue
        win = res_hist[-a.lookback:]
        dev = sum(win)
        sigma = math.sqrt(sum(x * x for x in win) / a.lookback * a.horizon)
        mu = -a.kappa * dev
        j = i + a.horizon
        actual = math.log(px[ts[j]] / px[t]) if j < len(ts) and day[j] == day[i] else None
        rows.append({"t": t, "price": px[t], "mu": mu, "sigma": sigma, "beta": beta, "actual": actual})

    evals = [r for r in rows if r and r["actual"] is not None]
    pend = [r for r in rows if r and r["actual"] is None]

    def stats(rs):
        n = len(rs)
        if n < 3:
            return {"n": n}
        mu = [r["mu"] for r in rs]; ac = [r["actual"] for r in rs]
        mm, ma = sum(mu) / n, sum(ac) / n
        cv = sum((x - mm) * (y - ma) for x, y in zip(mu, ac))
        sx = math.sqrt(sum((x - mm) ** 2 for x in mu)); sy = math.sqrt(sum((y - ma) ** 2 for y in ac))
        corr = cv / (sx * sy) if sx > 0 and sy > 0 else 0.0
        hit = sum(1 for x, y in zip(mu, ac) if x * y > 0) / sum(1 for x, y in zip(mu, ac) if x * y != 0)
        strong = [(x, y) for x, y, r in zip(mu, ac, rs) if abs(x) / r["sigma"] >= 1.0]
        hit_strong = (sum(1 for x, y in strong if x * y > 0) / len(strong)) if strong else None
        rmse_model = math.sqrt(sum((x - y) ** 2 for x, y in zip(mu, ac)) / n)
        rmse_zero = math.sqrt(sum(y * y for y in ac) / n)
        return {"n": n, "corr": corr, "hit_rate": hit, "strong_signals": len(strong), "hit_rate_strong": hit_strong,
                "rmse_model_bps": rmse_model * 1e4, "rmse_zero_bps": rmse_zero * 1e4}

    days = sorted({datetime.fromtimestamp(r["t"], ny).date() for r in evals + pend})
    today = days[-1] if days else None
    summary = {
        "symbol": a.symbol, "reference": a.ref, "source": "Yahoo Finance chart API (unofficial, delayed/personal-use)",
        "params": vars(a), "bars": len(ts), "first": datetime.fromtimestamp(ts[0], ny).isoformat(),
        "last": datetime.fromtimestamp(ts[-1], ny).isoformat(), "last_price": px[ts[-1]],
        "all_days": stats(evals),
        "today": stats([r for r in evals if datetime.fromtimestamp(r["t"], ny).date() == today]),
        "by_day": {str(d): stats([r for r in evals if datetime.fromtimestamp(r["t"], ny).date() == d]) for d in days},
        "pending_predictions": len(pend),
    }
    with open(os.path.join(a.dir, f"{a.symbol}-eval.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": [r for r in rows if r]}, f)
    json.dump(summary, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
