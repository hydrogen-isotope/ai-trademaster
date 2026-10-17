"""
Buy / sell indicators from the 30-minute cross-stock model (research only, not trading advice).

At each checkpoint (every 30 minutes from 09:35) the model predicts each stock's return over the next 30 minutes
relative to the average stock. A signal fires only when the prediction is both unusual and bigger than the cost:
  BUY / SELL           prediction beyond the training week's 95th / 5th percentile, and |prediction| > round-trip cost
  STRONG BUY / SELL    beyond the 99th / 1st percentile
Thresholds and weights come from the previous week only (frozen), so every signal shown is out of sample.
"BUY" means "expected to beat the other stocks over the next 30 minutes", not "the market will rise".

Usage: python -m trademaster_ml.signals --out private/reports/signals
"""
import argparse, json, os

import numpy as np

from .weektest import load, session
from .rolling import weeks_of, fit, features, target, SKIP

H = 30
STEP = 30
COST_ONE_WAY = 1.0             # bps, scenario (no quote data yet)


def checkpoints(T):
    return list(range(SKIP, T - H, STEP))


def predict_rel(sess, d, beta, w, ref, stocks):
    c, v = sess[d]
    mu = (features(c, v, beta, ref) @ w)[:, stocks]
    return mu - np.nanmean(mu, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist_1m_30d")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.abspath(a.out).startswith(os.path.abspath("private") + os.sep):
        raise SystemExit("--out must be inside private/")
    syms, ts, days, close, vol, ref, stocks = load(a.hist)
    names = [s for s, k in zip(syms, stocks) if k]
    wk = weeks_of(sorted(set(days)))
    sess = {d: session(close, vol, days, d) for w in wk for d in w}
    sigs = []
    for k in range(len(wk) - 1):
        train, test = wk[k], wk[k + 1]
        beta, w = fit(sess, train, ref, stocks, H)
        # thresholds from the training week's own predictions at the same checkpoints
        tr = np.concatenate([predict_rel(sess, d, beta, w, ref, stocks)[checkpoints(len(sess[d][0]))].ravel() for d in train])
        tr = tr[np.isfinite(tr)]
        q = dict(zip(["p1", "p5", "p95", "p99"], np.percentile(tr, [1, 5, 95, 99])))
        floor = 2 * COST_ONE_WAY
        for d in test:
            c, _ = sess[d]
            mu = predict_rel(sess, d, beta, w, ref, stocks)
            y = target(c, H)[:, stocks]
            yrel = y - np.nanmean(y, axis=1, keepdims=True)
            dts = ts[days == d]
            for t in checkpoints(len(c)):
                for i in range(len(names)):
                    m = mu[t, i]
                    if not np.isfinite(m):
                        continue
                    if m >= max(q["p95"], floor):
                        kind = "STRONG BUY" if m >= q["p99"] else "BUY"
                    elif m <= min(q["p5"], -floor):
                        kind = "STRONG SELL" if m <= q["p1"] else "SELL"
                    else:
                        continue
                    side = 1 if "BUY" in kind else -1
                    sigs.append(dict(date=str(d), ts=int(dts[t]), t=t, symbol=names[i], signal=kind, pred_bps=float(m),
                                     price=float(c[t, stocks][i]), actual_rel_bps=float(yrel[t, i]), actual_bps=float(y[t, i]),
                                     net_rel_bps=float(side * yrel[t, i] - 2 * COST_ONE_WAY)))
    out = dict(horizon_min=H, cost_one_way_bps=COST_ONE_WAY, signals=sigs, summary={})
    for kind in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        s = [x for x in sigs if x["signal"] == kind and np.isfinite(x["actual_rel_bps"])]   # scorable only
        if s:
            side = 1 if "BUY" in kind else -1
            out["summary"][kind] = dict(count=len(s), per_day=len(s) / sum(len(w) for w in wk[1:]),
                                        hit_rate=float(np.mean([side * x["actual_rel_bps"] > 0 for x in s])),
                                        avg_pred_bps=float(np.mean([x["pred_bps"] for x in s])),
                                        avg_actual_rel_bps=float(np.mean([x["actual_rel_bps"] for x in s])),
                                        avg_net_bps=float(np.mean([x["net_rel_bps"] for x in s])))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "signals.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
