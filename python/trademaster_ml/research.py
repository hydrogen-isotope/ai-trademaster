"""
Research harness with a locked hold-out (CPU only).

1. Development period (all sessions except the last --holdout): walk-forward folds train ridge and a B1 ensemble;
   B0 needs no training. Out-of-sample predictions from every fold are pooled.
2. Every (model x trading rule) combination is simulated as an intraday portfolio on the pooled development
   predictions: dollar-neutral, gross exposure 1, flat at each session close, costs charged on every weight change.
   The single best combination by development Sharpe is selected. The full grid is reported (multiple testing).
3. The hold-out sessions are touched exactly once: each model is retrained on the development period and the
   pre-selected combination is evaluated. Other combinations are shown on the hold-out only as context.

Target: forward return over --horizon bars minus the cross-sectional mean of stocks (market move removed).

Usage: python -m trademaster_ml.research --hist data/hist_1h --horizon 1 --holdout 120 --out models/research/1h
"""
import argparse, json, os, time

import numpy as np
import torch

from .dataset import FEATURES, build_panel
from .train import write_f32
from .train_wf import Ensemble, apply_norm, fit, norm_stats, rank_ic

torch.set_num_threads(max(1, os.cpu_count() // 2))

RULES = [("quintile", {}), ("hyst", {"enter": 0.1, "exit": 0.3}), ("hyst", {"enter": 0.2, "exit": 0.5}),
         ("signal", {"smooth": 0.1}), ("signal", {"smooth": 0.3})]


def rule_name(r):
    k, p = r
    return k if not p else k + "(" + ",".join(f"{a}={b}" for a, b in p.items()) + ")"


def cs_demean(Y, stock):
    m = np.nanmean(np.where(stock[None, :], Y, np.nan), axis=1, keepdims=True)
    return Y - m


def target_weights(rule, score, valid, prev, horizon, t):
    """Dollar-neutral weights with sum|w| = 1 over valid names; 0 elsewhere."""
    kind, p = rule
    n = len(score)
    w = np.zeros(n)
    idx = np.where(valid)[0]
    if len(idx) < 10 or np.std(score[idx]) == 0:
        return w
    pct = np.empty(n); pct[idx] = score[idx].argsort().argsort() / (len(idx) - 1)
    if kind == "quintile":
        if t % horizon != 0:                      # rebalance once per horizon
            return np.where(valid, prev, 0.0)
        w[idx[pct[idx] >= 0.8]] = 1; w[idx[pct[idx] <= 0.2]] = -1
    elif kind == "hyst":
        e, x = p["enter"], p["exit"]
        held_long, held_short = prev > 0, prev < 0
        long_ = valid & ((pct >= 1 - e) | (held_long & (pct >= 1 - x)))
        short = valid & ((pct <= e) | (held_short & (pct <= x)))
        w[long_ & ~short] = 1; w[short & ~long_] = -1
    elif kind == "signal":
        z = np.zeros(n); z[idx] = pct[idx] - 0.5
        tgt = z / np.abs(z).sum() if np.abs(z).sum() > 0 else z
        w = (1 - p["smooth"]) * np.where(valid, prev, 0.0) + p["smooth"] * tgt
        w[~valid] = 0
        w -= w[valid].mean() * valid if valid.any() else 0
        s = np.abs(w).sum()
        return w / s if s > 0 else w
    lw, sw = w[w > 0].sum(), -w[w < 0].sum()
    if lw > 0: w[w > 0] *= 0.5 / lw
    if sw > 0: w[w < 0] *= 0.5 / sw
    return w


def simulate(pred, y1, valid, session, horizon, cost_bps, rule):
    """pred [T,N] score, y1 [T,N] next-bar demeaned return (bps), valid [T,N]. Returns daily net/gross/turnover."""
    T, N = pred.shape
    w_prev = np.zeros(N)
    daily = {}
    for t in range(T):
        last_bar = t + 1 >= T or session[t + 1] != session[t]
        ok = valid[t] & np.isfinite(y1[t])
        w = np.zeros(N) if last_bar else target_weights(rule, pred[t], ok, w_prev, horizon, t)
        trade = np.abs(w - w_prev).sum()
        pnl = float(np.nansum(w * np.where(ok, y1[t], 0.0)))
        d = daily.setdefault(int(session[t]), [0.0, 0.0, 0.0])
        d[0] += pnl; d[1] += trade * cost_bps; d[2] += trade
        w_prev = w
    g = np.array([v[0] for v in daily.values()]); c = np.array([v[1] for v in daily.values()])
    to = np.array([v[2] for v in daily.values()])
    net = g - c
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    return {
        "days": int(len(net)), "gross_bps_day": float(g.mean()), "cost_bps_day": float(c.mean()), "net_bps_day": float(net.mean()),
        "turnover_day": float(to.mean()), "sharpe_ann": float(net.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "t_stat": float(net.mean() / sd * np.sqrt(len(net))) if sd > 0 else 0.0,
        "positive_days": float((net > 0).mean()) if len(net) else 0.0, "cum_net_bps": float(net.sum()),
    }


def train_models(P, Z, M, Y, V, stock, tr, va, seeds):
    N = len(P.symbols)
    models = [fit(Z, M, Y, V, tr, va, N, s)[0] for s in range(seeds)]
    ens = Ensemble(models).eval()
    def xy(idx):
        v = (V[idx] > 0) & stock[None, :]
        return Z[idx][v], Y[idx][v]
    Xtr, ytr = xy(tr)
    best = (-1e9, None)
    for lam in [1e1, 1e3, 1e5]:
        A = np.c_[Xtr, np.ones(len(Xtr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
        pv = (np.c_[Z[va].reshape(-1, Z.shape[-1]), np.ones(len(va) * N)] @ w).reshape(len(va), N)
        ic = rank_ic(pv, Y[va], (V[va] > 0) & stock[None, :])
        if ic > best[0]:
            best = (ic, w)
    return ens, best[1]


def predict(P, ens, w_ridge, Z, M, rows):
    N = len(P.symbols)
    with torch.no_grad():
        out = ens(torch.from_numpy(Z[rows]), torch.from_numpy(M[rows])).numpy()
    mu, sig = out[..., 0], np.exp(0.5 * out[..., 1])
    ridge = (np.c_[Z[rows].reshape(-1, Z.shape[-1]), np.ones(len(rows) * N)] @ w_ridge).reshape(len(rows), N)
    b0 = -0.5 * np.nan_to_num(P.X[rows][..., FEATURES.index("resid")]) * 1e4
    return {"b0": b0, "ridge": ridge, "b1": mu, "b1_score": mu / np.maximum(sig, 1e-6)}, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--holdout", type=int, default=120, help="final sessions locked away until the end")
    ap.add_argument("--first-train", type=int, default=250)
    ap.add_argument("--fold-val", type=int, default=20)
    ap.add_argument("--fold-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    P = build_panel(a.hist, horizon=a.horizon)
    S, N = P.session.max() + 1, len(P.symbols)
    stock = ~P.is_etf
    Y = cs_demean(P.y * 1e4, stock)
    Y1 = cs_demean(P.y1 * 1e4, stock)
    dev_end = S - a.holdout
    print(f"{a.hist}: {P.X.shape[0]} bars, {S} sessions ({stock.sum()} stocks); dev sessions [0,{dev_end}), "
          f"hold-out [{dev_end},{S}) locked; horizon {a.horizon} bars; cost {a.cost_bps} bps one-way")

    # ---------------- development walk-forward
    pooled = {k: [] for k in ("b0", "ridge", "b1", "b1_score")}
    pooled_rows, sig_all = [], []
    for start in range(a.first_train, dev_end - a.fold_val - a.fold_test + 1, a.fold_test):
        sel = lambda lo, hi: (P.session >= lo) & (P.session < hi)
        mean, std = norm_stats(P.X, sel(0, start))
        Z, M = apply_norm(P.X, mean, std)
        V = (np.isfinite(Y) & (M > 0)).astype(np.float32)
        Yf = np.where(np.isfinite(Y), Y, 0).astype(np.float32)
        tr = np.where(sel(0, start) & (V.sum(1) > 0))[0]
        va = np.where(sel(start, start + a.fold_val) & (V.sum(1) > 0))[0]
        te = np.where(sel(start + a.fold_val, start + a.fold_val + a.fold_test))[0]
        ens, w_r = train_models(P, Z, M, Yf, V, stock, tr, va, a.seeds)
        preds, sig = predict(P, ens, w_r, Z, M, te)
        for k in pooled: pooled[k].append(preds[k])
        pooled_rows.append(te); sig_all.append(sig)
        print(f"  fold test sessions [{start + a.fold_val},{start + a.fold_val + a.fold_test}) done  ({time.time() - t0:.0f}s)")
    rows = np.concatenate(pooled_rows)
    valid = np.isfinite(P.X[rows]).all(-1) & stock[None, :]
    preds = {k: np.concatenate(v) for k, v in pooled.items()}
    sig = np.concatenate(sig_all)

    # sigma calibration of B1 on development out-of-sample bars
    yv, mv, sv = Y[rows][valid & np.isfinite(Y[rows])], preds["b1"][valid & np.isfinite(Y[rows])], sig[valid & np.isfinite(Y[rows])]
    zres = (yv - mv) / sv
    calib = {"std_z": float(zres.std()), "within_1sigma": float((np.abs(zres) < 1).mean()), "ideal_within_1sigma": 0.683}

    grid = []
    for model in ("b0", "ridge", "b1", "b1_score"):
        ic = rank_ic(preds[model], np.where(np.isfinite(Y[rows]), Y[rows], 0), valid & np.isfinite(Y[rows]))
        for rule in RULES:
            r = simulate(preds[model], Y1[rows], valid, P.session[rows], a.horizon, a.cost_bps, rule)
            r.update(model=model, rule=rule_name(rule), rank_ic=ic)
            grid.append((model, rule, r))
    grid.sort(key=lambda g: g[2]["sharpe_ann"], reverse=True)
    print(f"\nDevelopment out-of-sample ({rows.size} bars, {len(np.unique(P.session[rows]))} sessions), all {len(grid)} combinations:")
    for model, rule, r in grid:
        print(f"  {model:8s} {rule_name(rule):26s} IC {r['rank_ic']:+.4f}  net {r['net_bps_day']:+7.2f} bps/day "
              f"(gross {r['gross_bps_day']:+6.2f}, cost {r['cost_bps_day']:5.2f})  Sharpe {r['sharpe_ann']:+5.2f}  t {r['t_stat']:+5.2f}")
    best_model, best_rule, best_dev = grid[0]
    print(f"Selected on development data: {best_model} + {rule_name(best_rule)}")
    print(f"B1 sigma calibration (dev): std of standardized residual {calib['std_z']:.2f} (ideal 1.00), "
          f"within 1 sigma {calib['within_1sigma']:.2f} (ideal 0.68)")

    # ---------------- hold-out, evaluated once
    mean, std = norm_stats(P.X, P.session < dev_end)
    Z, M = apply_norm(P.X, mean, std)
    V = (np.isfinite(Y) & (M > 0)).astype(np.float32)
    Yf = np.where(np.isfinite(Y), Y, 0).astype(np.float32)
    tr = np.where((P.session < dev_end - a.fold_val) & (V.sum(1) > 0))[0]
    va = np.where((P.session >= dev_end - a.fold_val) & (P.session < dev_end) & (V.sum(1) > 0))[0]
    ho = np.where(P.session >= dev_end)[0]
    ens, w_r = train_models(P, Z, M, Yf, V, stock, tr, va, a.seeds)
    hp, _ = predict(P, ens, w_r, Z, M, ho)
    hvalid = np.isfinite(P.X[ho]).all(-1) & stock[None, :]
    holdout = {}
    for model in ("b0", "ridge", "b1", "b1_score"):
        for rule in RULES:
            r = simulate(hp[model], Y1[ho], hvalid, P.session[ho], a.horizon, a.cost_bps, rule)
            holdout[f"{model} {rule_name(rule)}"] = r
    sel_key = f"{best_model} {rule_name(best_rule)}"
    h = holdout[sel_key]
    print(f"\nHOLD-OUT ({a.holdout} sessions, untouched until now) for the pre-selected {sel_key}:")
    print(f"  net {h['net_bps_day']:+.2f} bps/day (gross {h['gross_bps_day']:+.2f}, cost {h['cost_bps_day']:.2f}), "
          f"Sharpe {h['sharpe_ann']:+.2f}, t {h['t_stat']:+.2f}, positive days {h['positive_days']:.0%}, cum {h['cum_net_bps']:+.0f} bps")
    for c in (3.0,):
        hc = simulate(hp[best_model], Y1[ho], hvalid, P.session[ho], a.horizon, c, best_rule)
        print(f"  same at {c} bps one-way cost: net {hc['net_bps_day']:+.2f} bps/day, Sharpe {hc['sharpe_ann']:+.2f}")
        h[f"net_bps_day_at_{c}bps"] = hc["net_bps_day"]; h[f"sharpe_at_{c}bps"] = hc["sharpe_ann"]

    if best_model.startswith("b1"):
        onnx_path = os.path.join(a.out, "model.onnx")
        torch.onnx.export(ens, (torch.from_numpy(Z[ho[:1]]), torch.from_numpy(M[ho[:1]])), onnx_path, input_names=["x", "mask"],
                          output_names=["out"], dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "out": {0: "batch"}},
                          opset_version=17, dynamo=False)
        k = min(64, len(ho))
        with torch.no_grad():
            ref = ens(torch.from_numpy(Z[ho[:k]]), torch.from_numpy(M[ho[:k]])).numpy()
        for n, arr in (("x", Z[ho[:k]]), ("mask", M[ho[:k]]), ("out", ref)):
            write_f32(os.path.join(a.out, f"parity_{n}.bin"), arr)
    result = {
        "hist": a.hist, "horizon_bars": a.horizon, "cost_bps_one_way": a.cost_bps, "sessions": int(S), "dev_end": int(dev_end),
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "symbols": P.symbols,
        "selected": {"model": best_model, "rule": rule_name(best_rule), "dev": best_dev, "holdout": h},
        "dev_grid": [g[2] for g in grid], "holdout_all": holdout, "b1_sigma_calibration_dev": calib,
        "ridge_weights_final": w_r.tolist(), "elapsed_s": round(time.time() - t0),
    }
    with open(os.path.join(a.out, "research.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"saved {a.out}/research.json; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
