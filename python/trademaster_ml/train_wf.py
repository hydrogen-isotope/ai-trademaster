"""
Walk-forward evaluation of B1 (3-seed ensemble per fold) against baselines, on a cross-sectional target:
each stock's forward return minus the average forward return of all stocks at that bar (market move removed).
Reports per-fold and pooled out-of-sample metrics plus a long/short quintile backtest net of costs.
CPU only.

Usage: python -m trademaster_ml.train_wf --hist data/hist2 --out models/b1/v2
"""
import argparse, json, os, time

import numpy as np
import torch

from .b1 import B1
from .dataset import FEATURES, build_panel
from .train import write_f32

torch.set_num_threads(max(1, os.cpu_count() // 2))


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, mask):
        return torch.stack([m(x, mask) for m in self.models], 0).mean(0)


def norm_stats(X, rows):
    flat = X[rows].reshape(-1, X.shape[-1])
    mean, std = np.nanmean(flat, 0), np.nanstd(flat, 0)
    std[std < 1e-12] = 1.0
    return mean, std


def apply_norm(X, mean, std):
    Z = np.clip((X - mean) / std, -5, 5)
    valid = np.isfinite(Z).all(-1)
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32), valid.astype(np.float32)


def fit(Z, M, Y, V, tr, va, n_inst, seed, epochs=30, patience=5, d=32):
    torch.manual_seed(seed); np.random.seed(seed)
    model = B1(n_inst, Z.shape[-1], d=d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Zt, Mt, Yt, Vt = map(torch.from_numpy, (Z, M, Y, V))
    best, state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        for i in range(0, len(tr), 64):
            b = np.random.permutation(tr)[:64] if i == 0 else tr[np.random.randint(0, len(tr), 64)]
            out = model(Zt[b], Mt[b])
            mu, lv = out[..., 0], out[..., 1]
            loss = (0.5 * (lv + (Yt[b] - mu) ** 2 / lv.exp()) * Vt[b]).sum() / Vt[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Zt[va], Mt[va])[..., 0].numpy()
        v = V[va] > 0
        ic = rank_ic(p, Y[va], v)
        if ic > best:
            best, state, bad = ic, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model.eval(), best


def rank_ic(pred, y, valid):
    ics = []
    for t in range(pred.shape[0]):
        m = valid[t]
        if m.sum() < 10 or pred[t, m].std() == 0:
            continue
        ics.append(np.corrcoef(pred[t, m].argsort().argsort(), y[t, m].argsort().argsort())[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def evaluate(pred, y, valid, horizon, cost_bps):
    """pred/y [T, N] bps (cross-sectionally demeaned target), valid [T, N]."""
    ics, rows = [], []
    for t in range(pred.shape[0]):
        m = valid[t]
        if m.sum() < 10 or pred[t, m].std() == 0:
            continue
        ics.append(np.corrcoef(pred[t, m].argsort().argsort(), y[t, m].argsort().argsort())[0, 1])
    ics = np.array(ics)
    # Long top quintile / short bottom quintile, rebalanced every `horizon` bars (non-overlapping holds).
    gross, turnover, prev_long, prev_short = [], [], set(), set()
    for t in range(0, pred.shape[0], horizon):
        m = np.where(valid[t])[0]
        if len(m) < 10 or pred[t, m].std() == 0:
            continue
        q = max(1, len(m) // 5)
        o = m[pred[t, m].argsort()]
        longs, shorts = set(o[-q:]), set(o[:q])
        gross.append(y[t, list(longs)].mean() - y[t, list(shorts)].mean())
        changed = (len(longs - prev_long) + len(shorts - prev_short)) / (2 * q)
        turnover.append(changed)
        prev_long, prev_short = longs, shorts
    gross, turnover = np.array(gross), np.array(turnover)
    # cost: each changed name trades once out and once in on each side -> 2 trades x 2 legs per changed slot
    net = gross - turnover * 2 * 2 * cost_bps / 2
    p, a = pred[valid], y[valid]
    return {
        "bars": int(pred.shape[0]), "rank_ic": float(ics.mean()) if len(ics) else 0.0,
        "rank_ic_t_eff": float(ics.mean() / ics.std() * np.sqrt(len(ics) / horizon)) if len(ics) > 2 else 0.0,
        "ic": float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 else 0.0,
        "hit": float(((p > 0) == (a > 0))[(p != 0) & (a != 0)].mean()) if ((p != 0) & (a != 0)).any() else 0.0,
        "ls_gross_bps": float(gross.mean()) if len(gross) else 0.0,
        "ls_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "ls_net_bps": float(net.mean()) if len(net) else 0.0,
        "ls_net_cum_bps": float(net.sum()) if len(net) else 0.0,
        "rebalances": int(len(net)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist2")
    ap.add_argument("--out", default="private/models/b1/v2")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-train", type=int, default=30)
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=1.5, help="one-way cost per trade, bps (half spread + fees)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    P = build_panel(a.hist, horizon=a.horizon)
    N, S = len(P.symbols), P.session.max() + 1
    stock = ~P.is_etf
    Yraw = P.y * 1e4
    # cross-sectional target: subtract the mean forward return of valid stocks at each bar
    mkt = np.nanmean(np.where(stock[None, :], Yraw, np.nan), axis=1, keepdims=True)
    Y = Yraw - mkt
    print(f"panel {P.X.shape} ({S} sessions, {stock.sum()} stocks + {(~stock).sum()} ETFs), built {time.time() - t0:.1f}s")

    names = ["zero", "b0", "ridge", "b1"]
    pooled = {k: {"pred": [], "y": [], "v": []} for k in names}
    folds = []
    last = None
    for start in range(a.first_train, S - a.val - a.test + 1, a.test):
        tr_s, va_s, te_s = (0, start), (start, start + a.val), (start + a.val, start + a.val + a.test)
        rows = lambda r: (P.session >= r[0]) & (P.session < r[1])
        mean, std = norm_stats(P.X, rows(tr_s))
        Z, M = apply_norm(P.X, mean, std)
        V = (np.isfinite(Y) & (M > 0)).astype(np.float32)
        Yf = np.where(np.isfinite(Y), Y, 0).astype(np.float32)
        tr, va, te = (np.where(rows(r) & (V.sum(1) > 0))[0] for r in (tr_s, va_s, te_s))

        models, vals = [], []
        for seed in range(a.seeds):
            m, v = fit(Z, M, Yf, V, tr, va, N, seed)
            models.append(m); vals.append(v)
        ens = Ensemble(models).eval()
        with torch.no_grad():
            p_b1 = ens(torch.from_numpy(Z[te]), torch.from_numpy(M[te]))[..., 0].numpy()
        vt = (V[te] > 0) & stock[None, :]
        # ridge on pooled stock rows, lambda on validation rank IC
        def xy(idx):
            v = (V[idx] > 0) & stock[None, :]
            return Z[idx][v], Yf[idx][v]
        Xtr, ytr = xy(tr)
        best = (-1e9, None)
        for lam in [1e1, 1e3, 1e5]:
            A = np.c_[Xtr, np.ones(len(Xtr))]
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
            pv = (np.c_[Z[va].reshape(-1, Z.shape[-1]), np.ones(len(va) * N)] @ w).reshape(len(va), N)
            ic = rank_ic(pv, Yf[va], (V[va] > 0) & stock[None, :])
            if ic > best[0]:
                best = (ic, w)
        p_ridge = (np.c_[Z[te].reshape(-1, Z.shape[-1]), np.ones(len(te) * N)] @ best[1]).reshape(len(te), N)
        p_b0 = -0.5 * np.nan_to_num(P.X[te][..., FEATURES.index("resid")]) * 1e4
        preds = {"zero": np.zeros_like(p_b1), "b0": p_b0, "ridge": p_ridge, "b1": p_b1}
        fold = {"train_sessions": tr_s, "val_sessions": va_s, "test_sessions": te_s, "b1_val_rank_ic": vals}
        for k in names:
            fold[k] = evaluate(preds[k], Yf[te], vt, a.horizon, a.cost_bps)
            pooled[k]["pred"].append(preds[k]); pooled[k]["y"].append(Yf[te]); pooled[k]["v"].append(vt)
        folds.append(fold)
        print(f"fold test sessions {te_s}: " + "  ".join(
            f"{k} IC {fold[k]['rank_ic']:+.3f} net {fold[k]['ls_net_bps']:+.2f}" for k in names[1:]))
        last = (ens, Z, M, te, mean, std, tr_s, va_s, te_s)

    summary = {}
    for k in names:
        summary[k] = evaluate(np.concatenate(pooled[k]["pred"]), np.concatenate(pooled[k]["y"]), np.concatenate(pooled[k]["v"]),
                              a.horizon, a.cost_bps)
    print(f"\nPooled out-of-sample ({len(folds)} folds x {a.test} sessions), cost {a.cost_bps} bps one-way:")
    for k in names[1:]:
        s = summary[k]
        print(f"  {k:6s} rank IC {s['rank_ic']:+.4f} (t_eff {s['rank_ic_t_eff']:+.2f})  hit {s['hit']:.3f}  "
              f"L/S gross {s['ls_gross_bps']:+.2f} bps/rebalance, turnover {s['ls_turnover']:.2f}, net {s['ls_net_bps']:+.2f} "
              f"(cum {s['ls_net_cum_bps']:+.0f} bps over {s['rebalances']} rebalances)")

    # export the most recent fold's ensemble
    ens, Z, M, te, mean, std, tr_s, va_s, te_s = last
    onnx_path = os.path.join(a.out, "model.onnx")
    torch.onnx.export(ens, (torch.from_numpy(Z[te[:1]]), torch.from_numpy(M[te[:1]])), onnx_path,
                      input_names=["x", "mask"], output_names=["out"],
                      dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "out": {0: "batch"}}, opset_version=17, dynamo=False)
    k = min(64, len(te))
    with torch.no_grad():
        ref = ens(torch.from_numpy(Z[te[:k]]), torch.from_numpy(M[te[:k]])).numpy()
    write_f32(os.path.join(a.out, "parity_x.bin"), Z[te[:k]])
    write_f32(os.path.join(a.out, "parity_mask.bin"), M[te[:k]])
    write_f32(os.path.join(a.out, "parity_out.bin"), ref)
    spec = {
        "name": "b1_cross_attention_ensemble", "version": "2", "framework": f"torch {torch.__version__} (CPU)",
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "clip": 5.0, "symbols": P.symbols,
        "is_etf": P.is_etf.tolist(), "interval": "5m", "horizon_bars": a.horizon,
        "target": "forward return minus cross-sectional mean of stocks (bps)", "output": ["mu_bps", "logvar_bps2"],
        "export_fold": {"train": tr_s, "val": va_s, "test": te_s}, "seeds": a.seeds, "cost_bps_one_way": a.cost_bps,
        "walk_forward": {"folds": folds, "pooled": summary},
        "data_source": json.load(open(os.path.join(a.hist, "universe.json"), encoding="utf-8")).get("source", "Yahoo Finance chart API (older download)"),
    }
    with open(os.path.join(a.out, "model.spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"exported {onnx_path}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
