"""
Train B1 on CPU, compare against baselines on held-out sessions, export ONNX + spec + parity vectors.

Usage: python -m trademaster_ml.train --hist data/hist --out models/b1/v1 [--seeds 3]
Baselines on the same test bars: zero ("no change"), B0 (-0.5 x 30-min residual vs ref ETF), ridge (linear on features).
"""
import argparse, json, os, struct, time

import numpy as np
import torch

from .b1 import B1
from .dataset import FEATURES, build_panel, split_sessions

torch.set_num_threads(max(1, os.cpu_count() // 2))  # leave CPU headroom for the desktop
assert not torch.cuda.is_available() or True       # CPU build; we never move tensors to a GPU


def normalize(panel, train_mask):
    X = panel.X
    flat = X[train_mask].reshape(-1, X.shape[-1])
    mean = np.nanmean(flat, 0)
    std = np.nanstd(flat, 0)
    std[std < 1e-12] = 1.0
    Z = np.clip((X - mean) / std, -5, 5)
    valid = np.isfinite(Z).all(-1)
    Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)
    return Z, valid.astype(np.float32), mean, std


def metrics(pred, y, valid):
    """pred, y in bps, [T, N]; valid [T, N] bool. Stocks only are passed in."""
    p, a = pred[valid], y[valid]
    ic = float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 else 0.0
    nz = (p != 0) & (a != 0)
    hit = float(((p[nz] > 0) == (a[nz] > 0)).mean()) if nz.any() else float("nan")
    sse, sse0 = float(((p - a) ** 2).sum()), float((a ** 2).sum())
    # per-bar cross-sectional rank IC and top-minus-bottom quintile spread (before costs)
    ics, spreads = [], []
    for t in range(pred.shape[0]):
        m = valid[t]
        if m.sum() < 10:
            continue
        pr, ar = pred[t, m], y[t, m]
        if pr.std() == 0:
            continue
        rp, ra = pr.argsort().argsort(), ar.argsort().argsort()
        ics.append(np.corrcoef(rp, ra)[0, 1])
        q = max(1, len(pr) // 5)
        o = pr.argsort()
        spreads.append(ar[o[-q:]].mean() - ar[o[:q]].mean())
    ics = np.array(ics)
    return {
        "n": int(valid.sum()), "ic": ic, "hit_rate": hit,
        "rmse_bps": (sse / len(a)) ** 0.5, "rmse_zero_bps": (sse0 / len(a)) ** 0.5, "r2_oos": 1 - sse / sse0,
        "rank_ic_mean": float(ics.mean()) if len(ics) else float("nan"),
        "rank_ic_t": float(ics.mean() / ics.std() * np.sqrt(len(ics))) if len(ics) > 1 else float("nan"),
        "ls_spread_bps_per_bar": float(np.mean(spreads)) if spreads else float("nan"),
    }


def train_one(Z, M, Y, tr_idx, va_idx, n_inst, seed, epochs=40, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    model = B1(n_inst, Z.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Zt, Mt = torch.from_numpy(Z), torch.from_numpy(M)
    Yt = torch.from_numpy(np.where(np.isfinite(Y), Y, 0.0).astype(np.float32))
    Vt = torch.from_numpy((np.isfinite(Y) & (M > 0)).astype(np.float32))
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            out = model(Zt[b], Mt[b])
            mu, lv = out[..., 0], out[..., 1]
            nll = 0.5 * (lv + (Yt[b] - mu) ** 2 / lv.exp())
            loss = (nll * Vt[b]).sum() / Vt[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Zt[va_idx], Mt[va_idx])[..., 0].numpy()
        v = Vt[va_idx].numpy() > 0
        ic = float(np.corrcoef(pv[v], Y[va_idx][v])[0, 1])
        if ic > best:
            best, best_state, bad = ic, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, ep + 1


def write_f32(path, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}i", *arr.shape))
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="private/data/hist")
    ap.add_argument("--out", default="private/models/b1/v1")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    P = build_panel(a.hist, horizon=a.horizon)
    tr, va, te = split_sessions(P)
    Z, M, mean, std = normalize(P, tr)
    Y = P.y * 1e4  # bps
    has_y = np.isfinite(Y) & (M > 0)
    tr_idx = np.where(tr & has_y.any(1))[0]
    va_idx = np.where(va & has_y.any(1))[0]
    te_idx = np.where(te & has_y.any(1))[0]
    N = len(P.symbols)
    stock = ~P.is_etf
    print(f"panel: {P.X.shape[0]} bars x {N} instruments x {len(FEATURES)} features, sessions {P.session.max() + 1} "
          f"(train {len(tr_idx)}, val {len(va_idx)}, test {len(te_idx)} bars), built in {time.time() - t0:.1f}s")

    # ---- baselines on test (stocks only)
    Yte, Vte = Y[te_idx][:, stock], has_y[te_idx][:, stock]
    res = {"zero": metrics(np.zeros_like(Yte), np.where(Vte, Yte, 0), Vte)}
    b0 = -0.5 * P.X[te_idx][:, stock, FEATURES.index("resid")] * 1e4
    Vb0 = Vte & np.isfinite(b0)
    res["b0"] = metrics(np.where(Vb0, b0, 0), np.where(Vb0, Yte, 0), Vb0)
    # ridge on pooled features, lambda picked on validation
    def xy(idx):
        v = has_y[idx][:, stock]
        return Z[idx][:, stock][v], Y[idx][:, stock][v]
    Xtr, ytr = xy(tr_idx); Xva, yva = xy(va_idx)
    best_l, best_ic, w_best = None, -1e9, None
    for lam in [1e-1, 1e1, 1e3, 1e5]:
        A = np.c_[Xtr, np.ones(len(Xtr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
        ic = np.corrcoef(np.c_[Xva, np.ones(len(Xva))] @ w, yva)[0, 1]
        if ic > best_ic:
            best_l, best_ic, w_best = lam, ic, w
    pr = (np.c_[Z[te_idx][:, stock].reshape(-1, Z.shape[-1]), np.ones(Vte.size)] @ w_best).reshape(Vte.shape)
    res["ridge"] = metrics(np.where(Vte, pr, 0), np.where(Vte, Yte, 0), Vte)
    res["ridge"]["lambda"] = best_l

    # ---- B1, several seeds; the export is chosen by validation IC only (test never used for selection)
    runs = []
    for seed in range(a.seeds):
        ts = time.time()
        model, val_ic, eps = train_one(Z, M, Y, tr_idx, va_idx, N, seed)
        with torch.no_grad():
            pt = model(torch.from_numpy(Z[te_idx]), torch.from_numpy(M[te_idx]))[..., 0].numpy()[:, stock]
        m = metrics(np.where(Vte, pt, 0), np.where(Vte, Yte, 0), Vte)
        m.update(seed=seed, val_ic=val_ic, epochs=eps, train_seconds=round(time.time() - ts, 1))
        runs.append((val_ic, model, m))
        print(f"B1 seed {seed}: val IC {val_ic:+.4f}  test IC {m['ic']:+.4f}  rank IC {m['rank_ic_mean']:+.4f}  "
              f"hit {m['hit_rate']:.3f}  R2 {m['r2_oos']:+.4f}  ({eps} epochs, {m['train_seconds']}s)")
    runs.sort(key=lambda r: r[0], reverse=True)
    _, model, best = runs[0]
    res["b1_selected"] = best
    res["b1_all_seeds"] = [r[2] for r in runs]

    for k in ["zero", "b0", "ridge", "b1_selected"]:
        m = res[k]
        print(f"{k:12s} IC {m['ic']:+.4f}  rank IC {m['rank_ic_mean']:+.4f} (t {m['rank_ic_t']:+.2f})  hit {m['hit_rate']:.3f}  "
              f"RMSE {m['rmse_bps']:.2f} vs zero {m['rmse_zero_bps']:.2f}  R2 {m['r2_oos']:+.4f}  L/S {m['ls_spread_bps_per_bar']:+.2f} bps")

    # ---- export ONNX (fixed N = universe size; batch is dynamic)
    onnx_path = os.path.join(a.out, "model.onnx")
    x0, m0 = torch.from_numpy(Z[te_idx[:1]]), torch.from_numpy(M[te_idx[:1]])
    torch.onnx.export(model, (x0, m0), onnx_path, input_names=["x", "mask"], output_names=["out"],
                      dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "out": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # parity vectors for the C# engine
    k = min(64, len(te_idx))
    with torch.no_grad():
        ref = model(torch.from_numpy(Z[te_idx[:k]]), torch.from_numpy(M[te_idx[:k]])).numpy()
    write_f32(os.path.join(a.out, "parity_x.bin"), Z[te_idx[:k]])
    write_f32(os.path.join(a.out, "parity_mask.bin"), M[te_idx[:k]])
    write_f32(os.path.join(a.out, "parity_out.bin"), ref)

    sess_dates = sorted({str(np.datetime64(int(t), 's').astype('datetime64[D]')) for t in P.ts})
    spec = {
        "name": "b1_cross_attention", "version": "1", "framework": f"torch {torch.__version__} (CPU)",
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "clip": 5.0,
        "symbols": P.symbols, "is_etf": P.is_etf.tolist(), "interval": "5m", "horizon_bars": a.horizon,
        "output": ["mu_bps", "logvar_bps2"], "inputs": {"x": ["batch", len(P.symbols), len(FEATURES)], "mask": ["batch", len(P.symbols)]},
        "sessions": {"train": sess_dates[:40], "val": sess_dates[40:50], "test": sess_dates[50:]},
        "data_source": json.load(open(os.path.join(a.hist, "universe.json"), encoding="utf-8")).get("source", "Yahoo Finance chart API (older download)"), "results_test": res,
    }
    with open(os.path.join(a.out, "model.spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, default=float)
    print(f"exported {onnx_path} + spec + parity vectors ({k} test bars); total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
