"""
Train the B1 cross-stock model on the ENGINE's own features and export it for stage 5 (CPU only).

Input: one or more feature dumps written by the engine
  trademaster run --dump-features --out private/reports/<run>        ->  <run>/features.bin + features.json
Target: log return of the mid over --horizon cycles minus the cross-sectional mean of stocks (bps), so the model
learns relative moves across the universe. Split chronologically: train / validation / test (by cycle).
Output (to --out, under private/models/):
  model.onnx            inputs x[B,N,F], mask[B,N]; outputs out[B,N,2] (mu_bps, logvar) and attn[B,N,N]
  model.spec.json       feature_source="engine", features, mean/std, clip, horizon_cycles, sigma_scale, symbols, metrics
  parity_x/mask/out.bin vectors for the C# parity test

Usage: python -m trademaster_ml.engine_train --dumps private/reports/run1 private/reports/run2 --out private/models/engine/b1-v1
Synthetic dumps embed known signals by construction; a model trained on them proves nothing about real markets.
"""
import argparse, json, os, struct, time

import numpy as np
import torch

from .b1 import B1, WithAttention
from .train import write_f32
from .train_wf import fit, rank_ic

torch.set_num_threads(max(1, os.cpu_count() // 2))


def read_dump(run_dir):
    meta = json.load(open(os.path.join(run_dir, "features.json"), encoding="utf-8"))
    with open(os.path.join(run_dir, "features.bin"), "rb") as f:
        if f.read(4) != b"TMFD":
            raise SystemExit(f"{run_dir}/features.bin is not a TMFD feature dump")
        version, n, nf = struct.unpack("<iii", f.read(12))
        if version != 1:
            raise SystemExit(f"unsupported dump version {version}")
        rec = np.dtype([("ts", "<i8"), ("open", "<i4"), ("mid", "<f4", (n,)), ("mask", "u1", (n,)), ("x", "<f4", (n * nf,))])
        data = np.frombuffer(f.read(), dtype=rec)
    x = data["x"].reshape(len(data), n, nf).astype(np.float32)
    return meta, data["ts"], data["open"].astype(bool), data["mid"].astype(np.float64), data["mask"].astype(bool), x


def build(dumps, horizon):
    """Concatenate dumps; target = horizon-cycle log return minus stock cross-sectional mean (bps)."""
    metas, X, M, Y, OPEN = [], [], [], [], []
    for d in dumps:
        meta, ts, op, mid, mask, x = read_dump(d)
        if metas and (meta["symbols"] != metas[0]["symbols"] or meta["features"] != metas[0]["features"]):
            raise SystemExit("all dumps must share the same universe and feature list")
        metas.append(meta)
        T = len(ts)
        y = np.full(mid.shape, np.nan)
        cad = meta["cadence_ms"] * 1_000_000
        for t in range(T - horizon):
            if ts[t + horizon] - ts[t] == horizon * cad and op[t] and op[t + horizon]:   # contiguous, both in session
                with np.errstate(all="ignore"):
                    y[t] = np.log(mid[t + horizon] / mid[t]) * 1e4
        X.append(x); M.append(mask & op[:, None]); Y.append(y); OPEN.append(op)
    meta = metas[0]
    X, M, Y = np.concatenate(X), np.concatenate(M), np.concatenate(Y)
    stock = np.array([t == "Stock" for t in meta["types"]])
    with np.errstate(all="ignore"):
        Y = Y - np.nanmean(np.where(stock[None, :], Y, np.nan), axis=1, keepdims=True)
    return meta, X, M, Y, stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=10, help="prediction horizon in engine cycles")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    a = ap.parse_args()
    if not os.path.normpath(a.out).replace("\\", "/").startswith("private/"):
        raise SystemExit("--out must be under private/ (trained models are never committed)")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    meta, X, M, Y, stock = build(a.dumps, a.horizon)
    T, N, Fn = X.shape
    i_tr, i_va = int(T * a.train), int(T * (a.train + a.val))
    rows = np.arange(T)
    has = np.isfinite(Y) & M
    tr = rows[(rows < i_tr) & has.any(1)]
    va = rows[(rows >= i_tr) & (rows < i_va) & has.any(1)]
    te = rows[(rows >= i_va) & has.any(1)]
    flat = X[:i_tr][M[:i_tr]]
    mean, std = flat.mean(0), flat.std(0)
    std[std < 1e-12] = 1.0
    Z = np.clip((X - mean) / std, -a.clip, a.clip)
    Z = np.where(np.isfinite(Z) & M[..., None], Z, 0.0).astype(np.float32)
    Mf = M.astype(np.float32)
    V = (np.isfinite(Y) & M).astype(np.float32)
    Yf = np.where(np.isfinite(Y), Y, 0.0).astype(np.float32)
    print(f"{len(a.dumps)} dump(s): {T} cycles x {N} instruments x {Fn} features; train {len(tr)}, val {len(va)}, test {len(te)}")

    models = [fit(Z, Mf, Yf, V, tr, va, N, seed)[0] for seed in range(a.seeds)]
    ens = WithAttention(models).eval()
    with torch.no_grad():
        out_va, _ = ens(torch.from_numpy(Z[va]), torch.from_numpy(Mf[va]))
        out_te, _ = ens(torch.from_numpy(Z[te]), torch.from_numpy(Mf[te]))
    out_va, out_te = out_va.numpy(), out_te.numpy()

    # sigma calibration on validation: scale so standardized residuals have std 1
    v = (V[va] > 0) & stock[None, :]
    z = (Yf[va][v] - out_va[..., 0][v]) / np.exp(0.5 * out_va[..., 1][v])
    sigma_scale = float(np.std(z)) if z.size else 1.0
    vt = (V[te] > 0) & stock[None, :]
    metrics = {
        "val_rank_ic": rank_ic(out_va[..., 0], Yf[va], v), "test_rank_ic": rank_ic(out_te[..., 0], Yf[te], vt),
        "sigma_scale": sigma_scale, "cycles": int(T), "test_cycles": int(len(te)),
    }

    onnx_path = os.path.join(a.out, "model.onnx")
    k = max(1, min(64, len(te)))
    x0, m0 = torch.from_numpy(Z[te[:1]]), torch.from_numpy(Mf[te[:1]])
    torch.onnx.export(ens, (x0, m0), onnx_path, input_names=["x", "mask"], output_names=["out", "attn"],
                      dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"}, "out": {0: "batch"}, "attn": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref, _ = ens(torch.from_numpy(Z[te[:k]]), torch.from_numpy(Mf[te[:k]]))
    write_f32(os.path.join(a.out, "parity_x.bin"), Z[te[:k]])
    write_f32(os.path.join(a.out, "parity_mask.bin"), Mf[te[:k]])
    write_f32(os.path.join(a.out, "parity_out.bin"), ref.numpy())
    spec = {
        "name": "b1_cross_attention_engine", "version": "1", "feature_source": "engine",
        "features": meta["features"], "mean": mean.tolist(), "std": std.tolist(), "clip": a.clip,
        "horizon_cycles": a.horizon, "sigma_scale": sigma_scale, "symbols": meta["symbols"],
        "target": f"log mid return over {a.horizon} cycles minus stock cross-sectional mean (bps)",
        "outputs": ["out: mu_bps, logvar", "attn: last-block head-averaged attention (inferred relationships)"],
        "trained_on": [os.path.basename(os.path.normpath(d)) for d in a.dumps],
        "synthetic_data": bool(meta.get("synthetic", True)), "framework": f"torch {torch.__version__} (CPU)",
        "seeds": a.seeds, "metrics": metrics,
    }
    json.dump(spec, open(os.path.join(a.out, "model.spec.json"), "w", encoding="utf-8"), indent=1)
    print(f"val rank IC {metrics['val_rank_ic']:+.4f}  test rank IC {metrics['test_rank_ic']:+.4f}  "
          f"sigma scale {sigma_scale:.2f}  -> {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
