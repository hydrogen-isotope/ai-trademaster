# M2 (CPU) — B1 cross-stock model: training, export, C# inference, latency

Run 2026-10-17 on the development machine, **CPU only** (`torch 2.14.1+cpu`, ONNX Runtime 1.31 CPU EP). The GPU stays gated off (docs/GPU-STABILITY.md).

## Pipeline (all implemented)
`python -m trademaster_ml.fetch` → 60 sessions of 5-minute bars, 25 NASDAQ tech/semi stocks + 5 ETFs (Yahoo, unofficial, personal research)
→ `trademaster_ml.dataset` point-in-time features (r1, r3, r6, r12, vol12, residual vs reference ETF, volume surprise, time of day); target = next 10 minutes (2 bars), same session only
→ `trademaster_ml.train`: chronological split by session (40 train / 10 validation / 10 test). Baselines and B1 are scored on the same test bars, the model is chosen by validation only, then exported to ONNX with `model.spec.json` and parity vectors
→ C# `OnnxRunner` (preallocated OrtValues, CPU EP): its outputs match PyTorch to within 2.4e-6 (Python ORT) and < 1e-3 (C# parity test, `OnnxParityTests`)
→ `trademaster bench-onnx`: latency for inference modes A/B/C.

## Prediction quality on held-out sessions (stocks only, 641 test bars x 25 stocks)
| Model | IC | Rank IC (t) | Direction right | RMSE vs "no change" (bps) | R² out-of-sample | Top-minus-bottom quintile (bps/10 min, before costs) |
|---|---:|---:|---:|---|---:|---:|
| No change | 0 | — | — | 25.00 vs 25.00 | 0 | — |
| B0 (residual reversion) | +0.021 | −0.001 (−0.1) | 50.3% | 29.18 vs 25.00 | **−0.36** | −0.13 |
| Ridge (linear, same features) | +0.016 | +0.024 (+2.0) | 50.5% | 24.99 vs 25.00 | +0.0002 | +1.30 |
| **B1** (selected seed) | +0.030 | +0.001 (+0.1) | 50.6% | 24.99 vs 25.00 | +0.0003 | +0.22 |

B1 test IC across 3 seeds: +0.030 / +0.020 / +0.022.

**Reading:** no model has a usable edge. B1 and ridge barely beat "no change" (R² ≈ 0.0003), and that's within noise. The ridge rank-IC t-stat of 2.0 is overstated, because 10-minute targets on 5-minute bars overlap. A spread of about 1 bp before costs would not survive real spreads and fees. B0 is clearly harmful. The pipeline is ready; the signal is not. More data (months, more symbols, 1-minute or quote data) and better features are needed before any paper trading.

## CPU inference latency (B1 shape, random weights, 4 threads; report: `private/reports/m2-bench-onnx-cpu/summary.md`)
| N instruments | A: one pass p50 | C: per sequential step | C: 1,000 steps | B: per sample when batched |
|---:|---:|---:|---:|---:|
| 30 | 0.15 ms | 0.11 ms | 0.11 s | 0.020 ms (K=1000) |
| 300 | 0.39 ms | 0.39 ms | 0.39 s | 0.61 ms (K=100) |
| 1,000 | 1.99 ms | 2.06 ms | 2.06 s | 5.7 ms (K=10) |
| 3,000 | 49.9 ms | 50.2 ms | 50.2 s | (skipped: memory cap) |

- A single batched pass (mode A) fits the 1-second cycle comfortably at every N tested.
- **1,000 sequential steps (mode C) fit on this CPU only up to about 300 instruments.** Cost grows roughly with N² because of full attention.
- Options for larger N: sparse or sector-local attention, fewer steps, batching independent steps (mode B), or the GPU later, after a week without driver resets.

## Not yet done
- B1 is not wired into the live 1-second engine loop. It was trained on 5-minute bars, while the engine computes quote-level features each second, so it needs a matching feature pipeline (or 5-minute cadence) first.
- FP16/BF16/FP8 comparisons (M4) need the GPU path.
