# Trademaster — Plan

Research and engineering prototype of a neural-network cross-stock trading engine with a 1-second decision cycle.
**Not a proven strategy. Simulation only; no live orders.**

- Requirements: [docs/SPEC.md](docs/SPEC.md)
- Architecture (proposal): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Machine inventory: [docs/HARDWARE.md](docs/HARDWARE.md)

Stack: **C# / .NET 10** real-time engine · **ONNX Runtime** inference (CPU → DirectML/Windows ML on RX 7800 XT → CUDA/TensorRT later) · **Python/PyTorch** offline training, export and precision reference · no HLSL unless profiling requires it.

## Milestones

| # | Milestone | Done when | Status |
|---|---|---|---|
| A0 | Hardware inspection, architecture proposal | Docs above exist | ✅ |
| **M1** | **End-to-end synthetic simulation (CPU, C# only)** | Seeded generator → replay → S1–S6 with B0 baseline → simulated exchange; virtual and paced clocks; JSONL decision records; per-stage timings + deadline report; tests pass; replay is deterministic | ✅ 2026-10-16 — see [docs/BENCHMARK-M1.md](docs/BENCHMARK-M1.md) |
| M2 | ONNX inference path | Python env; small cross-stock NN (B1) trained; exported; runs in the engine on ORT CPU (and later GPU); GPU counters in report | 🟡 CPU part done 2026-10-17 (trained on real 5-min bars, ONNX parity, CPU benchmark) — see [docs/M2-CPU.md](docs/M2-CPU.md). GPU part blocked until the driver is stable |
| UI | Live dashboard | `trademaster serve`: dark terminal UI, 1-second SSE stream, watchlist, candles with fills, quote/model/account, positions, orders & fills, risk, system telemetry | ✅ 2026-10-17 (simulation feed; real-data mode pending) |
| AI | Research harness + locked hold-out | 2.9 y hourly + 60 d 5-min, 4 models x 5 rules, full portfolio sim with costs — see [docs/AI-RESEARCH.md](docs/AI-RESEARCH.md) | ❌ no net edge (hold-out −6.3 bps/day); gross edge needs ≤ ~0.7 bps one-way, i.e. passive execution research with quote data |
| EX | Execution research | Queue-position fill model, execution analytics (fill odds, spread capture, markouts, adverse selection, shortfall), signal-free probe, informed-flow generator, passive strategy style — see [docs/EXECUTION-RESEARCH.md](docs/EXECUTION-RESEARCH.md) and [docs/QUOTE-DATA-REQUIREMENTS.md](docs/QUOTE-DATA-REQUIREMENTS.md) | 🟡 machinery done and validated on synthetic data; calibration needs real L1–L3 quote data |
| FT | Frozen-model forward test | Hashed frozen B1/ridge/B0, live 5-min logger + scorer, dashboard page `/forward`, Friday rehearsal — see [docs/FORWARD-TEST.md](docs/FORWARD-TEST.md) | 🟡 ready; first live session Mon 19 Oct 09:30 ET (21:30 SGT) |
| M3 | First benchmark report | Sweep universe size N, window W, inference steps K (modes A/B/C) on CPU vs GPU; p50/p95/p99/max per stage; missed-deadline rate | |
| M4 | Precision study | FP64/FP32 reference vs FP16/BF16 (GPU) vs FP8 (emulated); decision flips and tick changes | |
| M5 | Optimize (only what profiling shows) | Pipelining/double buffering, GPU-resident inputs, IOBinding; before/after measurements | |
| M6 | Historical-data validation | Licensed data source chosen; real exchange config filled from official sources; chronological splits; costs/slippage; stress periods | |
| M7 | Paper trading | Behind explicit enablement and approval; after M6 is stable | |

## M1 — delivered
Solution skeleton and `doctor` · fixed-point prices, tick tables, calendar, config · seeded synthetic generator (factor model, hidden ETF baskets, lead-lag, injected defects and scenarios) with a binary replay format and hash-verified manifest · S1 ingest (sequence dedup, reorder window, gap loss) · S2 evaluation (crossed/locked, jumps, outlier trades, halts, staleness mask, feed-down) · S3 incremental lazy-decay features · S4 ring-buffer packing, observed beta cache, input hash · S5 B0 factor-residual baseline · S6 policy v0 + independent risk checks · simulated exchange (latency, marketable/through/touch fills, participation cap, partial fills, rejects, cancel races, disconnect + cancel-on-disconnect, reconcile) · cycle runner (virtual + paced, deadlines, missed/skipped policy, expiry cancels, kill switch) · JSONL audit log, summary.md, cycles.csv · 30 xUnit tests.

## M2 build order (proposed)
1. Install Python 3.12 venv (`torch`, `onnx`, `onnxruntime`, `numpy`) — needs your OK
2. Python dataset builder: replays the same S1–S4 pipeline (exported features) so training/serving features are identical
3. B1: per-instrument temporal encoder + attention across instruments/ETFs; train on synthetic data; export ONNX + `model.spec.json`
4. C# `OnnxModel : IModel` with preallocated inputs; backends CPU and DirectML/Windows ML (pick after testing on the RX 7800 XT)
5. Report: H2D/inference/D2H split, GPU utilisation + dedicated VRAM via Windows PDH counters
6. Parity test: Python vs C# ONNX outputs on the same batches; B1 vs B0 comparison on held-out synthetic days

## Open questions (needed before or during M1–M3)
1. **"Inference step"**: which mode did the original 1,000+ step design mean — A (batched pass), B (parallel samples) or C (sequential, dependent steps)? See ARCHITECTURE §8.
2. **Data granularity**: L1 quotes + trades (assumed), 1-second bars, or L2 order book depth?
3. **Prediction horizon**: what should `μ` forecast — next 1 s, 10 s, 60 s?
4. **Benchmark universe sizes**: proposed sweep N = 50 / 200 / 1,000 / 3,000 instruments.
5. **Python 3.12 install** via winget for M2 — OK?
6. **Data source** for M6 (SGX first or NASDAQ) and whether you already have access to any feed.
7. **Resting entries**: policy v0 makes every entry marketable (limit ≥ ask). Should the "cutoff price" design post passive entries that rest until the next cycle? That needs a queue/fill-probability-aware policy.
8. **Kill switch behaviour**: it currently cancels orders and stops trading but keeps positions. Should it also flatten positions, and how (aggressively or passively)?
