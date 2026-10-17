# M1 benchmark report — SYNTHETIC DATA, CPU only

Measured 2026-10-16/17 on the development machine ([HARDWARE.md](HARDWARE.md)): Ryzen 7 9700X, 64 GB DDR5, Windows 11 Pro 26200, .NET 10.0.13, tiered JIT off, workstation GC.
Single run per configuration. All data comes from the seeded synthetic generator; every number below describes **this engine on this synthetic workload**, not real-market behaviour.

## What was measured

- The full M1 pipeline, one thread, CPU only: ingest → evaluate → features → pack → **B0 baseline** (non-neural factor-residual, a few FLOPs per stock) → simulated venue sync → decision → risk → order interface → audit write.
- **No neural network and no GPU were involved.** Inference cost in these tables is the B0 rule, not a neural network. This report says nothing yet about the 1,000-step inference design, FP8, or the RX 7800 XT; those are M2–M4.
- `ExecSync` is the *simulated* venue processing the true market stream inside the process. It stands in for, but does not measure, real order-gateway or network latency.
- `Ingest` includes reading the replay file from disk.

## Results

### Real-time paced run (200 stocks + 9 ETFs, 30 minutes, speed 1×)

| | Value |
|---|---|
| Cycles | 1,801 (Ok 1,167 · FeedDown 6 · KillSwitch 628) |
| **Missed deadlines** | **0** of 1,167 trading cycles · skipped (overrun) 0 |
| Cutoff → order interface | p50 0.264 ms · p95 0.458 ms · p99 0.636 ms · max 7.007 ms |
| Full cycle incl. audit | p50 0.353 ms · p99 0.719 ms · max 7.095 ms |
| Scheduler start lag | p50 0.002 ms · p99 0.007 ms · max 0.204 ms |
| Warm-up (first 30 cycles, JIT) | max 30.4 ms |
| Peak working set / GC pause | 46 MB / 0.0 ms total |

### Virtual-mode scaling (compute-only latency, trading cycles after warm-up)

| Universe | Feed events/s (synthetic) | Ingest p50 | Pack p50 | B0 inference p50 | ExecSync p50 | **Cutoff → order p50 / p99 / max** | Would-miss |
|---|---:|---:|---:|---:|---:|---|---:|
| 200 + 9 ETFs | ~500 | 0.078 ms | 0.018 ms | 0.010 ms | 0.069 ms | **0.192 / 0.524 / 6.77 ms** | 0 / 1,137 |
| 1,000 + 21 ETFs | ~2,200 | 0.428 ms | 0.082 ms | 0.049 ms | 0.285 ms | **0.916 / 1.648 / 8.77 ms** | 0 / 355 |
| 3,000 + 31 ETFs | ~6,300 | 1.227 ms | 0.245 ms | 0.152 ms | 0.843 ms | **2.701 / 5.081 / 10.42 ms** | 0 / 162 |

Per-stage tables: `private/reports/m1-*/summary.md` (git-ignored, reproducible with the commands in the README).

### Determinism

Two virtual runs over the same data, config and model produced byte-identical `decisions.jsonl` (sha256 `5f2d94af…328d6`). The hash was also unchanged with tiered JIT on vs off. The paced run's decisions differ slightly from the virtual run (2,928 vs 2,925 fill reports), as expected: in paced mode, real decision latency sets the order submission time.

## Interpretation

- With the B0 baseline, the CPU pipeline's p99 stays **under 1% of the 900 ms budget** (worst single cycle about 1.2%) up to 3,000 instruments. Cost grows roughly linearly with instrument and event count, dominated by ingest and the simulated venue.
- **The budget is effectively all available for the cross-stock network.** The open question is whether the neural inference step (and the 1,000-step variant) fits in roughly 890 ms on this hardware. That's what M2/M3 measure.
- The p99 → max tails (6–10 ms) are single outliers. Likely sources are OS scheduling, first-time code paths and LINQ allocations in order handling; profiling is deferred to Phase C.

## Caveats and known gaps

- **Synthetic event rates are low.** Real full-depth feeds at market open can be far higher. Ingest needs re-measuring with real data, and probably a dedicated ingest thread.
- **Fewer trading cycles than it looks:** the injected −3% market shock trips the 5,000 daily-loss kill switch in every run, so about a third of cycles are `KillSwitch` (cheaper, no decisions). Measured trading cycles are 1,137 / 355 / 162.
- Single run per configuration, on a desktop with other processes running. No CPU pinning or power-plan control.
- **Not measured:** GPU utilisation, VRAM, power draw (no GPU work yet; power needs AMD ADLX), and real network/order-gateway latency.
- The P&L figures in the run summaries come from a generator with embedded signals and are **not evidence of anything**.
