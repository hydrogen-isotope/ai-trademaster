# AI Trademaster

A research project building an AI-driven, cross-stock equity trading engine with a 1-second decision cycle:
a low-latency C#/.NET engine, ONNX neural-network inference, and Python research tooling.

**Status: research stage.** The long-term goal is real trading on dedicated hardware (NVIDIA-class GPUs), but only
after strategies prove themselves on licensed data, in backtests and in forward tests. Today the engine trades only
against a built-in **simulated exchange**, used for backtesting, latency benchmarks and tests. **There is no live
broker connection.**

> **Read [DISCLAIMER.md](DISCLAIMER.md) first.** Not investment advice. The project's own research found **no edge after
> trading costs** so far. Automated trading can lose money quickly.

- Requirements: [docs/SPEC.md](docs/SPEC.md) · Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Plan: [PLAN.md](PLAN.md)
- Research: [AI](docs/AI-RESEARCH.md) · [execution](docs/EXECUTION-RESEARCH.md) · [forward test](docs/FORWARD-TEST.md) · [latency benchmark](docs/BENCHMARK-M1.md)
- [Security & data licensing](SECURITY.md) · [Licence](LICENSE)

## Bring your own data and models

**This repository contains no market data, no trained model weights and no generated datasets.** You must create
them yourself, under your own data licences. Everything you create goes in the git-ignored `private/` folder:

| You need | How | Where |
|---|---|---|
| Synthetic test market (for the engine, dashboard, tests) | `trademaster generate` (seeded, created locally) | `private/data/synth-*` |
| Real market data for research | your own licensed source, or `python/trademaster_ml/fetch.py` (Yahoo, unofficial, personal use; **you** must comply with its terms) | `private/data/` |
| Trained models | `python -m trademaster_ml.train_wf` / `research` (CPU), then ONNX export | `private/models/` |
| Run outputs | produced by every run | `private/reports/` |

Nothing trained here is guaranteed to work. Validate every model with the research harness (locked hold-out,
costs included) and a forward test before trusting it.

## Setup

Requires the **.NET 10 SDK** (`dotnet --list-sdks`). No other dependencies for M1.

```bash
dotnet build -c Release
```

```bash
dotnet test -c Release
```

## Run

All commands are run from the repository root.

```bash
dotnet run -c Release --project src/Trademaster.Cli -- doctor
```

```bash
dotnet run -c Release --project src/Trademaster.Cli -- generate --out private/data/synth-default --stocks 200 --minutes 30 --seed 42
```

```bash
dotnet run -c Release --project src/Trademaster.Cli -- run --mode virtual
```

```bash
dotnet run -c Release --project src/Trademaster.Cli -- run --mode paced --speed 1 --max-cycles 120
```

Live dashboard (dark trading-terminal UI, updates every cycle, http://127.0.0.1:5080, local only):

```bash
dotnet run -c Release --project src/Trademaster.Cli -- serve --speed 1
```

- **virtual**: simulated time, deterministic: the same data, config and model always give a byte-identical `decisions.jsonl`. Latency is still measured (compute only) and over-budget cycles are flagged as *would-miss*.
- **paced**: the replay is released at wall-clock speed (`--speed` multiplies it). A cycle that reaches the order interface after its deadline is `missed` and submits nothing, and cutoffs that passed during an overrun are `skipped_overrun`.

Each run writes `private/reports/<run-id>/`:

| File | Content |
|---|---|
| `summary.md` | Deadlines, per-stage p50/p95/p99/max, data quality, simulated execution, P&L (labelled synthetic) |
| `cycles.csv` | One row of timings per cycle |
| `decisions.jsonl` | One audit record per cycle: cutoff, status, model and data versions, input hash, orders with inputs and provenance, cancels, risk rejects |

## Layout

```
config/                default.json (engine), exchanges/*.json (SYNTH; SGX/NASDAQ skeletons, refused until verified)
src/Trademaster.Core       prices (fixed-point), tick tables, calendar, events, replay format, contracts, config
src/Trademaster.Sim        synthetic market generator, simulated exchange
src/Trademaster.Telemetry  stage clock, percentiles, system inventory
src/Trademaster.Engine     Stages/S1..S6, Risk, Orders, CycleRunner, AuditLog, RunReport
src/Trademaster.Cli        console entry point (doctor | generate | run)
tests/Trademaster.Tests    xUnit: ticks, calendar, ingest, evaluation, decision, risk, venue, end-to-end
private/data/  private/reports/            generated (git-ignored)
```

## Safety defaults

- `trading.live_enabled` is `false`, and no live adapter exists yet; the engine refuses any non-simulated adapter. Requirements for a future broker connection are in [SECURITY.md](SECURITY.md#4-live-trading-safeguards-for-the-ibkr-adapter--not-built-yet).
- Real exchange configs (`SGX.json`, `NASDAQ.json`) are empty skeletons and are refused until filled from official sources and marked `verified`.
- Synthetic data embeds known signals by construction; P&L on it says nothing about real markets.
- GPU work is gated off until the display driver has been stable for 7 days ([docs/GPU-STABILITY.md](docs/GPU-STABILITY.md)).

## Licence

Copyright © 2026 Hollander-Lawn. Released under the **[AI Trademaster Non-Commercial Research License](LICENSE)**:
- **free** for personal, research, hobby and educational use, including modifying it and sharing modified versions
  (keep the licence, the disclaimer and a note of your changes);
- **commercial use requires written permission** from the copyright holder. That includes selling it, paid services
  or signals built on it, managing other people's money, and business use.

Using it for your own personal account is allowed, entirely at your own risk (licence sections 4–6).
Third-party components keep their own licences.
