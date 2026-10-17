# AI Trademaster

Research and engineering prototype of a cross-stock trading engine with a 1-second decision cycle.
**Simulation only — no live brokerage connection exists. Not a proven trading strategy.**

- Requirements: [docs/SPEC.md](docs/SPEC.md) · Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Plan: [PLAN.md](PLAN.md)
- Machine inventory: [docs/HARDWARE.md](docs/HARDWARE.md) · M1 benchmark: [docs/BENCHMARK-M1.md](docs/BENCHMARK-M1.md)
- Research: [AI](docs/AI-RESEARCH.md) · [execution](docs/EXECUTION-RESEARCH.md) · [forward test](docs/FORWARD-TEST.md) · [security & data licensing](SECURITY.md)

> **Disclaimer.** This is a research and engineering project, not investment advice and not a trading system to rely on.
> The research in `docs/` found **no edge after trading costs**. Nothing here sends real orders.
>
> **No market data or trained models are in this repository.** They live in a git-ignored `private/` folder because market data
> (Yahoo, exchange and broker feeds) is licensed for personal use and can't be redistributed. Synthetic data can be generated
> locally (`trademaster generate`). If you fetch real data with `python/trademaster_ml/fetch.py`, you're responsible for
> complying with the data provider's terms.

## Status: M1 — end-to-end synthetic simulation (CPU, C#)

Seeded synthetic market → replay → six stages (ingest, evaluate, features, pack, B0 cross-stock baseline,
decision + risk) → simulated exchange. Every cycle writes an auditable decision record and is timed per stage
against its deadline.

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

- `trading.live_enabled` is `false`, and no live adapter exists; the engine refuses any non-simulated adapter.
- Real exchange configs (`SGX.json`, `NASDAQ.json`) are empty skeletons and are refused until filled from official sources and marked `verified`.
- Synthetic data embeds known signals by construction; P&L on it says nothing about real markets.
