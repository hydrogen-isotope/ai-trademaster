# Trademaster — Prototype Architecture (proposal)

Status: **proposal, 2026-10-16**. Requirements: [SPEC.md](SPEC.md). Machine: [HARDWARE.md](HARDWARE.md). Milestones: [../PLAN.md](../PLAN.md).
Nothing here has been measured yet. Every performance number in this repo must come from a benchmark run on the stated hardware.

---

## 1. Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Real-time engine | **C# / .NET 10** (SDK 10.0.402 installed) | Predictable latency, real threads, typed stage contracts, runs on Windows and Linux/ARM64 |
| NN inference | **ONNX Runtime** in-process from C# | One model format; switch backend by config: CPU, DirectML / Windows ML (RX 7800 XT now), CUDA / TensorRT (NVIDIA later) |
| Training, export, precision reference | **Python + PyTorch**, offline only | ML ecosystem; never runs in the live loop |
| Custom GPU kernels (HLSL) | **Not used** unless profiling shows ONNX Runtime/CPU can't keep up | Avoid building a DL framework |
| Prototype data | **Synthetic, seeded, deterministic** replay | No licensed feed yet; known ground truth for testing |
| Execution | **Simulated exchange adapter only** | Live trading needs separate explicit enablement (spec §7) |

Hand-off between the two languages is a single **model artifact**: `model.onnx` plus `model.spec.json` (feature list and order, window length, normalization stats, input/output names, training data hash, precision). The engine refuses to load a model whose spec doesn't match its feature pipeline.

---

## 2. Process picture

```
                        OFFLINE (Python)                                  ONLINE (C# engine, one process)
  replay files ──▶ train.py ──▶ export_onnx.py ──▶ model.onnx + spec.json ──┐
                        └──▶ precision_study.py (FP64/FP32/FP16/BF16/FP8)  │
                                                                           ▼
  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
  │ Source  │──▶│ S1 Ingest│──▶│ S2 Eval  │──▶│ S3 Per-  │──▶│ S4 Cross-│──▶│ S5 Cross-│──▶│ S6 Decide │──▶ Execution
  │ (replay)│   │ dedup,   │   │ quality, │   │ instrument│  │ sectional│   │ stock    │   │ policy +  │    adapter
  └─────────┘   │ order,   │   │ weights, │   │ features │   │ packing, │   │ inference│   │ risk      │    (sim)
                │ state    │   │ masks    │   │ (incr.)  │   │ caches   │   │ (ONNX)   │   └───────────┘
                └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘         │
                                                                                                 ▼
                         Telemetry (per-stage timers, deadlines, RAM/VRAM/GPU)      Audit log (JSONL decision records)
```

---

## 3. Time model

Every event carries three timestamps (int64 nanoseconds, UTC):

| Field | Meaning |
|---|---|
| `ExchangeTs` | When the exchange says it happened |
| `RecvTs` | When our process received it (in replay: simulated feed delay applied) |
| `Seq` | Per-source sequence number, used for gap and duplicate detection |

**Point-in-time rule (no look-ahead):** cycle *k* may use only events with `RecvTs ≤ T_k`, where `T_k = t0 + k × cadence`. Features never read `ExchangeTs` of the future, and training uses the identical snapshot rule.

Two clocks behind one `IClock` interface:

| Clock | Purpose | Deterministic? |
|---|---|---|
| `VirtualClock` | Correctness replay. Simulated time advances by events and cycles; compute time does not affect results | **Yes**: same input + seed + model ⇒ byte-identical decision log (CPU backend) |
| `PacedClock` | Latency benchmark. Replay is released at wall-clock speed; deadlines are real | No by nature: misses depend on real timing. Decisions for cycles that *do* complete must match `VirtualClock` |

GPU backends may not be bit-deterministic; the replay test checks GPU results against CPU within a tolerance and reports any decision that changes.

---

## 4. Cycle lifecycle, deadlines and stale policy

```
T_k ─── snapshot cutoff ───┬── S3 ── S4 ── S5 ── S6 ── risk ── order interface ──┐
                           │◀──────────────── budget (default 900 ms) ──────────▶│ D_k = T_k + budget
```

- **On time:** the decision record is written and orders go to the execution adapter.
- **Missed (finished after `D_k`):** the record is marked `MISSED`, **no orders are submitted** from it, and the miss is counted. The next cycle starts from a fresh snapshot at `T_{k+1}`. Missed cycles are skipped, never queued, so the engine never catches up on old data.
- **Resting orders:** each order carries `ValidUntil = T_k + cadence + grace`. The engine actively cancels expired orders, since exchanges don't support millisecond expiry. On a miss, the policy is configurable: `on_miss = cancel_expired` (default) or `cancel_all`.
- **Instrument staleness:** if an instrument's last update is older than its `max_age` (configurable per liquidity class), it's masked out of inference and can't get new orders; the reason is recorded.
- **Feed staleness:** no events for `feed_timeout` during an open session ⇒ trading state `FEED_DOWN`, all new orders blocked.
- **Market status:** pre-open / open / lunch break / auction / halt / closed come from exchange config plus halt events. Orders are only generated in states that allow them.

---

## 5. Stage responsibilities (with a proposed S3/S4 split)

**S1 Ingest.** Parse, validate ranges and types, sequence-gap detection, deduplication by `(source, seq)` plus a content hash fallback, a bounded reorder window for out-of-order events (late beyond the window ⇒ counted and dropped), latest L1 state per instrument (bid/ask/size/last trade), and ingestion latency `now − RecvTs`.

**S2 Evaluate.** Per-observation quality flags and weights: crossed or locked quotes, price jumps beyond N×volatility, outlier trades, stale quotes. Builds a validity mask per instrument. Never alters prices; it only flags, weights and masks.

**S3 Per-instrument features (incremental).** Updates only instruments in the *dirty set* (those that received events this cycle), in O(1) per update: multi-horizon EWMA returns and volatility, spread, microprice, order-flow imbalance, trade intensity, time since last update. State lives in preallocated struct-of-arrays.

**S4 Cross-sectional assembly and packing.**
- Normalization using frozen stats from `model.spec.json`.
- Rolling history window `[N, W, F]` kept as a ring buffer with a write index, so nothing gets shifted.
- Masks, plus market-level tokens such as ETFs and indices where data exists.
- Incrementally updated **observed statistics caches** (rolling covariance/correlation, betas to ETFs) for S5.
- Packs into a preallocated, double-buffered input batch and records its content hash.

**S5 Cross-stock inference.** Runs the model over the whole universe at once, so it sees relationships rather than isolated stocks. Output per instrument: expected return `μ` over horizon `h`, uncertainty `σ`, and optionally relationship/attention summaries.
- B0 (no NN, pure C#): factor-residual model (instrument vs rolling beta to market/sector ETF). This is the reference every NN must beat.
- B1 (NN, ONNX): a per-instrument temporal encoder plus attention across instrument and ETF tokens, trained in PyTorch.

**S6 Decide** (policy v0, explicit and configurable):
- Act only if `|μ| > cost_est + margin` (half-spread + fees + slippage estimate) **and** `|μ|/σ > θ`.
- Cutoff (limit) price: buy `= min(ask, mid × (1 + μ − required_edge))`, sell mirrored, then **rounded to a valid tick in the conservative direction** (buys down, sells up).
- Size is proportional to the score, capped by limits. Orders are good for one cycle (cancel/replace).
- Risk and order validation (independent of the model; the model can't bypass it): max position per instrument, gross/net exposure, daily loss limit and kill-switch, price collar vs last trade, max order rate, lot size, tick validity, market status, staleness.

**Observed vs inferred labelling** (spec §2, stage 5). Every input and output in the decision record is tagged with one of three classes:
- `observed`: prices, sizes, ETF quotes
- `estimated`: statistics computed from observed data, e.g. a rolling correlation
- `inferred`: model outputs and attention-based relationships

ETF holdings are never an input to the engine.

---

## 6. Data contracts (C#, sketch)

```csharp
readonly record struct Price(long Raw);                             // fixed-point, 1/10,000 of the currency; never float
                                                                   // (tick sizes vary by price band, so prices are not "tick counts")
readonly record struct MarketEvent(long Seq, long ExchangeTs, long RecvTs, int Instrument,   // dense instrument index
                                   EventKind Kind, long Px1, long Px2, long Sz1, long Sz2, TradingStatus Status);

sealed record CycleSnapshot(long CycleId, long CutoffTs, ReadOnlyMemory<InstrumentId> Dirty,
                            QualityReport Quality);                  // S1+S2 → S3

sealed class ModelInputBatch                                       // S4 → S5 (preallocated, double-buffered)
{ public long CycleId; public Device Device; public float[] X; public byte[] Mask; public ulong InputHash; }

readonly record struct Prediction(InstrumentId Id, float Mu, float Sigma, string HorizonTag);

sealed record DecisionRecord(long CycleId, long CutoffTs, long DecidedTs, CycleStatus Status,
                             string ModelVersion, ulong InputHash, string DataVersion,
                             IReadOnlyList<OrderIntent> Orders, IReadOnlyList<SkipReason> Skips,
                             StageTimings Timings);                  // → audit log, one JSON line per cycle
```

Prices are fixed-point integers end to end, and tick validity comes from the exchange's tick table. Floating point is used only inside features and the model, which keeps tick-boundary rounding explicit for the precision study.

---

## 7. Devices, memory ownership, pipelining

- Each stage declares its placement in config: `s34.device = cpu|gpu:0`, `s5.device = cpu|dml:0|cuda:0|cuda:1`.
- Every tensor buffer has one owning **device and stage**. A cross-device move happens only through an explicit, timed `Transfer`. Nothing assumes shared VRAM between GPUs.
- ONNX Runtime uses IOBinding / preallocated `OrtValue`s, so the steady state allocates nothing per cycle and inputs can stay GPU-resident.
- **M1 runs the stages synchronously on one thread**, which is simplest and deterministic. Pipelined mode comes in Phase C, and only if profiling says it helps: S4 fills buffer A while S5 reads buffer B, handed off by a bounded channel of capacity 1 that keeps the newest batch.
- GC: preallocated buffers, `GCLatencyMode.SustainedLowLatency`, and GC pause time reported per cycle (`GC.GetTotalPauseDuration`).

---

## 8. What is an "inference step"? (must be defined — spec §4)

The engine will support all three as a configurable `K` so they can be benchmarked:

| Mode | Meaning | Cost behaviour |
|---|---|---|
| **A. Batched pass** | One forward pass over the whole universe per cycle; `K` = number of instruments in the batch | Throughput-bound; GPU-friendly |
| **B. Parallel samples** | `K` independent passes, e.g. Monte-Carlo / ensemble members for uncertainty | Can be batched into one call |
| **C. Sequential steps** | `K` dependent passes: each step needs the previous one (iterative refinement, rollouts, message-passing rounds), with possible early exit | **Latency-bound**: 1,000 steps in < 1 s leaves < 1 ms per step *including* dispatch overhead |

The original "1,000+ steps, data-dependent time" sounds like **C**, the hardest case. The benchmark will sweep `K` and report the per-step dispatch overhead on each backend before any model is designed around it.

---

## 9. Precision plan

| Precision | RX 7800 XT (RDNA3) | Later (RTX Pro 6000, Blackwell) |
|---|---|---|
| FP64 reference | Python / NumPy on CPU | same |
| FP32 | ORT CPU and DirectML | CUDA / TensorRT |
| FP16 / BF16 | DirectML (to verify which ops run natively) | TensorRT |
| **FP8** | **No FP8 hardware on RDNA3** ⇒ *emulated* only (quantize → dequantize in PyTorch): accuracy study, **no speed claims** | Native via TensorRT: speed + accuracy |

Metrics per precision vs the FP64/FP32 reference: max/mean absolute deviation of `μ` and `σ`, rank correlation, **decision flips** (buy/sell/no-trade changes), **cutoff price changes in ticks**, and the share of cases within ε of a threshold or tick boundary.

---

## 10. Simulated execution

- The simulated exchange runs against the same replay: orders arrive after a configurable latency, and limit orders fill only if later replay trades go **through** the price. Touches fill with a configurable queue-position probability.
- **Partial fills** are capped at a participation fraction of traded volume.
- Fees are taken from exchange config. Rejects cover invalid tick, price collar, lot size and closed/halted markets.
- Cancel and replace race with fills, so a cancel may arrive after a fill.
- Injected disconnects and reconnects; state is reconciled on reconnect.
- A cutoff price is never treated as a guaranteed fill price.
- Live adapters stay unimplemented and require `trading.live_enabled = true` **plus** a separate confirmation setting before they can load.

---

## 11. Telemetry and benchmark report

- **Per cycle:** wall time per stage (QPC via `Stopwatch.GetTimestamp`): ingest, evaluate, S3, S4, host→device, S5, device→host, S6, risk, order interface, total. Also deadline met/missed, data age at cutoff, dirty-set size and GC pause.
- **Distributions:** all raw samples are kept (23,400 cycles per 6.5-hour session is small), giving exact p50/p95/p99/max per stage and end to end.
- **System:** process RAM; GPU utilization and dedicated VRAM via Windows PDH counters (`GPU Engine`, `GPU Adapter Memory`). **Power: not measured in M1** — to add later via AMD ADLX (NVML on NVIDIA). Reports say "not measured" and never print an estimate.
- **Output:** a console summary each N cycles, plus `private/reports/<run>/summary.md`, `cycles.csv` and `decisions.jsonl`. Every report states hardware, backend, precision, model hash, data hash, and **SYNTHETIC** when applicable.

---

## 12. Synthetic market generator

Seeded and reproducible. Generates L1 quotes and trades for a configurable universe:
- **Price structure:** market factor + sector factors + idiosyncratic noise, volatility clustering.
- **ETFs** as synthetic baskets. Their composition is written *only* to a ground-truth file for evaluation; the engine never reads it.
- **Injected lead-lag effects** (e.g. instrument A leads its sector by a few seconds), so we can check that the cross-stock model finds a real, known relationship.
- **Injected defects:** duplicates, out-of-order events, sequence gaps, delayed bursts, crossed quotes, halts, a market-wide shock, a feed outage.
- **Output:** a compact binary replay file plus a CSV export for inspection, and a manifest with the seed, parameters and hash. All results produced from it are labelled **SYNTHETIC**.

---

## 13. Exchange and universe configuration

`config/exchanges/*.json`: timezone, sessions (including lunch breaks), holiday calendar file, tick-size table by price band, lot size, price limits/collars, fee schedule. `config/universes/*.json`: instrument list with exchange, symbol, type (stock/ETF/index) and liquidity class.

Starting entries: `SYNTH` (synthetic exchange), plus `SGX` and `NASDAQ` skeletons. **Their tick tables, sessions, fees and holidays must be filled in from official exchange sources before any historical run; they won't be guessed.**

---

## 14. Project structure

```
Trademaster/
  Trademaster.slnx
  src/
    Trademaster.Core/      ids, ticks, time/clocks, contracts, config, exchange metadata
    Trademaster.Engine/    Stages/S1Ingest S2Evaluate S3Features S4Pack S5Inference S6Decision, Risk/, CycleRunner
    Trademaster.Sim/       synthetic generator, replay reader/writer, simulated exchange
    Trademaster.Telemetry/ timers, histograms, system counters, report writers
    Trademaster.Cli/       console app: doctor | generate | run | bench
  tests/
    Trademaster.Tests/     xUnit: dedup/reorder, tick rounding, staleness, determinism, risk, fills
  python/                  training, ONNX export, precision study (from M2)
  config/  private/models/  private/data/  private/reports/  docs/
```

**Dependencies.**
- M1: .NET 10 SDK, xUnit, `System.IO.Hashing` (xxHash3 for input hashes).
- M2: `Microsoft.ML.OnnxRuntime` plus a GPU execution provider (DirectML or Windows ML; pick after testing on the 7800 XT); Python 3.12 venv with `torch`, `onnx`, `onnxruntime`, `numpy`.

Console output is plain text first; `Spectre.Console` is optional.

---

## 15. M1 implementation notes (as built, 2026-10-16)

Where M1 differs from, or makes concrete, the proposal above:

- **Project layout:** Core / Sim / Telemetry / Engine / Cli / Tests, as in §14. The only third-party package is `System.IO.Hashing`.
- **Deduplication is by feed sequence number only.** The content-hash fallback is deferred until a real feed with per-channel sequences is known.
- **Ingestion is batched at each cycle start** (all records with `RecvTs ≤ T_k`), so the ingest stage time includes reading the replay from disk. A live feed would be ingested continuously on its own thread.
- **Single thread, synchronous stages**; no double buffering yet (Phase C, only if profiling asks for it).
- **Tiered JIT disabled** in the CLI (`TieredCompilation=false`). With tiering on, hot methods were promoted mid-run and distorted stage timings (Pack measured 0.33 ms tiered vs 0.017 ms fully optimised). Warm-up cycles are excluded from percentiles and reported separately.
- **Two data streams per data set:** `exchange.tmrp` (truth, drives the simulated venue) and `feed.tmrp` (delayed, lossy, duplicated, reordered, partly corrupted — what the engine sees).
- **Decision policy v0 consequence:** because the entry hurdle includes the half spread, an entry's limit is always ≥ the ask, so **entries are always marketable**. Exits can rest above the bid. Resting entries need a different policy (open question).
- **Kill switch** cancels all working orders and blocks new ones, but **does not liquidate positions**. A liquidation policy is still to be defined.
- **Reconnect handling:** open orders are reconciled with the venue snapshot immediately; positions are verified one cycle later, after reports held during the disconnect have arrived (the venue wins on mismatch).
- **Paced-mode waits** sleep coarsely, then spin for the last 20 ms. Start lag is measured and reported.
