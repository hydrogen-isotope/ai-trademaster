# Quantitative Trading System — Prototype Specification

*Source: provided by the project owner, 2026-10-16. Kept as the reference requirements document.*

## 1. Project objective

Build a prototype of a neural-network-based quantitative trading system that processes market data, evaluates relationships across stocks, and generates buy/sell decisions with associated cutoff prices.

The initial target is a one-second decision cadence. The system should aim to complete each decision cycle within its deadline, while measuring and reporting missed deadlines rather than assuming every cycle will finish on time.

The immediate goal is to validate the software architecture and benchmark performance on existing hardware. Do not purchase new hardware or assume the final deployment configuration before obtaining measurements.

This is a research and engineering prototype, not a proven profitable trading strategy.

## 2. Core architecture

The system is conceptually divided into six stages.

### Stage 1 — Market data collection

- Ingest market data from the selected data source.
- Timestamp and validate incoming records.
- Deduplicate events.
- Maintain the latest known state for each instrument.
- Handle missing, delayed, duplicated and out-of-order events.
- Record data freshness and ingestion latency.

### Stage 2 — Data evaluation

- Evaluate incoming data for quality, relevance and validity.
- Weight or score observations where appropriate.
- Produce structured, timestamped inputs for subsequent stages.
- Avoid introducing look-ahead bias into features.

Stages 1 and 2 form the upstream data preparation pipeline.

### Stages 3 and 4 — Intermediate processing

- Transform prepared market data into model-ready representations.
- Compute and maintain reusable per-stock features and intermediate state.
- Preprocess and cache data that later stages would otherwise have to recompute.
- Where possible, update only changed features rather than rebuilding all features each cycle.

The precise responsibilities of stages 3 and 4 remain to be defined during implementation.

### Stage 5 — Cross-stock inference

- Analyse relationships across the stock universe rather than evaluating every instrument in isolation.
- Consider market-wide dependencies and correlated movements.
- Include ETFs where suitable data is available.
- Account for the possibility that an individual stock's movement may be relevant to other stocks or broader market conditions.
- Use cached and preprocessed representations from stages 3 and 4 to reduce redundant computation.

The motivation for cross-stock analysis is that movements in one instrument can propagate through correlated assets, index products, ETFs and broader market conditions.

Do not assume ETF holdings or other non-public portfolio data is available. Use only appropriately licensed and accessible data, and clearly distinguish observed information from model-inferred relationships.

### Stage 6 — Final decision

- Convert model outputs into candidate trading actions.
- Determine buy, sell and cutoff prices using an explicitly defined decision policy.
- Apply risk and order-validation checks.
- Produce an auditable decision record with timestamps, model version, input-data version and confidence or uncertainty measures where appropriate.

Stages 5 and 6 form the inference and decision portion of the system. Their exact internal boundaries may be refined during prototyping.

## 3. Proposed hardware mapping

The intended eventual dual-GPU architecture is:

**GPU 1: preprocessing and cache preparation**
- Feature extraction and per-stock updates.
- Tensor construction, packing and normalization.
- Cache maintenance and preparation of the next inference batch.

**GPU 2: cross-stock inference and decision computation**
- Execute the cross-stock neural network.
- Compute final model outputs.
- Pass results to the decision and risk-management layer.

**System RAM: staging and persistent process state**
- Store rolling market-data history and large host-side caches.
- Stage packed tensors and exchange data between GPU stages.
- Use bounded queues and double buffering to overlap data preparation with inference.
- Avoid repeatedly transferring large tensors when they can remain resident in GPU memory.

This is a proposed deployment architecture, not a requirement to implement multi-GPU code immediately. Start with modular interfaces so the stages can initially run on one GPU or CPU and later be distributed across devices.

Do not assume that two GPUs share one unified VRAM pool. Explicitly manage allocations, transfers and ownership of tensors.

## 4. Inference and timing requirements

The design target is one decision cycle per second.

An earlier conceptual design involved more than 1,000 inference steps, with execution time expected to vary depending on the input data. This has not yet been benchmarked, and the meaning of an "inference step" must be defined precisely.

Requirements:

- Measure end-to-end cycle latency, not just neural-network execution time.
- Record p50, p95, p99 and maximum observed latency.
- Measure data ingestion, preprocessing, tensor packing, host-to-device transfers, inference, decision computation and order-interface latency separately.
- Measure GPU utilisation, VRAM usage, RAM usage, power consumption and missed deadlines.
- If a cycle cannot complete on time, record the miss and apply an explicit stale-result policy.
- Never silently submit a decision derived from stale market data.
- Support deterministic historical replay for reproducible testing.

Do not assume that a one-second cadence is automatically achievable until measured on the existing hardware.

## 5. Neural-network precision

FP8 is a candidate inference precision because of its potential throughput and memory advantages.

Implement precision as a configurable option where supported:

- FP8 inference.
- BF16 or FP16 comparison runs.
- Higher-precision reference calculations for validation where practical.

Measure:

- Throughput and latency.
- Numerical deviation from the reference.
- Changes in predicted outputs.
- Changes in buy/sell decisions and cutoff prices.
- Sensitivity around decision thresholds and market-price tick boundaries.

Do not assume FP8 is sufficiently accurate merely because it is faster. Training precision, accumulation precision and inference precision may differ.

## 6. Market coverage and data limitations

The initial exchange target and exact instrument universe remain to be finalised. Singapore Exchange (SGX) was considered as a smaller initial universe, with NASDAQ as a possible later target.

The system should support a configurable instrument universe rather than hardcoding assumptions about one exchange.

Implement:

- Instrument identifiers and exchange-specific metadata.
- Trading sessions, holidays and market status.
- Market-data timestamps and sequence handling.
- Configurable instrument lists.
- Missing-data and stale-data detection.
- Exchange-specific price increments and order constraints where applicable.

For the prototype, use historical data, synthetic data or an appropriately licensed feed. Do not connect to a live brokerage or submit real orders by default.

## 7. Execution and risk management

The conceptual trading system generates buy, sell and cutoff prices, with the intention that orders may be resting at the market before the next decision cycle.

This execution design is not yet fully specified. Do not assume that a cutoff price guarantees an execution price or that an order will fill.

Keep the execution interface separate from model inference. Define explicit handling for:

- Partial fills.
- Unfilled orders.
- Order cancellation and replacement.
- Stale signals.
- Market closures and halts.
- Rejected orders.
- Connection failures and reconnects.
- Position, exposure and loss limits.

For the prototype, implement a simulated execution adapter first. Use realistic transaction costs, fees, spread and slippage in backtests. Require explicit configuration and approval before any future live-trading capability can be enabled.

## 8. Prototype development plan

### Phase A — Establish the baseline
1. Inspect the existing hardware, operating system, GPU models, VRAM, driver versions and available RAM.
2. Select a suitable ML framework based on hardware support.
3. Create a minimal data-ingestion pipeline.
4. Build a deterministic synthetic or historical market-data replay.
5. Implement a simple baseline model and decision policy.
6. Measure CPU-only and available GPU performance.

### Phase B — Implement the six-stage pipeline
1. Define typed data contracts between stages.
2. Implement ingestion, deduplication and data evaluation.
3. Add incremental per-stock feature computation.
4. Add cache and tensor-packing components.
5. Implement a simple cross-stock baseline model.
6. Add decision generation and simulated order execution.
7. Add structured logging and performance instrumentation.

### Phase C — Optimize
1. Profile CPU and GPU bottlenecks.
2. Cache reusable intermediate results.
3. Introduce asynchronous transfers and double buffering where supported.
4. Compare FP8 against BF16/FP16 and a higher-precision reference.
5. Test batching and the latency/throughput trade-off.
6. Test concurrent preprocessing and inference.
7. Explore two-GPU separation only if profiling demonstrates a benefit.

### Phase D — Validate
1. Test against historical data with chronological train/validation/test separation.
2. Prevent look-ahead bias and survivorship bias.
3. Include realistic fees, spreads, slippage and market-data delays.
4. Test volatile periods, missing feeds, sudden price movements and market-wide shocks.
5. Report predictive quality, simulated trading performance, drawdowns, latency distributions and missed deadlines separately.
6. Paper-trade only after the data pipeline and simulation are stable.

## 9. Implementation instructions

Begin by inspecting the existing machine and proposing a minimal prototype architecture. Do not assume that the final dual-RTX PRO 6000 workstation or a six-node DGX Spark cluster has been purchased.

Priorities:
1. Correctness and reproducibility.
2. Clear separation of pipeline stages.
3. Accurate timing and profiling.
4. Data freshness and failure handling.
5. A working CPU/single-GPU baseline.
6. Optimization only after profiling.

Keep the code modular, testable and configurable. Provide a clear project structure, dependency list, setup instructions, automated tests, a synthetic-data demonstration and a benchmark report.

Do not invent measured performance, trading returns or hardware capabilities. Label synthetic results clearly. Ask for the existing hardware specifications and preferred language/framework if these are not available from the project environment.

The first milestone is a functioning end-to-end simulation that consumes timestamped market data, produces auditable decisions and reports whether each cycle meets its deadline.
