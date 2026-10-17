# Stage 5: cross-stock inference in the live engine

> **Status (2026-10-17): code written and compiled. Not yet run or tested.** Nothing here has produced results;
> all instructions below are for the first run.

## How it works
- `model.name = "onnx"` loads `OnnxCrossStockModel` from `model.path`, a folder containing `model.onnx` and `model.spec.json`.
- **The network:** a B1 attention model. Every stock and ETF in the universe is one token, so each prediction attends to the state of the whole market at the same second.
- **Input:** the newest stage-4 time slice, normalized and clipped with the model's own training statistics, and mapped into the model's instrument order. Masked instruments (stale, halted, no quote) are excluded from attention.
- **Output per instrument:**
  - **μ:** expected log return over `horizon_cycles`;
  - **σ:** uncertainty, scaled by the `sigma_scale` calibrated on validation data (earlier research found the raw σ was about 4× too small);
  - **peer:** if the model exports its attention map, the instrument it attended to most, with the weight. It's logged in each order's audit record as an **inferred** relationship. ETF holdings are never used.
- **Safety checks:**
  - The model is refused unless `feature_source = "engine"`. Models trained on 5-minute bars can't drive the 1-second engine.
  - The feature list must match the engine's exactly.
  - GPU backends go through the safety gate (`docs/GPU-BACKENDS.md`).

## Training loop (engine features → model → engine; no train/serve skew)
1. **Export features** from engine runs. This writes `features.bin` and `features.json` into the run folder under `private/`:
   ```bash
   dotnet run -c Release --project src/Trademaster.Cli -- run --dump-features --out private/reports/feat-1
   ```
2. **Train** (CPU) on one or more dumps:
   ```bash
   python -m trademaster_ml.engine_train --dumps private/reports/feat-1 private/reports/feat-2 --out private/models/engine/b1-v1
   ```
   - Target: the mid's log return over 10 cycles minus the stock cross-sectional mean.
   - Chronological 70/15/15 split, 3-seed ensemble.
   - Calibrates σ and exports `model.onnx` (outputs `out` + `attn`), `model.spec.json` and parity vectors.
3. **Run** with it:
   ```bash
   dotnet run -c Release --project src/Trademaster.Cli -- run --model onnx --model-path private/models/engine/b1-v1
   ```

On synthetic data the model can only learn the generator's planted signals. Real use needs real quote data
(`docs/QUOTE-DATA-REQUIREMENTS.md`).

## Other pipeline additions (same change)
| Stage | Addition | Config |
|---|---|---|
| 1 Ingest | **Content-based dedup** for feeds without reliable sequence numbers: drops exact repeats within a window and quotes older than the last applied one | `ingest.dedup = sequence \| content`, `ingest.content_window_ms` |
| 4 Pack | Feature **clipping** to the model's training range; non-finite features → 0 | from the model spec (`clip`) |
| 6 Decide/Risk | **Kill-switch action**: `hold` (default, keep positions) or `flatten` (close every position with marketable limits at the opposite quote until flat) | `risk.kill_switch_action` |
| Audit | **Hash-chained decision log**: every record carries `prev_hash` and `hash`; `trademaster verify-audit <run>/decisions.jsonl` finds edits, deletions and reordering | always on |
| Audit | Orders record the model's inferred **peer** and weight | when the model exports attention |

## Tests written (compiled, not yet run)
`tests/Trademaster.Tests/PipelineV2Tests.cs`:
- the audit chain verifies, and detects an edited or deleted record;
- content dedup drops repeats and stale quotes;
- an unknown dedup mode is rejected;
- the ONNX model refuses missing files, non-engine models and feature-list mismatches;
- the model factory builds B0 by default and rejects unknown names.

Not yet covered by a test: kill-switch flatten, the feature dump round-trip into Python, and an end-to-end run with an ONNX model.
