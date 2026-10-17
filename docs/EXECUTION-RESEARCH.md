# Execution research: passive limit orders (2026-10-17, CPU only, SYNTHETIC data)

**Why:** the AI research ([AI-RESEARCH.md](AI-RESEARCH.md)) found gross edges that only survive at about 0.6–1.0 bps one-way cost. Taker execution can't get there, so the question is whether resting limit orders (the spec's "orders resting at the market with cutoff prices") can.

## What was built
| Piece | Where | What it does |
|---|---|---|
| Queue-position fill model | `SimExchange` (`execution.fill_model = queue`) | Joining a level puts the **displayed size ahead of us**. Trades at our price consume it. Size decreases without trades are treated as cancellations, **pro rata** to our position. Improving the price puts us first. We fill only after the queue ahead is gone, or completely when the market trades or quotes through our price. |
| Order/fill records + true market path | `SimExchange.Orders`, `FillLog`, `QuoteAt/MidAt` | Every order: decision mid, queue ahead on arrival, filled qty, time to fill, end reason. Every fill: maker or taker. |
| Execution analytics | `ExecutionAnalytics` | Fill rate, time to fill, **spread capture**, **markouts** and **adverse selection** at 1/5/30/60 s, fees, and **implementation shortfall incl. clean-up** (unfilled remainder crossed at expiry). |
| Signal-free probe | `trademaster exec-study` | Random stocks, sides and times. Three placements: aggressive / passive join / passive improve. Isolates execution cost from alpha. |
| Informed-flow generator | `generate --impact-bps X` | Each trade moves the latent price in its own direction by X bps per average trade (then decays). This creates **known** adverse selection for passive orders. |
| Passive strategy style | `run --exec-style passive` | Joins the bid (buys) or ask (sells), capped by the cutoff price. Entry hurdle = fee + assumed adverse selection instead of half spread. **Keeps its queue position** while still at the best price and within the cutoff. |
| Tests | `SimExchangeTests` | Queue consumption, pro-rata cancels, improve-first, trade-through, and exact markout/adverse-selection/clean-up arithmetic. |

## Results (synthetic; 30-min sessions, about 1,000 probes per placement)
**Probe, queue model, 30 s expiry — shortfall incl. clean-up (bps, + = cost):**

| Injected impact | Aggressive | Passive improve | Passive join | Join: adverse selection at 60 s | Join fill rate |
|---|---:|---:|---:|---:|---:|
| 0 bps | 11.6 | 3.9 | **2.6** | −1.6 | 79% |
| 1 bp | 11.6 | 3.2 | **3.6** | −2.2 | 76% |
| 3 bps | 11.7 | 5.2 | **5.7** | −5.1 | 74% |

- **The measurement recovers the injected effect.** Adverse selection on passive fills grows monotonically with impact.
- There's also some adverse selection at **zero** impact. Passive orders tend to fill exactly when the price moves through them (the "winner's curse" of resting liquidity).

**How much the answer depends on assumptions (1 bp impact, passive join):**

| Variant | Fill rate | Shortfall (bps) |
|---|---:|---:|
| Queue model, expiry 5 s | 41% | 8.1 |
| Queue model, expiry 10 s | 57% | 6.6 |
| Queue model, expiry 30 s | 76% | 3.6 |
| Queue model, expiry 60 s | 85% | 1.8 |
| Queue model, expiry 120 s | 91% | 0.6 |
| **Touch-probability model**, 30 s | 69% | **6.6** |

The fill model alone moves the 30-second result from 3.6 to 6.6 bps. **Conclusions about passive execution are only as good as the fill model, and only real order-book data can calibrate it.**

**Strategy-level (B0 engine, queue model, loss limit lifted):**

| Synthetic market | Aggressive P&L (fees) | Passive P&L (fees) |
|---|---:|---:|
| 0 bps impact | −9,539 (4,208) | **−4,119** (2,270) |
| 1 bp impact | −13,739 (4,612) | **−3,792** (2,418) |
| 3 bps impact | −14,068 (6,073) | **−6,183** (3,357) |

Passive trades about 41% of the shares and loses 55–72% less. It's still a loss.

## What this does and doesn't show
- ✅ The machinery works and measures what it should: queue position, fill odds, spread capture, adverse selection, clean-up cost and opportunity cost.
- ✅ Directionally, passive execution can cut costs a lot. The saving shrinks as adverse selection grows, and unfilled orders cost the most when you're forced to cross later.
- ❌ **Synthetic numbers aren't real numbers.**
  - Synthetic spreads average about 18 bps, versus about 1–2 bps for NASDAQ mega-caps, which flatters passive execution.
  - Synthetic queue sizes and cancellation behaviour aren't realistic.
  - The probe doesn't model our own market impact, hidden orders, or other venues (US equities trade on many exchanges).
- ❌ Signal-free probes understate opportunity cost. A real strategy's unfilled passive orders are disproportionately the ones where the signal was right.
- **Next:** calibrate the queue model and the adverse-selection markouts on real quote/trade data (see [QUOTE-DATA-REQUIREMENTS.md](QUOTE-DATA-REQUIREMENTS.md)). Then re-run the AI research with *measured* passive costs instead of a flat 1.5 bps.

Commands:

```bash
dotnet run -c Release --project src/Trademaster.Cli -- generate --out private/data/synth-impact1 --impact-bps 1
```

```bash
dotnet run -c Release --project src/Trademaster.Cli -- exec-study --data private/data/synth-impact1 --fill-model queue --expiry 30
```

```bash
dotnet run -c Release --project src/Trademaster.Cli -- run --data private/data/synth-impact1 --exec-style passive --fill-model queue --max-daily-loss 1e9
```
