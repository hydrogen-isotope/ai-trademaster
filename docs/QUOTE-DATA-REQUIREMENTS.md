# Quote data needed for execution research

Goal: measure, on real markets, what [EXECUTION-RESEARCH.md](EXECUTION-RESEARCH.md) can only simulate:
- fill probability and time to fill by queue position;
- cancellation rates ahead of us;
- adverse selection (markouts after passive fills);
- the true all-in cost of passive vs aggressive execution.

Free Yahoo bars **cannot** answer any of this.

## Three tiers (each adds to the one before)
| Tier | Content | What it lets us measure | Limitation |
|---|---|---|---|
| **1. Top of book + trades (L1 / BBO)** | Every quote update: bid/ask price **and size**. Every trade: price, size, sale conditions. Exchange timestamps (µs or ns) and sequence numbers. | Spreads, the L1 queue model as built (queue ahead = displayed size), markouts and adverse selection, fill-model calibration under the pro-rata-cancel assumption | Queue position is only estimated; nothing behind the best price is visible |
| **2. Depth by price (L2, e.g. 10 levels)** | Aggregate size at each price level, per update | Queue depth behind the best, deeper placement, book-imbalance signals | Still no per-order queue; cancellations ahead vs behind us are indistinguishable |
| **3. Market-by-order (L3, full order-by-order)** | Every add / modify / cancel / execute with order IDs, e.g. a Nasdaq TotalView-ITCH-style feed | **Exact queue reconstruction** for a hypothetical order, true cancel-ahead rates, realistic fill simulation | Largest data volume and cost |

**Recommendation:** start with **tier 3 for 1 venue × 20–30 names × 3 months**. That's the gold standard for passive-order research, and tiers 1–2 can be derived from it. If cost is a constraint, tier 1 is the minimum useful step: it calibrates exactly the model we already have.

## Coverage
- **Symbols:** 20–50 of the names in our research universe (mega-caps plus mid-liquidity), and a few semiconductors because the AI work focused there.
- **Period:** at least 3 months, ideally 6+, including high-volatility days, earnings days, the open and close auctions, and index-rebalance days.
- **Venues (US):** volume is spread across many exchanges. Calibrating on one venue (e.g. Nasdaq) is a start. Consolidated BBO/trades (SIP-style) are needed to see where trades actually happen.
- **Timestamps:** exchange (matching-engine) time, plus sequence numbers to detect gaps. Our feed handler already does gap and duplicate detection.

## Reference data also needed
- **Fee/rebate schedule** for the venue and our broker. US exchanges often pay maker rebates and charge taker fees, with tiering. This decides whether passive execution earns or pays.
- **Tick-size rules** and **order types available through the broker**: post-only, IOC, hidden or midpoint.
- **Broker round-trip latency** to the venue. Our simulator assumes 5 ms each way.
- **Corporate actions and halts** (LULD bands, trading halts).
- **For SGX:** the equivalent full-depth market-by-order product, board lots, tick table, fees, and sessions/auctions. These must come from SGX's official documentation (the config skeleton refuses to load until then).

## Possible sources (verify licensing, coverage and current pricing before buying)
- **Databento** — historical and live US equities, including Nasdaq order-by-order (MBO), depth (MBP-10) and top-of-book schemas, via API.
- **Nasdaq** directly — TotalView-ITCH historical and real-time products.
- **LSEG Tick History**, **AlgoSeek**, **Polygon.io** — quotes and trades history for US equities (depth varies by product).
- **NYSE TAQ** — consolidated trades and quotes history (often accessed via academic platforms such as WRDS).
- **SGX** — market data products via SGX or its licensed distributors.

Check whether personal or research use is allowed, and whether redistribution or display (e.g. in the dashboard) needs a separate licence.

## Rough sizing (orders of magnitude only — confirm with the vendor)
- Liquid mega-caps generate a very large number of book updates per day: hundreds of thousands to millions of messages per symbol on order-by-order feeds.
- Expect **tens to hundreds of GB compressed per month** for tier 3 across ~30 names; tier 1 is much smaller.
- The replay format (`.tmrp`) and the fixed-point price pipeline are ready. We'd add an importer per vendor format and an L3 order book to the simulator.

## After the data: the only fully real test
Simulated fills can't capture our own impact or the live queue. The last step, with your explicit approval, is a **tiny-size live passive test through a broker** that compares realised fills and markouts with the simulator's predictions. Paper-trading accounts generally don't simulate queues realistically, so they can't stand in for this.
