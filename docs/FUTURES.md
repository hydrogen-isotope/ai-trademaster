# Futures support (COMEX gold and silver) and 5-day forecasts

## What was added
| Area | What |
|---|---|
| C# core | `InstrumentType.Future`. Instruments carry `Root`, `Multiplier`, `TickSize` and `LastTradeDate`, plus a tick table per instrument (GC $0.10 and SI $0.005, inside the same exchange). |
| C# calendar | Overnight sessions such as CME Globex 18:00–17:00 ET: Sunday-evening open, daily 60-minute break, Friday 17:00 close, and the trading date rolls forward at the evening open. |
| Config | `config/exchanges/COMEX.json`: contract specs checked against CME; holidays, fees and margins **not** verified, so `verified=false`. |
| Python `futures.py` | GC/SI specs, contract codes (GCZ6), last trading day, open-interest roll rule, ratio-adjusted continuous settlement series |
| Python `databento_client.py` | Licensed CME data (Databento GLBX.MDP3): final daily settlements and open interest; quotes the cost before every download and caps the spend |
| Python `daily_forecast.py` | 5-trading-day price forecasts with walk-forward validation and a pass/fail rule fixed in advance |

Futures are **not tradable** in the 1-second engine. The decision stage only trades stocks, and the COMEX config is
unverified. Trading futures would also need margin, short positions and multiplier-based P&L, which is a separate step.

## Verified contract specs (CME Group pages, read 2026-10-17)
| | Gold (GC) | Silver (SI) |
|---|---|---|
| Contract unit | 100 troy oz | 5,000 troy oz |
| Tick | $0.10 per oz = $10 | $0.005 per oz = $25 |
| Globex hours | Sun–Fri 18:00–17:00 ET, 60-min break from 17:00 | same |
| Listed contracts | 26 consecutive months, plus Jun and Dec within 72 months | 26 consecutive months, plus Jul and Dec within 60 months |
| Last trade | 12:30 CT, third last business day of the month | 12:25 CT, third last business day of the month |
| Settlement | deliverable | deliverable |

Sources: [gold specs](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html),
[silver specs](https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html).

## Futures prices: why settlements, and how rolls are handled
- **The daily "close" of a futures contract is its official settlement price.** The series uses only *final*
  settlements (Databento `statistics`, `stat_type` 3, flag bit 0 set, intraday flag clear). CME re-publishes these, and
  the last final value is used.
- **Rolls follow open interest.** Each day the series uses the contract with the highest open interest on the
  previous day. It never rolls back to an earlier expiry, and never holds a contract within 2 business days of its
  last trade.
- **Earlier prices are ratio-adjusted** at each roll, so the switch from one contract to the next doesn't create a
  fake return. The newest price is always the real price of the contract held. Databento's own continuous symbols are
  unadjusted, which is why the series is built here.

## Getting the data
1. Create a Databento account; new accounts get free usage credit. Create an API key.
2. Save it as `private/config/databento.json` (`{"api_key": "..."}`), or set the env var `DATABENTO_API_KEY`.
3. Get a cost quote. Nothing is spent until you add `--download`, and anything above `--max-cost-usd` (default $1) is
   refused:
   ```bash
   python -m trademaster_ml.databento_client --roots GC SI --start 2019-01-01
   ```

## 5-day forecasts
See the module docstring of `python/trademaster_ml/daily_forecast.py` for the model and the pass/fail rule.
`tools\daily-forecast.cmd` runs the end-of-day job: refresh data, score earlier forecasts, write the next one.

Forecasts are probabilities, not promises. Five trading days ahead, the expected price is close to today's price, so
the useful output is the range, and validation checks that the stated ranges hold as often as they claim.
