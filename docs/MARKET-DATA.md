# Market data for the dashboard

The markets page (`trademaster app`, or `tools\AI-Trademaster.cmd`) shows a watchlist, charts, quotes and trades.
It is **read-only**: it never places orders.

## Why not Yahoo Finance?
Yahoo has no official public API. Its Terms of Service (section 2(d)) forbid accessing or collecting data from its
services "using any automated means" without Yahoo's express prior permission. So the dashboard does not use Yahoo.

The older research scripts (`fetch.py`, `fetch1m.py`, `forward.py`, `signals_live.py`) still use Yahoo's chart
endpoint, and that use has the same problem. Moving them to Alpaca is a pending task (Alpaca's free plan also
provides years of historical 1-minute bars).

## Live data: Alpaca (free)
[Alpaca](https://alpaca.markets) Market Data API v2, free Basic plan, via its official, documented API:
- real-time quotes, trades and bars from the **IEX exchange** (one exchange, so prices are real but volumes only
  cover IEX);
- historical bars from **all US exchanges** (SIP), except the most recent 15 minutes;
- 200 requests per minute. The dashboard caches responses so that polling every 1–2 s stays well under this.

Check Alpaca's current plan terms yourself when you sign up. Free tiers change.

### Setup (you do this; the app never asks for your password)
1. Create a free Alpaca account and generate **API keys** for market data in the Alpaca dashboard. Paper-trading
   keys work for market data.
2. Give the keys to the app in **one** of these two ways:
   - environment variables `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`; or
   - a file `private/config/alpaca.json`. The `private/` folder is git-ignored, and the pre-commit and pre-push
     scanners block keys:
     ```json
     { "key_id": "YOUR_KEY_ID", "secret_key": "YOUR_SECRET", "feed": "iex" }
     ```
3. Restart the app. The top bar then reads **LIVE · ALPACA** instead of **DEMO · RECORDED DATA**.

The keys stay on this PC. They are sent only to `data.alpaca.markets` and are never shown in the browser or the logs.

## Without a key: demo mode
The app replays the recorded 1-minute data in `private/data/hist_1m_30d`: the latest session, looping, at 30× speed.
It is clearly labelled **DEMO** and has no bid/ask quotes.

## Later: Interactive Brokers
IBKR market data (TWS API, paid subscription) can be added as another provider behind the same `IMarketData`
interface (`src/Trademaster.Cli/MarketData.cs`). That is the natural next step once a broker account exists.
