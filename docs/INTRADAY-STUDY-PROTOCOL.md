# Intraday study: two documented day-trading rules (rules written 2026-10-20, before any result)

## Question
Do the two best-documented intraday rules that a retail algorithm can run (5-minute bars, market orders, no speed edge)
make money after realistic costs on QQQ, MU and NVDA?

## Data
Alpaca 5-minute bars (SIP, split-adjusted), 2016-01 → 2026-10-19: `private/data/intraday5/{QQQ,SPY,MU,NVDA}.json`.
Regular session only (09:30–16:00 New York); a day is used only if its 09:30, 10:00, 15:30 and last bars exist.
Bar times are bar starts: the 09:30 bar covers 09:30–09:35.

## Rules (fixed; nothing is tuned)
- **R1, intraday momentum** (Gao, Han, Li, Zhou, JFE 2018): signal = log(close of the 09:55 bar, i.e. 10:00 /
  previous regular-session close). At 15:30 (open of the 15:30 bar) go long if the signal > 0, short if < 0; exit at
  the session's last close. QQQ and SPY.
- **R2, opening-range breakout** (Zarattini & Aziz 2023): if the 09:30 bar closes above its open go long, below go
  short (no trade if equal), entering at the 09:35 bar's open. Stop: the 09:30 bar's low (long) or high (short); if a
  later bar's low (high) reaches it, exit at the stop price, or at that bar's open if it gapped through. Otherwise exit
  at the session's last close. QQQ, MU, NVDA.
- Both are reported long-and-short and long-only (the paper account is long-only).

## Costs (round trip, per trade)
QQQ and SPY: 2 bps base (also 1 and 4). MU and NVDA: 5 bps base (also 2 and 10).

## Periods and success (decided now)
- Development 2016-01-04 → 2022-12-30; hold-out 2023-01-03 → 2026-10-19. With no parameters to choose, both periods
  are reported together, but a rule counts as working only if **both** periods have mean net return per trade > 0 with
  t ≥ 2 at the base cost (t over trades; one trade per day).
- Also reported: hit rate, annualised net return (sum of daily trade returns per year) and the cost at which the
  mean breaks even.

## Results (added 2026-10-20 after the complete run; nothing above was changed)
Base costs (QQQ/SPY 2 bps, MU/NVDA 5 bps round trip); mean per trade, t over trades.

| rule | stock | development gross / net (t) | hold-out gross / net (t) | long-only net dev / hold-out | verdict |
|---|---|---|---|---|---|
| R1 intraday momentum | QQQ | +0.2 / −1.8 bps (−1.9) | −0.3 / −2.3 bps (−2.6) | −2.2 / −1.9 | fails |
| R1 intraday momentum | SPY | −0.6 / −2.6 bps (−3.1) | +0.5 / −1.6 bps (−2.2) | −3.4 / −1.5 | fails |
| R2 opening-range breakout | QQQ | +3.9 / +1.9 bps (+1.2) | +2.1 / +0.1 bps (+0.0) | +2.3 / −0.0 | fails |
| R2 opening-range breakout | MU | −1.3 / −6.3 bps (−1.7) | −1.8 / −6.8 bps (−1.2) | −11.2 / −7.4 | fails |
| R2 opening-range breakout | NVDA | +6.7 / +1.7 bps (+0.4) | +7.1 / +2.1 bps (+0.4) | +1.5 / +1.4 | fails |

**No rule works.** Intraday momentum has no edge even before costs on QQQ or SPY in 2016–26. The opening-range breakout
has a small, consistent edge before costs on QQQ (2–4 bps) and NVDA (about 7 bps), but costs leave about zero and the
results are far from significant; on MU it loses before costs. This matches the independent replications.
