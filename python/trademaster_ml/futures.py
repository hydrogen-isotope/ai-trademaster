"""
Futures support: contract specifications, contract codes, last trading days, roll rules and roll-adjusted daily
settlement series.

Specs for GC and SI were taken from CME Group's official contract-spec pages on 2026-10-17:
  https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html
  https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html

The daily "close" of a futures contract is its official daily SETTLEMENT price, not the last trade. A continuous
series is built by picking, for each trading date, the contract with the highest open interest on the previous date
(never rolling back to an earlier expiry), then ratio-adjusting all earlier settlements so that a roll does not
create a fake return. Returns are therefore always measured within one contract.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta

MONTH_CODES = "FGHJKMNQUVXZ"          # Jan..Dec


@dataclass(frozen=True)
class FutureSpec:
    root: str
    name: str
    exchange: str
    unit: str
    multiplier: float                  # contract size in units (troy ounces)
    tick: float                        # minimum price fluctuation (USD per unit)
    globex_hours_et: str
    last_trade_rule: str               # human-readable, from the spec page
    last_trade_time_ct: str
    settlement_time_et: str            # end of the settlement window (TAS cutoff)
    source: str
    verified_on: str
    active_months: tuple = field(default=())   # liquid delivery months by market convention (not exchange rules)

    @property
    def tick_value(self):
        return self.tick * self.multiplier


SPECS = {
    "GC": FutureSpec(
        root="GC", name="COMEX Gold", exchange="COMEX", unit="troy ounce", multiplier=100, tick=0.10,
        globex_hours_et="Sun-Fri 18:00-17:00 ET, 60-minute break from 17:00",
        last_trade_rule="third last business day of the contract month", last_trade_time_ct="12:30",
        settlement_time_et="13:30", verified_on="2026-10-17",
        source="https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html",
        active_months=(2, 4, 6, 8, 10, 12)),
    "SI": FutureSpec(
        root="SI", name="COMEX Silver", exchange="COMEX", unit="troy ounce", multiplier=5000, tick=0.005,
        globex_hours_et="Sun-Fri 18:00-17:00 ET, 60-minute break from 17:00",
        last_trade_rule="third last business day of the contract month", last_trade_time_ct="12:25",
        settlement_time_et="13:25", verified_on="2026-10-17",
        source="https://www.cmegroup.com/markets/metals/precious/silver.contractSpecs.html",
        active_months=(3, 5, 7, 9, 12)),
}

# CME Group full-day closures relevant to COMEX metals settlements (verify each year against CME's holiday calendar).
# Days with an early close still settle and are trading days.
CME_HOLIDAYS = {date(2026, 1, 1), date(2026, 4, 3), date(2026, 12, 25), date(2027, 1, 1), date(2027, 3, 26),
                date(2027, 12, 24)}


def is_business_day(d, holidays=CME_HOLIDAYS):
    return d.weekday() < 5 and d not in holidays


def contract_code(root, year, month):
    """('GC', 2026, 12) -> 'GCZ6' (CME Globex style: single-digit year)."""
    return f"{root}{MONTH_CODES[month - 1]}{year % 10}"


def parse_contract(symbol, first_seen):
    """'GCZ6' -> (root, year, month). The single-digit year is reused every 10 years, so it is resolved to the
    earliest year whose contract month has not ended before the contract was first seen in the data."""
    root, mcode, ydigit = symbol[:-2], symbol[-2], int(symbol[-1])
    month = MONTH_CODES.index(mcode) + 1
    year = first_seen.year - first_seen.year % 10 + ydigit
    while date(year, month, 28) < first_seen:
        year += 10
    while date(year - 10, month, 28) >= first_seen:
        year -= 10
    return root, year, month


def last_trading_day(year, month, holidays=CME_HOLIDAYS):
    """Third last business day of the contract month (GC and SI rule)."""
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    seen = 0
    while True:
        if is_business_day(d, holidays):
            seen += 1
            if seen == 3:
                return d
        d -= timedelta(days=1)


def trading_dates_after(d, n, holidays):
    out, x = [], d
    while len(out) < n:
        x += timedelta(days=1)
        if x.weekday() < 5 and x not in holidays:
            out.append(x)
    return out


def continuous_series(settles, open_interest, min_days_to_expiry=2):
    """
    settles:       {contract_symbol: {date: settlement_price}}
    open_interest: {contract_symbol: {date: open_interest}}
    Returns a list of dicts sorted by date: {date, contract, settle (raw), adj (ratio back-adjusted), rolled (bool)}.

    Rule: on each date use the contract that had the highest open interest on the PREVIOUS date with data, among
    contracts with a settlement today, never an earlier expiry than the one held, and never a contract within
    `min_days_to_expiry` business days of its last trading day. The roll ratio is new/old settlement on the roll date.
    """
    dates = sorted({d for s in settles.values() for d in s})
    if not dates:
        return []
    expiry = {}
    for sym in settles:
        root, y, m = parse_contract(sym, min(settles[sym]))
        expiry[sym] = (y, m, last_trading_day(y, m))
    rows, held, prev_date = [], None, None
    for d in dates:
        live = [s for s in settles if d in settles[s]
                and sum(1 for k in range(1, 40) if is_business_day(d + timedelta(days=k)) and d + timedelta(days=k) <= expiry[s][2]) >= min_days_to_expiry]
        if not live:
            continue
        def oi(s):
            return open_interest.get(s, {}).get(prev_date, -1) if prev_date else open_interest.get(s, {}).get(d, -1)
        best = max(live, key=lambda s: (oi(s), -expiry[s][0] * 12 - expiry[s][1]))
        if held is not None and held in live and (expiry[best][:2] < expiry[held][:2] or oi(best) <= oi(held)):
            best = held                                   # no roll backwards, no roll without a strictly better OI
        rolled = held is not None and best != held
        rows.append({"date": d, "contract": best, "settle": settles[best][d], "rolled": rolled,
                     "prev_contract": held if rolled else None,
                     "prev_contract_settle": settles[held].get(d) if rolled and held else None})
        held, prev_date = best, d
    # ratio back-adjust: walk backwards, multiply earlier prices by new/old on each roll date
    factor = 1.0
    for i in range(len(rows) - 1, -1, -1):
        rows[i]["adj"] = rows[i]["settle"] * factor
        r = rows[i]
        if r["rolled"]:
            old = r["prev_contract_settle"]
            if old:                                       # both contracts settled on the roll date
                factor *= r["settle"] / old
    return rows
