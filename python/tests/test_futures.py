"""
Offline checks for futures support (contract rules, rolls, settlement parsing). Run: python python/tests/test_futures.py
"""
import math, os, sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trademaster_ml import futures, databento_client  # noqa: E402


def test_specs_match_cme_pages():
    gc, si = futures.SPECS["GC"], futures.SPECS["SI"]
    assert (gc.multiplier, gc.tick, gc.tick_value) == (100, 0.10, 10.0)
    assert (si.multiplier, si.tick) == (5000, 0.005) and abs(si.tick_value - 25.0) < 1e-9


def test_contract_codes_and_year_decoding():
    assert futures.contract_code("GC", 2026, 12) == "GCZ6"
    assert futures.parse_contract("GCZ6", date(2021, 3, 1)) == ("GC", 2026, 12)
    assert futures.parse_contract("GCG1", date(2020, 12, 1)) == ("GC", 2021, 2)
    assert futures.parse_contract("SIH7", date(2026, 10, 1)) == ("SI", 2027, 3)


def test_last_trading_day_is_third_last_business_day():
    assert futures.last_trading_day(2026, 12) == date(2026, 12, 29)     # 31 Thu, 30 Wed, 29 Tue
    assert futures.last_trading_day(2026, 10) == date(2026, 10, 28)     # 30 Fri, 29 Thu, 28 Wed


def _days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_roll_adjustment_keeps_within_contract_returns():
    days = _days(date(2026, 9, 1), 30)
    near, far = "GCV6", "GCZ6"
    settles = {near: {}, far: {}}
    oi = {near: {}, far: {}}
    for k, d in enumerate(days):
        settles[near][d] = 100.0 * math.exp(0.001 * k)
        settles[far][d] = 102.0 * math.exp(0.001 * k)               # contango: far trades 2% higher
        oi[near][d] = 1000 - 40 * k
        oi[far][d] = 200 + 40 * k                                   # open interest migrates; crosses at k = 10
    rows = futures.continuous_series(settles, oi)
    rolls = [r for r in rows if r["rolled"]]
    assert len(rolls) == 1 and rolls[0]["contract"] == far
    adj = [r["adj"] for r in rows]
    rets = [math.log(adj[i + 1] / adj[i]) for i in range(len(adj) - 1)]
    assert max(abs(x - 0.001) for x in rets) < 1e-12               # no fake 2% jump at the roll
    assert abs(adj[-1] - settles[far][days[-1]]) < 1e-9            # latest price is the real contract price


def test_parse_records_keeps_final_settlements_only():
    recs = [
        {"symbol": "GCZ6", "stat_type": 3, "ts_ref": "2026-10-16T00:00:00.000000000Z", "price": "4180.0", "stat_flags": 2, "ts_recv": "1", "sequence": 1},
        {"symbol": "GCZ6", "stat_type": 3, "ts_ref": "2026-10-16T00:00:00.000000000Z", "price": "4186.7", "stat_flags": 3, "ts_recv": "2", "sequence": 2},
        {"symbol": "GCZ6", "stat_type": 3, "ts_ref": "2026-10-16T00:00:00.000000000Z", "price": "4190.0", "stat_flags": 11, "ts_recv": "3", "sequence": 3},
        {"symbol": "GCZ6-GCG7", "stat_type": 3, "ts_ref": "2026-10-16T00:00:00.000000000Z", "price": "-12.0", "stat_flags": 3, "ts_recv": "4", "sequence": 4},
        {"symbol": "GCZ6", "stat_type": 9, "ts_ref": "2026-10-16T00:00:00.000000000Z", "quantity": 400000, "ts_recv": "5", "sequence": 5},
        {"symbol": "GCZ6", "stat_type": 9, "ts_ref": "2026-10-16T00:00:00.000000000Z", "quantity": 401234, "ts_recv": "6", "sequence": 6},
    ]
    s, oi = databento_client.parse_records(recs)
    assert s == {"GCZ6": {date(2026, 10, 16): 4186.7}}             # preliminary and intraday dropped, spread ignored
    assert oi == {"GCZ6": {date(2026, 10, 16): 401234}}            # last open-interest message wins


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
