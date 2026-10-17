"""
Offline checks for the Alpaca client (no network, no key needed). Run: python python/tests/test_alpaca.py
Fixtures are hand-made responses in Alpaca's documented /v2/stocks/bars shape.
"""
import json, os, sys, tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trademaster_ml import alpaca, dataset  # noqa: E402

NY = ZoneInfo("America/New_York")


def et(h, m, day=16):
    return int(datetime(2026, 10, day, h, m, tzinfo=NY).timestamp())


def bar(t, c, v=100):
    return {"t": datetime.fromtimestamp(t, ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"), "o": c, "h": c + 1, "l": c - 1, "c": c, "v": v, "n": 1, "vw": c}


def test_parse_ts():
    assert alpaca.parse_ts("2026-10-16T13:30:00Z") == et(9, 30)
    assert alpaca.parse_ts("2026-10-16T13:30:00.123456789Z") == et(9, 30)
    assert alpaca.parse_ts("2026-10-16T09:30:00-04:00") == et(9, 30)
    assert alpaca.rfc3339(et(9, 30)) == "2026-10-16T13:30:00Z"


def test_regular_session_filters_pre_and_after_hours():
    rows = [(et(9, 0), 1, 1, 1, 1, 1), (et(9, 30), 2, 2, 2, 2, 1), (et(15, 59), 3, 3, 3, 3, 1), (et(16, 0), 4, 4, 4, 4, 1)]
    assert [r[0] for r in alpaca.regular_session(rows)] == [et(9, 30), et(15, 59)]


def test_session_hours_are_aligned_to_the_open():
    rows = [(et(9, 0), 1, 1, 1, 1, 5),                                   # pre-market, dropped
            (et(9, 30), 10, 11, 9, 10.5, 1), (et(10, 0), 10.5, 12, 10, 11, 2),
            (et(10, 30), 11, 11.5, 10.8, 11.2, 3), (et(15, 30), 20, 21, 19, 20.5, 4)]
    h = alpaca.session_hours(rows)
    assert [r[0] for r in h] == [et(9, 30), et(10, 30), et(15, 30)]
    assert h[0] == (et(9, 30), 10, 12, 9, 11, 3)                         # o of first, max h, min l, c of last, sum v


def test_get_bars_paginates_and_merges():
    pages = [
        {"bars": {"AAA": [bar(et(9, 30), 10), bar(et(9, 31), 11)]}, "next_page_token": "p2"},
        {"bars": {"AAA": [bar(et(9, 32), 12)], "BBB": [bar(et(9, 30), 50)]}, "next_page_token": None},
    ]
    calls = []

    def fake_get(path, params, creds, retries=5):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    orig = alpaca._get
    alpaca._get = fake_get
    try:
        out = alpaca.get_bars(["AAA", "BBB"], "1Min", et(9, 30), et(10, 0), creds=("k", "s", "iex"), pause=0)
    finally:
        alpaca._get = orig
    assert len(calls) == 2 and calls[1]["page_token"] == "p2" and calls[0]["feed"] == "sip"
    assert [r[4] for r in out["AAA"]] == [10, 11, 12] and out["BBB"][0][0] == et(9, 30)


def test_chart_json_loads_in_build_panel():
    with tempfile.TemporaryDirectory() as d:
        rows = {s: [(et(9, 30) + 300 * k, 0, 0, 0, 100 + k + j, 1000) for k in range(30)] for j, s in enumerate(["AAA", "BBB", "ETF"])}
        for s, r in rows.items():
            json.dump(alpaca.chart_json(r), open(os.path.join(d, f"{s}-5m.json"), "w"))
        json.dump({"stocks": ["AAA", "BBB"], "etfs": ["ETF"], "reference_etf": {"AAA": "ETF", "BBB": "ETF"}, "interval": "5m"},
                  open(os.path.join(d, "universe.json"), "w"))
        P = dataset.build_panel(d, horizon=2)
        assert P.symbols == ["AAA", "BBB", "ETF"] and P.close.shape == (30, 3) and P.close[0, 0] == 100


def test_missing_key_is_a_clear_error():
    saved = {k: os.environ.pop(k, None) for k in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]}
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                alpaca.credentials(root=d)
                raise AssertionError("expected AlpacaConfigError")
            except alpaca.AlpacaConfigError as e:
                assert "docs/MARKET-DATA.md" in str(e)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
