"""
Checks for trademaster_ml.indexes (look-back returns, sector mapping, ETF forecast; no network, no private data).
Run: python python/tests/test_indexes.py
"""
import math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trademaster_ml import indexes as ix  # noqa: E402


def test_lookback_returns_and_ytd():
    dates = [f"2025-12-{d:02d}" for d in range(22, 32)] + [f"2026-01-{d:02d}" for d in range(1, 31)]
    closes = [100.0] * 10 + [100.0 + i for i in range(1, 31)]
    st = ix.stats(dates, closes)
    assert math.isclose(st["d1"], 130 / 129 - 1) and math.isclose(st["w1"], 130 / 125 - 1)
    assert math.isclose(st["ytd"], 130 / 100 - 1)                      # from the last close of the previous year
    assert st["m3"] is None and st["y1"] is None                       # not enough history


def test_every_sector_code_maps_once():
    seen = {}
    for key, _, codes, _ in ix.SECTORS:
        for c in codes:
            assert c not in seen, f"SIC {c} in {seen.get(c)} and {key}"
            seen[c] = key
    assert ix.SECTOR_OF[36] == "semis" and ix.SECTOR_OF[73] == "software" and 99 not in ix.SECTOR_OF


def test_etf_forecast_scales_with_beta():
    a = ix.forecast(100.0, "2026-10-16", 0.002, 0.0014, beta=1.0)
    b = ix.forecast(100.0, "2026-10-16", 0.002, 0.0014, beta=2.0)
    assert b["hi80"] - b["lo80"] > a["hi80"] - a["lo80"] and b["median"] > a["median"] > 100
    assert a["lo95"] < a["lo80"] < a["lo50"] < a["median"] < a["hi50"] < a["hi80"] < a["hi95"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
