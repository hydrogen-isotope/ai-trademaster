import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from trademaster_ml import fairvalue as FVm


def _setup(n=10):
    rows = [{"symbol": f"P{i}", "close": 100.0} for i in range(n)] + [{"symbol": "X", "close": 50.0}]
    sector = {r["symbol"]: "semis" for r in rows}
    # peers: P/E 20 (yield 0.05), P/S 4 (0.25), P/B 2 (0.5)
    ratios = {"pe": {f"P{i}": 0.05 for i in range(n)}, "ps": {f"P{i}": 0.25 for i in range(n)}, "pb": {f"P{i}": 0.5 for i in range(n)}}
    return rows, sector, ratios


def test_fair_value_is_peer_multiple_times_own_value():
    rows, sector, ratios = _setup()
    ratios["pe"]["X"], ratios["ps"]["X"], ratios["pb"]["X"] = 0.10, 0.25, 0.25     # X: P/E 10, P/S 4, P/B 4
    r = FVm.compute(rows, ratios, sector)["X"]
    assert r["methods"]["pe"]["fair"] == 100.0          # EPS 5 x peer P/E 20
    assert r["methods"]["ps"]["fair"] == 50.0
    assert r["methods"]["pb"]["fair"] == 25.0           # book 12.5 x peer P/B 2
    assert r["fair"] == 50.0 and r["verdict"] == "fair" and r["n_methods"] == 3


def test_negative_earnings_skip_pe_and_verdicts():
    rows, sector, ratios = _setup()
    ratios["pe"]["X"], ratios["ps"]["X"], ratios["pb"]["X"] = -0.02, 0.05, 0.1
    r = FVm.compute(rows, ratios, sector)["X"]
    assert r["methods"]["pe"]["own"] is None and "fair" not in r["methods"]["pe"]
    assert r["n_methods"] == 2 and r["verdict"] == "overvalued"


def test_few_sector_peers_fall_back_to_universe():
    rows, sector, ratios = _setup()
    sector["X"] = "utilities"
    for k in ratios:
        ratios[k]["X"] = ratios[k]["P0"]
    r = FVm.compute(rows, ratios, sector)["X"]
    assert all(m["scope"] == "universe" for m in r["methods"].values())
    assert np.isclose(r["fair"], 50.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
