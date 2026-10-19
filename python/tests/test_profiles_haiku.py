"""
Checks for trademaster_ml.profiles_haiku (recency, clean-up, segments from the input, peers added; no network).
Run: python python/tests/test_profiles_haiku.py
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trademaster_ml import profiles as P, profiles_haiku as H  # noqa: E402

TICKERS = {"NVDA": "NVIDIA CORP", "SNDK": "Sandisk Corp", "MU": "MICRON TECHNOLOGY INC"}


def test_check_one_applies_recency_cleanup_segments_and_peers():
    P.market_status = lambda name: None                                    # no network in tests
    with tempfile.TemporaryDirectory() as tmp:
        H.DIR = tmp
        os.makedirs(os.path.join(tmp, "nasdaq"))
        os.makedirs(os.path.join(tmp, "input"))
        json.dump({"segments": {"period": "2025-08-28", "axis": "products and services", "items": [{"label": "DRAM", "pct": 76.5, "prev_pct": 70.1}]},
                   "peers": [{"name": "Sandisk Corp", "ticker": "SNDK"}, {"name": "NVIDIA Corp", "ticker": "NVDA"}],
                   "filing": {"source": {"kind": "SEC 10-K", "filed": "2026-10-09", "url": "u"}}},
                  open(os.path.join(tmp, "input", "MU.json"), "w"))
        raw = {"symbol": "MU", "market": "nasdaq", "name": "Micron Technology", "summary": "Memory maker.",
               "products": [{"name": "DRAM", "revenue_share_pct": 70}],
               "customers": [{"name": "NVIDIA", "ticker": "NVDA", "evidence": "HBM supply", "source_url": "x", "source_date": "2025-06"},
                             {"name": "Old Customer Inc", "ticker": "", "evidence": "old", "source_url": "y", "source_date": "2019"},
                             {"name": "Cloud providers", "ticker": "", "evidence": "cat", "source_url": "z", "source_date": "2025"}],
               "suppliers": [], "competitors": [{"name": "Micron Technology", "ticker": "MU", "evidence": "", "source_url": "", "source_date": "2025"}],
               "drivers": ["HBM demand"]}
        f = os.path.join(tmp, "nasdaq", "MU.json")
        json.dump(raw, open(f, "w"))
        pr = H.check_one(f, TICKERS)
    assert [e["name"] for e in pr["customers"]] == ["NVIDIA"]
    assert pr["removed"]["stale_source"] == ["Old Customer Inc"] and pr["removed"]["category"] == ["Cloud providers"]
    assert [e["ticker"] for e in pr["competitors"]] == ["SNDK", "NVDA"]   # itself dropped; Eulerpool peers added
    assert pr["segments"]["items"][0]["pct"] == 76.5 and pr["products"][0]["revenue_share_pct"] == 0
    assert pr["source"]["filed"] == "2026-10-09" and "Haiku" in pr["source"]["kind"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
