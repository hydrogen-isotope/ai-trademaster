"""
Checks for trademaster_ml.profiles (size limits, name-to-ticker matching, prompt block; no model, no network).
Run: python python/tests/test_profiles.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trademaster_ml import profiles as P  # noqa: E402

TICKERS = {"MSFT": "MICROSOFT CORP", "AAPL": "Apple Inc.", "APLE": "Apple Hospitality REIT, Inc.", "SKM": "SK TELECOM CO.,LTD",
           "NVDA": "NVIDIA CORP", "INTC": "INTEL CORP", "NTLA": "Intellia Therapeutics, Inc.", "BRK.B": "BERKSHIRE HATHAWAY INC"}


def test_clean_enforces_amendment_g_sizes():
    pr = P.clean({"summary": "x" * 500, "products": [{"name": f"p{i}", "revenue_share_pct": 150} for i in range(9)],
                  "customers": [{"name": f"c{i}", "ticker": ""} for i in range(8)] + ["not a dict"], "suppliers": [], "competitors": [],
                  "drivers": ["d"] * 7})
    assert len(pr["summary"]) == 300 and len(pr["products"]) == 6 and len(pr["customers"]) == 6 and len(pr["drivers"]) == 5
    assert all(p["revenue_share_pct"] == 100 for p in pr["products"])


def test_ticker_matching_is_strict():
    idx = {}
    f = lambda n: P.find_ticker(n, TICKERS, idx)
    assert f("Microsoft") == "MSFT" and f("Apple") == "AAPL" and f("Intel") == "INTC" and f("Nvidia Corporation") == "NVDA"
    assert f("SK Hynix") is None                                           # not SK Telecom
    assert f("Google Cloud") == "GOOGL" and f("Amazon Web Services") == "AMZN"   # listed parents
    assert P.map_ticker({"name": "SK Hynix", "ticker": "SKM"}, TICKERS) is None or P.map_ticker({"name": "SK Hynix", "ticker": "SKM"}, TICKERS) != "SKM"
    assert P.map_ticker({"name": "Microsoft", "ticker": "MSFT"}, TICKERS) == "MSFT"


def test_prompt_block_lists_profile_and_related_news():
    pr = {"summary": "Makes memory chips.", "products": [{"name": "DRAM", "revenue_share_pct": 70}, {"name": "NAND", "revenue_share_pct": 0}],
          "customers": [{"name": "Apple", "ticker": "AAPL"}], "suppliers": [], "competitors": [{"name": "Samsung", "ticker": None}],
          "drivers": ["memory prices"], "source": {"kind": "SEC 10-K", "filed": "2026-10-09"}}
    rel = [{"ticker": "AAPL", "role": "customer", "name": "Apple", "headlines": [{"t": "2026-10-15T10:00:00Z", "headline": "Apple cuts orders"}]}]
    b = P.prompt_block(pr, rel)
    assert "SEC 10-K filed 2026-10-09" in b and "DRAM (~70% of revenue)" in b and "NAND;" not in b.split("products: ")[1][:5]
    assert "Apple (AAPL)" in b and "Samsung" in b and "AAPL (customer), 2026-10-15: Apple cuts orders" in b
    assert P.prompt_block(None, []) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
