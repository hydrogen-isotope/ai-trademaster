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


def test_tidy_drops_categories_itself_duplicates_and_note_issuers():
    tk = dict(TICKERS, GFS="GLOBALFOUNDRIES Inc.", AMD="ADVANCED MICRO DEVICES INC", GLDI="CREDIT SUISSE AG", USOI="CREDIT SUISSE AG",
              SLVO="CREDIT SUISSE AG", MU="MICRON TECHNOLOGY INC", GPUS="Hyperscale Data, Inc.")
    P.note_issuers.__defaults__[0].clear()                                # caches built from the real list in other runs
    P.find_ticker.__defaults__[0].clear()
    pr = {"customers": [{"name": "Hyperscale data centers", "ticker": "GPUS"}, {"name": "Microsoft", "ticker": ""}, {"name": "PC OEMs", "ticker": ""}],
          "suppliers": [{"name": "GLOBALFOUNDRIES Inc. (GF)", "ticker": ""}],
          "competitors": [{"name": "Advanced Micro Devices, Inc.", "ticker": "AMD"}, {"name": "AMD", "ticker": ""},
                          {"name": "Micron Technology", "ticker": "MU"}, {"name": "Credit Suisse", "ticker": "GLDI"}]}
    P.tidy(pr, "MU", "Micron Technology, Inc.", "We buy wafers from GLOBALFOUNDRIES and compete with AMD.", tk)
    assert [e["name"] for e in pr["customers"]] == ["Microsoft"] and pr["customers"][0]["ticker"] == "MSFT"
    assert pr["suppliers"][0]["ticker"] == "GFS" and pr["suppliers"][0]["in_filing"]
    assert [(e["name"], e["ticker"]) for e in pr["competitors"]] == [("Advanced Micro Devices, Inc.", "AMD"), ("Credit Suisse", None)]
    assert pr["removed"]["category"] == ["Hyperscale data centers", "PC OEMs"] and pr["removed"]["duplicate"] == ["AMD"]
    assert pr["removed"]["itself"] == ["Micron Technology"]
    P.note_issuers.__defaults__[0].clear()
    P.find_ticker.__defaults__[0].clear()


def test_select_text_finds_relationships_deep_in_a_long_filing():
    filler = " ".join(f"Our products are designed for many uses and markets number {i}." for i in range(600))   # ~38k characters
    deep = "We utilize Taiwan Semiconductor Manufacturing Company Limited (TSMC) for the production of wafers."
    rev = "The Data Center segment represented 48% of net revenue in 2025."
    section = "ITEM 1. BUSINESS Overview We design chips. " + filler + " " + deep + " " + rev + " " + filler
    t = P.select_text(section, section + " Exhibit 21 Advanced Micro Devices, Inc. subsidiaries.", budget=6000)
    assert len(t) <= 6000 and t.startswith("ITEM 1. BUSINESS") and deep in t and rev in t
    assert "number 599." not in t                                         # filler is not selected


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
