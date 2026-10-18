"""
Point-in-time checks for trademaster_ml.fundamentals (no network, no private data).
Run: python python/tests/test_fundamentals.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from trademaster_ml import fundamentals as fu  # noqa: E402


def test_value_only_after_filing_and_first_filed_wins():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    # period ending 2023-12-31 filed on 2024-01-03 (value 10), restated on 2024-01-05 (value 99): keep 10
    rows = [[None, "2023-12-31", 10.0, "10-K", "2024-01-03"], [None, "2023-12-31", 99.0, "10-K/A", "2024-01-05"]]
    s = fu.to_days(dates, fu.instants(rows))
    assert np.isnan(s[0]) and np.isnan(s[1]), "value must not appear on or before its filing day"
    assert s[2] == 10 and s[4] == 10, "first filed value is used; the restatement is ignored"


def test_newer_period_wins_and_late_old_filing_does_not_override():
    dates = ["2024-05-01", "2024-05-02", "2024-05-03"]
    rows = [[None, "2024-03-31", 5.0, "10-Q", "2024-04-30"], [None, "2023-12-31", 7.0, "10-K", "2024-05-01"]]
    s = fu.to_days(dates, fu.instants(rows))
    assert list(s) == [5.0, 5.0, 5.0]


def test_quarters_from_year_to_date_and_ttm():
    rows = [  # fiscal year starting 2023-01-01: Q1 3-month, then 6M, 9M and FY cumulative
        ["2023-01-01", "2023-03-31", 1.0, "10-Q", "2023-05-01"],
        ["2023-01-01", "2023-06-30", 3.0, "10-Q", "2023-08-01"],
        ["2023-01-01", "2023-09-30", 6.0, "10-Q", "2023-11-01"],
        ["2023-01-01", "2023-12-31", 10.0, "10-K", "2024-02-15"],
    ]
    q = fu.quarters(rows)
    assert [q[e][0] for e in sorted(q)] == [1.0, 2.0, 3.0, 4.0]
    t = fu.ttm(q)
    assert t == [("2024-02-15", "2023-12-31", 10.0)]


def test_split_factor_converts_old_counts():
    splits = [("2024-06-10", 10.0)]
    assert fu.split_factor(splits, "2024-03-31") == 10.0
    assert fu.split_factor(splits, "2024-06-30") == 1.0


def test_stale_values_are_dropped():
    dates = ["2024-01-02", "2025-06-02"]
    s = fu.to_days(dates, [("2023-12-01", "2023-11-30", 1.0)])
    assert s[0] == 1.0 and np.isnan(s[1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
