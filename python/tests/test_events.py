import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import date

import numpy as np

from trademaster_ml import earnings, holidays, v7, v7r


def test_us_holidays_known_dates():
    h = holidays.us_holidays(2026)
    for d in ("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
              "2026-11-26", "2026-12-25"):
        assert date.fromisoformat(d) in h, d
    assert len(h) == 10
    assert date(2022, 12, 31) not in holidays.us_holidays(2022)          # Saturday New Year: no Friday closure
    assert date(2022, 12, 26) in holidays.us_holidays(2022)              # Sunday Christmas -> Monday
    assert holidays.next_sessions("2026-11-25", 2) == ["2026-11-27", "2026-11-30"]


def test_report_dates_one_per_quarter_and_reaction_day():
    with tempfile.TemporaryDirectory() as d:
        old = earnings.BY_SYM, earnings.UPCOMING
        earnings.BY_SYM, earnings.UPCOMING = d, os.path.join(d, "up.json")
        try:
            json.dump([{"date": "2026-07-29", "hour": "amc", "year": 2026, "quarter": 2, "epsActual": 1.0},
                       {"date": "2026-10-28", "hour": None, "year": 2026, "quarter": 3, "epsActual": None},
                       {"date": "2026-11-04", "hour": None, "year": 2026, "quarter": 3, "epsActual": None},
                       {"date": "2026-08-03", "hour": "bmo", "year": 2026, "quarter": 99, "epsActual": 2.0}],
                      open(os.path.join(d, "X.json"), "w"))
            json.dump({"rows": [{"symbol": "X", "date": "2026-11-04"}]}, open(earnings.UPCOMING, "w"))
            r = earnings.report_dates("X", json.load(open(earnings.UPCOMING))["rows"])
            assert ("2026-11-04", "unknown") in r and all(x[0] != "2026-10-28" for x in r)  # the weekly calendar's date wins
            assert ("2026-07-29", "after") in r and ("2026-08-03", "before") in r
            dates = ["2026-07-29", "2026-07-30", "2026-07-31", "2026-08-03"]
            ev, nd = earnings.reaction_events(dates, ["X"])
            assert sorted(map(tuple, ev.tolist())) == [(1, 1, 0), (3, 3, 0)]               # amc -> next session; bmo same day
            # window counts: from close 0 over 1 day sees the amc reaction; from close 1 over 1 day sees nothing
            t, n = np.array([0, 1, 2]), np.array([0, 0, 0])
            assert earnings.window_counts(ev, nd, 1, t, n, 1).tolist() == [1, 0, 1]
            unk = np.array([[1, 2, 0]])                                                   # unknown time: day 1 or 2
            assert earnings.window_counts(unk, 4, 1, np.array([0, 1, 2]), np.array([0, 0, 0]), 1).tolist() == [1, 1, 0]
        finally:
            earnings.BY_SYM, earnings.UPCOMING = old


def test_blend_and_event_variance():
    v7.set_horizon(5)
    sd_e = np.array([0.05, 0.05])
    iv = np.array([0.40, np.nan])
    assert np.allclose(v7r.blend(sd_e, iv, 0.0), sd_e)
    full = v7r.blend(sd_e, iv, 1.0)
    assert np.isclose(full[0], 0.40 * np.sqrt(5 / 252)) and full[1] == 0.05
    half = v7r.blend(sd_e, iv, 0.5)
    assert np.isclose(half[0], np.sqrt(0.05 * 0.40 * np.sqrt(5 / 252)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
