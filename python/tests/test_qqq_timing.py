import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from trademaster_ml import qqq_timing as Q


def make(close, rf=0.0):
    dates = [f"2000-01-{i + 1:02d}" if i < 31 else f"d{i:04d}" for i in range(len(close))]
    return Q.Data(dates, close, [rf] * len(close))


def hold(cap=1.0, trend=0):
    return dict(target=None, window=None, cap=cap, trend=trend)


def test_full_exposure_equals_qqq():
    rng = np.random.default_rng(0)
    D = make(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    r, tr, c = Q.simulate(D, hold(), 5, 299)
    assert np.allclose(r, D.R[5:300]) and tr.sum() == 0
    m = Q.metrics(r, D.rf[5:300], D.R[5:300])
    assert abs(m["alpha_ann"]) < 1e-12 and abs(m["beta"] - 1) < 1e-12


def test_cash_earns_rf_and_leverage_pays_spread():
    D = make([100.0] * 50, rf=2.52)                                 # flat QQQ, 2.52%/yr = 1 bp a day
    r, _, _ = Q.simulate(D, hold(cap=2.0), 5, 49)
    assert abs(r[0] + (D.rf[5] + Q.BORROW_SPREAD / 252)) < 1e-15     # -1x borrowed at rf + spread (then drifts slightly)
    D2 = make([100.0] * 50 + [90.0] + [80.0] * 30, rf=2.52)          # falls below its 5-day average -> cash
    r2, _, _ = Q.simulate(D2, hold(trend=5), 10, 80)
    assert np.allclose(r2[-20:], D2.rf[0])


def test_one_day_delay():
    close = [100.0 + i for i in range(30)] + [50.0] + [50.0] * 10 + [200.0]            # drop on day 30: below the SMA from close 30
    D = make(close)
    tg = Q.target_series(D, hold(trend=5))
    assert tg[29] == 1 and tg[30] == 0
    r, tr, _ = Q.simulate(D, hold(trend=5), 10, 41)
    assert r[30 - 10] == D.R[30]                                         # day 30's fall is fully held
    assert tr[31 - 10] > 0 and tr[30 - 10] == 0                        # the exit trades at close 31, not 30


def test_vol_target_caps_and_band():
    rng = np.random.default_rng(1)
    D = make(100 * np.cumprod(1 + rng.normal(0, 0.005, 400)))        # ~8% vol: 20% target hits the cap
    tg = Q.target_series(D, dict(target=0.20, window=20, cap=1.5, trend=0))
    assert np.nanmax(tg) == 1.5
    r, tr, _ = Q.simulate(D, dict(target=0.20, window=20, cap=1.5, trend=0), 60, 399)
    assert (tr[1:] == 0).mean() > 0.8                                  # drift inside the 0.10 band does not trade


def test_grid_size():
    assert len(Q.designs()) == 78


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
