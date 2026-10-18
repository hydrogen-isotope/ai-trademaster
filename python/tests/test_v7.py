"""
Checks for trademaster_ml.v7's distribution maths (no private data). Run: python python/tests/test_v7.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from trademaster_ml import v7  # noqa: E402
from trademaster_ml.daily_v2 import GRID, recal_levels_sym  # noqa: E402


def sample(n, m, sd, rng):
    return m + sd * rng.standard_t(v7.NU, n) * v7.TSCALE


def test_quantiles_are_ordered_and_median_is_mean():
    q = v7.quantiles(np.array([0.01, -0.02]), np.array([0.05, 0.10]), GRID)
    assert np.all(np.diff(q, axis=1) > 0)
    assert np.allclose(q[:, v7.I50], [0.01, -0.02])


def test_p_up_increases_with_the_mean():
    sd = np.full(3, 0.05)
    p = 1 - v7.p_le(0.0, np.array([-0.02, 0.0, 0.02]), sd, GRID)
    assert p[0] < p[1] < p[2] and abs(p[1] - 0.5) < 1e-9


def test_calibrated_model_covers_80_percent():
    rng = np.random.default_rng(0)
    n = 100_000
    m, sd = np.zeros(n), np.full(n, 0.04)
    y = sample(n, 0.0, 0.04, rng)
    q = v7.quantiles(m, sd, GRID)
    cov = np.mean((y >= q[:, v7.I10]) & (y <= q[:, v7.I90]))
    assert abs(cov - 0.80) < 0.01, cov


def test_recalibration_repairs_too_narrow_ranges():
    rng = np.random.default_rng(1)
    n = 100_000
    m, sd = np.zeros(n), np.full(n, 0.04)
    past = sample(n, 0.0, 0.06, rng)                    # reality is 1.5x wider than the model thinks
    R = recal_levels_sym(v7.t_cdf((past - m) / sd), GRID)
    future = sample(n, 0.0, 0.06, rng)
    q = v7.quantiles(m, sd, R)
    raw = v7.quantiles(m, sd, GRID)
    cov_raw = np.mean((future >= raw[:, v7.I10]) & (future <= raw[:, v7.I90]))
    cov_cal = np.mean((future >= q[:, v7.I10]) & (future <= q[:, v7.I90]))
    assert cov_raw < 0.7 and abs(cov_cal - 0.80) < 0.02, (cov_raw, cov_cal)


def test_crps_prefers_the_right_mean():
    rng = np.random.default_rng(2)
    y = sample(20_000, 0.01, 0.04, rng)
    sd = np.full(len(y), 0.04)
    right = v7.crps_rows(v7.quantiles(np.full(len(y), 0.01), sd, GRID), y).mean()
    wrong = v7.crps_rows(v7.quantiles(np.full(len(y), -0.01), sd, GRID), y).mean()
    assert right < wrong


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
