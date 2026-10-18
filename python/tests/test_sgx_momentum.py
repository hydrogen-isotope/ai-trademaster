"""
Checks for trademaster_ml.sgx_momentum (synthetic data; no network, no private data).
Run: python python/tests/test_sgx_momentum.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from trademaster_ml import sgx_momentum as sm  # noqa: E402


def test_month_end_detection():
    dates = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02"]
    assert [sm.is_month_end(dates, i) for i in range(4)] == [False, False, True, False]
    assert sm.is_month_end(["2026-10-30"], 0)            # latest close: next weekday is in November
    assert not sm.is_month_end(["2026-10-29"], 0)


def test_picks_use_only_past_data_and_pick_winners():
    rng = np.random.default_rng(0)
    T, N = 300, 40
    drift = np.linspace(-0.002, 0.002, N)                  # stock N-1 trends up most
    close = np.exp(np.cumsum(drift + 0.01 * rng.standard_normal((T, N)), axis=0)) * 5
    volume = np.full((T, N), 1e6)
    t = 280
    picks, nu = sm.picks_at(t, close, volume)
    assert nu == N and len(picks) == 10
    lc = np.log(close)
    signal = lc[t - 21] - lc[t - 252]
    assert sorted(int(j) for j in picks) == sorted(int(j) for j in np.argsort(-signal)[:10])   # top 10 by 12-1 momentum
    assert min(picks) >= N // 2                            # all from the up-trending half
    future = close.copy()
    future[t + 1:] *= rng.uniform(0.1, 10, (T - t - 1, N))  # changing the future must not change the picks
    assert sm.picks_at(t, future, volume)[0] == picks


def test_price_filter_excludes_penny_stocks():
    T, N = 300, 20
    close = np.exp(np.cumsum(np.full((T, N), 0.001), axis=0))
    close[:, :5] *= 0.05                                    # five stocks under 0.20
    picks, nu = sm.picks_at(280, close, np.full((T, N), 1e6))
    assert nu == 15 and all(j >= 5 for j in picks)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
