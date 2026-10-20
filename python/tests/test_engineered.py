import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from trademaster_ml import engineered as E


class FakeP:
    def __init__(self, T, N):
        self.N, self.T = N, T


class FakeD:
    """3 stocks over 6 days; stock 0 gains 1% a day, stock 1 is flat, stock 2 loses 1%; QQQ flat."""
    def __init__(self):
        T, N = 6, 3
        self.P = FakeP(T, N)
        self.U = np.ones((T, N), bool)
        self.rG = np.tile([1.0, 0.5, 0.0], (T, 1))            # stock 2 is the predicted loser
        self.r_att = np.full((T, N), 0.5)
        self.vol = np.tile([0.02, 0.02, 0.04], (T, 1))
        self.earn = np.zeros((T, N), bool)
        self.r_cc = np.tile([0.01, 0.0, -0.01], (T, 1))
        self.r_co = np.tile([0.004, 0.0, -0.004], (T, 1))
        self.q_cc = np.zeros(T)
        self.dv = np.tile([100.0, 10.0, 1.0], (T, 1))


def base(**k):
    p = dict(alpha=0.0, q=0.0, gamma=0.0, lam=0.0, e="keep", mode="day")
    p.update(k)
    return p


def test_equal_weight_day_mode_returns_and_costs():
    D = FakeD()
    ex, tr, c = E.simulate(D, base(), 0, 5)
    assert abs(tr[0] - 1.0) < 1e-12 and abs(c[0] - E.COST_STOCK) < 1e-12        # first day buys the whole sleeve
    assert abs(ex[0] - (0.0 - E.COST_STOCK)) < 1e-12                             # (+1% + 0 - 1%) / 3 = 0, minus cost
    assert all(t < 0.05 for t in tr[1:])                                         # small drifts stay inside the band


def test_loser_filter_drops_the_bottom_and_reweights():
    D = FakeD()
    w = E.target(D, 0, base(q=0.3))
    assert w[2] == 0 and abs(w[0] - 0.5) < 1e-12 and abs(w[1] - 0.5) < 1e-12


def test_risk_sizing_and_alpha_core():
    D = FakeD()
    w = E.target(D, 0, base(gamma=1.0, alpha=0.5))
    assert abs(w.sum() - 0.5) < 1e-12 and w[2] < w[0]                           # the volatile stock gets less


def test_overnight_mode_pays_both_legs_every_day():
    D = FakeD()
    ex, tr, c = E.simulate(D, base(q=0.3, mode="overnight"), 0, 3)
    assert all(abs(t - 2.0) < 1e-12 for t in tr)
    assert abs(ex[0] - (0.5 * 0.004 - 2 * E.COST_STOCK)) < 1e-12


def test_size_weighting_and_cap():
    D = FakeD()
    w = E.target(D, 0, base(base="size"))                                       # 3 stocks: the cap becomes 1/3
    assert abs(w.sum() - 1) < 1e-9 and np.allclose(w, 1 / 3)
    x = np.array([50.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 20 stocks, one giant
    T, N = 2, len(x)
    D.P, D.U, D.rG, D.r_att = FakeP(T, N), np.ones((T, N), bool), np.full((T, N), 0.5), np.full((T, N), 0.5)
    D.vol, D.earn, D.dv = np.full((T, N), 0.02), np.zeros((T, N), bool), np.tile(x, (T, 1))
    w = E.target(D, 0, base(base="size"))
    assert abs(w.sum() - 1) < 1e-9 and abs(w[0] - E.SIZE_CAP) < 1e-9 and np.allclose(w[1:], 0.9 / 19)


def test_dsr_penalises_many_trials():
    sr = 0.05
    d_few, _ = E.dsr(sr, 1500, 0.0, 3.0, np.random.default_rng(0).normal(0, 0.005, 5))
    d_many, _ = E.dsr(sr, 1500, 0.0, 3.0, np.random.default_rng(0).normal(0, 0.03, 300))
    assert d_many < d_few


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
