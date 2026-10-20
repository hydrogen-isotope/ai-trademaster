import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from trademaster_ml import earnings_study as es

DATES = np.array(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])


def test_reaction_day_rule():
    assert es.reaction_index(DATES, "2020-01-03", "before") == 1
    assert es.reaction_index(DATES, "2020-01-03", "after") == 2
    assert es.reaction_index(DATES, "2020-01-04", "before") == 2        # Saturday report: next session
    gaps = {1: 0.01, 2: 0.05, 3: 0.02}
    assert es.reaction_index(DATES, "2020-01-03", "unknown", lambda i: gaps[i]) == 2     # larger gap is next day
    assert es.reaction_index(DATES, "2020-01-03", "unknown", lambda i: {1: 0.06, 2: 0.05}[i]) == 1
    assert es.reaction_index(DATES, "2020-01-08", "after") is None
    assert es.timing_of("amc") == "after" and es.timing_of("dmh") == "before" and es.timing_of(None) == "unknown"
    assert es.timing_of("time-not-supplied") == "unknown"


def _ev(rets, qrets, signs_gap, dates, period="dev", surprise=None):
    n = len(rets)
    z = np.zeros(n)
    return dict(gap=np.array(signs_gap, float), gap_size=np.full(n, 5.0), sigma=np.full(n, .02),
                surprise=np.array(surprise if surprise is not None else [1.0] * n),
                tercile=np.full(n, 2), known_time=np.ones(n, int), r_o2c=np.array(rets, float), q_o2c=np.array(qrets, float),
                r_c5=z.copy(), q_c5=z.copy(), r_c20=z.copy(), q_c20=z.copy(), r_d0=np.array(rets, float), q_d0=z.copy(),
                period=np.array([period] * n, dtype=object), reaction_date=np.array(dates, dtype=object),
                quarter=np.array(["2020-Q1"] * n, dtype=object))


def test_cost_arithmetic_and_clustering():
    # continuation, long: stock +2%, QQQ +0.5% -> gross adj 1.5%; cost 15 bp open+close -> net 1.35%
    ev = _ev([0.02, 0.02], [0.005, 0.005], [1, 1], ["d1", "d2"])
    r = es.evaluate(ev, "H1_cont_g1", "dev")
    assert abs(r["mean_gross_adj"] - 0.015) < 1e-12 and abs(r["mean_net_adj"] - 0.0135) < 1e-12
    assert abs(es.evaluate(ev, "H1_cont_g1", "dev", cost_mult=2.0)["mean_net_adj"] - 0.012) < 1e-12
    assert abs(r["mean_net_raw"] - 0.0185) < 1e-12 and r["hit_rate"] == 1.0
    # reversal short on a positive gap, stock +2%: loses
    r = es.evaluate(ev, "H1_rev_g1", "dev")
    assert r["mean_net_adj"] < 0 and r["hit_rate"] == 0.0
    # long-only drops shorts
    ev2 = _ev([0.02, -0.02], [0, 0], [1, -1], ["d1", "d2"])
    assert es.evaluate(ev2, "H1_cont_g1", "dev")["n_trades"] == 2
    assert es.evaluate(ev2, "H1_cont_g1", "dev", long_only=True)["n_trades"] == 1
    # clustering: day A has 3 trades (net 0.01 each), day B one (net 0.03): day means 0.01, 0.03 -> t over 2 days
    ev3 = _ev([0.0115] * 3 + [0.0315], [0] * 4, [1] * 4, ["A", "A", "A", "B"])
    r = es.evaluate(ev3, "H1_cont_g1", "dev")
    assert r["n_trades"] == 4 and r["n_days"] == 2
    dm = np.array([0.01, 0.03])
    assert abs(r["t_clustered"] - dm.mean() / (dm.std(ddof=1) / np.sqrt(2))) < 1e-9
    assert abs(r["mean_net_adj"] - 0.01 * 3 / 4 - 0.03 / 4) < 1e-12           # per-trade mean, not per-day
    # H3 cost = 5 + 3 bp
    ev4 = _ev([0, 0], [0, 0], [1, 1], ["a", "b"])
    ev4["r_c5"] = np.array([0.01, 0.01])
    assert abs(es.evaluate(ev4, "H3_k5_a", "dev")["mean_net_adj"] - (0.01 - 0.0008)) < 1e-12


def test_holdout_guard():
    ev = _ev([0.01], [0], [1], ["d1"], period="holdout")
    try:
        es.evaluate(ev, "H1_cont_g1", "holdout")
        raise AssertionError("hold-out should be refused")
    except ValueError:
        pass
    assert es.evaluate(ev, "H1_cont_g1", "holdout", allow_holdout=True)["n_trades"] == 1
    assert es.evaluate(ev, "H1_cont_g1", "dev")["n_trades"] == 0               # hold-out rows never leak into dev


def test_terciles_dev_only_cutoffs():
    n = 30
    q = np.array(["2022-Q4"] * n + ["2023-Q1"] * 3, dtype=object)
    per = np.array(["dev"] * n + ["holdout"] * 3, dtype=object)
    s = np.concatenate([np.arange(n, dtype=float), [-100.0, 15.0, 100.0]])
    ev = dict(quarter=q, period=per, surprise=s)
    cuts, last = es.assign_terciles(ev)
    assert list(cuts) == ["2022-Q4"] and ev["tercile"][-3:].tolist() == [0, 1, 2]     # hold-out used 2022-Q4 cut-offs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
