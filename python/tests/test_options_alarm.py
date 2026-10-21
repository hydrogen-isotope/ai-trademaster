import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from trademaster_ml import options_alarm as A

MINS = list(range(570, 960, 5))


def day(prices, iv=0.5, pc=0.6, vol=1000, first_up=True, low=None):
    o = list(prices)
    c = list(prices)
    if first_up:
        c[0] = o[0] + 1
    else:
        c[0] = o[0] - 1
    ivs = iv if isinstance(iv, list) else [iv] * len(MINS)
    return {"m": MINS, "o": o, "c": c, "l": low or o, "iv": ivs, "pc": [pc] * len(MINS), "vol": [vol] * len(MINS),
            "has_opt": True}


def test_sells_at_the_close_when_up():
    D = {"d1": day([100.0] + [101.0] * 77)}
    r = A.trade(D, ["d1"], 0)
    assert r[2] == "close" and abs(r[0] - (101 / 101 - 1 - 2 * A.COST)) < 1e-12     # bought at the 9:35 open (101)


def test_no_entry_when_first_bar_down():
    D = {"d1": day([100.0] * 78, first_up=False)}
    assert A.trade(D, ["d1"], 0) is None


def test_keeps_a_loser_then_sells_at_cost():
    p1 = [100.0, 100.0] + [99.0] * 76                                               # bought at 100, closes lower
    p2 = [99.0] * 10 + [100.5] * 68
    D = {"d1": day(p1), "d2": day(p2)}
    r = A.trade(D, ["d1", "d2"], 0)
    assert r[2] == "cost" and r[1] == 1 and abs(r[0] - (100.5 / 100 - 1 - 2 * A.COST)) < 1e-12


def test_carry_stop_at_minus_4pct():
    p1 = [100.0, 100.0] + [99.0] * 76
    p2 = [97.0] * 78
    lows = [97.0] * 5 + [95.0] * 73
    D = {"d1": day(p1), "d2": day(p2, low=lows)}
    r = A.trade(D, ["d1", "d2"], 0)
    assert r[2] == "stop" and abs(r[0] - (96 / 100 - 1 - 2 * A.COST)) < 1e-12


def test_iv_alarm_sells_at_next_open_even_at_a_loss():
    prices = [100.0, 100.0, 100.0, 99.0, 98.0] + [97.0] * 73
    ivs = [0.5, 0.5, 0.5, 0.5, 0.56] + [0.6] * 73                                 # base 0.5 at the 9:35 bar; +12% at bar 4
    D = {"d1": day(prices, iv=ivs)}
    r = A.trade(D, ["d1"], 0, alarm=(0.08, False))
    assert r[2] == "alarm" and abs(r[0] - (97 / 100 - 1 - 2 * A.COST)) < 1e-12     # sold at bar 5's open
    assert A.trade(D, ["d1"], 0, alarm=(0.20, False)) is None or A.trade(D, ["d1"], 0, alarm=(0.20, False))[2] != "alarm"


def test_iv_from_black_scholes_round_trip():
    from trademaster_ml.options import bs_price
    S, K, T, sig = 100.0, 102.0, 10 / 365, 0.45
    for call in (True, False):
        px = float(bs_price(np.array([S]), np.array([K]), np.array([T]), np.array([A.RATE]), np.array([sig]), np.array([call]))[0])
        iv = A._bs_iv([px], [S], [K], [T], [call])[0]
        assert abs(iv - sig) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
