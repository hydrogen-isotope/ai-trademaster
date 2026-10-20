import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trademaster_ml import papertrade as T


def _fake_orders():
    sent = []
    T._order = lambda book, sym, side, qty=None, notional=None, log=print: sent.append((book, sym, side, qty, notional))
    return sent


def test_rank_book_buys_equal_weight_and_sells_dropped():
    sent = _fake_orders()
    b = {"cash": 1000.0, "holdings": {"OLD": 2.0}}
    T._rebalance("rank", b, ["A", "B"], {"A": 10.0, "B": 20.0, "OLD": 50.0})
    assert ("rank", "OLD", "sell", 2.0, None) in sent
    buys = {s: n for _, s, side, _, n in sent if side == "buy"}
    assert set(buys) == {"A", "B"} and abs(buys["A"] - 550) < 1 and abs(buys["B"] - 539) < 1   # (1000 + 100) / 2 each; the last buy keeps 2% of the cash


def test_ai_book_caps_each_stock_at_one_twentieth():
    sent = _fake_orders()
    b = {"cash": 10000.0, "holdings": {}}
    T._rebalance("ai", b, ["A", "B"], {"A": 10.0, "B": 10.0})
    assert [round(n) for *_, n in sent] == [500, 500]


def test_refuses_non_paper_keys():
    real = T.alpaca.credentials
    T.alpaca.credentials = lambda: ("AKLIVE", "x", None)
    try:
        T._creds()
        raise AssertionError("a live key was accepted")
    except SystemExit:
        pass
    finally:
        T.alpaca.credentials = real


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
