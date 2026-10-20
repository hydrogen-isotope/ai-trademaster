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


def _night(held, ts, is_open, last_buy=None):
    sent = []
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, tif="day": sent.append((side, qty, tif))
    T._open_orders = lambda book: []
    T._latest_prices = lambda syms: {"QQQ": 700.0}
    st = {"books": {"night": {"cash": 10000.0 if not held else 0.0, "holdings": {"QQQ": held} if held else {}, "last_buy": last_buy}}}
    T.night_step(st, {"timestamp": ts, "is_open": is_open})
    return sent, st


def test_night_buys_whole_shares_in_the_closing_auction_before_1550():
    sent, st = _night(0, "2026-10-21T15:30:00-04:00", True)
    assert sent == [("buy", 14.0, "cls")] and st["books"]["night"]["last_buy"] == "2026-10-21"
    assert _night(0, "2026-10-21T15:55:00-04:00", True)[0] == []                         # too late for the closing auction
    assert _night(0, "2026-10-21T15:30:00-04:00", True, last_buy="2026-10-21")[0] == []  # once per day


def test_night_sells_at_the_open_or_right_away_if_missed():
    assert _night(14.0, "2026-10-21T20:00:00-04:00", False)[0] == [("sell", 14.0, "opg")]
    assert _night(14.0, "2026-10-22T08:00:00-04:00", False)[0] == [("sell", 14.0, "opg")]
    assert _night(14.0, "2026-10-21T17:00:00-04:00", False)[0] == []                     # opening orders open at 19:00
    assert _night(14.0, "2026-10-22T10:15:00-04:00", True)[0] == [("sell", 14.0, "day")]


def test_manual_orders_refuse_shorts_and_bad_input():
    st = {"books": {"manual": {"cash": 0.0, "start": 0.0, "holdings": {"MU": 2.0}, "cost": {"MU": 1800.0}}}}
    T._state = lambda c, log=print: st
    T._settle = lambda s, log=print: None
    T._save = lambda path, obj: None
    T._latest_prices = lambda syms: {s: 900.0 for s in syms}
    T.status = lambda *a, **k: None
    T._append = lambda path, rec: None
    T.context = lambda sym: {}
    sent = []
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, tif="day", limit=None: sent.append((book, sym, side, qty, notional, limit)) or {"id": "x"}
    T.time.sleep = lambda s: None
    for spec in ({"symbol": "MU", "side": "sell", "qty": 3}, {"symbol": "NVDA", "side": "sell", "qty": 1}, {"symbol": "MU", "side": "buy"},
                 {"symbol": "MU", "side": "buy", "notional": 100, "limit": 850}, {"symbol": "MU X", "side": "buy", "qty": 1}):
        try:
            T.manual_order(spec)
            raise AssertionError(f"accepted {spec}")
        except SystemExit:
            pass
    assert sent == []
    T.manual_order({"symbol": "mu", "side": "sell", "notional": 900})            # $900 of a $900 stock = 1 share of the 2 held
    T.manual_order({"symbol": "MU", "side": "buy", "qty": 1, "limit": 850})
    assert sent == [("manual", "MU", "sell", 1.0, None, None), ("manual", "MU", "buy", 1.0, None, 850.0)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
