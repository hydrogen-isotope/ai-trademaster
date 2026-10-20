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
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, tif="day", limit=None, take_profit=None, stop_loss=None:         sent.append((book, sym, side, qty, notional, limit) + ((take_profit, stop_loss) if take_profit or stop_loss else ())) or {"id": "x"}
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
    # target / stop: whole shares, target above and stop below the price, both needed to protect a holding
    for spec in ({"symbol": "MU", "side": "buy", "notional": 900, "take_profit": 950}, {"symbol": "MU", "side": "buy", "qty": 1.5, "take_profit": 950},
                 {"symbol": "MU", "side": "buy", "qty": 1, "take_profit": 890}, {"symbol": "MU", "side": "buy", "qty": 1, "stop_loss": 910},
                 {"symbol": "MU", "side": "sell", "qty": 1, "take_profit": 950}):
        try:
            T.manual_order(spec)
            raise AssertionError(f"accepted {spec}")
        except SystemExit:
            pass
    T.manual_order({"symbol": "MU", "side": "buy", "qty": 2, "take_profit": 930, "stop_loss": 880})
    T.manual_order({"symbol": "MU", "side": "sell", "qty": 1, "take_profit": 930, "stop_loss": 880})
    assert sent[-2:] == [("manual", "MU", "buy", 2.0, None, None, 930.0, 880.0), ("manual", "MU", "sell", 1.0, None, None, 930.0, 880.0)]
    tifs = []
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, tif="day", limit=None, take_profit=None, stop_loss=None: tifs.append(tif) or {"id": "x"}
    T.manual_order({"symbol": "MU", "side": "buy", "qty": 1, "take_profit": 930, "stop_loss": 880})
    T.manual_order({"symbol": "MU", "side": "buy", "qty": 1})
    assert tifs == ["gtc", "day"]


def test_bracket_legs_settle_once_each():
    import importlib
    importlib.reload(T)                                                          # undo the earlier tests' stubs
    T._req = lambda method, path, body=None, params=None: {
        "id": "P", "status": "filled", "side": "buy", "filled_qty": "10", "filled_avg_price": "200",
        "legs": [{"id": "TP", "status": "filled", "side": "sell", "type": "limit", "filled_qty": "10", "filled_avg_price": "204"},
                 {"id": "SL", "status": "canceled", "side": "sell", "type": "stop", "filled_qty": "0", "filled_avg_price": None}]}
    import json, os, tempfile
    d = tempfile.mkdtemp()
    T.LEDGER, T.MANUAL_CLOSED = os.path.join(d, "l.jsonl"), os.path.join(d, "c.jsonl")
    T._append = lambda path, rec: open(path, "a").write(json.dumps(rec) + "\n")
    T._append(T.LEDGER, {"book": "manual", "symbol": "NVDA", "side": "buy", "order_id": "P", "class": "bracket"})
    st = {"books": {"manual": {"cash": 0.0, "start": 0.0, "holdings": {}, "cost": {}}}}
    T._settle(st); T._settle(st)
    m = st["books"]["manual"]
    assert m["holdings"] == {} and abs(m["cash"] - 40.0) < 1e-9                  # bought 10 at 200, target sold 10 at 204
    closed = [json.loads(l) for l in open(T.MANUAL_CLOSED)]
    assert len(closed) == 1 and abs(closed[0]["pnl"] - 40.0) < 1e-9 and "P:done" in st["settled"]


def test_intraday_slice_top5_every_30_min_and_flat_at_1550():
    import importlib, json, os, tempfile, time as _t
    importlib.reload(T)
    d = tempfile.mkdtemp()
    T.FORWARD = d
    syms = [f"S{i}" for i in range(30)]
    T.FROZEN_MANIFEST = os.path.join(d, "m.json")
    json.dump({"symbols": syms, "is_etf": [False] * 30}, open(T.FROZEN_MANIFEST, "w"))
    os.makedirs(os.path.join(d, "live-2026-10-20"))
    now = _t.time()
    rec = {"t": int(now) - 400, "label": "10:30", "valid": [True] * 30, "b1": [float(i) for i in range(30)]}
    open(os.path.join(d, "live-2026-10-20", "predictions.jsonl"), "w").write(json.dumps(rec) + "\n")
    sent = []
    T._open_orders = lambda book: []
    T._latest_prices = lambda ss: {x: 100.0 for x in ss}
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, **k: sent.append((sym, side, round(notional or qty)))
    st = {"books": {"intraday": {"cash": 10000.0, "start": 10000.0, "holdings": {}}}}
    T.intraday_step(st, {"timestamp": "2026-10-20T10:36:00-04:00", "is_open": True})
    assert sorted(s_ for s_, side, _ in sent) == ["S25", "S26", "S27", "S28", "S29"] and all(n == 2000 for *_, n in sent[:4])
    sent.clear()
    T.intraday_step(st, {"timestamp": "2026-10-20T10:41:00-04:00", "is_open": True})       # same bar: no new trades
    assert sent == []
    st["books"]["intraday"]["holdings"] = {"S29": 20.0}
    st["books"]["intraday"]["cost"] = {"S29": 1900.0}                                     # bought at 95, now 100: a gain
    T.intraday_step(st, {"timestamp": "2026-10-20T15:51:00-04:00", "is_open": True})       # flat by the close
    assert sent == [("S29", "sell", 20)]


def test_intraday_keeps_losers_overnight_until_breakeven_stop_or_3_days():
    import importlib
    importlib.reload(T)
    sent, px = [], {"A": 98.0, "B": 101.0}
    T._open_orders = lambda book: []
    T._latest_prices = lambda ss: {x: px[x] for x in ss if x in px}
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, **k: sent.append((sym, side))
    b = {"cash": 0.0, "start": 10000.0, "holdings": {"A": 10.0, "B": 10.0}, "cost": {"A": 1000.0, "B": 1000.0}, "day": "2026-10-20"}
    st = {"books": {"intraday": b}}
    T.intraday_step(st, {"timestamp": "2026-10-20T15:51:00-04:00", "is_open": True})       # A is -2%: kept; B is +1%: sold
    assert sent == [("B", "sell")] and b["carry"] == {"A": "2026-10-20"}
    b["holdings"].pop("B"); sent.clear()
    T.intraday_step(st, {"timestamp": "2026-10-21T09:40:00-04:00", "is_open": True})       # next day, still -2%: kept
    assert ("A", "sell") not in sent
    px["A"] = 100.5; sent.clear()
    T.intraday_step(st, {"timestamp": "2026-10-21T10:05:00-04:00", "is_open": True})       # back above cost: sold
    assert ("A", "sell") in sent
    for price, day, expect in ((95.9, "2026-10-21", True), (98.0, "2026-10-23", True), (98.0, "2026-10-22", False)):
        px["A"] = price; sent.clear()
        b["holdings"], b["cost"], b["carry"], b["day"] = {"A": 10.0}, {"A": 1000.0}, {"A": "2026-10-20"}, day
        T.intraday_step(st, {"timestamp": f"{day}T10:05:00-04:00", "is_open": True})       # -4.1% stop; 3 sessions; 2 sessions
        assert (("A", "sell") in sent) == expect, (price, day)


def test_focus_buys_up_openers_once_and_keeps_losers():
    import importlib
    importlib.reload(T)
    sent, px = [], {"NVDA": 216.0, "MU": 900.0}
    T._open_orders = lambda book: []
    T._latest_prices = lambda ss: {x: px[x] for x in ss if x in px}
    T._first_bar = lambda sym, day: {"NVDA": (215.0, 216.0), "MU": (905.0, 900.0)}[sym]     # NVDA up, MU down
    T._order = lambda book, sym, side, qty=None, notional=None, log=print, **k: sent.append((sym, side, round(notional or qty)))
    b = {"cash": 10000.0, "start": 10000.0, "holdings": {}}
    st = {"books": {"focus": b}}
    T.focus_step(st, {"timestamp": "2026-10-21T09:30:00-04:00", "is_open": True})          # before the window: nothing
    assert sent == []
    T.focus_step(st, {"timestamp": "2026-10-21T09:36:00-04:00", "is_open": True})
    assert sent == [("NVDA", "buy", 5000)]
    sent.clear()
    T.focus_step(st, {"timestamp": "2026-10-21T09:41:00-04:00", "is_open": True})          # once per day
    assert sent == []
    b["holdings"], b["cost"] = {"NVDA": 23.0}, {"NVDA": 23.0 * 220.0}                      # bought at 220, now 216: a loss
    T.focus_step(st, {"timestamp": "2026-10-21T15:51:00-04:00", "is_open": True})
    assert sent == [] and b["carry"] == {"NVDA": "2026-10-21"}
    px["NVDA"] = 221.0
    T.focus_step(st, {"timestamp": "2026-10-22T09:31:00-04:00", "is_open": True})          # back above cost: sold
    assert ("NVDA", "sell", 23) in sent


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
