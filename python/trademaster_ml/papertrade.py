"""
Automatic PAPER trading on Alpaca's paper account (simulated money only), to see whether the signals would make money.

Safety: orders go only to https://paper-api.alpaca.markets, and only with a paper key (key id starting "PK"); any other
key or URL is refused. Long only. US stocks only (Alpaca has no SGX). Runs only while the app is open (the app calls
`run` every few minutes; nothing happens unless it is switched on in the Auto-trade page).

Two books, each with its own share of the paper account, tracked in our own ledger (Alpaca holds one combined account):
  ai    holds the stocks whose latest 5-day Qwen + models research signal is BUY (latest close's evaluations), up to
        MAX_AI by chance of a rise, equal weight; rebalanced once per trading day.
  rank  holds the top decile of the v6 cross-stock ranking, equal weight; rebalanced on the first run of each week.
  night holds QQQ only overnight: a market-on-close buy (sent before 15:50 New York time) and a market-on-open sell
        (sent between 19:00 and 09:28 New York time). If the app was not open in time, the open sell is replaced by an
        ordinary market sell at the next run during the session. Compared with simply holding QQQ.

Files (private/reports/paper/): control.json (switches, set by the app), book.json (holdings and cash per book),
ledger.jsonl (every order and fill), equity.jsonl (daily value per book and QQQ), status.json (what the page shows).

Manual trades (the "manual" book): orders the user places from the Auto-trade page, long only, sized against the
paper account's buying power (no fixed share). Each order is saved with what the models said at that moment
(manual.jsonl), and each closed position with its result (manual_closed.jsonl), so the user's habits can later be
compared with, and learned next to, the models.

Usage: python -m trademaster_ml.papertrade run | status | close --book ai|rank|night|all
       python -m trademaster_ml.papertrade order --json '{"symbol": "MU", "side": "buy", "notional": 1000}'
       python -m trademaster_ml.papertrade cancel --id <order id>
"""
import argparse, json, os, time, urllib.error, urllib.request
from datetime import datetime, timezone

from . import alpaca, holidays, options_watch

PAPER = "https://paper-api.alpaca.markets"
DIR = os.path.join("private", "reports", "paper")
CONTROL, BOOK, LEDGER, EQUITY, STATUS = (os.path.join(DIR, f) for f in ("control.json", "book.json", "ledger.jsonl", "equity.jsonl", "status.json"))
BOOKS = {"ai": "AI signals (Qwen + models, 5-day BUY)", "rank": "Top-ranked stocks (LightGBM top 20, daily)",
         "night": "Overnight QQQ (buy at the close, sell at the next open)",
         "intraday": "Intraday 30-minute model (top 5, every 30 min; keeps losers until break-even)",
         "focus": "NVDA & MU opening-range (buy if the first 5 minutes close up; keeps losers until break-even; options recorded)"}
SPLIT = {"ai": 0.40, "rank": 0.40, "night": 0.10, "intraday": 0.10, "focus": 0.10}   # focus is on margin (1.1x)
FOCUS_SYMS, FOCUS_ENTRY = ("NVDA", "MU"), (9 * 60 + 35, 9 * 60 + 50)
OPTIONS_ALARM = False            # off: it lost 0.15-0.27% per trade in the 2025-26 test (docs/OPTIONS-ALARM-PROTOCOL.md);
                                 # readings are still recorded for learning
INTRA_TOP, INTRA_EVERY_S, INTRA_FLAT_MIN = 5, 1800, 15 * 60 + 50
CARRY_STOP, CARRY_MAX_DAYS = -0.04, 3        # a position kept to avoid selling at a loss: sold at -4% or after 3 sessions
FORWARD = os.path.join("private", "data", "forward")
FROZEN_MANIFEST = os.path.join("private", "models", "forward", "frozen-2026-10-17", "manifest.json")
LABELS = {**BOOKS, "manual": "Your manual trades"}
MANUAL, MANUAL_CLOSED = os.path.join(DIR, "manual.jsonl"), os.path.join(DIR, "manual_closed.jsonl")
OUTCOMES = os.path.join(DIR, "manual_outcomes.json")         # what happened after each decision (learning labels)
FINAL_STATES = ("filled", "canceled", "expired", "rejected", "done_for_day")
OPEN_STATES = ("new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding")
MAX_AI = 20
MIN_TRADE = 25.0                 # dollars: smaller differences from the target are left alone
DRIFT = 0.20                     # rebalance a holding only when it is more than 20% away from its target value


# ---------------------------------------------------------------- the paper account (refuses anything that is not paper)

def _creds():
    kid, sec, _ = alpaca.credentials()
    if not str(kid).startswith("PK"):
        raise SystemExit("refused: the Alpaca key is not a paper-trading key (paper keys start with PK)")
    return kid, sec


def _req(method, path, body=None, params=None):
    kid, sec = _creds()
    url = PAPER + path + ("?" + "&".join(f"{k}={v}" for k, v in params.items()) if params else "")
    assert url.startswith(PAPER + "/"), "paper endpoint only"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
                                                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Alpaca paper {method} {path}: HTTP {e.code} {e.read()[:200]!r}") from None


def _load(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def _save(path, obj):
    os.makedirs(DIR, exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def _append(path, rec):
    os.makedirs(DIR, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def control():
    c = _load(CONTROL, {})
    c.setdefault("enabled", False)
    c.setdefault("books", {k: True for k in BOOKS})
    c.setdefault("split", dict(SPLIT))
    for k in BOOKS:
        c["books"].setdefault(k, True)
        c["split"].setdefault(k, SPLIT[k])
    return c


def _today_ny():
    return datetime.now(alpaca.NY).date().isoformat() if hasattr(alpaca, "NY") else datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------- targets

def _latest_prices(symbols):
    kid, sec = _creds()
    snaps = {}
    for i in range(0, len(symbols), 100):
        group = symbols[i:i + 100]
        d = alpaca._get("/v2/stocks/snapshots", {"symbols": ",".join(group), "feed": "iex"}, (kid, sec, None))
        for s, x in (d or {}).items():
            p = (x.get("latestTrade") or {}).get("p") or (x.get("dailyBar") or {}).get("c") or (x.get("prevDailyBar") or {}).get("c")
            if p:
                snaps[s] = float(p)
    return snaps


def targets_ai():
    from . import stock_eval
    asof = stock_eval.latest_asof("nasdaq")
    idx = _load(os.path.join(stock_eval.EVALS, "nasdaq", f"index_{asof}.json"), {})
    buys = [(s, e["h"]["5"]["p_up"]) for s, e in idx.items() if (e.get("h") or {}).get("5", {}).get("signal") == "buy"]
    buys.sort(key=lambda x: -x[1])
    return asof, [s for s, _ in buys[:MAX_AI]], len(idx)


RANK_TOP = 20


def targets_rank():
    """The LightGBM ranking's top RANK_TOP (rankings_g.json, refit after each close); falls back to the ridge top decile."""
    g = _load(os.path.join("private", "reports", "v6", "rankings_g.json"), {})
    if g.get("ranking"):
        return g.get("asof"), [r["symbol"] for r in g["ranking"][:RANK_TOP]]
    rk = _load(os.path.join("private", "reports", "v6", "rankings.json"), {})
    return rk.get("asof"), [r["symbol"] for r in rk.get("rankings", []) if r.get("decile") == 10]


# ---------------------------------------------------------------- orders

def _order(book, sym, side, qty=None, notional=None, log=print, tif="day", limit=None, take_profit=None, stop_loss=None):
    body = {"symbol": sym, "side": side, "type": "limit" if limit else "market", "time_in_force": tif,
            "client_order_id": f"tm-{book}-{int(time.time() * 1000)}-{sym}"[:48]}
    exits = None
    if take_profit or stop_loss:                    # buy: bracket (entry + target + stop); sell: one-cancels-other exit
        exits = "bracket" if side == "buy" else "oco"
        body["order_class"] = exits
        if take_profit:
            body["take_profit"] = {"limit_price": f"{take_profit:.2f}"}
        if stop_loss:
            body["stop_loss"] = {"stop_price": f"{stop_loss:.2f}"}
        if exits == "oco":                          # the OCO parent is the take-profit limit itself
            body["type"], body["limit_price"] = "limit", f"{take_profit:.2f}"
    if limit:
        body["limit_price"] = f"{limit:.2f}"
    if notional is not None:
        body["notional"] = f"{notional:.2f}"
    else:
        body["qty"] = f"{qty:.9f}".rstrip("0").rstrip(".")
    o = _req("POST", "/v2/orders", body)
    _append(LEDGER, {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "book": book, "symbol": sym, "side": side,
                     "qty": qty, "notional": notional, "tif": tif, "order_id": o.get("id"), "status": o.get("status"),
                     **({"class": exits, "take_profit": take_profit, "stop_loss": stop_loss} if exits else {})})
    log(f"[paper] {book}: {side} {sym} " + (f"${notional:.2f}" if notional is not None else f"{qty} sh") + (f" ({tif})" if tif != "day" else ""))
    return o


def _settle(book_state, log=print):
    """Apply fills of our orders to the books (filled quantity and average price), once per order."""
    seen = set(book_state.get("settled", []))
    pending = [json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []
    for rec in pending:
        oid = rec.get("order_id")
        if not oid or oid in seen or f"{oid}:done" in seen:
            continue
        o = _req("GET", f"/v2/orders/{oid}", params={"nested": "true"})
        parts = [o] + list(o.get("legs") or [])                   # bracket / OCO: the target and stop legs fill separately
        if all(x.get("status") in FINAL_STATES for x in parts):
            seen.add(f"{oid}:done")
        for x in parts:
            if x.get("id") in seen or x.get("status") not in FINAL_STATES:
                continue
            seen.add(x["id"])
            _apply_fill(book_state, rec, x, log)
    book_state["settled"] = sorted(seen)


def _apply_fill(book_state, rec, o, log=print):
    oid = o.get("id")
    q, px = float(o.get("filled_qty") or 0), float(o.get("filled_avg_price") or 0)
    b = book_state["books"].setdefault(rec["book"], {"cash": 0.0, "start": 0.0, "holdings": {}})
    sign = 1 if o.get("side", rec["side"]) == "buy" else -1
    if q > 0:                                                       # cost basis for every book
        cost = b.setdefault("cost", {})
        held = b["holdings"].get(rec["symbol"], 0.0)
        if sign > 0:
            cost[rec["symbol"]] = cost.get(rec["symbol"], 0.0) + q * px
        elif held > 0:
            avg = cost.get(rec["symbol"], 0.0) / held
            cost[rec["symbol"]] = cost.get(rec["symbol"], 0.0) - avg * q
            if rec["book"] == "manual":
                _append(MANUAL_CLOSED, {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": rec["symbol"], "qty": q,
                                        "avg_entry": avg, "exit": px, "pnl": (px - avg) * q, "ret": px / avg - 1 if avg else None,
                                        "order_id": oid})
    b["holdings"][rec["symbol"]] = round(b["holdings"].get(rec["symbol"], 0.0) + sign * q, 9)
    if abs(b["holdings"][rec["symbol"]]) < 1e-9:
        b["holdings"].pop(rec["symbol"])
        b.get("cost", {}).pop(rec["symbol"], None)
    b["cash"] -= sign * q * px


def _rebalance(name, b, want, prices, log=print):
    value = b["cash"] + sum(q * prices.get(s, 0) for s, q in b["holdings"].items())
    slots = max(len(want), MAX_AI) if name == "ai" else len(want)    # ai: at most 5% of the book per stock, rest in cash
    tgt = value / slots if want else 0.0
    n = 0
    for s, q in list(b["holdings"].items()):                       # sell what is no longer wanted, trim overweights
        cur = q * prices.get(s, 0)
        if s not in want:
            _order(name, s, "sell", qty=q, log=log); n += 1
        elif cur - tgt > max(MIN_TRADE, DRIFT * tgt) and prices.get(s):
            _order(name, s, "sell", qty=min(q, (cur - tgt) / prices[s]), log=log); n += 1
    cash = b["cash"] + sum(b["holdings"].get(s, 0) * prices.get(s, 0) for s in b["holdings"] if s not in want)
    for s in want:                                                 # buy new names and top up underweights
        cur = b["holdings"].get(s, 0) * prices.get(s, 0)
        need = tgt - cur
        if need > max(MIN_TRADE, DRIFT * tgt) and prices.get(s) and cash > MIN_TRADE:
            amt = min(need, cash * 0.98)
            _order(name, s, "buy", notional=round(amt, 2), log=log); n += 1
            cash -= amt
    return n


def _open_orders(book):
    """Our not-yet-final orders for a book (from the ledger, checked with Alpaca)."""
    out = []
    for rec in ([json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []):
        if rec.get("book") == book and rec.get("order_id"):
            o = _req("GET", f"/v2/orders/{rec['order_id']}", params={"nested": "true"})
            for x in [o] + list(o.get("legs") or []):
                if x.get("status") in OPEN_STATES:
                    out.append({**x, "role": "entry" if x is o and rec.get("class") != "oco" else
                                ("target" if x.get("type") == "limit" else "stop" if x.get("type") in ("stop", "stop_limit") else "")})
    return out


def night_step(st, clock, log=print):
    """The overnight QQQ book (see module doc). Whole shares (auction orders cannot be fractional)."""
    b = st["books"]["night"]
    now = datetime.fromisoformat(str(clock["timestamp"])).astimezone(alpaca.NY)
    mins = now.hour * 60 + now.minute
    held = b["holdings"].get("QQQ", 0.0)
    pend = _open_orders("night")
    if any(o.get("side") == "buy" for o in pend) or any(o.get("side") == "sell" for o in pend):
        return 0
    session = clock.get("is_open")
    if held > 0:
        if session and mins >= 9 * 60 + 30:                                     # the open sell was missed: sell now
            _order("night", "QQQ", "sell", qty=held, log=log)
            return 1
        if not session and (mins >= 19 * 60 or mins < 9 * 60 + 28):            # market-on-open sell for the next open
            _order("night", "QQQ", "sell", qty=held, log=log, tif="opg")
            return 1
        return 0
    if session and mins < 15 * 60 + 50 and b.get("last_buy") != now.date().isoformat():
        px = _latest_prices(["QQQ"]).get("QQQ")
        n = int(b["cash"] * 0.98 // px) if px else 0
        if n >= 1:
            _order("night", "QQQ", "buy", qty=float(n), log=log, tif="cls")
            b["last_buy"] = now.date().isoformat()
            return 1
    return 0


def _state(c, log=print):
    """The books; created on first use by splitting the paper account between the automatic books (manual has none)."""
    st = _load(BOOK, None)
    if st is None:
        eq = float(_req("GET", "/v2/account")["equity"])
        st = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "start_equity": eq, "settled": [],
              "books": {k: {"cash": eq * c["split"][k], "start": eq * c["split"][k], "holdings": {}, "last_rebalance": None} for k in BOOKS}}
        _save(BOOK, st)
        log(f"[paper] started: ${eq:,.2f} paper equity split {c['split']}")
    st["books"].setdefault("manual", {"cash": 0.0, "start": 0.0, "holdings": {}, "cost": {}})
    for k in BOOKS:
        if k not in st["books"]:
            amt = st.get("start_equity", 0.0) * c["split"].get(k, 0.0)
            st["books"][k] = {"cash": amt, "start": amt, "holdings": {}, "last_rebalance": None}
    return st


def _rebuild_cost(st, book):
    """Cost basis from the book's own filled orders (for holdings bought before cost tracking covered every book)."""
    b = st["books"][book]
    cost, held = {}, {}
    for line in (open(LEDGER) if os.path.exists(LEDGER) else []):
        rec = json.loads(line)
        if rec.get("book") != book or not rec.get("order_id"):
            continue
        o = _req("GET", f"/v2/orders/{rec['order_id']}")
        q, px = float(o.get("filled_qty") or 0), float(o.get("filled_avg_price") or 0)
        if q <= 0:
            continue
        sym = rec["symbol"]
        if rec["side"] == "buy":
            cost[sym], held[sym] = cost.get(sym, 0.0) + q * px, held.get(sym, 0.0) + q
        elif held.get(sym):
            cost[sym] -= cost[sym] / held[sym] * min(q, held[sym])
            held[sym] -= min(q, held[sym])
    b["cost"] = {k: v for k, v in cost.items() if k in b["holdings"]}


def _sessions_since(day, today):
    from . import holidays
    n, d = 0, day
    while d < today:
        d = holidays.next_sessions(d, 1)[0]
        n += 1
    return n


def _first_bar(sym, day):
    """The 09:30-09:35 New York 5-minute bar of `day` (IEX feed, real time on the free plan): (open, close) or None."""
    t0 = int(datetime.fromisoformat(f"{day}T09:30:00").replace(tzinfo=alpaca.NY).timestamp())
    kid, sec = _creds()
    bars = alpaca.get_bars([sym], "5Min", t0, t0 + 299, feed="iex", creds=(kid, sec, None), pause=0.1).get(sym, [])
    b0 = next((b for b in bars if b[0] == t0), None)
    return (b0[1], b0[4]) if b0 else None


def focus_step(st, clock, log=print):
    """The NVDA & MU slice: at 09:35-09:50 New York, buy each of FOCUS_SYMS (half the slice each) whose first 5-minute
    bar closed above its open; exits as the intraday slice: at 15:50 sell what is at or above cost, keep losers until
    back at cost, CARRY_STOP below it, or CARRY_MAX_DAYS sessions; never add to a kept loser."""
    b = st["books"]["focus"]
    now = datetime.fromisoformat(str(clock["timestamp"])).astimezone(alpaca.NY)
    today, mins = now.date().isoformat(), now.hour * 60 + now.minute
    if not clock.get("is_open") or _open_orders("focus"):
        return 0
    if b["holdings"] and set(b["holdings"]) - set(b.get("cost", {})):
        _rebuild_cost(st, "focus")
    carry = b.setdefault("carry", {})
    for sym in list(carry):
        if sym not in b["holdings"]:
            carry.pop(sym)
    prices = _latest_prices(sorted(set(FOCUS_SYMS) | set(b["holdings"])))

    def ret(sym):
        c, q, px = b.get("cost", {}).get(sym), b["holdings"].get(sym), prices.get(sym)
        return px / (c / q) - 1 if c and q and px else None

    def may_sell(sym):
        r = ret(sym)
        if r is None or r >= 0:
            return True
        return sym in carry and (r <= CARRY_STOP or _sessions_since(carry[sym], today) >= CARRY_MAX_DAYS)

    n, sold = 0, set()
    for sym in list(carry):                                        # kept losers: sell when a rule says so
        if may_sell(sym):
            _order("focus", sym, "sell", qty=b["holdings"][sym], log=log)
            sold.add(sym)
            n += 1
    ent = b.setdefault("opt_entry", {})
    for sym in list(ent):
        if sym not in b["holdings"]:
            ent.pop(sym)
    for sym, q in list(b["holdings"].items()):                     # options watch: the first reading of the day is the base
        if sym in sold or not prices.get(sym):
            continue
        try:
            r = options_watch.reading(sym, prices[sym])
        except Exception as e:  # noqa: BLE001 - the watch must not stop the slice
            log(f"[paper] options reading {sym} failed: {type(e).__name__}: {str(e)[:100]}")
            continue
        base = ent.get(sym)
        if not base or base.get("day") != today:
            if r and r.get("iv"):
                ent[sym] = {**r, "day": today}
            options_watch.record(sym, prices[sym], r)
            continue
        why = options_watch.alarm(base, r)
        options_watch.record(sym, prices[sym], r, why)
        if why and OPTIONS_ALARM:                                                    # the alarm overrides the keep-losers rule
            _order("focus", sym, "sell", qty=q, log=log)
            carry.pop(sym, None)
            sold.add(sym)
            b["note"] = f"sold {sym}: {why}"
            n += 1
    if mins >= INTRA_FLAT_MIN or (b.get("day") and b["day"] != today and not b.get("entered_today")):
        kept = []
        for sym, q in list(b["holdings"].items()):                 # the close (or leftovers from a day the app missed)
            if sym in carry or sym in sold:
                continue
            if may_sell(sym):
                _order("focus", sym, "sell", qty=q, log=log)
                n += 1
            else:
                carry[sym] = today
                kept.append(sym)
        if mins >= INTRA_FLAT_MIN:
            b["note"] = "sold for the close" + (f"; kept overnight at a loss: {', '.join(kept)}" if kept else "")
            return n
    if b.get("entered") == today or not (FOCUS_ENTRY[0] <= mins < FOCUS_ENTRY[1]):
        if b.get("entered") != today and mins >= FOCUS_ENTRY[1]:
            b["note"] = "no entry today (the app was not open at 9:35-9:50 New York)"
        return n
    b["entered"], b["day"] = today, today
    value = b["cash"] + sum(q * prices.get(s_, 0) for s_, q in b["holdings"].items())
    per = value / len(FOCUS_SYMS)
    took = []
    for sym in FOCUS_SYMS:
        fb = _first_bar(sym, today)
        if not fb or fb[1] <= fb[0] or sym in carry or b["holdings"].get(sym):
            continue
        amt = min(per, b["cash"] * 0.98)
        if amt > MIN_TRADE:
            _order("focus", sym, "buy", notional=round(amt, 2), log=log)
            took.append(sym)
            n += 1
    b["note"] = f"bought {', '.join(took)} at the open" if took else "first 5 minutes were down: no entry today"
    b["last_rebalance"], b["signals_asof"] = today, "09:30-09:35 bar"
    return n


def intraday_step(st, clock, log=print):
    """The intraday slice: every 30 minutes hold the frozen 30-minute model's (B1) top INTRA_TOP stocks, long only, equal
    weight, from the live forward test's latest prediction (private/data/forward/live-<date>/predictions.jsonl, under
    10 minutes old). It never sells at a loss just to follow the schedule: at 15:50 New York it sells what is at or above
    cost and keeps the rest overnight; the 30-minute reshuffle also keeps losers. A kept ("carried") position is sold
    once it is back at or above cost, at CARRY_STOP below cost, or after CARRY_MAX_DAYS sessions. Carried stocks are
    never added to."""
    b = st["books"]["intraday"]
    now = datetime.fromisoformat(str(clock["timestamp"])).astimezone(alpaca.NY)
    today, mins = now.date().isoformat(), now.hour * 60 + now.minute
    if _open_orders("intraday"):
        return 0
    if b["holdings"] and set(b["holdings"]) - set(b.get("cost", {})):
        _rebuild_cost(st, "intraday")
    carry = b.setdefault("carry", {})
    for sym in list(carry):
        if sym not in b["holdings"]:
            carry.pop(sym)
    prices = _latest_prices(sorted(b["holdings"])) if b["holdings"] else {}

    def pnl(sym):
        c, q, px = b.get("cost", {}).get(sym), b["holdings"].get(sym), prices.get(sym)
        return px / (c / q) - 1 if c and q and px else None

    def may_sell(sym):                                              # not at a loss, or a carry limit was reached
        r = pnl(sym)
        if r is None or r >= 0:
            return True
        return sym in carry and (r <= CARRY_STOP or _sessions_since(carry[sym], today) >= CARRY_MAX_DAYS)

    n = 0
    if clock.get("is_open"):
        for sym in list(carry):                                    # carried positions: sell when a rule says so
            if sym in b["holdings"] and may_sell(sym):
                _order("intraday", sym, "sell", qty=b["holdings"][sym], log=log)
                n += 1
    stale = b.get("day") and b["day"] != today
    if b["holdings"] and clock.get("is_open") and (stale or mins >= INTRA_FLAT_MIN):
        kept = []
        for sym, q in list(b["holdings"].items()):
            if sym in carry:
                continue
            if may_sell(sym):
                _order("intraday", sym, "sell", qty=q, log=log)
                n += 1
            else:
                carry[sym] = today
                kept.append(sym)
        b["day"] = today
        b["note"] = ("flat for the close" if not kept else f"kept overnight at a loss: {', '.join(kept)}") if not stale \
            else "sold yesterday's leftovers"
        return n
    if mins >= INTRA_FLAT_MIN or not clock.get("is_open"):
        return n
    path = os.path.join(FORWARD, f"live-{today}", "predictions.jsonl")
    if not os.path.exists(path):
        b["note"] = "waiting for the live 30-minute model"
        return 0
    last = None
    with open(path) as f:
        for line in f:
            last = line
    rec = json.loads(last) if last else None
    if not rec or time.time() - rec["t"] > 600 + 300:              # the bar starts at t, completes at t + 300
        b["note"] = "the live model's latest prediction is too old"
        return 0
    if b.get("last_t") and rec["t"] - b["last_t"] < INTRA_EVERY_S:
        return 0
    m = _load(FROZEN_MANIFEST, {})
    syms, etf = m.get("symbols", []), m.get("is_etf", [])
    cand = [(rec["b1"][i], s_) for i, s_ in enumerate(syms) if rec["valid"][i] and not (etf and etf[i])]
    if len(cand) < 20:
        b["note"] = f"model warming up ({len(cand)} stocks valid)"
        return 0
    want = [s_ for _, s_ in sorted(cand, reverse=True)[:INTRA_TOP]]
    prices.update(_latest_prices(sorted(set(want) - set(prices))))
    for sym in b["holdings"]:                                       # dropped from the top 5 at a loss: keep (carry)
        if sym not in want and sym not in carry and not may_sell(sym):
            carry[sym] = today
    free = {"cash": b["cash"], "holdings": {k: v for k, v in b["holdings"].items() if k not in carry}}
    n += _rebalance("intraday", free, [s_ for s_ in want if s_ in prices and s_ not in carry], prices, log)
    b["last_t"], b["day"], b["last_rebalance"], b["signals_asof"] = rec["t"], today, today, f"{rec['label']} ET bar"
    b["note"] = f"holding the model's top {INTRA_TOP} from the {rec['label']} ET bar" + \
        (f"; kept at a loss: {', '.join(sorted(carry))}" if carry else "")
    return n


def run(a=None, log=print):
    c = control()
    st = _load(BOOK, None)
    if st:
        _settle(st, log)                                           # manual fills are settled even while auto-trade is off
        _save(BOOK, st)
    if not c["enabled"]:
        return {"done": "auto-trade is off"}
    clock = _req("GET", "/v2/clock")
    if c.get("reset") in ("pending", "submitted"):
        waiting = _reset_step(c, clock, log)
        if waiting:
            return waiting
    st = _state(c, log)
    _settle(st, log)
    night = night_step(st, clock, log) if c["books"].get("night") and "night" in st["books"] else 0
    intra = intraday_step(st, clock, log) if c["books"].get("intraday") and "intraday" in st["books"] else 0
    intra += focus_step(st, clock, log) if c["books"].get("focus") and "focus" in st["books"] else 0
    if not clock.get("is_open"):
        _save(BOOK, st)
        status(st, note="market closed: next open " + str(clock.get("next_open")))
        return {"done": "market closed", "night_orders": night, "intraday_orders": intra}
    day = str(clock["timestamp"])[:10]
    week = datetime.fromisoformat(day).isocalendar()[:2]
    done = {}
    for name in BOOKS:
        b = st["books"][name]
        if name in ("night", "intraday", "focus") or not c["books"].get(name):
            continue
        last = b.get("last_rebalance")
        if name == "ai" and last == day:
            continue
        if name == "rank" and last == day:                         # daily (was weekly with the ridge top decile)
            continue
        asof, want = (targets_ai()[:2] if name == "ai" else targets_rank())
        prices = _latest_prices(sorted(set(want) | set(b["holdings"])))
        done[name] = _rebalance(name, b, [s for s in want if s in prices], prices, log)
        b["last_rebalance"], b["signals_asof"] = day, asof
    _save(BOOK, st)
    time.sleep(3)
    _settle(st, log)
    _save(BOOK, st)
    status(st)
    return {"orders": done, "night_orders": night, "intraday_orders": intra}


def _reset_step(c, clock, log=print):
    """Account reset requested from control.json ("reset": "pending"): at the open, cancel every order and sell every
    position (manual ones too); once all are filled, move the old books, ledger and equity history aside to
    private/reports/paper/archive-<time>/ and start fresh books from the current equity with the current split."""
    if not clock.get("is_open"):
        return {"done": "reset waits for the open: " + str(clock.get("next_open"))}
    if c["reset"] == "pending":
        _req("DELETE", "/v2/positions", params={"cancel_orders": "true"})
        c["reset"] = "submitted"
        _save(CONTROL, c)
        log("[paper] reset: cancelled all orders and sent sells for every position")
    for _ in range(12):                                            # market sells at the open fill in seconds
        if not (_req("GET", "/v2/positions") or _req("GET", "/v2/orders", params={"status": "open"})):
            break
        time.sleep(10)
    else:
        return {"done": "reset: waiting for the sells to fill"}
    stamp = datetime.now(alpaca.NY).strftime("%Y-%m-%d_%H%M")
    arch = os.path.join(DIR, f"archive-{stamp}")
    os.makedirs(arch, exist_ok=True)
    for f in (BOOK, LEDGER, EQUITY, STATUS):
        if os.path.exists(f):
            os.replace(f, os.path.join(arch, os.path.basename(f)))
    c["reset"], c["reset_at"] = "done", stamp
    _save(CONTROL, c)
    st = _state(c, log)
    _save(BOOK, st)
    log(f"[paper] reset done: old books moved to {arch}")
    return None                                                    # the normal run continues with the fresh books


def close(book, log=print):
    st = _load(BOOK, None)
    if not st:
        return
    _settle(st, log)
    for name in (BOOKS if book == "all" else [book]):
        for s, q in list(st["books"][name]["holdings"].items()):
            _order(name, s, "sell", qty=q, log=log)
    _save(BOOK, st)
    time.sleep(3)
    _settle(st, log)
    _save(BOOK, st)
    status(st)


def status(st=None, note=None, log=print):
    """Account, books and positions for the Auto-trade page; one equity line per day."""
    if st is None:
        st = _load(BOOK, None)
        if st:                                                     # record fills (manual limit orders too) on every refresh
            _settle(st, log)
            _save(BOOK, st)
    c = control()
    acct = _req("GET", "/v2/account")
    out = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "control": c, "note": note, "paper": True,
           "account": {k: acct.get(k) for k in ("equity", "cash", "buying_power", "status")}, "books": {}}
    if st:
        held = sorted({s for b in st["books"].values() for s in b["holdings"]} | {"QQQ"})
        prices = _latest_prices(held)
        for name, b in st["books"].items():
            pos = [{"symbol": s, "qty": q, "price": prices.get(s), "value": q * prices.get(s, 0)} for s, q in sorted(b["holdings"].items())]
            val = b["cash"] + sum(p["value"] for p in pos)
            if name == "manual":
                for p_ in pos:
                    cst = b.get("cost", {}).get(p_["symbol"])
                    p_["avg_entry"] = cst / p_["qty"] if cst and p_["qty"] else None
                    p_["pnl"] = p_["value"] - cst if cst is not None else None
            out["books"][name] = {"label": LABELS.get(name, name), "enabled": c["books"].get(name, name == "manual"), "value": val, "start": b["start"],
                                  "return": val / b["start"] - 1 if b["start"] else None, "cash": b["cash"], "positions": pos,
                                  "last_rebalance": b.get("last_rebalance"), "signals_asof": b.get("signals_asof"), "note": b.get("note")}
        out["started"], out["qqq"] = st["started"], prices.get("QQQ")
        hist = [json.loads(l) for l in open(EQUITY)] if os.path.exists(EQUITY) else []
        today = datetime.now(alpaca.NY).date().isoformat() if hasattr(alpaca, "NY") else out["time"][:10]
        if prices.get("QQQ") and (not hist or hist[-1]["date"] != today):
            _append(EQUITY, {"date": today, "qqq": prices["QQQ"], **{n: out["books"][n]["value"] for n in out["books"]}})
        elif hist and hist[-1]["date"] == today and prices.get("QQQ"):
            hist[-1] = {"date": today, "qqq": prices["QQQ"], **{n: out["books"][n]["value"] for n in out["books"]}}
            open(EQUITY, "w").write("".join(json.dumps(h) + "\n" for h in hist))
        out["equity_history"] = [json.loads(l) for l in open(EQUITY)] if os.path.exists(EQUITY) else []
        out["ledger"] = ([json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else [])[-60:]
        if not os.path.exists(OUTCOMES) or time.time() - os.path.getmtime(OUTCOMES) > 1800:
            try:
                label_decisions(log)
            except Exception as e:  # noqa: BLE001
                log(f"[paper] outcomes skipped: {type(e).__name__}: {e}")
        outc = _load(OUTCOMES, {})
        closed = [json.loads(l) for l in open(MANUAL_CLOSED)] if os.path.exists(MANUAL_CLOSED) else []
        wins = [x for x in closed if (x.get("pnl") or 0) > 0]
        out["manual"] = {"decisions": sum(1 for _ in open(MANUAL)) if os.path.exists(MANUAL) else 0,
                         "labelled": sum(1 for v in outc.values() if "ret_1d" in v),
                         "closed": len(closed), "wins": len(wins), "realized_pnl": sum(x.get("pnl") or 0 for x in closed),
                         "recent_closed": closed[-20:], "open_orders": [{k: o.get(k) for k in ("id", "symbol", "side", "qty", "notional", "type",
                                                                                                   "limit_price", "status", "submitted_at")}
                                                                          for o in _open_orders("manual")]}
    _save(STATUS, out)
    return out


def market_snapshot(sym):
    """Market conditions at the moment of a decision: the stock, QQQ, SPY and VIXY (volatility fear gauge), and the
    time of day."""
    kid, sec = _creds()
    now = datetime.now(alpaca.NY)
    today = now.date().isoformat()
    out = {"time_ny": now.isoformat(timespec="minutes"), "minutes_since_open": now.hour * 60 + now.minute - (9 * 60 + 30),
           "weekday": now.strftime("%a")}
    try:
        d = alpaca._get("/v2/stocks/snapshots", {"symbols": ",".join(dict.fromkeys([sym, "QQQ", "SPY", "VIXY"])), "feed": "iex"}, (kid, sec, None))
    except Exception:  # noqa: BLE001 - conditions are best effort
        return out
    for s_, x in (d or {}).items():
        last = (x.get("latestTrade") or {}).get("p")
        day, prev = x.get("dailyBar") or {}, x.get("prevDailyBar") or {}
        is_today = str(day.get("t", ""))[:10] == today
        prev_close = (prev if is_today else day).get("c")
        rec = {"last": last, "prev_close": prev_close,
               "chg_pct": round((last / prev_close - 1) * 100, 3) if last and prev_close else None}
        if is_today:
            rec.update({"open": day.get("o"), "high": day.get("h"), "low": day.get("l"), "volume": day.get("v"), "vwap": day.get("vw"),
                        "range_pct": round((day["h"] / day["l"] - 1) * 100, 3) if day.get("h") and day.get("l") else None,
                        "from_open_pct": round((last / day["o"] - 1) * 100, 3) if last and day.get("o") else None})
        out[s_] = rec
    return out


def intraday_view(sym):
    """The live 30-minute model's latest score for `sym` and its rank among the stocks it could score."""
    today = datetime.now(alpaca.NY).date().isoformat()
    path = os.path.join(FORWARD, f"live-{today}", "predictions.jsonl")
    m = _load(FROZEN_MANIFEST, {})
    syms = m.get("symbols", [])
    if not os.path.exists(path) or sym not in syms:
        return None
    last = None
    with open(path) as f:
        for line in f:
            last = line
    if not last:
        return None
    rec = json.loads(last)
    i = syms.index(sym)
    if not rec["valid"][i]:
        return {"bar": rec["label"], "valid": False}
    vals = sorted((rec["b1"][k] for k, ok in enumerate(rec["valid"]) if ok), reverse=True)
    return {"bar": rec["label"], "b1_bps": rec["b1"][i], "rank": vals.index(rec["b1"][i]) + 1, "of": len(vals)}


def context(sym, px=None):
    """What the models said about `sym` right now, plus market conditions (saved with each manual order, so the user's
    decisions can later be learned from)."""
    out = {"market": market_snapshot(sym), "intraday_model": intraday_view(sym)}
    try:                                                           # the 5-minute model's 117 inputs at this moment
        from . import intraday_big
        out["model_inputs"] = intraday_big.live_features(sym)
    except Exception as e:  # noqa: BLE001 - recording must never block a trade
        out["model_inputs"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    try:                                                           # near-the-money options: ATM IV, put/call ratio
        out["options"] = options_watch.reading(sym, px) if px else None
    except Exception as e:  # noqa: BLE001
        out["options"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    for h, path in (("1", os.path.join("private", "reports", "v7", "h1", "forecast.json")), ("5", os.path.join("private", "reports", "v7", "forecast.json"))):
        f = _load(path, {})
        r = (f.get("rows") or {}).get(sym)
        if r:
            out[f"v7_{h}d"] = {"asof": f.get("asof"), "p_up": r.get("p_up"), "median": r.get("median"), "lo80": r.get("lo80"),
                              "hi80": r.get("hi80"), "p_beat_qqq": r.get("p_beat_qqq"), "ranges": f.get("ranges")}
    rk = _load(os.path.join("private", "reports", "v6", "rankings.json"), {})
    r = next((x for x in rk.get("rankings", []) if x["symbol"] == sym), None)
    if r:
        out["rank"] = {"asof": rk.get("asof"), "score_pct": r.get("score_pct"), "decile": r.get("decile")}
    try:
        from . import stock_eval
        asof = stock_eval.latest_asof("nasdaq")
        e = _load(os.path.join(stock_eval.EVALS, "nasdaq", f"index_{asof}.json"), {}).get(sym)
        if e:
            out["ai"] = {"asof": asof, "h": e.get("h"), "lean": e.get("lean")}
    except Exception:  # noqa: BLE001 - context is best effort
        pass
    fv = (_load(os.path.join("private", "reports", "v6", "fairvalue.json"), {}).get("stocks") or {}).get(sym)
    if fv:
        out["fair_value"] = {k: fv.get(k) for k in ("fair", "low", "high", "verdict")}
    try:
        from . import earnings
        up = (_load(earnings.UPCOMING, {}) or {}).get("rows", [])
        today = datetime.now(alpaca.NY).date().isoformat()
        nxt = [d for d, _ in earnings.report_dates(sym, up) if d >= today]
        out["next_earnings"] = nxt[0] if nxt else None
    except Exception:  # noqa: BLE001
        pass
    return out


def manual_order(spec, log=print):
    """One order from the Auto-trade page: {symbol, side: buy|sell, qty | notional, limit?}. Long only."""
    sym = str(spec.get("symbol", "")).strip().upper()
    side = spec.get("side")
    if not sym.replace(".", "").isalnum() or len(sym) > 10 or side not in ("buy", "sell"):
        raise SystemExit("refused: bad symbol or side")
    qty = float(spec["qty"]) if spec.get("qty") not in (None, "") else None
    notional = float(spec["notional"]) if spec.get("notional") not in (None, "") else None
    limit = float(spec["limit"]) if spec.get("limit") not in (None, "") else None
    note = str(spec.get("note") or "")[:300]
    tp = float(spec["take_profit"]) if spec.get("take_profit") not in (None, "") else None
    sl = float(spec["stop_loss"]) if spec.get("stop_loss") not in (None, "") else None
    if (qty is None) == (notional is None) or (qty or notional or 0) <= 0:
        raise SystemExit("refused: give either a share quantity or a dollar amount")
    if notional is not None and limit:
        raise SystemExit("refused: a limit order needs a share quantity")
    c = control()
    if c.get("reset") in ("pending", "submitted"):
        raise SystemExit("refused: the account reset runs at the open (it sells everything); trade once it has finished, "
                         "a few minutes after 9:30 AM New York")
    st = _state(c, log)
    _settle(st, log)
    px = _latest_prices([sym]).get(sym)
    if not px:
        raise SystemExit(f"refused: no price for {sym}")
    if side == "sell":
        held = st["books"]["manual"]["holdings"].get(sym, 0.0)
        want = qty if qty is not None else notional / px
        if held <= 0 or want > held * 1.0001:
            raise SystemExit(f"refused: you hold {held:g} {sym} in manual trades (long only; nothing to short)")
        qty, notional = min(want, held), None
    if tp or sl:
        if qty is None or abs(qty - round(qty)) > 1e-9:
            raise SystemExit("refused: a target/stop order needs a whole number of shares")
        qty = float(round(qty))
        base = limit or px
        if tp and tp <= base + 0.01:
            raise SystemExit(f"refused: the target {tp:.2f} must be above the {'limit' if limit else 'current'} price {base:.2f}")
        if sl and sl >= base - 0.01:
            raise SystemExit(f"refused: the stop {sl:.2f} must be below the {'limit' if limit else 'current'} price {base:.2f}")
        if side == "sell" and not (tp and sl):
            raise SystemExit("refused: protecting a holding needs both a target and a stop (or use a plain limit sell)")
        if side == "sell" and limit:
            raise SystemExit("refused: a protective exit uses the target as its limit; leave the limit price empty")
    # target / stop orders stay active across days until they fill or are cancelled (good till cancelled)
    o = _order("manual", sym, side, qty=qty, notional=notional, log=log, limit=limit, take_profit=tp, stop_loss=sl,
               tif="gtc" if (tp or sl) else "day")
    _append(MANUAL, {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "order_id": o.get("id"), "symbol": sym, "side": side,
                     "qty": qty, "notional": notional, "limit": limit, "take_profit": tp, "stop_loss": sl, "price_at_order": px, "note": note,
                     "context": context(sym, px)})
    _save(BOOK, st)
    time.sleep(2)
    _settle(st, log)
    _save(BOOK, st)
    status(st)
    return {"order_id": o.get("id"), "status": o.get("status"), "symbol": sym, "side": side}


def label_decisions(log=print):
    """For each manual decision, what happened next (stock and QQQ): 30 minutes after the fill, at that day's close, and
    1 and 5 trading days later. Filled in as each horizon matures -> manual_outcomes.json."""
    if not os.path.exists(MANUAL):
        return {}
    out = _load(OUTCOMES, {})
    kid, sec = _creds()
    creds = (kid, sec, None)
    now = time.time()
    for line in open(MANUAL):
        r = json.loads(line)
        oid = r.get("order_id")
        o = out.get(oid) or {"symbol": r["symbol"], "side": r["side"], "note": r.get("note")}
        if all(k in o for k in ("ret_30m", "ret_close", "ret_1d", "ret_5d")):
            continue
        if "fill_t" not in o:
            try:
                x = _req("GET", f"/v2/orders/{oid}")
            except Exception:  # noqa: BLE001
                continue
            if not x.get("filled_at"):
                continue
            o["fill_price"] = float(x["filled_avg_price"])
            o["fill_t"] = int(datetime.fromisoformat(x["filled_at"].replace("Z", "+00:00")).timestamp())
        t0, sign = o["fill_t"], 1 if r["side"] == "buy" else -1
        for s_ in (r["symbol"], "QQQ"):
            key = "" if s_ == r["symbol"] else "qqq_"
            try:
                if f"{key}ret_30m" not in o and now > t0 + 1800 + 1000:   # the free plan serves SIP data only after 15 minutes
                    b = alpaca.get_bars([s_], "5Min", t0 - 300, t0 + 1800, feed="sip", creds=creds, pause=0.1)[s_]
                    if len(b) >= 2:
                        o[f"{key}ret_30m"] = sign * (b[-1][4] / b[0][4] - 1)
                daily = alpaca.get_bars([s_], "1Day", t0 - 5 * 86400, min(now - 1000, t0 + 12 * 86400), feed="sip", creds=creds, pause=0.1)[s_]
                day0 = datetime.fromtimestamp(t0, alpaca.NY).date()
                after = [x for x in daily if datetime.fromtimestamp(x[0], alpaca.NY).date() >= day0]
                base = o["fill_price"] if not key else (next((x[1] for x in after), None))
                if after and base and now > datetime.combine(day0, datetime.min.time(), alpaca.NY).timestamp() + 16.5 * 3600:
                    o[f"{key}ret_close"] = sign * (after[0][4] / base - 1)
                for n in (1, 5):
                    if len(after) > n and base:
                        o[f"{key}ret_{n}d"] = sign * (after[n][4] / base - 1)
            except Exception as e:  # noqa: BLE001 - try again on the next refresh
                log(f"[paper] outcome {r['symbol']}: {type(e).__name__}: {str(e)[:80]}")
        out[oid] = o
    _save(OUTCOMES, out)
    return out


def cancel(order_id, log=print):
    mine = {json.loads(l).get("order_id") for l in open(MANUAL)} if os.path.exists(MANUAL) else set()
    for parent in list(mine):                                      # a target or stop leg of one of the user's orders
        if parent and order_id not in mine:
            o = _req("GET", f"/v2/orders/{parent}", params={"nested": "true"})
            if any(x.get("id") == order_id for x in o.get("legs") or []):
                mine.add(order_id)
    if order_id not in mine:
        raise SystemExit("refused: not one of your manual orders")
    _req("DELETE", f"/v2/orders/{order_id}")
    log(f"[paper] canceled {order_id}")
    status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "close", "order", "cancel"])
    ap.add_argument("--book", choices=["ai", "rank", "night", "intraday", "focus", "manual", "all"], default="all")
    ap.add_argument("--json", default="{}")
    ap.add_argument("--id", default="")
    a = ap.parse_args()
    if a.cmd == "order":
        print(json.dumps(manual_order(json.loads(a.json))))
    elif a.cmd == "cancel":
        cancel(a.id)
    elif a.cmd == "run":
        print(json.dumps(run(a)))
    elif a.cmd == "close":
        close(a.book)
    else:
        status()


if __name__ == "__main__":
    main()
