"""
Automatic PAPER trading on Alpaca's paper account (simulated money only), to see whether the signals would make money.

Safety: orders go only to https://paper-api.alpaca.markets, and only with a paper key (key id starting "PK"); any other
key or URL is refused. Long only. US stocks only (Alpaca has no SGX). Runs only while the app is open (the app calls
`run` every few minutes; nothing happens unless it is switched on in the Auto-trade page).

Two books, each with its own share of the paper account, tracked in our own ledger (Alpaca holds one combined account):
  ai    holds the stocks whose latest 5-day Qwen + models research signal is BUY (latest close's evaluations), up to
        MAX_AI by chance of a rise, equal weight; rebalanced once per trading day.
  rank  holds the top decile of the v6 cross-stock ranking, equal weight; rebalanced on the first run of each week.

Files (private/reports/paper/): control.json (switches, set by the app), book.json (holdings and cash per book),
ledger.jsonl (every order and fill), equity.jsonl (daily value per book and QQQ), status.json (what the page shows).

Usage: python -m trademaster_ml.papertrade run | status | close --book ai|rank|all
"""
import argparse, json, os, time, urllib.error, urllib.request
from datetime import datetime, timezone

from . import alpaca, holidays

PAPER = "https://paper-api.alpaca.markets"
DIR = os.path.join("private", "reports", "paper")
CONTROL, BOOK, LEDGER, EQUITY, STATUS = (os.path.join(DIR, f) for f in ("control.json", "book.json", "ledger.jsonl", "equity.jsonl", "status.json"))
BOOKS = {"ai": "AI signals (Qwen + models, 5-day BUY)", "rank": "Top-ranked stocks (v6 top 10%)"}
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
    c.setdefault("split", {"ai": 0.5, "rank": 0.5})
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


def targets_rank():
    rk = _load(os.path.join("private", "reports", "v6", "rankings.json"), {})
    return rk.get("asof"), [r["symbol"] for r in rk.get("rankings", []) if r.get("decile") == 10]


# ---------------------------------------------------------------- orders

def _order(book, sym, side, qty=None, notional=None, log=print):
    body = {"symbol": sym, "side": side, "type": "market", "time_in_force": "day",
            "client_order_id": f"tm-{book}-{int(time.time() * 1000)}-{sym}"[:48]}
    if notional is not None:
        body["notional"] = f"{notional:.2f}"
    else:
        body["qty"] = f"{qty:.9f}".rstrip("0").rstrip(".")
    o = _req("POST", "/v2/orders", body)
    _append(LEDGER, {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "book": book, "symbol": sym, "side": side,
                     "qty": qty, "notional": notional, "order_id": o.get("id"), "status": o.get("status")})
    log(f"[paper] {book}: {side} {sym} " + (f"${notional:.2f}" if notional is not None else f"{qty} sh"))
    return o


def _settle(book_state, log=print):
    """Apply fills of our orders to the books (filled quantity and average price), once per order."""
    seen = set(book_state.get("settled", []))
    pending = [json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []
    for rec in pending:
        oid = rec.get("order_id")
        if not oid or oid in seen:
            continue
        o = _req("GET", f"/v2/orders/{oid}")
        if o.get("status") not in ("filled", "canceled", "expired", "rejected", "done_for_day"):
            continue
        q, px = float(o.get("filled_qty") or 0), float(o.get("filled_avg_price") or 0)
        b = book_state["books"][rec["book"]]
        sign = 1 if rec["side"] == "buy" else -1
        b["holdings"][rec["symbol"]] = round(b["holdings"].get(rec["symbol"], 0.0) + sign * q, 9)
        if abs(b["holdings"][rec["symbol"]]) < 1e-9:
            b["holdings"].pop(rec["symbol"])
        b["cash"] -= sign * q * px
        seen.add(oid)
    book_state["settled"] = sorted(seen)


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


def run(a=None, log=print):
    c = control()
    if not c["enabled"]:
        return {"done": "auto-trade is off"}
    clock = _req("GET", "/v2/clock")
    st = _load(BOOK, None)
    acct = _req("GET", "/v2/account")
    if st is None:                                                 # first run: split the paper account between the books
        eq = float(acct["equity"])
        st = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "start_equity": eq, "settled": [],
              "books": {k: {"cash": eq * c["split"][k], "start": eq * c["split"][k], "holdings": {}, "last_rebalance": None} for k in BOOKS}}
        _save(BOOK, st)
        log(f"[paper] started: ${eq:,.2f} paper equity split {c['split']}")
    _settle(st, log)
    if not clock.get("is_open"):
        _save(BOOK, st)
        status(st, note="market closed: next open " + str(clock.get("next_open")))
        return {"done": "market closed"}
    day = str(clock["timestamp"])[:10]
    week = datetime.fromisoformat(day).isocalendar()[:2]
    done = {}
    for name in BOOKS:
        b = st["books"][name]
        if not c["books"].get(name):
            continue
        last = b.get("last_rebalance")
        if name == "ai" and last == day:
            continue
        if name == "rank" and last and datetime.fromisoformat(last).isocalendar()[:2] == week:
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
    return {"orders": done}


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
    st = st or _load(BOOK, None)
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
            out["books"][name] = {"label": BOOKS[name], "enabled": c["books"].get(name, False), "value": val, "start": b["start"],
                                  "return": val / b["start"] - 1 if b["start"] else None, "cash": b["cash"], "positions": pos,
                                  "last_rebalance": b.get("last_rebalance"), "signals_asof": b.get("signals_asof")}
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
    _save(STATUS, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "close"])
    ap.add_argument("--book", choices=["ai", "rank", "all"], default="all")
    a = ap.parse_args()
    if a.cmd == "run":
        print(json.dumps(run(a)))
    elif a.cmd == "close":
        close(a.book)
    else:
        status()


if __name__ == "__main__":
    main()
