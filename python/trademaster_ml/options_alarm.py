"""
Options alarm study under docs/OPTIONS-ALARM-PROTOCOL.md: NVDA & MU 5-minute stock and near-the-money option bars,
the live slice's trades with and without the alarm, and a learned exit model. Research only; no orders.

Usage: python -m trademaster_ml.options_alarm fetch | search | final | model
"""
import json, math, os, sys, time
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import alpaca
from .options import STANDARD, list_contracts, implied_vol

DIR = os.path.join("private", "data", "options_alarm")
OUT = os.path.join("private", "reports", "options_alarm")
SYMS = ("NVDA", "MU")
START, END = date(2024, 2, 1), date(2026, 10, 20)
DEV_END, FINAL_FROM = "2025-06-30", "2025-07-01"
FIT_END = "2025-03-31"                      # model: fit up to here, early stopping / threshold on Apr-Jun 2025
BAND, MIN_DAYS, RATE, COST = 0.05, 5, 0.04, 3e-4
CARRY_STOP, CARRY_DAYS = -0.04, 3
GRID = [(j, pc) for j in (0.05, 0.08, 0.12, 0.20) for pc in (False, True)]
LIVE = (0.08, True)


# ---------------------------------------------------------------- data

def _ny(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(alpaca.NY)


def _bars(path, params, creds):
    out, token = {}, None
    while True:
        p = dict(params)
        if token:
            p["page_token"] = token
        d = alpaca._get(path, p, creds) or {}
        bars = d.get("bars") or {}
        if isinstance(bars, list):
            bars = {params.get("symbols", "_"): bars}
        for s, b in bars.items():
            out.setdefault(s, []).extend(b)
        token = d.get("next_page_token")
        time.sleep(0.3)
        if not token:
            return out


def fetch(log=print):
    os.makedirs(DIR, exist_ok=True)
    creds = alpaca.credentials()
    for sym in SYMS:
        path = os.path.join(DIR, f"{sym}.json")
        have = json.load(open(path)) if os.path.exists(path) else {"stock": {}, "options": {}}
        if not have["stock"]:
            stock = {}
            d0 = START
            while d0 <= END:                                       # quarter by quarter
                d1 = min(END, d0 + timedelta(days=90))
                b = _bars("/v2/stocks/bars", {"symbols": sym, "timeframe": "5Min", "feed": "sip", "adjustment": "raw",
                                              "start": f"{d0}T13:00:00Z", "end": f"{d1}T21:00:00Z", "limit": 10000}, creds)
                for x in b.get(sym, []):
                    t = _ny(x["t"])
                    m = t.hour * 60 + t.minute
                    if 570 <= m < 960:                             # 9:30 -> 15:55 bars
                        stock.setdefault(str(t.date()), []).append([m, x["o"], x["h"], x["l"], x["c"], x["v"]])
                d0 = d1 + timedelta(days=1)
            have["stock"] = stock
            json.dump(have, open(path, "w"))
            log(f"{sym}: {len(stock)} days of 5-minute stock bars", flush=True)
        cons = [c for c in list_contracts(sym, creds, END + timedelta(days=30)) if STANDARD.match(c["symbol"])]
        by_exp = {}
        for c in cons:
            by_exp.setdefault(c["expiration_date"], []).append((float(c["strike_price"]), c["type"] == "call", c["symbol"]))
        exps = sorted(by_exp)

        def one(day):
            bars = have["stock"][day]
            first = next((x for x in bars if x[0] == 575), None)       # the 9:35 bar
            e = first and next((x for x in exps if (date.fromisoformat(x) - date.fromisoformat(day)).days >= MIN_DAYS), None)
            if not e:
                return day, None
            S0 = first[1]
            pick = [c for c in by_exp[e] if abs(c[0] / S0 - 1) <= BAND]
            got = {}
            for i in range(0, len(pick), 100):
                b_ = _bars("/v1beta1/options/bars", {"symbols": ",".join(c[2] for c in pick[i:i + 100]), "timeframe": "5Min",
                                                     "start": f"{day}T13:00:00Z", "end": f"{day}T21:00:00Z", "limit": 10000}, creds)
                for s_, xs in b_.items():
                    got[s_] = [[_ny(x["t"]).hour * 60 + _ny(x["t"]).minute, x["c"], x["v"]] for x in xs]
            return day, {"expiry": e, "contracts": [[k, call, s_] for k, call, s_ in pick], "bars": got}

        todo = [d for d in sorted(have["stock"]) if d not in have["options"] and d >= str(START)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            for n, (day, v) in enumerate(ex.map(one, todo), 1):
                have["options"][day] = v
                if n % 100 == 0:
                    json.dump(have, open(path, "w"))
                    log(f"  {sym} {day}: {n}/{len(todo)} days of option bars", flush=True)
        json.dump(have, open(path, "w"))
        log(f"{sym}: option bars for {sum(1 for v in have['options'].values() if v)} days", flush=True)


if __name__ == "__main__":
    {"fetch": fetch}[sys.argv[1]]()
