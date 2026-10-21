"""
Options watch for the NVDA & MU slice: a live reading of each stock's near-the-money options (at-the-money implied
volatility and the put/call volume ratio) and an alarm when either jumps against the reading taken after entry.
Read-only market data (Alpaca indicative options feed). The rule is an untested paper experiment: there is no intraday
options history to back-test it on.
"""
import json, math, os
from datetime import date, datetime, timedelta

import numpy as np

from . import alpaca
from .options import implied_vol

LOG = os.path.join("private", "reports", "paper", "options_watch.jsonl")
IV_JUMP = 0.08          # sell when at-the-money IV is 8% (relative) above the entry reading
PC_JUMP = 1.5           # ... or the put/call volume ratio is 1.5x the entry reading (and at least 1.0)
MIN_VOLUME = 500        # contracts traded today near the money before the put/call ratio counts
BAND, MIN_DAYS, MAX_DAYS = 0.05, 5, 21          # skips the last few days before expiry (noisy IV)


def reading(sym, S, today=None, creds=None):
    """{'iv': ATM implied vol, 'pc': put/call volume ratio, 'vol': contracts, 'expiry': ...} or None."""
    today = today or datetime.now(alpaca.NY).date()
    creds = creds or alpaca.credentials()
    snaps, token = {}, None
    for _ in range(5):
        p = {"feed": "indicative", "limit": 250, "expiration_date_gte": str(today + timedelta(days=MIN_DAYS)),
             "expiration_date_lte": str(today + timedelta(days=MAX_DAYS)),
             "strike_price_gte": f"{S * (1 - BAND):.2f}", "strike_price_lte": f"{S * (1 + BAND):.2f}"}
        if token:
            p["page_token"] = token
        d = alpaca._get(f"/v1beta1/options/snapshots/{sym}", p, creds) or {}
        snaps.update(d.get("snapshots") or {})
        token = d.get("next_page_token")
        if not token:
            break
    if not snaps:
        return None
    parsed = []
    for occ, v in snaps.items():
        root = len(occ) - 15                                           # OCC: ROOT + YYMMDD + C/P + strike*1000
        exp = date(2000 + int(occ[root:root + 2]), int(occ[root + 2:root + 4]), int(occ[root + 4:root + 6]))
        call, K = occ[root + 6] == "C", int(occ[root + 7:]) / 1000
        q = v.get("latestQuote") or {}
        bar = v.get("dailyBar") or {}
        vol = bar.get("v", 0) if str(bar.get("t", ""))[:10] == str(today) else 0
        parsed.append((exp, call, K, q.get("bp"), q.get("ap"), v.get("impliedVolatility"), vol))
    expiry = min({x[0] for x in parsed})
    T = max((expiry - today).days, 1) / 365.0
    near = [x for x in parsed if x[0] == expiry]
    ivs = {}
    for call in (True, False):
        side = [x for x in near if x[1] == call]
        if not side:
            continue
        exp, c, K, bp, ap, iv, _ = min(side, key=lambda x: abs(x[2] - S))
        if not iv and bp and ap and ap >= bp > 0:
            iv = float(implied_vol([(bp + ap) / 2], [S], [K], [T], [0.04], [c])[0])
        if iv and math.isfinite(iv):
            ivs[call] = float(iv)
    calls = sum(x[6] for x in parsed if x[1])
    puts = sum(x[6] for x in parsed if not x[1])
    return {"iv": float(np.mean(list(ivs.values()))) if ivs else None, "pc": puts / calls if calls else None,
            "vol": int(calls + puts), "expiry": str(expiry)}


def alarm(entry, now):
    """Why the options say sell (a short text), or None."""
    if not entry or not now:
        return None
    if entry.get("iv") and now.get("iv") and now["iv"] / entry["iv"] - 1 >= IV_JUMP:
        return f"options alarm: implied volatility {entry['iv']:.0%} -> {now['iv']:.0%}"
    if (now.get("vol", 0) >= MIN_VOLUME and now.get("pc") and entry.get("pc") and now["pc"] >= 1.0
            and now["pc"] >= PC_JUMP * entry["pc"]):
        return f"options alarm: put/call volume {entry['pc']:.2f} -> {now['pc']:.2f}"
    return None


def record(sym, price, r, why=None):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": datetime.now(alpaca.NY).isoformat(timespec="seconds"), "symbol": sym, "price": price,
                            **(r or {}), "alarm": why}) + "\n")
