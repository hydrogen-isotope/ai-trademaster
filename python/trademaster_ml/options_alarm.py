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
        for wait in range(12):                                     # shared rate limit: back off on HTTP 429
            try:
                d = alpaca._get(path, p, creds) or {}
                break
            except RuntimeError as e:
                if "429" not in str(e) or wait == 11:
                    raise
                time.sleep(15 + 5 * wait)
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
        with ThreadPoolExecutor(4) as ex:
            for n, (day, v) in enumerate(ex.map(one, todo), 1):
                have["options"][day] = v
                if n % 100 == 0:
                    json.dump(have, open(path, "w"))
                    log(f"  {sym} {day}: {n}/{len(todo)} days of option bars", flush=True)
        json.dump(have, open(path, "w"))
        log(f"{sym}: option bars for {sum(1 for v in have['options'].values() if v)} days", flush=True)


# ---------------------------------------------------------------- readings and trades

SPLITS = {"NVDA": [("2024-06-10", 10.0)]}              # raw bars before the date are divided by the factor


def _adj(sym, day):
    f = 1.0
    for d, k in SPLITS.get(sym, []):
        if day < d:
            f *= k
    return f


def _bs_iv(price, S, K, T, call, r=RATE):
    """Vectorised Black-Scholes implied volatility by bisection; NaN outside no-arbitrage bounds."""
    from scipy.special import ndtr
    price, S, K, T = (np.asarray(x, float) for x in (price, S, K, T))
    call = np.asarray(call, bool)
    disc = K * np.exp(-r * T)
    intrinsic = np.where(call, np.maximum(S - disc, 0), np.maximum(disc - S, 0))
    ok = (price > intrinsic + 1e-6) & (price < np.where(call, S, K)) & (T > 0)
    lo, hi = np.full(price.shape, 0.01), np.full(price.shape, 5.0)
    sq = np.sqrt(np.maximum(T, 1e-9))
    for _ in range(50):
        sig = 0.5 * (lo + hi)
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sq)
        c = S * ndtr(d1) - disc * ndtr(d1 - sig * sq)
        v = np.where(call, c, c - S + disc)
        hi = np.where(v > price, sig, hi)
        lo = np.where(v > price, lo, sig)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def day_readings(day, bars, opt):
    """Per stock bar: ATM IV, put/call ratio and contracts traded so far (raw prices)."""
    mins = [b[0] for b in bars]
    n = len(mins)
    iv, pc, vol = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    if not opt or not opt.get("bars"):
        return mins, iv, pc, vol
    exp, d0 = date.fromisoformat(opt["expiry"]), date.fromisoformat(day)
    cons = {c[2]: (c[0], c[1]) for c in opt["contracts"]}
    series = {occ: {x[0]: (x[1], x[2]) for x in xs} for occ, xs in opt["bars"].items() if occ in cons}
    cumc = cump = 0.0
    for i, m in enumerate(mins):
        for occ, ser in series.items():
            if m in ser:
                if cons[occ][1]:
                    cumc += ser[m][1]
                else:
                    cump += ser[m][1]
        vol[i] = cumc + cump
        pc[i] = cump / cumc if cumc > 0 else np.nan
        S = bars[i][4]
        T = ((exp - d0).days + (960 - (m + 5)) / 1440) / 365
        best = {}
        for occ, ser in series.items():
            recent = [mm for mm in ser if m - 30 < mm <= m]
            if not recent:
                continue
            K, call = cons[occ]
            if call not in best or abs(K - S) < abs(best[call][0] - S):
                best[call] = (K, ser[max(recent)][0])
        if len(best) == 2:
            v = _bs_iv([best[True][1], best[False][1]], [S, S], [best[True][0], best[False][0]], [T, T], [True, False])
            if np.isfinite(v).all():
                iv[i] = float(v.mean())
    return mins, iv, pc, vol


def load_sym(sym):
    """Days with split-adjusted bars and per-bar option readings; cached after the first build."""
    cache = os.path.join(DIR, f"{sym}_readings.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    have = json.load(open(os.path.join(DIR, f"{sym}.json")))
    out = {}
    for day in sorted(have["stock"]):
        bars = sorted(have["stock"][day])
        opt = have["options"].get(day)
        mins, iv, pc, vol = day_readings(day, bars, opt)
        f = _adj(sym, day)
        out[day] = {"m": mins, "o": [b[1] / f for b in bars], "l": [b[3] / f for b in bars], "c": [b[4] / f for b in bars],
                    "iv": [float(x) if np.isfinite(x) else None for x in iv],
                    "pc": [float(x) if np.isfinite(x) else None for x in pc], "vol": [float(x) for x in vol],
                    "has_opt": bool(opt and opt.get("bars"))}
    json.dump(out, open(cache, "w"))
    return out


def _alarm(base, iv, pc, vol, jump, use_pc):
    if base[0] and iv and iv / base[0] - 1 >= jump:
        return True
    return bool(use_pc and vol >= 500 and pc and base[1] and pc >= 1.0 and pc >= 1.5 * base[1])


def trade(D, days, k0, alarm=None, signal=None, entry_ok=None):
    """One entry on days[k0] if its first bar closed up. alarm=(jump, use_pc): variant B; signal(day, i) -> True sells at
    the next bar open (rule C); entry_ok(day) -> bool (rule D). Returns (net return, sessions held, why) or None."""
    d = D[days[k0]]
    if not d["has_opt"] or not d["m"] or d["m"][0] != 570 or 575 not in d["m"] or d["c"][0] <= d["o"][0]:
        return None
    if entry_ok is not None and not entry_ok(days[k0]):
        return None
    i0 = d["m"].index(575)
    cost = d["o"][i0]
    net = lambda px: px / cost - 1 - 2 * COST                     # noqa: E731
    for k in range(k0, min(k0 + CARRY_DAYS + 1, len(days))):
        x = D[days[k]]
        base, pending = None, False
        for i in range(i0 if k == k0 else 0, len(x["m"])):
            m, o = x["m"][i], x["o"][i]
            if pending:
                return net(o), k - k0, "alarm"
            if k > k0:
                if o >= cost:
                    return net(o), k - k0, "cost"
                stop = cost * (1 + CARRY_STOP)
                if x["l"][i] <= stop:
                    return net(min(o, stop)), k - k0, "stop"
                if k - k0 >= CARRY_DAYS and m >= 950:
                    return net(o), k - k0, "time"
            elif m >= 950:
                if o >= cost:
                    return net(o), 0, "close"
                break                                             # kept overnight at a loss
            if m >= 575 and alarm:
                if base is None:
                    if x["iv"][i]:
                        base = (x["iv"][i], x["pc"][i])
                elif _alarm(base, x["iv"][i], x["pc"][i], x["vol"][i], *alarm):
                    pending = True
            if m >= 575 and signal and signal(days[k], i):
                pending = True
    return None


def _cluster_t(rows):
    by = {}
    for day, x in rows:
        by.setdefault(day, []).append(x)
    v = np.array([np.mean(x) for x in by.values()])
    t = float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if len(v) > 2 and v.std() > 0 else float("nan")
    return float(v.mean()), t, len(v)


def evaluate(d0, d1, variants, signal=None, entry_ok=None):
    """variants: {name: alarm tuple or None}. signal(sym, day, i), entry_ok(sym, day) for rules C and D."""
    res = {}
    for name, alarm in variants.items():
        diffs, a_r, b_r, why = [], [], [], {}
        for sym in SYMS:
            D = load_sym(sym)
            days = sorted(D)
            for k, day in enumerate(days):
                if not (d0 <= day <= d1):
                    continue
                a = trade(D, days, k)
                if a is None:
                    continue
                b = trade(D, days, k, alarm=alarm,
                          signal=(lambda dd, i, s=sym: signal(s, dd, i)) if signal else None,
                          entry_ok=(lambda dd, s=sym: entry_ok(s, dd)) if entry_ok else None)
                b_ret = b[0] if b else 0.0                        # rule D skipping an entry earns 0
                diffs.append((day, b_ret - a[0])); a_r.append(a[0]); b_r.append(b_ret)
                w = b[2] if b else "no entry"
                why[w] = why.get(w, 0) + 1
        m, t, nd = _cluster_t(diffs)
        res[name] = {"trades": len(diffs), "days": nd, "mean_diff_pct": m * 100, "t": t,
                     "mean_A_pct": float(np.mean(a_r) * 100), "mean_B_pct": float(np.mean(b_r) * 100),
                     "hit_A": float(np.mean(np.array(a_r) > 0)), "hit_B": float(np.mean(np.array(b_r) > 0)), "exits_B": why}
    return res


def _show(log, name, r):
    log(f"  {name}: {r['trades']} trades, B-A {r['mean_diff_pct']:+.3f}% per trade (t {r['t']:+.2f}); "
        f"A {r['mean_A_pct']:+.3f}% (hit {r['hit_A']:.0%}), B {r['mean_B_pct']:+.3f}% (hit {r['hit_B']:.0%}); exits {r['exits_B']}",
        flush=True)


def _grid():
    return {f"iv{round(j * 100)}_pc{'on' if pc else 'off'}": (j, pc) for j, pc in GRID}


def search(log=print):
    os.makedirs(OUT, exist_ok=True)
    res = evaluate(str(START), DEV_END, _grid())
    for k, r in res.items():
        _show(log, k, r)
    best = max(res, key=lambda k: res[k]["mean_diff_pct"])
    json.dump({"dev": res, "selected": best}, open(os.path.join(OUT, "search.json"), "w"), indent=1)
    log(f"selected on development: {best}")
    return res


def final(log=print):
    if os.path.exists(os.path.join(OUT, "final.json")):
        raise SystemExit("the final check was already run once")
    s = json.load(open(os.path.join(OUT, "search.json")))
    g = _grid()
    live = f"iv{round(LIVE[0] * 100)}_pc{'on' if LIVE[1] else 'off'}"
    res = evaluate(FINAL_FROM, str(END), {k: g[k] for k in dict.fromkeys([s["selected"], live])})
    for k, r in res.items():
        _show(log, k, r)
    r = res[s["selected"]]
    out = {"final": res, "selected": s["selected"], "pass": bool(r["mean_diff_pct"] > 0 and r["t"] >= 2)}
    json.dump(out, open(os.path.join(OUT, "final.json"), "w"), indent=1)
    log(f"final check for {s['selected']}: {'PASS' if out['pass'] else 'fails'}")
    return out


if __name__ == "__main__":
    {"fetch": fetch, "search": search, "final": final}[sys.argv[1]]()
