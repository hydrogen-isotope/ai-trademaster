"""
Intraday study under docs/INTRADAY-STUDY-PROTOCOL.md: R1 intraday momentum (QQQ, SPY) and R2 opening-range breakout
(QQQ, MU, NVDA) on Alpaca 5-minute bars, 2016-2026. Research only.

Usage: python -m trademaster_ml.intraday_study
"""
import json, math, os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from .alpaca import NY

DIR = os.path.join("private", "data", "intraday5")
OUT = os.path.join("private", "reports", "intraday_study.json")
DEV_END = "2022-12-30"
COSTS = {"QQQ": (2, 1, 4), "SPY": (2, 1, 4), "MU": (5, 2, 10), "NVDA": (5, 2, 10)}     # base, low, high (bps round trip)


def sessions(sym):
    """{date: [(minute of day, o, h, l, c)]} regular-session 5-minute bars, by bar start time in New York."""
    days = defaultdict(list)
    for t, o, h, l, c, v in json.load(open(os.path.join(DIR, f"{sym}.json"))):
        d = datetime.fromtimestamp(t, timezone.utc).astimezone(NY)
        m = d.hour * 60 + d.minute
        if 9 * 60 + 30 <= m < 16 * 60:
            days[d.date().isoformat()].append((m, o, h, l, c))
    return {k: sorted(v) for k, v in sorted(days.items())}


def r1_trades(S):
    out, prev_close = [], None
    for day, bars in S.items():
        by = {m: (o, h, l, c) for m, o, h, l, c in bars}
        if prev_close and 9 * 60 + 30 in by and 9 * 60 + 55 in by and 15 * 60 + 30 in by:
            sig = math.log(by[9 * 60 + 55][3] / prev_close)
            entry, exit_ = by[15 * 60 + 30][0], bars[-1][4]
            if sig != 0:
                out.append((day, (1 if sig > 0 else -1) * math.log(exit_ / entry), 1 if sig > 0 else -1))
        prev_close = bars[-1][4] if bars and bars[-1][0] >= 15 * 60 + 50 else None
    return out


def r2_trades(S):
    out = []
    for day, bars in S.items():
        if len(bars) < 10 or bars[0][0] != 9 * 60 + 30 or bars[1][0] != 9 * 60 + 35:
            continue
        m0, o0, h0, l0, c0 = bars[0]
        if c0 == o0:
            continue
        side = 1 if c0 > o0 else -1
        stop = l0 if side > 0 else h0
        entry, exit_ = bars[1][1], bars[-1][4]
        for m, o, h, l, c in bars[1:]:
            if side > 0 and l <= stop:
                exit_ = min(o, stop); break
            if side < 0 and h >= stop:
                exit_ = max(o, stop); break
        out.append((day, side * math.log(exit_ / entry), side))
    return out


def stats(trades, cost_bps, long_only=False):
    r = np.array([x for _, x, side in trades if side > 0 or not long_only])
    net = r - cost_bps / 1e4
    if len(net) < 2:
        return {"n": int(len(net))}
    t = net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))
    return {"n": int(len(net)), "gross_bps": float(r.mean() * 1e4), "net_bps": float(net.mean() * 1e4), "t": float(t),
            "hit": float(np.mean(net > 0)), "net_per_year": float(net.mean() * 252), "breakeven_bps": float(r.mean() * 1e4)}


def main():
    res = {}
    for rule, syms, fn in (("R1 intraday momentum", ("QQQ", "SPY"), r1_trades), ("R2 opening-range breakout", ("QQQ", "MU", "NVDA"), r2_trades)):
        for s in syms:
            if not os.path.exists(os.path.join(DIR, f"{s}.json")):
                continue
            S = sessions(s)
            tr = fn(S)
            base, lo, hi = COSTS[s]
            row = {}
            for per, sel in (("dev", lambda d: d <= DEV_END), ("holdout", lambda d: d > DEV_END)):
                t = [x for x in tr if sel(x[0])]
                row[per] = {"base": stats(t, base), "low_cost": stats(t, lo), "high_cost": stats(t, hi), "long_only": stats(t, base, long_only=True)}
            res[f"{rule} | {s}"] = row
            b = row["dev"]["base"], row["holdout"]["base"]
            works = all(x.get("t", 0) >= 2 and x.get("net_bps", 0) > 0 for x in b)
            row["works"] = works
            print(f"{rule:28} {s:5} dev: n {b[0]['n']} gross {b[0]['gross_bps']:+.2f} net {b[0]['net_bps']:+.2f} bps t {b[0]['t']:+.2f} | "
                  f"holdout: n {b[1]['n']} gross {b[1]['gross_bps']:+.2f} net {b[1]['net_bps']:+.2f} bps t {b[1]['t']:+.2f} | "
                  f"{'WORKS' if works else 'fails'} | long-only net dev {row['dev']['long_only'].get('net_bps', 0):+.2f} hold {row['holdout']['long_only'].get('net_bps', 0):+.2f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
