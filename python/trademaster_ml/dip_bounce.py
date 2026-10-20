"""
Dip-and-bounce intraday study under docs/DIP-BOUNCE-STUDY-PROTOCOL.md (NVDA, MU, QQQ; 5-minute bars). Research only.

Usage: python -m trademaster_ml.dip_bounce
"""
import json, math, os

import numpy as np

from .intraday_study import sessions

OUT = os.path.join("private", "reports", "dip_bounce.json")
SYMS = ("NVDA", "MU", "QQQ")
DS, US = (0.005, 0.01, 0.015, 0.02), (0.005, 0.01, 0.02)
THROUGH = 0.0005
COST, COSTS_ALSO = 1.0, (3.0, 5.0)
DEV_END = "2022-12-30"
ENTRY_FROM, ENTRY_TO = 10 * 60, 15 * 60


def run():
    raw = {}
    for s in SYMS:
        raw[s] = sessions_with_volume(s)
    res = {}
    for s in SYMS:
        days = raw[s]
        dev = [x for x in days if x[0] <= DEV_END]
        hold = [x for x in days if x[0] > DEV_END]
        grid = []
        for d in DS:
            for u in US:
                r = trades(dev, d, u)
                grid.append({"d": d, "u": u, **stats(r, len(dev))})
        ok = [g for g in grid if g.get("t", 0) >= 2]
        sel = max(ok, key=lambda g: g["net_per_day_bps"]) if ok else None
        out = {"dev_grid": grid, "selected": sel and {"d": sel["d"], "u": sel["u"]}}
        if sel:
            r = trades(hold, sel["d"], sel["u"])
            h = stats(r, len(hold))
            h.update({f"net_bps_at_{int(c)}": float(np.mean(np.array(r) * 1e4 - c)) if r else None for c in COSTS_ALSO})
            h["success"] = bool(h.get("t", 0) >= 2 and h.get("net_bps", 0) > 0 and (h.get("net_bps_at_3") or 0) > 0)
            out["holdout"] = h
        res[s] = out
        best = max(grid, key=lambda g: g.get("net_per_day_bps", -1e9))
        print(f"{s}: best dev setting d={best['d']:.1%} u={best['u']:.1%}: {best['trades']} trades on {len(dev)} days, "
              f"net {best.get('net_bps', 0):+.2f} bps/trade (t {best.get('t', 0):+.2f}), {best.get('net_per_day_bps', 0):+.2f} bps/day; "
              f"selected: {out['selected']}", flush=True)
        if sel:
            hh = out["holdout"]
            print(f"   HOLD-OUT d={sel['d']:.1%} u={sel['u']:.1%}: {hh['trades']} trades, net {hh['net_bps']:+.2f} bps/trade (t {hh['t']:+.2f}), "
                  f"{hh['net_per_day_bps']:+.2f} bps/day, at 3 bps {hh['net_bps_at_3']:+.2f}, hit {hh['hit']:.0%} -> "
                  f"{'SUCCESS' if hh['success'] else 'fails'}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


def sessions_with_volume(sym):
    """[(date, bars, volumes)] regular-session days with their 09:30 bar."""
    from collections import defaultdict
    from datetime import datetime, timezone
    from .alpaca import NY
    from .intraday_study import DIR
    days = defaultdict(list)
    for t, o, h, l, c, v in json.load(open(os.path.join(DIR, f"{sym}.json"))):
        dt = datetime.fromtimestamp(t, timezone.utc).astimezone(NY)
        m = dt.hour * 60 + dt.minute
        if 9 * 60 + 30 <= m < 16 * 60:
            days[dt.date().isoformat()].append((m, o, h, l, c, v))
    out = []
    for day, rows in sorted(days.items()):
        rows.sort()
        if rows[0][0] == 9 * 60 + 30 and rows[-1][0] >= 15 * 60 + 50 and len(rows) >= 70:
            out.append((day, [r[:5] for r in rows], [r[5] or 0.0 for r in rows]))
    return out


def trades(days, d, u):
    out = []
    for day, bars, vols in days:
        r = one(bars, vols, d, u)
        if r is not None:
            out.append(r)
    return out


def one(bars, vols, d, u):
    pv = vol = 0.0
    entry = target = None
    for k, (m, o, h, l, c) in enumerate(bars):
        if entry is None:
            if ENTRY_FROM <= m <= ENTRY_TO and vol > 0:
                limit = pv / vol * (1 - d)
                if l < limit * (1 - THROUGH):
                    entry, target = limit, limit * (1 + u)       # the exit can fill from the next bar on
        elif h > target:
            return math.log(target / entry)
        pv += (h + l + c) / 3 * vols[k]
        vol += vols[k]
    return math.log(bars[-1][4] / entry) if entry is not None else None


def stats(r, n_days):
    r = np.array(r)
    if len(r) < 2:
        return {"trades": int(len(r)), "days": n_days}
    net = r * 1e4 - COST
    t = net.mean() / (net.std(ddof=1) / math.sqrt(len(net)))
    return {"trades": int(len(r)), "days": n_days, "trade_share": len(r) / n_days, "gross_bps": float(r.mean() * 1e4),
            "net_bps": float(net.mean()), "t": float(t), "hit": float(np.mean(net > 0)),
            "net_per_day_bps": float(net.sum() / n_days)}


if __name__ == "__main__":
    run()
