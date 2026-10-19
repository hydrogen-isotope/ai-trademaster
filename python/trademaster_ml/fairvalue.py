"""
Fair value from peer multiples (research only), for the analysis-report summary on the Markets stock page.

For each ranked NASDAQ stock: its own P/E, P/S and P/B (from the point-in-time SEC ratios in fundamentals.npz at the
rankings date) and the same multiples of its sector group (indexes.SECTORS; the whole universe when fewer than
MIN_PEERS peers have a value). Each method's fair price = the peers' median multiple times the stock's own earnings,
sales or book value per share; the low/high use the peers' 25th/75th percentile multiple. A method is skipped when the
stock's own value is not positive (e.g. P/E for a loss-making company). The fair value is the median over the methods
that apply.

Limits, shown on the page: peer multiples ignore growth and quality differences, SGX has no fundamentals, and this is
not validated against later prices.

Output: private/reports/v6/fairvalue.json.  Usage: python -m trademaster_ml.fairvalue
"""
import json, os
from datetime import datetime

import numpy as np

from . import indexes, v6

OUT = os.path.join("private", "reports", "v6", "fairvalue.json")
FUND = os.path.join("private", "data", "us", "fundamentals.npz")
MIN_PEERS = 8
# method key, label, ratio in fundamentals.npz (value / market cap) — the multiple is its inverse
METHODS = [("pe", "P/E", "earnings_yield"), ("ps", "P/S", "sales_to_price"), ("pb", "P/B", "book_to_market")]


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def compute(rows, ratios, sector_of):
    """rows: [{symbol, close}], ratios: {method: {symbol: value/price ratio}}, sector_of: {symbol: sector key}.
    Returns {symbol: fair-value record}."""
    out = {}
    for r in rows:
        s, close = r["symbol"], r["close"]
        sec = sector_of.get(s, "other")
        meth = {}
        for key, label, _ in METHODS:
            own = ratios[key].get(s)
            if own is None or not np.isfinite(own) or own <= 0:
                meth[key] = {"label": label, "own": None, "note": "own value not positive"}
                continue
            peer = [v for p, v in ratios[key].items() if p != s and sector_of.get(p, "other") == sec and np.isfinite(v) and v > 0]
            scope = "sector"
            if len(peer) < MIN_PEERS:
                peer = [v for p, v in ratios[key].items() if p != s and np.isfinite(v) and v > 0]
                scope = "universe"
            if len(peer) < MIN_PEERS:
                continue
            mult = 1.0 / np.array(peer)
            q25, q50, q75 = np.percentile(mult, [25, 50, 75])
            per = close * own                                  # own earnings / sales / book per share
            meth[key] = {"label": label, "own": _r(1.0 / own, 1), "peer_median": _r(q50, 1), "peers": len(peer), "scope": scope,
                         "fair": _r(per * q50), "low": _r(per * q25), "high": _r(per * q75)}
        ok = [m for m in meth.values() if m.get("fair") is not None]
        rec = {"close": close, "sector": sec, "methods": meth}
        if ok:
            fair, low, high = (float(np.median([m[k] for m in ok])) for k in ("fair", "low", "high"))
            rec.update({"fair": _r(fair), "low": _r(low), "high": _r(high), "upside": _r(fair / close - 1, 4), "n_methods": len(ok),
                        "verdict": "undervalued" if close < low else "overvalued" if close > high else "fair"})
        out[s] = rec
    return out


def update(log=print):
    rk = json.load(open(os.path.join(v6.OUT, "rankings.json")))
    asof = rk["asof"]
    F = np.load(FUND)
    dates = [str(d) for d in F["dates"]]
    t = dates.index(asof) if asof in dates else len(dates) - 1
    col = {str(s): j for j, s in enumerate(F["symbols"])}
    rows = [{"symbol": r["symbol"], "close": r["close"]} for r in rk["rankings"] if r["symbol"] in col]
    ratios = {key: {r["symbol"]: float(F[name][t, col[r["symbol"]]]) for r in rows} for key, _, name in METHODS}
    idx = json.load(open(indexes.OUT)) if os.path.exists(indexes.OUT) else {}
    sector_of = idx.get("members") or {}
    labels = {k: lab for k, lab, *_ in indexes.SECTORS}
    res = compute(rows, ratios, sector_of)
    for rec in res.values():
        rec["sector_label"] = labels.get(rec["sector"], "Other / unclassified")
    doc = {"asof": asof, "fundamentals_date": dates[t], "built": datetime.now().isoformat(timespec="seconds"),
           "method": "peer multiples (P/E, P/S, P/B; sector median, 25th-75th percentile range); not validated", "stocks": res}
    tmp = OUT + ".tmp"
    json.dump(doc, open(tmp, "w"), separators=(",", ":"))
    os.replace(tmp, OUT)
    log(f"[fairvalue] {sum(1 for r in res.values() if 'fair' in r)} of {len(res)} stocks valued (as of {asof})")
    return doc


if __name__ == "__main__":
    update()
