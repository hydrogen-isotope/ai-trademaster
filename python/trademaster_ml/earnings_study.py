"""Earnings reaction study, data part (rules in docs/EARNINGS-STUDY-PROTOCOL.md; research only, no orders).

build()    -> event table (dict of numpy columns) saved to private/reports/earnings_study/events.csv
evaluate() -> statistics of one candidate rule on one period; the hold-out is refused unless allow_holdout=True.
CLI: python -m trademaster_ml.earnings_study build | dev      (dev = development period only)

Implementation choices where the protocol is silent: sigma needs >= 50 valid daily returns among the 60; the 20-day
median dollar volume needs >= 15 valid days; the day-0 return used by H3 rule (b) is the raw stock return
close_{R-1} -> close_R; holding-window prices means every close from R to the exit close is present for the stock
and for QQQ; one event per (symbol, reaction day).
"""
import csv, glob, json, os, sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EARN_DIR = os.path.join(ROOT, "private", "data", "earnings", "by_symbol")
PANEL = os.path.join(ROOT, "private", "data", "us", "panel.npz")
OUT_DIR = os.path.join(ROOT, "private", "reports", "earnings_study")

DEV_START, DEV_END = "2016-01-04", "2022-12-30"
HOLD_START, HOLD_END = "2023-01-03", "2026-10-16"
AFTER = {"amc", "time-after-hours"}
BEFORE = {"bmo", "time-pre-market", "dmh"}
COST_OPEN, COST_CLOSE, COST_LATER = 10e-4, 5e-4, 3e-4        # one-way, per leg

NUM_COLS = ["gap", "sigma", "gap_size", "surprise", "r_o2c", "q_o2c", "r_c5", "q_c5", "r_c20", "q_c20", "r_d0", "q_d0"]
STR_COLS = ["symbol", "report_date", "reaction_date", "hour", "period", "quarter"]
INT_COLS = ["known_time", "tercile"]          # tercile: 0 bottom, 1 middle, 2 top, -1 undefined


def timing_of(hour):
    h = (hour or "").strip().lower()
    return "after" if h in AFTER else "before" if h in BEFORE else "unknown"


def reaction_index(dates, rep, timing, abs_gap=None):
    """Index of reaction day R in sorted `dates`. rep = report date string. before: first session >= rep; after: first
    session > rep; unknown: whichever of (first session >= rep, the next session) has the larger |gap| (abs_gap(i))."""
    i0 = int(np.searchsorted(dates, rep, side="left"))              # first session >= rep
    i1 = int(np.searchsorted(dates, rep, side="right"))             # first session > rep
    if timing == "before":
        i = i0
    elif timing == "after":
        i = i1
    else:
        if i0 >= len(dates):
            return None
        i1u = i0 + 1
        if i1u >= len(dates):
            return i0
        g0, g1 = abs_gap(i0), abs_gap(i1u)
        if not np.isfinite(g0):
            return i1u if np.isfinite(g1) else i0
        if not np.isfinite(g1):
            return i0
        return i0 if g0 >= g1 else i1u
    return i if i < len(dates) else None


def surprise_of(actual, est):
    return (actual - est) / max(abs(est), 0.05)


def quarter_label(d):
    y, m = int(d[:4]), int(d[5:7])
    return "%d-Q%d" % (y, (m - 1) // 3 + 1)


def period_of(rdate):
    if DEV_START <= rdate <= DEV_END:
        return "dev"
    if HOLD_START <= rdate <= HOLD_END:
        return "holdout"
    return None


def _logret(a, b):
    with np.errstate(all="ignore"):
        return float(np.log(b / a)) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan


def assign_terciles(ev):
    """Surprise terciles. Cut-offs (33.3 / 66.7 percentiles) per calendar quarter from DEVELOPMENT events only.
    Hold-out events (and any quarter without development events) use the cut-offs of the LAST development quarter.
    Returns the cut-off table {quarter: [c1, c2]} and the last-quarter cut-offs; fills ev['tercile']
    (0 bottom, 2 top, 1 middle, -1 undefined)."""
    q, per, s = ev["quarter"], ev["period"], ev["surprise"]
    cuts = {}
    for qq in sorted(set(q[per == "dev"])):
        m = (per == "dev") & (q == qq) & np.isfinite(s)
        if m.sum() >= 3:
            cuts[qq] = [float(np.quantile(s[m], 1 / 3)), float(np.quantile(s[m], 2 / 3))]
    last = cuts[sorted(cuts)[-1]]
    ter = np.full(len(s), -1, dtype=int)
    for i in range(len(s)):
        if not np.isfinite(s[i]):
            continue
        c = cuts.get(q[i], last) if per[i] == "dev" else last
        ter[i] = 0 if s[i] < c[0] else 2 if s[i] > c[1] else 1
    ev["tercile"] = ter
    return cuts, last


def build(verbose=True):
    z = np.load(PANEL, allow_pickle=True)
    dates = np.array([str(d) for d in z["dates"]])
    syms = [str(s) for s in z["symbols"]]
    O, C, V = z["open"].astype(float), z["close"].astype(float), z["volume"].astype(float)
    col = {s: j for j, s in enumerate(syms)}
    qj = col["QQQ"]
    qo, qc = O[:, qj], C[:, qj]
    T = len(dates)
    rows = []
    counts = dict(raw_reports=0, no_price_symbol=0, no_reaction=0, outside_periods=0, fail_price_or_volume=0,
                  fail_sigma=0, duplicate=0, eligible=0)
    for path in sorted(glob.glob(os.path.join(EARN_DIR, "*.json"))):
        sym = os.path.basename(path)[:-5]
        try:
            recs = json.load(open(path))
        except Exception:
            continue
        j = col.get(sym)
        if j is None:
            counts["no_price_symbol"] += 1
            continue
        o, c, v = O[:, j], C[:, j], V[:, j]
        seen = set()
        for r in recs:
            a, e = r.get("epsActual"), r.get("epsEstimate")
            if a is None or e is None or not r.get("date"):
                continue
            counts["raw_reports"] += 1
            tm = timing_of(r.get("hour"))

            def ag(i):
                return abs(_logret(c[i - 1], o[i])) if i >= 1 else np.nan
            R = reaction_index(dates, r["date"], tm, ag) if r["date"] >= dates[0] else None
            if R is not None and (np.datetime64(dates[R]) - np.datetime64(r["date"])).astype(int) > 7:
                R = None                      # report date not matched to a nearby session
            if R is None:
                counts["no_reaction"] += 1
                continue
            per = period_of(dates[R])
            if per is None:
                counts["outside_periods"] += 1
                continue
            if R < 61 or not (np.isfinite(o[R]) and np.isfinite(c[R]) and np.isfinite(c[R - 1])):
                counts["fail_price_or_volume"] += 1
                continue
            dv = c[R - 20:R] * v[R - 20:R]
            if c[R - 1] < 5 or np.isfinite(dv).sum() < 15 or np.nanmedian(dv) < 5e6:
                counts["fail_price_or_volume"] += 1
                continue
            lr = np.diff(np.log(c[R - 61:R]))
            lr = lr[np.isfinite(lr)]
            if len(lr) < 50 or np.std(lr, ddof=1) <= 0:
                counts["fail_sigma"] += 1
                continue
            if (sym, R) in seen:
                counts["duplicate"] += 1
                continue
            seen.add((sym, R))
            sigma = float(np.std(lr, ddof=1))
            gap = _logret(c[R - 1], o[R])
            row = dict(symbol=sym, report_date=r["date"], reaction_date=dates[R], hour=str(r.get("hour") or ""),
                       period=per, quarter=quarter_label(dates[R]), known_time=int(tm != "unknown"),
                       gap=gap, sigma=sigma, gap_size=abs(gap) / sigma, surprise=surprise_of(float(a), float(e)),
                       r_o2c=_logret(o[R], c[R]), q_o2c=_logret(qo[R], qc[R]),
                       r_d0=_logret(c[R - 1], c[R]), q_d0=_logret(qc[R - 1], qc[R]))
            for k in (5, 20):
                end = R + k
                if end < T and np.isfinite(c[R:end + 1]).all() and np.isfinite(qc[R:end + 1]).all():
                    row["r_c%d" % k], row["q_c%d" % k] = _logret(c[R], c[end]), _logret(qc[R], qc[end])
                else:
                    row["r_c%d" % k] = row["q_c%d" % k] = np.nan
            rows.append(row)
            counts["eligible"] += 1
    ev = {k: np.array([r[k] for r in rows], dtype=float) for k in NUM_COLS}
    for k in STR_COLS:
        ev[k] = np.array([r[k] for r in rows], dtype=object)
    ev["known_time"] = np.array([r["known_time"] for r in rows], dtype=int)
    cuts, last = assign_terciles(ev)
    n = len(rows)
    summary = dict(counts=counts, n_events=n, n_dev=int((ev["period"] == "dev").sum()),
                   n_holdout=int((ev["period"] == "holdout").sum()),
                   n_known_time=int(ev["known_time"].sum()), known_time_share=float(ev["known_time"].mean()),
                   tercile_cuts_rule="per calendar quarter, development events only; hold-out events use the cut-offs "
                                     "of the last development quarter",
                   last_dev_quarter=sorted(cuts)[-1], holdout_cuts=last, n_dev_quarters=len(cuts),
                   window_available={k: int(np.isfinite(ev[k]).sum()) for k in ("r_o2c", "r_d0", "r_c5", "r_c20")})
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = STR_COLS + INT_COLS + NUM_COLS
    with open(os.path.join(OUT_DIR, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([ev[k][i] if (k in STR_COLS or k in INT_COLS) else ("" if np.isnan(ev[k][i]) else "%.8g" % ev[k][i])
                        for k in cols])
    json.dump(summary, open(os.path.join(OUT_DIR, "events_summary.json"), "w"), indent=1)
    if verbose:
        print(json.dumps(summary, indent=1))
    return ev, summary


def load_events():
    with open(os.path.join(OUT_DIR, "events.csv")) as f:
        rd = list(csv.DictReader(f))
    ev = {}
    for k in STR_COLS:
        ev[k] = np.array([r[k] for r in rd], dtype=object)
    for k in INT_COLS:
        ev[k] = np.array([int(r[k]) for r in rd], dtype=int)
    for k in NUM_COLS:
        ev[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rd])
    return ev


# ---- rules -------------------------------------------------------------------------------------------------------

def _rule(rid, kind, **kw):
    return dict(id=rid, kind=kind, **kw)


RULES = [_rule("H1_cont_g1", "H1", d=1, g=1), _rule("H1_cont_g2", "H1", d=1, g=2), _rule("H1_cont_g3", "H1", d=1, g=3),
         _rule("H1_rev_g1", "H1", d=-1, g=1), _rule("H1_rev_g2", "H1", d=-1, g=2), _rule("H1_rev_g3", "H1", d=-1, g=3),
         _rule("H2_a", "H2", agree=False), _rule("H2_b", "H2", agree=True),
         _rule("H3_k5_a", "H3", k=5, agree=False), _rule("H3_k5_b", "H3", k=5, agree=True),
         _rule("H3_k20_a", "H3", k=20, agree=False), _rule("H3_k20_b", "H3", k=20, agree=True)]
RULE_BY_ID = {r["id"]: r for r in RULES}


def rule_trades(ev, rule):
    """Per event: sign (+1/-1/0 = no trade), raw log return, QQQ log return, one-way cost sum (log units)."""
    if isinstance(rule, str):
        rule = RULE_BY_ID[rule]
    gap, gs, sur, ter = ev["gap"], ev["gap_size"], ev["surprise"], ev["tercile"]
    with np.errstate(invalid="ignore"):
        sg, ss = np.sign(gap), np.sign(sur)
        extreme = (ter == 0) | (ter == 2)
        if rule["kind"] == "H1":
            sign = np.where(gs >= rule["g"], rule["d"] * sg, 0.0)
        elif rule["kind"] == "H2":
            sign = np.where(extreme, ss, 0.0)
            if rule["agree"]:
                sign = np.where(sg == ss, sign, 0.0)
        else:
            sign = np.where(extreme, ss, 0.0)
            if rule["agree"]:
                sign = np.where(np.sign(ev["r_d0"]) == ss, sign, 0.0)
    sign = np.nan_to_num(sign, nan=0.0)
    if rule["kind"] in ("H1", "H2"):
        r, q, cost = ev["r_o2c"], ev["q_o2c"], COST_OPEN + COST_CLOSE
    else:
        r, q, cost = ev["r_c%d" % rule["k"]], ev["q_c%d" % rule["k"]], COST_CLOSE + COST_LATER
    return sign, r, q, cost


def evaluate(events, rule, period, cost_mult=1.0, long_only=False, known_time_only=False, allow_holdout=False):
    if period != "dev" and not allow_holdout:
        raise ValueError("hold-out is closed: evaluate() on period %r needs allow_holdout=True" % (period,))
    if period not in ("dev", "holdout"):
        raise ValueError("period must be 'dev' or 'holdout'")
    sign, r, q, cost = rule_trades(events, rule)
    m = (events["period"] == period) & (sign != 0) & np.isfinite(r) & np.isfinite(q)
    if long_only:
        m &= sign > 0
    if known_time_only:
        m &= events["known_time"] == 1
    n = int(m.sum())
    if n == 0:
        return dict(n_trades=0, mean_net_adj=np.nan, mean_gross_adj=np.nan, mean_net_raw=np.nan, hit_rate=np.nan,
                    t_clustered=np.nan, n_days=0)
    s = sign[m]
    gross_adj = s * (r[m] - q[m])
    c = cost * cost_mult
    net_adj = gross_adj - c
    net_raw = s * r[m] - c
    ud, inv = np.unique(events["reaction_date"][m].astype(str), return_inverse=True)
    day_mean = np.bincount(inv, weights=net_adj) / np.bincount(inv)
    nd = len(ud)
    if nd > 1 and np.std(day_mean, ddof=1) > 0:
        t = float(np.mean(day_mean) / (np.std(day_mean, ddof=1) / np.sqrt(nd)))
    else:
        t = np.nan
    return dict(n_trades=n, mean_net_adj=float(net_adj.mean()), mean_gross_adj=float(gross_adj.mean()),
                mean_net_raw=float(net_raw.mean()), hit_rate=float((net_adj > 0).mean()), t_clustered=t, n_days=int(nd))


def dev_table(ev):
    """Development-only table of all 12 rules: all / long-only / 2x cost / known-time-only variants."""
    out = {}
    for rl in RULES:
        out[rl["id"]] = dict(
            all=evaluate(ev, rl, "dev"), long_only=evaluate(ev, rl, "dev", long_only=True),
            cost2x=evaluate(ev, rl, "dev", cost_mult=2.0),
            cost2x_long_only=evaluate(ev, rl, "dev", cost_mult=2.0, long_only=True),
            known_time_only=evaluate(ev, rl, "dev", known_time_only=True))
    return out


def _fmt(d):
    if not d["n_trades"]:
        return "n=0"
    return "n=%6d days=%4d net_adj=%+7.1fbp gross=%+7.1fbp netraw=%+7.1fbp hit=%.3f t=%+5.2f" % (
        d["n_trades"], d["n_days"], 1e4 * d["mean_net_adj"], 1e4 * d["mean_gross_adj"], 1e4 * d["mean_net_raw"],
        d["hit_rate"], d["t_clustered"])


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "dev":
        ev = load_events()
        keep = ev["period"] == "dev"
        ev = {k: v[keep] for k, v in ev.items()}      # development events only
        tab = dev_table(ev)
        for rid, t in tab.items():
            print(rid)
            for k, d in t.items():
                print("  %-17s %s" % (k, _fmt(d)))
        json.dump(tab, open(os.path.join(OUT_DIR, "dev_all.json"), "w"), indent=1)
        print("saved", os.path.join(OUT_DIR, "dev_all.json"))
    else:
        print("usage: python -m trademaster_ml.earnings_study build | dev")


if __name__ == "__main__":
    main(sys.argv)
