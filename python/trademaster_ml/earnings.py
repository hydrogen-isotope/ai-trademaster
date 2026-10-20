"""
Earnings report dates for v7 amendment B (event-aware ranges). Research only, read-only data from Eulerpool.

  fetch      per stock (every symbol ever in the NASDAQ universe): calendar/earnings-by-symbol/{symbol}
             -> private/data/earnings/by_symbol/{SYMBOL}.json (re-fetched when older than --max-age-days)
  upcoming   the next weeks' calendar (calendar/earnings/{date} returns Monday-Friday of that week)
             -> private/data/earnings/upcoming.json

Reaction day (amendment B, as clarified): the report date for before-open or during-market reports, the next trading
day for after-close reports, and either of the two for reports without a time (94% of them). Per stock and fiscal
quarter, one date: the one with a reported EPS if any, else the weekly calendar's date, else the earliest.

Usage: python -m trademaster_ml.earnings fetch [--max-age-days 30] | upcoming
"""
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np

from . import eulerpool, holidays

DIR = os.path.join("private", "data", "earnings")
BY_SYM = os.path.join(DIR, "by_symbol")
UPCOMING = os.path.join(DIR, "upcoming.json")
AFTER = {"amc", "time-after-hours"}
BEFORE = {"bmo", "time-pre-market", "dmh"}
UPCOMING_WEEKS = 5


def _get(path):
    for a in range(4):
        st, d = eulerpool.call("GET", path)
        if st == 200:
            return d
        if st == 404:
            return []
        time.sleep(2 + 3 * a)
    raise RuntimeError(f"Eulerpool {path}: HTTP {st}")


def fetch(max_age_days=30, symbols=None, log=print):
    from . import fundamentals
    os.makedirs(BY_SYM, exist_ok=True)
    syms = symbols or fundamentals.universe_symbols()
    now = time.time()
    todo = [s for s in syms if not os.path.exists(p := os.path.join(BY_SYM, f"{s}.json")) or now - os.path.getmtime(p) > max_age_days * 86400]
    log(f"[earnings] {len(syms) - len(todo)} up to date, {len(todo)} to fetch")
    failed = []

    def one(s):
        try:
            rows = _get(f"calendar/earnings-by-symbol/{s}")
            keep = [{k: r.get(k) for k in ("date", "hour", "year", "quarter", "epsActual")} for r in rows if isinstance(r, dict) and r.get("date")]
            tmp = os.path.join(BY_SYM, f"{s}.json.tmp")
            json.dump(keep, open(tmp, "w"))
            os.replace(tmp, os.path.join(BY_SYM, f"{s}.json"))
        except Exception as e:  # noqa: BLE001 - one stock must not stop the rest
            failed.append(s)
            log(f"[earnings] {s}: {type(e).__name__}: {str(e)[:120]}")

    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 250 == 0 and i:
                log(f"[earnings] {i}/{len(todo)}")
    log(f"[earnings] fetch done ({len(failed)} failed)")
    return failed


def upcoming(log=print, today=None):
    today = today or date.today()
    rows = []
    for w in range(UPCOMING_WEEKS):
        d = today + timedelta(days=7 * w)
        rows += [{"symbol": r.get("symbol"), "date": r.get("date")} for r in _get(f"calendar/earnings/{d}") if isinstance(r, dict)]
    os.makedirs(DIR, exist_ok=True)
    json.dump({"fetched": datetime.now().isoformat(timespec="seconds"), "rows": rows}, open(UPCOMING, "w"))
    log(f"[earnings] upcoming: {len(rows)} reports over {UPCOMING_WEEKS} weeks")
    return rows


def report_dates(sym, upcoming_rows=None):
    """[(report date, timing)] for one stock, timing in {"before", "after", "unknown"}: one per fiscal quarter."""
    p = os.path.join(BY_SYM, f"{sym}.json")
    rows = json.load(open(p)) if os.path.exists(p) else []
    cal = {r["date"] for r in (upcoming_rows or []) if r.get("symbol") == sym}
    by_q = {}
    for r in rows:
        by_q.setdefault((r.get("year"), r.get("quarter")) if r.get("year") else r["date"], []).append(r)
    out = []
    for lst in by_q.values():
        pick = [r for r in lst if r.get("epsActual") is not None] or [r for r in lst if r["date"] in cal] or sorted(lst, key=lambda r: r["date"])
        r = pick[0]
        hr = r.get("hour") or ""
        out.append((r["date"], "after" if hr in AFTER else "before" if hr in BEFORE else "unknown"))
    for d in cal:                                              # in the weekly calendar but not yet in the stock's list
        if not any(abs((date.fromisoformat(d) - date.fromisoformat(x)).days) < 20 for x, _ in out):
            out.append((d, "unknown"))
    return sorted(set(out))


def reaction_events(dates, symbols, future=0):
    """Earnings events as rows of (first, last, stock): the first and last trading-day index at which the report can be
    priced in (equal when the time is known). `future` extra days are the next US trading days after dates[-1], for
    live forecasts; indices refer to dates + those days."""
    extra = holidays.next_sessions(dates[-1], future) if future else []
    all_days = list(dates) + extra
    idx = {d: i for i, d in enumerate(all_days)}
    up = json.load(open(UPCOMING))["rows"] if os.path.exists(UPCOMING) else []
    ev = []
    for j, s in enumerate(symbols):
        for d, timing in report_dates(s, up):
            if d < all_days[0] or d > all_days[-1]:
                continue
            i = idx.get(d)
            if i is None:                                       # report on a non-trading day: priced in at the next session
                i = next((k for k, x in enumerate(all_days) if x > d), None)
                if i is None:
                    continue
                first = last = i
            else:
                first = i + 1 if timing == "after" else i
                last = i + 1 if timing in ("after", "unknown") else i
            last = min(last, len(all_days) - 1)
            if first < len(all_days):
                ev.append((first, last, j))
    return np.array(ev, dtype=np.int64).reshape(-1, 3), len(all_days)


def window_counts(ev, n_days, n_stocks, t, n, h):
    """Number of events of stock n[k] that can be priced in within (t[k], t[k] + h]."""
    A = np.zeros((n_days + 1, n_stocks))
    B = np.zeros((n_days + 1, n_stocks))
    np.add.at(A, (ev[:, 0] + 1, ev[:, 2]), 1)
    np.add.at(B, (ev[:, 1] + 1, ev[:, 2]), 1)
    A, B = np.cumsum(A, axis=0), np.cumsum(B, axis=0)
    hi = np.minimum(t + h + 1, n_days)
    return A[hi, n] - B[t + 1, n]                              # first <= t + h  and  last >= t + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "upcoming"])
    ap.add_argument("--max-age-days", type=float, default=30)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.max_age_days)
    else:
        upcoming()


if __name__ == "__main__":
    main()
