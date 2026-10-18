"""
Point-in-time company fundamentals and industries for v6, streamed from SEC EDGAR (research only, read-only).

  fetch   ticker -> CIK map, then per company (every symbol that was ever in the v6 universe): company facts (only the
          fields below are kept) and submissions (SIC industry code, 10-Q/10-K filing dates); plus split history from
          Alpaca. Resumable; <= 5 requests/s (SEC fair-access limit is 10).
  panel   private/data/us/fundamentals.npz aligned to panel.npz: sic, sic2 [N] and point-in-time [T, N] features.

Point in time: each reported value is usable from the first trading day AFTER its filing date, and only its FIRST
filed version is used (later restatements would be look-ahead). Values older than 400 days are dropped.

The SEC User-Agent (it must contain a contact email) is read from private/config/sec.json {"user_agent": "..."}.

Usage: python -m trademaster_ml.fundamentals fetch [--max-age-days 7] | panel
"""
import argparse, gzip, json, os, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import numpy as np

from . import alpaca

DIR = os.path.join("private", "data", "us")
EDGAR = os.path.join(DIR, "edgar")
GAAP = {
    "ni": ["NetIncomeLoss"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "gp": ["GrossProfit"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"],
    "shares": ["CommonStockSharesOutstanding"],
}
DEI_SHARES = "EntityCommonStockSharesOutstanding"
FLOWS = ("ni", "rev", "gp", "cfo", "eps")
STALE_DAYS = 400
NAMES = ["earnings_yield", "book_to_market", "sales_to_price", "gross_profitability", "roe", "asset_growth", "accruals",
         "share_issuance", "leverage", "sue", "days_since_filing"]


# ---------------------------------------------------------------- fetch

class _Rate:
    def __init__(self, per_s):
        self.gap, self.lock, self.next = 1.0 / per_s, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def _ua():
    p = os.path.join("private", "config", "sec.json")
    if not os.path.exists(p):
        raise SystemExit("private/config/sec.json with {\"user_agent\": \"Name contact@email\"} is required by SEC")
    return json.load(open(p))["user_agent"]


def _sec(url, ua, rate, retries=5):
    for a in range(retries):
        rate.wait()
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429) or e.code >= 500:
                time.sleep(5 + 10 * a)
                continue
            raise
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError):
            time.sleep(5 + 10 * a)
    raise RuntimeError(f"SEC request failed after {retries} tries: {url}")


def _keep(facts):
    out = {}
    g = facts.get("facts", {}).get("us-gaap", {})
    for key, tags in GAAP.items():
        # companies switch tags over time (e.g. SalesRevenueNet -> RevenueFromContract... with ASC 606), so all
        # tags are merged; for the same period and filing the earlier-listed tag wins
        seen, merged = set(), []
        for tag in tags:
            units = g.get(tag, {}).get("units", {})
            rows = units.get("USD") or units.get("USD/shares") or units.get("shares") or []
            for x in rows:
                k = (x.get("start"), x["end"], x["filed"])
                if x.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "10-KT") and k not in seen:
                    seen.add(k)
                    merged.append([x.get("start"), x["end"], x["val"], x.get("form"), x["filed"]])
        if merged:
            out[key] = merged
    d = facts.get("facts", {}).get("dei", {}).get(DEI_SHARES, {}).get("units", {}).get("shares")
    if d:
        out["dei_shares"] = [[None, x["end"], x["val"], x.get("form"), x["filed"]] for x in d]
    return out


def universe_symbols():
    meta = json.load(open(os.path.join(DIR, "gpu", "meta.json")))
    T, N = meta["T"], meta["N"]
    mask = np.fromfile(os.path.join(DIR, "gpu", "mask.f32"), np.float32).reshape(T, N)
    ever = mask.max(0) > 0.5
    return [s for s, e in zip(meta["symbols"], ever) if e]


def fetch(max_age_days=None):
    ua, rate = _ua(), _Rate(5)
    os.makedirs(EDGAR, exist_ok=True)
    tick = _sec("https://www.sec.gov/files/company_tickers.json", ua, rate)
    cik_of = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in tick.values()}
    syms = universe_symbols()
    have = {s: cik_of[s] for s in syms if s in cik_of}
    json.dump(have, open(os.path.join(EDGAR, "cik_map.json"), "w"))
    print(f"{len(syms)} symbols were ever in the universe; {len(have)} have a current SEC CIK "
          f"({len(syms) - len(have)} without: mostly delisted or renamed)", flush=True)
    fresh = {int(f[:-5]) for f in os.listdir(EDGAR) if f[:-5].isdigit()
             and (max_age_days is None or time.time() - os.path.getmtime(os.path.join(EDGAR, f)) < max_age_days * 86400)}
    todo = sorted(set(have.values()) - fresh)                  # new companies, plus stale ones when refreshing
    t0, done = time.time(), [0]

    def one(cik):
        facts = _sec(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", ua, rate)
        sub = _sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua, rate)
        rec = {"cik": cik, "facts": _keep(facts) if facts else {}, "sic": None, "filings": []}
        if sub:
            rec["sic"] = sub.get("sic")
            f = sub.get("filings", {}).get("recent", {})
            rec["filings"] = [[d, fm] for d, fm in zip(f.get("filingDate", []), f.get("form", []))
                              if fm in ("10-Q", "10-K", "20-F", "40-F", "10-KT")]
            for extra in sub.get("filings", {}).get("files", []):        # older filings live in extra pages
                more = _sec("https://data.sec.gov/submissions/" + extra["name"], ua, rate)
                if more:
                    rec["filings"] += [[d, fm] for d, fm in zip(more.get("filingDate", []), more.get("form", []))
                                       if fm in ("10-Q", "10-K", "20-F", "40-F", "10-KT")]
        tmp = os.path.join(EDGAR, f"{cik}.json.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(EDGAR, f"{cik}.json"))
        done[0] += 1
        if done[0] % 100 == 0:
            el = (time.time() - t0) / 60
            print(f"  {done[0]}/{len(todo)} companies, {el:.1f} min, ~{el / done[0] * (len(todo) - done[0]):.0f} min left", flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, todo))
    # split history (Alpaca corporate actions) for the same symbols
    creds, splits = alpaca.credentials(), []
    for k in range(0, len(syms), 100):
        params = {"symbols": ",".join(syms[k:k + 100]), "types": "forward_split,reverse_split",
                  "start": "2016-01-01", "end": date.today().isoformat(), "limit": 1000}
        while True:
            d = alpaca._get("/v1/corporate-actions", params, creds)
            for kind in ("forward_splits", "reverse_splits"):
                for x in d.get("corporate_actions", {}).get(kind, []):
                    splits.append([x["symbol"], x["ex_date"], float(x["new_rate"]) / float(x["old_rate"])])
            if not d.get("next_page_token"):
                break
            params["page_token"] = d["next_page_token"]
    json.dump(splits, open(os.path.join(EDGAR, "splits.json"), "w"))
    print(f"fetch complete: {len(todo)} companies streamed, {len(splits)} splits", flush=True)


# ---------------------------------------------------------------- point-in-time series

def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def first_filed(rows):
    """{(start, end): (value, filed, form)} keeping the earliest filing of each period."""
    out = {}
    for start, end, val, form, filed in rows:
        k = (start, end)
        if k not in out or filed < out[k][1]:
            out[k] = (val, filed, form)
    return out


def quarters(rows):
    """Discrete fiscal-quarter values {end: (value, available_date)} from 3-month entries, or from differences of
    cumulative year-to-date entries sharing a start date (this also yields Q4 = FY - 9M)."""
    ff = first_filed(rows)
    q = {}
    by_start = {}
    for (start, end), (val, filed, _) in ff.items():
        if not start:
            continue
        days = (_d(end) - _d(start)).days
        if 80 <= days <= 100:
            q[end] = (val, filed)
        by_start.setdefault(start, []).append((end, val, filed, days))
    for start, xs in by_start.items():
        xs.sort()
        for (e0, v0, f0, d0), (e1, v1, f1, d1) in zip(xs, xs[1:]):
            gap = (_d(e1) - _d(e0)).days
            if 80 <= gap <= 100 and 80 <= d0 <= 380 and e1 not in q:
                q[e1] = (v1 - v0, max(f0, f1))
    return q


def ttm(q):
    """[(available, period_end, sum of the 4 consecutive quarters ending there)]."""
    ends = sorted(q)
    out = []
    for i in range(3, len(ends)):
        w = ends[i - 3:i + 1]
        if all(80 <= (_d(b) - _d(a)).days <= 100 for a, b in zip(w, w[1:])):
            out.append((max(q[e][1] for e in w), ends[i], sum(q[e][0] for e in w)))
    return out


def instants(rows):
    """[(available, period_end, value)] for balance-sheet / share-count facts."""
    return sorted((filed, end, val) for (start, end), (val, filed, _) in first_filed(rows).items())


def split_factor(splits, end):
    """Product of split ratios with ex-date after `end`: converts a count reported at `end` into today's share units."""
    f = 1.0
    for ex, r in splits:
        if ex > end:
            f *= r
    return f


def to_days(dates, events, stale=STALE_DAYS):
    """Point-in-time daily series from (available, period_end, value) events: value usable from the first trading
    day strictly after `available`, newest period wins, dropped when the period end is older than `stale` days."""
    T = len(dates)
    out = np.full(T, np.nan)
    end_at = np.full(T, -1e9)
    ds = np.array(dates)
    dnum = np.array([_d(x).toordinal() for x in dates])
    for avail, end, val in sorted(events):
        i = int(np.searchsorted(ds, avail, side="right"))
        if i >= T or val is None or not np.isfinite(val):
            continue
        e = _d(end).toordinal()
        upd = (end_at[i:] <= e)
        out[i:][upd] = val
        end_at[i:][upd] = e
    out[dnum - end_at > stale] = np.nan
    return out


def panel():
    p = np.load(os.path.join(DIR, "panel.npz"))
    dates = [str(d) for d in p["dates"]]
    syms = [str(s) for s in p["symbols"]]
    T, N = len(dates), len(syms)
    close = p["close"].astype(np.float64)
    cmap = json.load(open(os.path.join(EDGAR, "cik_map.json")))
    sp = {}
    spath = os.path.join(EDGAR, "splits.json")
    if not os.path.exists(spath):
        print("warning: no splits.json yet (fetch not finished): share counts are not split-adjusted", flush=True)
    for s, ex, r in json.load(open(spath)) if os.path.exists(spath) else []:
        sp.setdefault(s, []).append((ex, r))
    F = {k: np.full((T, N), np.nan, np.float32) for k in NAMES}
    sic = np.full(N, -1.0, np.float32)
    covered = 0
    dnum = np.array([_d(x).toordinal() for x in dates])
    for j, s in enumerate(syms):
        if s not in cmap:
            continue
        path = os.path.join(EDGAR, f"{cmap[s]}.json")
        if not os.path.exists(path):
            continue
        rec = json.load(open(path))
        if rec.get("sic"):
            sic[j] = float(rec["sic"])
        fx = rec["facts"]
        splits = sp.get(s, [])
        ser = {}
        for k in ("ni", "rev", "gp", "cfo"):
            if k in fx:
                ser[k] = to_days(dates, ttm(quarters(fx[k])))
        for k in ("assets", "equity", "liab"):
            if k in fx:
                ser[k] = to_days(dates, instants(fx[k]))
        sh_rows = fx.get("dei_shares") or fx.get("shares")
        if sh_rows:
            ev = [(a, e, v * split_factor(splits, e)) for a, e, v in instants(sh_rows)]
            ser["shares"] = to_days(dates, ev)
        if not ser:
            continue
        covered += 1
        nan = np.full(T, np.nan)
        mcap = close[:, j] * ser.get("shares", nan)
        mcap = np.where(mcap > 0, mcap, np.nan)
        assets = ser.get("assets", nan)
        assets_pos = np.where(assets > 0, assets, np.nan)
        eq = ser.get("equity", nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            F["earnings_yield"][:, j] = ser.get("ni", nan) / mcap
            F["book_to_market"][:, j] = eq / mcap
            F["sales_to_price"][:, j] = ser.get("rev", nan) / mcap
            F["gross_profitability"][:, j] = ser.get("gp", nan) / assets_pos
            F["roe"][:, j] = ser.get("ni", nan) / np.where(eq > 0, eq, np.nan)
            F["accruals"][:, j] = (ser.get("ni", nan) - ser.get("cfo", nan)) / assets_pos
            F["leverage"][:, j] = ser.get("liab", nan) / assets_pos
            # vs the point-in-time value one year (252 trading days) earlier
            lag = np.clip(np.arange(T) - 252, 0, None)
            F["asset_growth"][:, j] = np.where(np.arange(T) >= 252, assets / assets_pos[lag] - 1, np.nan)
            sh = ser.get("shares", nan)
            F["share_issuance"][:, j] = np.where(np.arange(T) >= 252, np.log(sh / np.where(sh > 0, sh, np.nan)[lag]), np.nan)
        # standardized unexpected earnings from split-adjusted quarterly EPS: (q - same quarter last year) / std of
        # the previous 8 such changes; available from the filing of quarter q
        if "eps" in fx:
            q = quarters(fx["eps"])
            ends = sorted(q)
            adj = {e: (q[e][0] / split_factor(splits, e), q[e][1]) for e in ends}
            ch = []
            ev = []
            for i, e in enumerate(ends):
                prev = [x for x in ends[:i] if 350 <= (_d(e) - _d(x)).days <= 380]
                if not prev:
                    continue
                c = adj[e][0] - adj[prev[-1]][0]
                if len(ch) >= 4:
                    sd = np.std(ch[-8:])
                    if sd > 0:
                        ev.append((adj[e][1], e, c / sd))
                ch.append(c)
            F["sue"][:, j] = to_days(dates, ev, stale=200)
        fil = sorted(d for d, _ in rec.get("filings", []))
        if fil:
            idx = np.searchsorted(np.array(fil), np.array(dates), side="left") - 1     # last filing before day t
            last = np.array([_d(fil[i]).toordinal() if i >= 0 else np.nan for i in idx], float)
            F["days_since_filing"][:, j] = np.where(dnum - last <= STALE_DAYS, dnum - last, np.nan)
    for k in F:
        F[k][~np.isfinite(F[k])] = np.nan
    sic2 = np.where(sic >= 0, np.floor(sic / 100), -1).astype(np.float32)
    np.savez(os.path.join(DIR, "fundamentals.npz"), dates=np.array(dates), symbols=np.array(syms), sic=sic, sic2=sic2, **F)
    uni = universe_symbols()
    ui = np.array([s in set(uni) for s in syms])
    print(f"fundamentals: {covered} of {len(uni)} ever-in-universe symbols covered; SIC known for {int((sic[ui] >= 0).sum())}", flush=True)
    meta = json.load(open(os.path.join(DIR, "gpu", "meta.json")))
    mask = np.fromfile(os.path.join(DIR, "gpu", "mask.f32"), np.float32).reshape(meta["T"], meta["N"]) > 0.5
    for k in NAMES:
        print(f"  {k:<20} available on {np.isfinite(F[k][mask]).mean():.0%} of universe stock-days", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "panel"])
    ap.add_argument("--max-age-days", type=float, default=None, help="fetch: also re-fetch companies older than this")
    a = ap.parse_args()
    fetch(a.max_age_days) if a.cmd == "fetch" else panel()


if __name__ == "__main__":
    main()
