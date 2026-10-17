"""
Databento Historical API (licensed CME Globex data, GLBX.MDP3) for daily futures settlements and open interest.
Plain HTTPS + JSON (no extra package). Key: env DATABENTO_API_KEY or private/config/databento.json {"api_key"}.
The key is sent only to hist.databento.com and never printed.

Every download first asks Databento for its price (metadata.get_cost) and refuses to spend more than
`max_cost_usd` (default $1) unless told otherwise. Databento bills usage against the account's credit.

Usage:
  python -m trademaster_ml.databento_client --roots GC SI --start 2019-01-01            # quote the cost only
  python -m trademaster_ml.databento_client --roots GC SI --start 2019-01-01 --download # fetch (if under the cap)
"""
import argparse, base64, json, os, urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from .futures import SPECS, continuous_series

BASE = "https://hist.databento.com/v0"
DATASET = "GLBX.MDP3"
STAT_SETTLEMENT, STAT_OPEN_INTEREST = 3, 9
FLAG_FINAL, FLAG_INTRADAY = 1, 8
OUT = os.path.join("private", "data", "futures")


class DatabentoConfigError(RuntimeError):
    pass


def api_key(root="."):
    k = os.environ.get("DATABENTO_API_KEY")
    p = os.path.join(root, "private", "config", "databento.json")
    if not k and os.path.exists(p):
        k = json.load(open(p, encoding="utf-8")).get("api_key")
    if not k:
        raise DatabentoConfigError("no Databento key: set DATABENTO_API_KEY or create private/config/databento.json "
                                   '{"api_key": "..."} (see docs/FUTURES.md)')
    return k


def _post(endpoint, form, key):
    data = urllib.parse.urlencode(form).encode()
    auth = base64.b64encode(f"{key}:".encode()).decode()
    req = urllib.request.Request(f"{BASE}/{endpoint}", data=data, headers={"Authorization": "Basic " + auth})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Databento HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}") from None


def _request(roots, start, end):
    return {"dataset": DATASET, "symbols": ",".join(f"{r}.FUT" for r in roots), "stype_in": "parent",
            "schema": "statistics", "start": start, "end": end}


def cost(roots, start, end, key):
    return float(_post("metadata.get_cost", _request(roots, start, end), key))


def fetch_statistics(roots, start, end, key):
    form = _request(roots, start, end) | {"encoding": "json", "pretty_px": "true", "pretty_ts": "true", "map_symbols": "true"}
    text = _post("timeseries.get_range", form, key)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_records(records):
    """-> settles {symbol: {date: price}}, oi {symbol: {date: qty}} using only FINAL, end-of-day settlements and the
    last open-interest message per trading date (CME re-publishes; the last one is authoritative)."""
    settles, oi = defaultdict(dict), defaultdict(dict)
    order = defaultdict(dict)
    for r in records:
        sym = r.get("symbol")
        if not sym or "-" in sym or " " in sym:          # outrights only (no spreads / options)
            continue
        st, ref = r.get("stat_type"), r.get("ts_ref")
        if ref is None or str(ref).startswith("18446744073709551615"):
            continue
        d = date.fromisoformat(str(ref)[:10])
        seq = (str(r.get("ts_recv")), r.get("sequence", 0))
        if st == STAT_SETTLEMENT:
            flags = int(r.get("stat_flags", 0))
            if not flags & FLAG_FINAL or flags & FLAG_INTRADAY:
                continue
            key = ("s", sym, d)
            if key not in order or seq > order[key]:
                order[key] = seq
                settles[sym][d] = float(r["price"])
        elif st == STAT_OPEN_INTEREST:
            q = r.get("quantity")
            if q is None or int(q) >= 2 ** 63 - 1:
                continue
            key = ("o", sym, d)
            if key not in order or seq > order[key]:
                order[key] = seq
                oi[sym][d] = int(q)
    return dict(settles), dict(oi)


def save_continuous(root, settles, oi, out=OUT):
    rows = continuous_series({s: v for s, v in settles.items() if s.startswith(root)},
                             {s: v for s, v in oi.items() if s.startswith(root)})
    os.makedirs(out, exist_ok=True)
    doc = {"root": root, "spec_source": SPECS[root].source, "price": "official daily settlement (final)",
           "adjustment": "ratio back-adjusted at open-interest rolls", "source": f"Databento {DATASET} statistics",
           "dates": [str(r["date"]) for r in rows], "close": [r["adj"] for r in rows], "settle": [r["settle"] for r in rows],
           "contract": [r["contract"] for r in rows], "rolled": [r["rolled"] for r in rows],
           "built_at": datetime.now(timezone.utc).isoformat()}
    tmp = os.path.join(out, f"{root}.json.tmp")
    json.dump(doc, open(tmp, "w"))
    os.replace(tmp, os.path.join(out, f"{root}.json"))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["GC", "SI"])
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-cost-usd", type=float, default=1.0)
    ap.add_argument("--incremental", action="store_true",
                    help="only fetch from 14 days before the newest saved settlement and merge (daily use)")
    a = ap.parse_args()
    try:
        key = api_key()
    except DatabentoConfigError as e:
        raise SystemExit(str(e))
    raw_path = os.path.join(OUT, "raw_statistics.json")
    old = json.load(open(raw_path)) if os.path.exists(raw_path) else None
    if a.incremental and old and old["settles"]:
        newest = max(d for v in old["settles"].values() for d in v)
        a.start = str(date.fromisoformat(newest) - timedelta(days=14))
    c = cost(a.roots, a.start, a.end, key)
    print(f"Databento quote for {','.join(a.roots)} statistics {a.start}..{a.end}: ${c:.4f}")
    if not a.download:
        print("quote only; add --download to fetch")
        return
    if c > a.max_cost_usd:
        raise SystemExit(f"refusing: ${c:.2f} is above --max-cost-usd {a.max_cost_usd}")
    recs = fetch_statistics(a.roots, a.start, a.end, key)
    os.makedirs(OUT, exist_ok=True)
    settles, oi = parse_records(recs)
    if a.incremental and old:                                   # merge: new values win for overlapping dates
        for name, new in (("settles", settles), ("open_interest", oi)):
            for sym, v in old[name].items():
                merged = {date.fromisoformat(d): x for d, x in v.items()}
                merged.update(new.get(sym, {}))
                new[sym] = merged
    json.dump({"settles": {s: {str(d): p for d, p in v.items()} for s, v in settles.items()},
               "open_interest": {s: {str(d): q for d, q in v.items()} for s, v in oi.items()}},
              open(os.path.join(OUT, "raw_statistics.json"), "w"))
    for root in a.roots:
        doc = save_continuous(root, settles, oi)
        rolls = sum(doc["rolled"])
        print(f"{root}: {len(doc['dates'])} settlement days {doc['dates'][0]}..{doc['dates'][-1]}, {rolls} rolls, "
              f"last {doc['contract'][-1]} {doc['settle'][-1]}")


if __name__ == "__main__":
    main()
