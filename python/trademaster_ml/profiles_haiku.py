"""
v8 amendment I: company profiles researched by Claude Haiku subagents (web + filing excerpts), checked here.

The research itself is run from Claude Code (Haiku subagents write private/data/profiles_haiku/<market>/<SYM>.json from
the inputs in private/data/profiles_haiku/input/). This module checks each one with the same rules as the Qwen
profiles and writes <SYM>.checked.json, which variant B then prefers:
  - size limits, categories, the company itself, duplicates, strict ticker matching (profiles.tidy);
  - recency: a relationship whose source is dated before 2023 is dropped;
  - companies whose shares exist but no longer trade are dropped (Eulerpool market data);
  - the revenue split comes from Eulerpool segment data (from the input), never from the research text.

Usage: python -m trademaster_ml.profiles_haiku check
"""
import glob, json, os, re, sys
from datetime import datetime

from . import profiles as P

DIR = os.path.join("private", "data", "profiles_haiku")
MIN_YEAR = 2023
MAX_AGE_DAYS = 180


def _year(s):
    m = re.search(r"(19|20)\d\d", str(s or ""))
    return int(m.group(0)) if m else None


def check_one(path, tickers):
    raw = json.load(open(path, encoding="utf-8"))
    mk, sym, name = raw.get("market"), raw.get("symbol"), raw.get("name", "")
    inp = os.path.join(DIR, "input", f"{sym}.json")
    inp = json.load(open(inp, encoding="utf-8")) if os.path.exists(inp) else {}
    pr = P.clean(raw)
    stale = []
    for k in ("customers", "suppliers", "competitors"):
        src = {e.get("name"): e for e in raw.get(k, []) if isinstance(e, dict)}
        keep = []
        for e in pr[k]:
            r = src.get(e["name"]) or {}
            y = _year(r.get("source_date"))
            if y is not None and y < MIN_YEAR:
                stale.append(e["name"])
                continue
            e.update({"evidence": str(r.get("evidence", ""))[:200], "source_url": str(r.get("source_url", ""))[:300],
                      "source_date": str(r.get("source_date", ""))[:10], "ticker": str(r.get("ticker") or "")[:10]})
            keep.append(e)
        pr[k] = keep
    have = {(P._norm(e["name"]) or [""])[0] for e in pr["competitors"]}     # amendment H: Eulerpool peers added
    for x in inp.get("peers") or []:
        if (P._norm(x.get("name")) or [""])[0] not in have:
            pr["competitors"].append({"name": x["name"], "ticker": x.get("ticker") or "", "evidence": "listed as a peer by Eulerpool",
                                      "source_url": "", "source_date": "", "from": "peer list"})
    P.tidy(pr, sym, name, "", tickers)
    pr["competitors"] = pr["competitors"][:10]                             # researched rivals first, then peers
    pr["removed"]["stale_source"] = stale
    for k in ("customers", "suppliers", "competitors"):
        for e in list(pr[k]):
            if not e.get("ticker") and P.market_status(e["name"]) == "delisted":
                pr["removed"]["no_longer_exist"].append(e["name"])
                pr[k].remove(e)
    seg = inp.get("segments")
    if seg:
        for x in pr.get("products", []):
            x["revenue_share_pct"] = 0                                     # the real split is shown from segment data
    filing = (inp.get("filing") or {}).get("source") or {}
    pr.update({"symbol": sym, "market": mk, "name": name, "segments": seg, "mapped_v3": True,
               "source": {"kind": "Claude Haiku research (filing + web)", "filed": filing.get("filed"), "url": filing.get("url"),
                          "filing_kind": filing.get("kind")},
               "notes": str(raw.get("notes", ""))[:600], "searches_used": raw.get("searches_used"),
               "built": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")})
    out = path[:-5] + ".checked.json"
    json.dump(pr, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return pr


def check_all(log=print):
    tickers = P.sec_tickers()
    n = 0
    for f in sorted(glob.glob(os.path.join(DIR, "*", "*.json"))):
        if any(os.sep + d + os.sep in f for d in ("input", "batches", "pilot_extra")) or f.endswith(".checked.json") or os.path.basename(f) == "todo.json":
            continue
        try:
            check_one(f, tickers)
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the rest
            log(f"[haiku] {f}: {type(e).__name__}: {e}")
    log(f"[haiku] checked {n} profiles")
    return n


def load(market, sym):
    """The checked Haiku profile, or None if missing or older than MAX_AGE_DAYS."""
    p = os.path.join(DIR, market, f"{sym}.checked.json")
    if not os.path.exists(p):
        return None
    pr = json.load(open(p, encoding="utf-8"))
    if (datetime.now() - datetime.fromisoformat(pr["built"])).days > MAX_AGE_DAYS:
        return None
    return pr


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check_all()
