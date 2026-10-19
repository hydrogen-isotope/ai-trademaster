"""
v8 amendment G: a product profile per company, written by Qwen from the company's own description (research only).

Source text: NASDAQ, the "Business" section of the latest 10-K (Item 1; Item 4 of a 20-F) on SEC EDGAR filed on or
before the forecast close; otherwise, and for SGX, Eulerpool's company description. Qwen turns it into schema JSON:
products (with an approximate revenue share), customers, suppliers, competitors (with US tickers where it gives one),
sales drivers and a short summary. Related companies are mapped to tickers only when the ticker exists in SEC's list
and its company name agrees with Qwen's, so a made-up ticker never pulls in another company's news.

Profiles are cached in private/data/profiles/<market>/<SYM>.json and rebuilt after 120 days.
The SEC User-Agent (private/config/sec.json) is only ever sent to sec.gov.

Usage: python -m trademaster_ml.profiles build MU AMD --cpu | show MU
"""
import argparse, gzip, html, json, os, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

from . import alpaca, eulerpool, fundamentals, llm_news

DIR = os.path.join("private", "data", "profiles")
SEC_TICKERS = os.path.join("private", "data", "us", "edgar", "company_tickers.json")
MAX_AGE_DAYS = 120
MAX_TEXT = 12000                      # characters of source text given to Qwen (about 3k tokens)
ANNUAL = ("10-K", "10-K405", "10-KT", "20-F", "40-F")
_rate = fundamentals._Rate(4)

# Kept simple on purpose: length limits and number ranges make llama.cpp's grammar very slow (0.7 tokens/s on the
# CPU), so sizes are checked after generation instead (clean()).
ENT = {"type": "object", "properties": {"name": {"type": "string"}, "ticker": {"type": "string"}}, "required": ["name", "ticker"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "products": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "revenue_share_pct": {"type": "integer"}},
                                                  "required": ["name", "revenue_share_pct"]}},
        "customers": {"type": "array", "items": ENT},
        "suppliers": {"type": "array", "items": ENT},
        "competitors": {"type": "array", "items": ENT},
        "drivers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "products", "customers", "suppliers", "competitors", "drivers"],
}
LIMITS = {"products": 6, "customers": 6, "suppliers": 6, "competitors": 6, "drivers": 5}


def clean(pr):
    """Amendment G sizes: at most 6 products / customers / suppliers / competitors, 5 drivers, short texts."""
    out = {"summary": str(pr.get("summary", ""))[:300]}
    for k, n in LIMITS.items():
        xs = pr.get(k) or []
        if k == "drivers":
            out[k] = [str(x)[:120] for x in xs][:n]
        elif k == "products":
            out[k] = [{"name": str(x.get("name", ""))[:60], "revenue_share_pct": max(0, min(100, int(x.get("revenue_share_pct") or 0)))}
                      for x in xs if isinstance(x, dict) and x.get("name")][:n]
        else:
            out[k] = [{"name": str(x.get("name", ""))[:60], "ticker": str(x.get("ticker") or "")[:8]} for x in xs if isinstance(x, dict) and x.get("name")][:n]
    return out


SYSTEM = "You are a careful equity analyst. You summarise a company's own description of its business. Reply only with the JSON requested."


# ---------------------------------------------------------------- source text

def _sec_text(url):
    ua = fundamentals._ua()
    for a in range(4):
        _rate.wait()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"}), timeout=60)
            b = r.read()
            b = gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b
            t = re.sub(r"(?is)<(script|style|ix:header)\b.*?</\1>", " ", b.decode("utf-8", "replace"))
            return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
        except Exception:  # noqa: BLE001
            time.sleep(5 + 5 * a)
    return None


def _section(t, form):
    """The longest span from the section heading to the next item (the table of contents gives short spans)."""
    if form.startswith("20-F"):
        start, end = r"(?i)item\s*4\.?\s*information\s+on\s+the\s+company", r"(?i)item\s*4a\.?"
    else:
        start, end = r"(?i)item\s*1\.?\s*business", r"(?i)item\s*1a\.?\s*risk|item\s*2\.?\s*properties"
    best = None
    for m in re.finditer(start, t):
        e = re.search(end, t[m.start() + 20:])
        n = e.start() + 20 if e else min(len(t) - m.start(), 60000)
        if best is None or n > best[1]:
            best = (m.start(), n)
    return t[best[0]:best[0] + best[1]] if best and best[1] > 1500 else None


def sec_business(sym, asof):
    """(text, source dict) from the latest annual report filed on or before `asof`, or (None, None)."""
    cmap = os.path.join(fundamentals.EDGAR, "cik_map.json")
    cik = json.load(open(cmap)).get(sym) if os.path.exists(cmap) else None
    if not cik:
        return None, None
    sub = fundamentals._sec(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", fundamentals._ua(), _rate)
    r = (sub or {}).get("filings", {}).get("recent", {})
    for i, f in enumerate(r.get("form", [])):
        if f in ANNUAL and r["filingDate"][i] <= asof:
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{r['accessionNumber'][i].replace('-', '')}/{r['primaryDocument'][i]}"
            t = _sec_text(url)
            sec = _section(t, f) if t else None
            if sec:
                return sec[:MAX_TEXT], {"kind": f"SEC {f}", "filed": r["filingDate"][i], "url": url}
            return None, None                                                   # latest annual report only
    return None, None


def eulerpool_description(market, sym):
    key = sym
    if market == "sgx":
        a = next((x for x in json.load(open(os.path.join("private", "data", "sgx", "assets.json"))) if x["symbol"] == sym), None)
        key = a["isin"] if a and a.get("isin") else sym
    st, d = eulerpool.call("GET", f"equity/profile/{key}")
    if st != 200 or not isinstance(d, dict) or not d.get("description"):
        return None, None
    if market == "nasdaq" and d.get("country") not in (None, "US"):
        return None, None                                                       # ticker matched another country's stock
    text = f"{d.get('name', sym)}. Sector: {d.get('sector')}; industry: {d.get('industry')}; employees: {d.get('employees')}.\n{d['description']}"
    return text[:MAX_TEXT], {"kind": "Eulerpool description", "filed": None, "url": d.get("website")}


# ---------------------------------------------------------------- related companies -> tickers

def _norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return [w for w in s.split() if w not in ("inc", "corp", "corporation", "co", "ltd", "limited", "plc", "the", "group", "holdings", "company", "sa", "ag", "nv")]


def sec_tickers():
    if not os.path.exists(SEC_TICKERS) or time.time() - os.path.getmtime(SEC_TICKERS) > 30 * 86400:
        d = fundamentals._sec("https://www.sec.gov/files/company_tickers.json", fundamentals._ua(), _rate)
        if d:
            json.dump({v["ticker"]: v["title"] for v in d.values()}, open(SEC_TICKERS, "w"))
    return json.load(open(SEC_TICKERS)) if os.path.exists(SEC_TICKERS) else {}


# business units and brand names whose listed parent has a different legal name
PARENTS = {"amazon": "AMZN", "aws": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL", "facebook": "META",
           "instagram": "META", "ibm": "IBM", "hp": "HPQ", "dell": "DELL", "tsmc": "TSM", "azure": "MSFT", "xbox": "MSFT"}


def find_ticker(name, tickers, _index={}):
    """Ticker by company name: the title must start with the name's first word, contain its first two words, and be a
    plain share class; the shortest such title wins ("Apple" -> Apple Inc, not Apple Hospitality REIT)."""
    nw = _norm(name)[:2]
    if not nw:
        return None
    if nw[0] in PARENTS:                                                    # brands and units of big listed parents
        return PARENTS[nw[0]]
    if not _index:
        for t, title in tickers.items():
            w = _norm(title)
            if w and re.fullmatch(r"[A-Z]{1,5}", t):
                _index.setdefault(w[0], []).append((len(w), t, set(w)))
    cands = sorted(c for c in _index.get(nw[0], []) if set(nw) <= c[2])
    return cands[0][1] if cands else None


def map_ticker(ent, tickers):
    """Keep Qwen's ticker only if SEC lists it and the first two words of Qwen's company name are in SEC's name for it
    ("SK Hynix" must not map to SK Telecom)."""
    t = (ent.get("ticker") or "").upper().strip().replace("-", ".")
    nw = _norm(ent.get("name"))
    if t in tickers and nw and set(nw[:2]) <= set(_norm(tickers[t])):
        return t
    return find_ticker(ent.get("name"), tickers)


# ---------------------------------------------------------------- build / load

def path(market, sym):
    return os.path.join(DIR, market, f"{sym}.json")


def load(market, sym):
    p = path(market, sym)
    if not os.path.exists(p):
        return None
    pr = json.load(open(p, encoding="utf-8"))
    if datetime.now() - datetime.fromisoformat(pr["built"]) > timedelta(days=MAX_AGE_DAYS):
        return None
    if market == "nasdaq" and not pr.get("mapped_v2"):
        tickers = sec_tickers()
        for k in ("customers", "suppliers", "competitors"):
            for e in pr.get(k, []):
                e["ticker"] = e.get("ticker") or find_ticker(e["name"], tickers)
                if e["ticker"] == sym:
                    e["ticker"] = None
        pr["mapped_v2"] = True
    return pr


def build(market, sym, asof, name="", port=None):
    """Write and return the profile (the llama-server on `port` must be running)."""
    text, src = (sec_business(sym, asof) if market == "nasdaq" else (None, None))
    if not text:
        text, src = eulerpool_description(market, sym)
    if not text:
        return None
    user = (f"Company: {name or sym} (ticker {sym}). Below is {'its own annual report business section' if src['kind'].startswith('SEC') else 'a description of it'}.\n\n"
            f"{text}\n\nSummarise this company for an analyst who must judge how news affects its share price.\n"
            "- products: its main products or business lines (at most 6). revenue_share_pct: the share of revenue only if the text "
            "states it or it is well known; otherwise 0. Do not guess numbers.\n"
            "- customers, suppliers, competitors: specific named companies only (at most 6 each), never categories such as "
            "'data center customers'. You may add major ones that are widely known for this company, but only if you are sure; "
            "leave a list empty rather than guess. Give the US stock ticker if the company is listed in the US, otherwise an empty "
            "string. Do not list the company itself or companies that no longer exist.\n"
            "- drivers: the main things that move its sales and profits, specific to this company (at most 5).\n"
            "- summary: one sentence.")
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    body = {"prompt": prompt, "n_predict": 700, "temperature": 0, "cache_prompt": False, "id_slot": 1, "json_schema": SCHEMA}
    port = port or 8766
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    try:
        pr = clean(json.loads(r.get("content", "")))
    except (ValueError, TypeError, AttributeError):
        return None
    tickers = sec_tickers()
    me = (_norm(name) or _norm(sym))[:1]
    for k in ("customers", "suppliers", "competitors"):
        pr[k] = [e for e in pr.get(k, []) if _norm(e["name"])[:1] != me]            # never the company itself
        for e in pr.get(k, []):
            e["ticker"] = map_ticker(e, tickers)
            if e["ticker"] == sym:
                e["ticker"] = None
    pr.update({"mapped_v2": True, "symbol": sym, "market": market, "source": src, "built": datetime.now().isoformat(timespec="seconds"),
               "seconds": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(path(market, sym)), exist_ok=True)
    json.dump(pr, open(path(market, sym), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return pr


def related_news(pr, cutoff, days=7, per=3, max_companies=6):
    """Recent headlines (recap templates removed) for the profile's customers, suppliers and competitors with a ticker."""
    rel = []
    for role in ("customers", "suppliers", "competitors"):
        for e in pr.get(role, []):
            if e.get("ticker") and e["ticker"] not in [x[0] for x in rel]:
                rel.append((e["ticker"], role[:-1], e["name"]))
    rel = rel[:max_companies]
    if not rel:
        return []
    creds = alpaca.credentials()
    params = {"symbols": ",".join(t for t, _, _ in rel), "start": (cutoff - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "sort": "desc", "include_content": "false"}
    doc = alpaca._get("/v1beta1/news", params, creds)
    out = {t: [] for t, _, _ in rel}
    for n in doc.get("news", []):
        h = (n.get("headline") or "").strip()
        if not h or llm_news.RECAP.search(h):
            continue
        for t in n.get("symbols", []):
            if t in out and len(out[t]) < per and h not in [x["headline"] for x in out[t]]:
                out[t].append({"t": n["created_at"], "headline": h[:200]})
    return [{"ticker": t, "role": role, "name": nm, "headlines": out[t]} for t, role, nm in rel if out[t]]


def prompt_block(pr, rel):
    """Text added to the analysis prompt for variant B."""
    if not pr:
        return ""
    src = pr["source"]
    L = [f"Company profile (from its {src['kind']}{' filed ' + src['filed'] if src.get('filed') else ''}): {pr.get('summary', '')}"]
    if pr.get("products"):
        L.append("- products: " + "; ".join(p["name"] + (f" (~{p['revenue_share_pct']}% of revenue)" if p.get("revenue_share_pct") else "") for p in pr["products"]))
    for k in ("customers", "suppliers", "competitors"):
        if pr.get(k):
            L.append(f"- {k}: " + ", ".join(e["name"] + (f" ({e['ticker']})" if e.get("ticker") else "") for e in pr[k]))
    if pr.get("drivers"):
        L.append("- what drives its sales and profits: " + "; ".join(pr["drivers"]))
    if rel:
        L += ["", "Recent news about its customers, suppliers and competitors (last 5 sessions):"]
        for r in rel:
            for h in r["headlines"]:
                L.append(f"- {r['ticker']} ({r['role']}), {h['t'][:10]}: {h['headline']}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "show"])
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--market", choices=["nasdaq", "sgx"], default="nasdaq")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cmd == "show":
        for s in a.symbols:
            print(json.dumps(load(a.market, s), indent=1, ensure_ascii=False))
        return
    if not (a.allow_gpu or a.cpu):
        raise SystemExit("choose --allow-gpu (only while you are at the PC) or --cpu")
    from . import stock_eval
    srv = llm_news.Server(a.allow_gpu and not a.cpu, ctx_per_slot=stock_eval.CTX, slots=stock_eval.SLOTS, port=stock_eval.PORT)
    try:
        asof = stock_eval.latest_asof(a.market)
        for s in a.symbols:
            nm = {x["symbol"]: x.get("name") or "" for x in json.load(open(stock_eval._market_dirs(a.market)["names"]))}.get(s, "")
            pr = build(a.market, s, asof, name=nm, port=stock_eval.PORT)
            print(json.dumps(pr, indent=1, ensure_ascii=False) if pr else f"{s}: no source text", flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
