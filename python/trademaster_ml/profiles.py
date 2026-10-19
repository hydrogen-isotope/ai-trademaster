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
import argparse, glob, gzip, html, json, math, os, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

from . import alpaca, eulerpool, fundamentals, llm_news

DIR = os.path.join("private", "data", "profiles")
SEC_TICKERS = os.path.join("private", "data", "us", "edgar", "company_tickers.json")
MAX_AGE_DAYS = 120
MAX_TEXT = 17000                      # characters of source text given to Qwen (about 4.3k tokens; fits the 6k context with the answer)
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


REL = re.compile(r"(?i)\b(customers?|suppliers?|vendors?|foundr(y|ies)|wafers?|manufactur\w*|assembl\w*|partners?|partnership|compet\w*|"
                 r"rely|relies|depend\w*|sole source|single source|power(s|ed)? the|acquir\w*|largest|principal|"
                 r"accounted for|segments?)\b")
REV = re.compile(r"(?i)(\d{1,2}(\.\d)?\s?%|\d{1,2}(\.\d)? percent).{0,80}\b(revenue|sales)\b|\b(revenue|sales)\b.{0,80}(\d{1,2}(\.\d)?\s?%|percent)")
NAME = re.compile(r"(?<![.!?]\s)\b[A-Z][a-zA-Z&]+(?:\s[A-Z][a-zA-Z&]+)*(?:\s(?:Inc|Corp|Corporation|Ltd|Limited|Co|Company|AG|N\.V|plc))?")


def select_text(section, full, overview=3000, budget=MAX_TEXT):
    """The business overview, then the sentences (from the Business section first, then the whole filing) that name
    customers, suppliers, competitors or partners, or give revenue shares: a 50k-character section is otherwise cut
    before its supplier and customer paragraphs (AMD's TSMC and Sony mentions came after 12k characters)."""
    cands, seen = [], set()
    for rank, src in enumerate((section[overview:], full)):
        for k, snt in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z])", src)):
            if not 40 <= len(snt) <= 500 or snt in seen or re.search(r"(?i)forward-looking|could adversely|may not be able", snt):
                continue
            seen.add(snt)
            rev = bool(REV.search(snt))
            rel = bool(REL.search(snt))
            names = len(NAME.findall(snt))
            firms = len(re.findall(r"\b(Inc|Corp|Corporation|Limited|Ltd|Company|plc|AG|N\.V)\b|\([A-Z]{2,6}\)", snt))   # named companies
            if rev or (rel and names >= 2):
                score = 3 * rev + 2 * min(firms, 3) + min(names, 4) + (4 if rank == 0 else 0)   # the Business section first
                cands.append((-score, rank, k, snt))
    used, keep, rest = overview, [], 0
    for c in sorted(cands):                                                 # most specific sentences first
        if used + len(c[3]) < budget and (c[1] == 0 or rest + len(c[3]) < 0.35 * budget):   # rest of the filing: at most 35%
            keep.append(c)
            used += len(c[3]) + 1
            rest += len(c[3]) + 1 if c[1] else 0
    return "\n".join([section[:overview]] + [c[3] for c in sorted(keep, key=lambda c: (c[1], c[2]))])   # back in document order


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
                return select_text(sec, t), {"kind": f"SEC {f}", "filed": r["filingDate"][i], "url": url}
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

STOP = {"inc", "incorporated", "corp", "corporation", "co", "ltd", "limited", "plc", "the", "group", "holdings", "company", "sa", "ag", "nv",
        "de", "llc", "lp"}
# words of a category, not a company ("Hyperscale cloud providers", "PC OEMs", "Game console manufacturers")
GENERIC = {"customers", "customer", "providers", "provider", "oems", "oem", "odms", "odm", "manufacturers", "manufacturer", "makers",
           "companies", "suppliers", "supplier", "vendors", "vendor", "data", "center", "centers", "datacenters", "markets", "market",
           "hyperscale", "hyperscalers", "hyperscaler", "cloud", "neocloud", "industrial", "automotive", "smartphone", "smartphones", "mobile",
           "game", "gaming", "console", "consoles", "pc", "pcs", "enterprise", "enterprises", "government", "governments", "distributors",
           "retailers", "partners", "others", "other", "various", "banks", "fintech", "fintechs", "foundries", "service", "services", "telecom",
           "operators", "carriers", "consumers", "consumer", "businesses", "chip", "chips", "memory", "semiconductor", "electronics", "end", "users",
           "and", "of", "major", "large", "leading", "global", "local", "regional", "asian", "us", "chinese", "japanese", "korean", "european"}
FUNDS = re.compile(r"(?i)\b(etn|etns|x-links|trust|fund|etf|notes|index|warrants?|units?)\b")


def _norm(s):
    s = re.sub(r"\([^)]*\)", " ", (s or "").lower())                      # "GLOBALFOUNDRIES Inc. (GF)" -> globalfoundries
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return [w for w in s.split() if w not in STOP]


def is_category(name):
    w = _norm(name)
    return not w or all(x in GENERIC for x in w)


def sec_tickers():
    if not os.path.exists(SEC_TICKERS) or time.time() - os.path.getmtime(SEC_TICKERS) > 30 * 86400:
        d = fundamentals._sec("https://www.sec.gov/files/company_tickers.json", fundamentals._ua(), _rate)
        if d:
            json.dump({v["ticker"]: v["title"] for v in d.values()}, open(SEC_TICKERS, "w"))
    return json.load(open(SEC_TICKERS)) if os.path.exists(SEC_TICKERS) else {}


# business units and brand names whose listed parent has a different legal name
PARENTS = {"amazon": "AMZN", "aws": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL", "facebook": "META",
           "instagram": "META", "ibm": "IBM", "hp": "HPQ", "dell": "DELL", "tsmc": "TSM", "azure": "MSFT", "xbox": "MSFT"}


def note_issuers(tickers, _cache={}):
    """Tickers of issuers listed only through several unrelated notes/ETNs (e.g. CREDIT SUISSE AG: GLDI, USOI, SLVO):
    a title with 3+ tickers that share no 2-letter prefix is not a company's shares (GOOG/GOOGL share one)."""
    if "x" not in _cache:
        by = {}
        for t, title in tickers.items():
            by.setdefault(title, []).append(t)
        fin = re.compile(r"(?i)\b(ag|trust|l\.?p\.?|fund|bank|securities|capital)\b")   # issuers of notes, not operating firms
        _cache["x"] = {t for title, ts in by.items() if len(ts) >= 3 and len({x[:2] for x in ts}) == len(ts) and fin.search(title) for t in ts}
    return _cache["x"]


def find_ticker(name, tickers, _index={}):
    """Ticker by company name: the title must start with the name's first word, contain its first two words, and be a
    plain share class; the shortest such title wins ("Apple" -> Apple Inc, not Apple Hospitality REIT)."""
    nw = _norm(name)[:2]
    if not nw or is_category(name):
        return None
    if nw[0] in PARENTS:                                                    # brands and units of big listed parents
        return PARENTS[nw[0]]
    bare = (name or "").strip()
    if re.fullmatch(r"[A-Z]{2,5}", bare) and bare in tickers and bare not in note_issuers(tickers):
        return bare                                                         # the name is itself a ticker ("AMD")
    if not _index:
        notes = note_issuers(tickers)
        for t, title in tickers.items():
            w = _norm(title)
            if w and re.fullmatch(r"[A-Z]{1,5}", t) and not FUNDS.search(title) and t not in notes:   # shares only, not notes/ETNs
                _index.setdefault(w[0], []).append((len(w), len(t), t, set(w)))   # shortest name, then the main (shortest) ticker
    cands = sorted(c for c in _index.get(nw[0], []) if set(nw) <= c[3])
    return cands[0][2] if cands else None


def map_ticker(ent, tickers):
    """Keep Qwen's ticker only if SEC lists it and the first two words of Qwen's company name are in SEC's name for it
    ("SK Hynix" must not map to SK Telecom)."""
    t = (ent.get("ticker") or "").upper().strip().replace("-", ".")
    nw = _norm(ent.get("name"))
    if (t in tickers and nw and not is_category(ent.get("name")) and not FUNDS.search(tickers[t]) and t not in note_issuers(tickers)
            and set(nw[:2]) <= set(_norm(tickers[t]))):
        return t
    return find_ticker(ent.get("name"), tickers)


# ---------------------------------------------------------------- build / load

def path(market, sym):
    return os.path.join(DIR, market, f"{sym}.json")


def load(market, sym, asof=None):
    from . import profiles_haiku                                           # amendment I: the researched profile first
    hp = profiles_haiku.load(market, sym)
    if hp:
        hp["named_by"] = reverse_links(market, sym)
        return hp
    p = path(market, sym)
    if not os.path.exists(p):
        return None
    pr = json.load(open(p, encoding="utf-8"))
    sec_src = (pr.get("source") or {}).get("kind", "").startswith("SEC")
    if not sec_src and datetime.now() - datetime.fromisoformat(pr["built"]) > timedelta(days=MAX_AGE_DAYS):
        return None                                                        # descriptions: rebuilt after 120 days
    if sec_src and asof and newer_annual_report(pr, sym, asof):
        return None                                                        # filings: rebuilt when a newer annual report exists
    if not pr.get("mapped_v3"):                                            # older profiles: same clean-up, without a model call
        tidy(pr, sym, "", "", sec_tickers())
        pr["mapped_v3"] = True
    pr["named_by"] = reverse_links(market, sym)
    return pr


def _complete(prompt, schema, port, n_predict=700, slot=1):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False, "id_slot": slot, "json_schema": schema}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    return r.get("content", "")


def gone(names, port):
    """Second pass: which of these companies no longer exist on their own (bankrupt, dissolved, fully acquired)?
    Qwen can only pick from the given names, so it cannot add anything."""
    out = set()
    for n in names:                                                         # one yes/no per company, read from token probabilities
        user = (f"Does {n} still exist today as an independent company, i.e. it has not gone bankrupt, been dissolved, or been "
                "taken over or merged into another company? Answer yes or no.")
        prompt = f"<|im_start|>system\nYou answer questions about companies with one word.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 20, "cache_prompt": False, "id_slot": 1}
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                                                        {"Content-Type": "application/json"}), timeout=300))
        except Exception:  # noqa: BLE001
            continue
        p = {"yes": 0.0, "no": 0.0}
        for x in r["completion_probabilities"][0]["top_logprobs"]:
            w = x["token"].strip().lower()
            if w in p:
                p[w] += math.exp(x["logprob"])
        if p["yes"] + p["no"] > 0 and p["no"] / (p["yes"] + p["no"]) >= 0.6:
            out.add(n)
    return out


def tidy(pr, sym, name, text, tickers, port=None):
    """Clean the related companies:
    1. drop categories ("Hyperscale cloud providers") and the company itself;
    2. map tickers (strict name match; shares only, never notes or ETNs);
    3. drop duplicates (same ticker or name);
    4. companies that are neither named in the filing nor have a current US ticker came from Qwen's own (2024) memory:
       only these go through the "no longer exists" check (port given), so a re-listed company such as Sandisk or a
       supplier named in this year's filing can never be removed by out-of-date knowledge."""
    me = (_norm(name) or _norm(sym))[:1]
    words = set(_norm(text or ""))
    pr.setdefault("removed", {"category": [], "itself": [], "duplicate": [], "no_longer_exist": []})
    unverified = []
    for k in ("customers", "suppliers", "competitors"):
        keep, seen = [], set()
        for e in pr.get(k, []):
            nw = _norm(e["name"])
            if is_category(e["name"]):
                pr["removed"]["category"].append(e["name"])
                continue
            if nw[:1] == me:
                pr["removed"]["itself"].append(e["name"])
                continue
            e["ticker"] = e["ticker"] if str(e.get("ticker") or "").endswith(".SI") else map_ticker(e, tickers)   # SGX peers keep theirs
            if e["ticker"] == sym:
                pr["removed"]["itself"].append(e["name"])
                continue
            key = e["ticker"] or " ".join(nw[:2])
            if key in seen:
                pr["removed"]["duplicate"].append(e["name"])
                continue
            seen.add(key)
            e["in_filing"] = bool(nw) and nw[0] in words
            if not e["in_filing"] and not e["ticker"]:
                unverified.append(e["name"])
            keep.append(e)
        pr[k] = keep
    # acronym duplicates: "OCBC Bank" (no ticker) vs "Oversea-Chinese Banking Corporation" (O39.SI): keep the one with a ticker
    initials = lambda n: "".join(w[0] for w in re.split(r"[\s\-]+", re.sub(r"[^A-Za-z\s\-]", " ", n or "").lower()) if w)
    for k in ("customers", "suppliers", "competitors"):
        full = [initials(e["name"]) for e in pr[k] if e.get("ticker")]
        dup = [e for e in pr[k] if not e.get("ticker") and any(f.startswith(_norm(e["name"])[0]) and len(_norm(e["name"])[0]) >= 3 for f in full)]
        pr["removed"]["duplicate"] += [e["name"] for e in dup]
        pr[k] = [e for e in pr[k] if e not in dup]
        unverified = [n for n in unverified if n not in [e["name"] for e in dup]]
    if port and unverified:
        dead, ask = set(), []
        for n in sorted(set(unverified)):                                   # market data first, Qwen's memory only if unknown
            s_ = market_status(n)
            if s_ == "delisted":
                dead.add(n)
            elif s_ is None:
                ask.append(n)
        dead |= gone(ask, port)
        for k in ("customers", "suppliers", "competitors"):
            pr["removed"]["no_longer_exist"] += [e["name"] for e in pr[k] if e["name"] in dead]
            pr[k] = [e for e in pr[k] if e["name"] not in dead]
    return pr


def _ident(market, sym):
    """Eulerpool identifier: the ticker for US stocks, the ISIN for SGX."""
    if market == "sgx":
        a = next((x for x in json.load(open(os.path.join("private", "data", "sgx", "assets.json"))) if x["symbol"] == sym), None)
        return a.get("isin") if a else None
    return sym


def segments(market, sym, asof, lag_days=75):
    """Amendment H: revenue split from Eulerpool's segment data (the company's XBRL filings): the latest fiscal period
    ending >= lag_days before `asof` whose shares sum to 90-110%, business segments preferred over product lines,
    plus the same split about a year earlier."""
    ident = _ident(market, sym)
    if not ident:
        return None
    try:
        st, d = eulerpool.call("GET", f"equity/segments/{ident}")
    except Exception:  # noqa: BLE001
        return None
    if st != 200 or not isinstance(d, list):
        return None
    limit = (datetime.fromisoformat(asof) - timedelta(days=lag_days)).strftime("%Y-%m-%d")
    groups = {}
    for x in d:
        if x.get("period") and x["period"] <= limit and x.get("percentage") is not None and x.get("label"):
            groups.setdefault((x["axis"], x["period"]), {})[x["label"]] = float(x["percentage"])
    ok = {k: v for k, v in groups.items() if len(v) >= 2 and 90 <= sum(v.values()) <= 110}
    if not ok:
        return None
    pref = lambda k: (k[1], 1 if "BusinessSegments" in k[0] else 0)
    axis, period = max(ok, key=pref)
    cur = ok[(axis, period)]
    prev_keys = [k for k in ok if k[0] == axis and 300 <= (datetime.fromisoformat(period) - datetime.fromisoformat(k[1])).days <= 430]
    prev = ok[max(prev_keys, key=lambda k: k[1])] if prev_keys else {}
    items = sorted(cur.items(), key=lambda kv: -kv[1])[:8]
    return {"period": period, "axis": "business segments" if "BusinessSegments" in axis else "products and services",
            "items": [{"label": k, "pct": round(v, 1), "prev_pct": round(prev[k], 1) if k in prev else None} for k, v in items],
            "prev_period": max(prev_keys, key=lambda k: k[1])[1] if prev_keys else None, "source": "Eulerpool segment data (company XBRL filings)"}


def peers(market, sym):
    """Amendment H: Eulerpool's peer list, as {name, ticker}: US tickers by the strict name match, SGX by ISIN."""
    ident = _ident(market, sym)
    if not ident:
        return []
    try:
        st, d = eulerpool.call("GET", f"equity/peers/{ident}")
    except Exception:  # noqa: BLE001
        return []
    if st != 200 or not isinstance(d, list):
        return []
    sgx = {a.get("isin"): a["symbol"] for a in json.load(open(os.path.join("private", "data", "sgx", "assets.json")))} \
        if os.path.exists(os.path.join("private", "data", "sgx", "assets.json")) else {}
    tickers = sec_tickers()
    out = []
    for x in d[:8]:
        nm, isin = x.get("peer_name") or "", x.get("peer_isin") or ""
        t = sgx.get(isin) if isin.startswith("SG") else (find_ticker(nm, tickers) if isin.startswith("US") else None)
        if nm:
            out.append({"name": nm, "ticker": t or "", "from": "peer list"})
    return out


def reverse_links(market, sym):
    """Amendment H: companies whose own profile names this one: X lists it as a supplier -> X is its customer, and vice
    versa. Built from every stored profile, so it grows as more companies are profiled."""
    out = []
    haiku = glob.glob(os.path.join("private", "data", "profiles_haiku", "*", "*.checked.json"))
    have = {os.path.basename(f)[:-len(".checked.json")] for f in haiku}
    qwen = [f for f in glob.glob(os.path.join(DIR, "*", "*.json")) if os.path.basename(f)[:-5] not in have]
    for f in haiku + qwen:                                                 # a researched profile replaces the Qwen one
        try:
            other = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if other.get("symbol") in (None, sym):
            continue
        for k, role in (("suppliers", "customer"), ("customers", "supplier")):
            if any(e.get("ticker") == sym for e in other.get(k, [])):
                out.append({"name": other.get("name") or other["symbol"], "ticker": other["symbol"] if other.get("market") == "nasdaq" else "",
                            "role": role, "named_by": other["symbol"]})
    return out


def newer_annual_report(pr, sym, asof):
    """True if SEC has a newer annual report than the profile's source (checked at most weekly)."""
    src = pr.get("source") or {}
    if not src.get("kind", "").startswith("SEC") or not src.get("filed"):
        return False
    last = pr.get("checked")
    if last and datetime.now() - datetime.fromisoformat(last) < timedelta(days=7):
        return False
    cmap = os.path.join(fundamentals.EDGAR, "cik_map.json")
    cik = json.load(open(cmap)).get(sym) if os.path.exists(cmap) else None
    if not cik:
        return False
    try:
        sub = fundamentals._sec(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", fundamentals._ua(), _rate)
    except Exception:  # noqa: BLE001
        return False
    r = (sub or {}).get("filings", {}).get("recent", {})
    newest = max((r["filingDate"][i] for i, f in enumerate(r.get("form", [])) if f in ANNUAL and r["filingDate"][i] <= asof), default=None)
    pr["checked"] = datetime.now().isoformat(timespec="seconds")
    json.dump(pr, open(path(pr["market"], sym), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return bool(newest and newest > src["filed"])


def market_status(name):
    """'trading' if Eulerpool finds this company's shares and they traded in the last 30 days, 'delisted' if it finds
    the shares but no trades, None if it does not know the company (then Qwen's yes/no decides)."""
    nw = _norm(name)[:2]
    if not nw:
        return None
    try:
        st, d = eulerpool.call("GET", "equity/search", {"q": " ".join(nw)})
    except Exception:  # noqa: BLE001
        return None
    hits = [r for r in (d.get("results", []) if st == 200 and isinstance(d, dict) else [])
            if r.get("type") == "stock" and r.get("isin") and set(nw) <= set(_norm(r.get("name")))][:2]
    if not hits:
        return None
    now = int(time.time())
    for r in hits:
        st2, c = eulerpool.call("GET", f"charting/ohlcv/{r['isin']}", {"resolution": "D", "from": now - 30 * 86400, "to": now})
        if st2 == 200 and isinstance(c, dict) and c.get("t"):
            return "trading"
    return "delisted"


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
    pr["segments"] = segments(market, sym, asof)
    if pr["segments"]:                                                     # real revenue split replaces Qwen's shares
        for x in pr.get("products", []):
            x["revenue_share_pct"] = 0
    have = {_norm(e["name"])[:1] and _norm(e["name"])[0] for e in pr.get("competitors", [])}
    pr["competitors"] = pr.get("competitors", []) + [e for e in peers(market, sym) if _norm(e["name"])[:1] and _norm(e["name"])[0] not in have]
    tidy(pr, sym, name, text, sec_tickers(), port)
    pr.update({"mapped_v3": True, "name": name or sym, "symbol": sym, "market": market, "source": src, "built": datetime.now().isoformat(timespec="seconds"),
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
    for e in pr.get("named_by", []):                                       # amendment H reverse links
        if e.get("ticker") and e["ticker"] not in [x[0] for x in rel]:
            rel.append((e["ticker"], e["role"], e["name"]))
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


def prompt_block(pr, rel, moves=None):
    """Text added to the analysis prompt for variant B (moves: SGX peers' 5-session price moves, {ticker: return})."""
    if not pr:
        return ""
    src = pr["source"]
    if "Haiku" in src.get("kind", ""):
        L = [f"Company profile (researched from {'its ' + src['filing_kind'] + ' filed ' + src['filed'] + ' and ' if src.get('filed') and src.get('filing_kind') else 'its filings and '}"
             f"sources dated 2023 or later): {pr.get('summary', '')}"]
    else:
        L = [f"Company profile (from its {src['kind']}{' filed ' + src['filed'] if src.get('filed') else ''}): {pr.get('summary', '')}"]
    if pr.get("products"):
        L.append("- products: " + "; ".join(p["name"] + (f" (~{p['revenue_share_pct']}% of revenue)" if p.get("revenue_share_pct") else "") for p in pr["products"]))
    for k in ("customers", "suppliers", "competitors"):
        if pr.get(k):
            L.append(f"- {k}: " + ", ".join(e["name"] + (f" ({e['ticker']})" if e.get("ticker") else "") for e in pr[k]))
    seg = pr.get("segments")
    if seg:
        L.append(f"- revenue split by {seg['axis']} (fiscal period ending {seg['period']}): " + "; ".join(
            f"{x['label']} {x['pct']:.0f}%" + (f" (a year earlier {x['prev_pct']:.0f}%)" if x.get("prev_pct") is not None else "") for x in seg["items"]))
    if pr.get("named_by"):
        L.append("- named in other companies' filings as their supplier/customer: " + ", ".join(
            f"{e['name']} ({e['named_by']}, a {e['role']})" for e in pr["named_by"][:6]))
    if pr.get("drivers"):
        L.append("- what drives its sales and profits: " + "; ".join(pr["drivers"]))
    if moves:
        L.append("- its peers' share price moves over the last 5 sessions: " + ", ".join(f"{t} {r:+.1%}" for t, r in moves.items()))
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
