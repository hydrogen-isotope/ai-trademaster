"""
v8 amendments C-E: a per-stock Qwen analysis that leads the 5-day forecast (research only; no orders).

For one stock, Qwen (local, llama.cpp) reads the quantitative facts of stages 2-4 (price/momentum, fundamentals,
industry and cross-stock rank), the v7 model-only forecast, the market news index, same-industry peers with the
strongest recent news, and the stock's own headlines of the last 30 days. It answers in schema-enforced JSON (reasons
first, then lean / confidence / risk), then one more question: the chance (0-100) that the price is higher in 5 trading
days. Its probability P_qwen is post-processed by our models (amendment E): P_total = P_model + W_QWEN * (P_qwen - 0.5), and the v7
distribution is re-centred so that its calibrated P(up) equals P_total. Three views are kept: total, Qwen only, model
only, plus a research signal (buy / sell / none). Every forecast is logged before the outcome exists; `score` runs the
forward test.

  eval SYM [--market nasdaq|sgx] --allow-gpu|--cpu   evaluate one stock now and print the verdict
  serve --allow-gpu|--cpu [--app-url URL]            work through the queue while the app is open; the model is
                                                     unloaded after 5 idle minutes, when the app closes, or on a GPU
                                                     driver reset
  batch                                              queue the app-start set (top/bottom 25 by rank score, top/bottom
                                                     10 by model P(up), NASDAQ)
  score [--market nasdaq|sgx]                        forward test: advised vs model-only on matured forecasts

Usage: python -m trademaster_ml.stock_eval eval MU --allow-gpu
"""
import argparse, glob, json, math, os, re, subprocess, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

from . import alpaca, llm_news, profiles, v6, v7

EVALS = os.path.join("private", "reports", "evals")
QUEUE = os.path.join(EVALS, "queue")
STATE = os.path.join(EVALS, "daemon.json")
LOCK = os.path.join(EVALS, "daemon.lock")
REFRESH_STATUS = os.path.join("private", "reports", "v6", "refresh_status.json")
PORT, SLOTS, CTX = 8766, 2, 6144
MODEL_NAME = "Qwen2.5-7B-Instruct (Q4_K_M, local)"

# amendment C answer scale (its k values were replaced by amendment E; the lean is shown, not mapped)
LEAN_K = {"strong_down": -0.30, "down": -0.12, "neutral": 0.0, "up": 0.12, "strong_up": 0.30}
CONF_MULT = {"low": 0.5, "medium": 1.0, "high": 1.25}
RISK_SD = {"normal": 1.0, "elevated": 1.25}
# amendment E
W_QWEN = 0.5                        # amendment E rule: 0.12 / p90|P_qwen - 0.5| over the hand check = 0.12 / 0.155 = 0.77, capped at 0.5
BUY_AT, SELL_AT = 0.58, 0.42        # research signal thresholds on P_total
P_CLIP = (0.05, 0.95)
I25, I75 = (int(np.argmin(np.abs(v7.GRID - x))) for x in (0.25, 0.75))
FALLBACK = {"lean": "neutral", "confidence": "low", "risk": "normal"}
NEWS_DAYS, MAX_HEADLINES, MAX_PEERS = 30, 40, 5
HISTORY_DAYS, IDLE_EXIT_S = 260, 300
SAFE_SYM = re.compile(r"^[A-Z0-9.\-]{1,15}$")

SCHEMA = {
    "type": "object",
    "properties": {
        "positives": {"type": "array", "items": {"type": "string", "maxLength": 220}, "maxItems": 3},
        "negatives": {"type": "array", "items": {"type": "string", "maxLength": 220}, "maxItems": 3},
        "lean": {"type": "string", "enum": list(LEAN_K)},
        "confidence": {"type": "string", "enum": list(CONF_MULT)},
        "risk": {"type": "string", "enum": list(RISK_SD)},
    },
    "required": ["positives", "negatives", "lean", "confidence", "risk"],
    "additionalProperties": False,
}

SYSTEM = ("You are a careful equity analyst. You weigh the news and the numbers about one stock and judge its likely price "
          "move over the next 5 trading days. You reply only with the JSON requested.")

# (feature, label, format): ret = log return shown as %, ann = daily sd annualised, pct = ratio as %, num, days
FACTS = [("mom5", "1-week return", "ret"), ("mom21", "1-month return", "ret"), ("mom12_1", "12-month return excluding the last month", "ret"),
         ("vol60", "volatility, annualised (60 days)", "ann"), ("dist_252_high", "distance below the 52-week high", "ret"),
         ("beta252", "beta to the market", "num"), ("earnings_yield", "earnings yield", "pct"), ("sales_to_price", "sales / price", "num"),
         ("gross_profitability", "gross profitability", "num"), ("sue", "latest earnings surprise (standardised)", "num"),
         ("days_since_filing", "days since the last financial filing", "days"),
         ("ind_mom21", "its industry's 1-month move", "ret"), ("ind_mom126", "its industry's 6-month move", "ret")]

SIC2 = {1: "agriculture", 10: "metal mining", 13: "oil and gas extraction", 15: "construction", 20: "food products",
        23: "apparel", 24: "wood products", 26: "paper", 27: "printing and publishing", 28: "chemicals and pharmaceuticals",
        29: "petroleum refining", 30: "rubber and plastics", 32: "stone, clay and glass", 33: "primary metals",
        34: "fabricated metal products", 35: "industrial machinery and computer equipment",
        36: "electronic equipment and semiconductors", 37: "transportation equipment", 38: "instruments and medical devices",
        39: "miscellaneous manufacturing", 40: "railroads", 42: "trucking", 44: "water transport", 45: "air transport",
        47: "transport services", 48: "communications", 49: "utilities", 50: "wholesale (durable goods)",
        51: "wholesale (non-durable goods)", 52: "building materials retail", 53: "general merchandise stores", 54: "food stores",
        55: "auto dealers", 56: "apparel stores", 57: "home furniture and electronics stores", 58: "restaurants", 59: "retail",
        60: "banks", 61: "credit institutions", 62: "securities and brokers", 63: "insurance carriers", 64: "insurance agents",
        65: "real estate", 67: "holding and investment offices", 70: "hotels", 72: "personal services",
        73: "business services and software", 75: "auto services", 78: "motion pictures", 79: "amusement and recreation",
        80: "health services", 82: "educational services", 83: "social services", 87: "engineering and management services",
        99: "non-classifiable"}


class NotForecast(Exception):
    """The stock has no v7 forecast for the latest close (outside the forecast universe)."""


# ---------------------------------------------------------------- inputs (all point-in-time at the latest close)

def _market_dirs(market):
    v6.use_market(market)
    return {"gpu": v6.GPU_DIR, "v6": v6.OUT, "v7": v6.V7_OUT, "names": v6.NAMES_FILE, "out": os.path.join(EVALS, market)}


def latest_asof(market):
    d = _market_dirs(market)
    with open(os.path.join(d["v7"], "forecast.json")) as f:
        m = re.search(r'"asof":\s*"([0-9-]+)"', f.read(200))
    return m.group(1) if m else None


def _shadow_record(v7_dir, asof):
    """The shadow-log record (model-only m, sd, close per stock + recalibration levels) for close `asof`: the v7r one
    (v7 amendment B, the ranges the app shows) when it exists for that close, else v7's."""
    for name in ("shadow_r.jsonl", "shadow.jsonl"):
        path = os.path.join(v7_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            if f'"asof": "{asof}"' in line[:40]:
                return json.loads(line)
    return None


def next_open_utc(asof):
    """09:30 New York on the first weekday after the close `asof`: news later than this is not used (no look-ahead)."""
    nxt = datetime.fromisoformat(asof).replace(tzinfo=llm_news.NY) + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.replace(hour=9, minute=30).astimezone(timezone.utc)


def _fmt(kind, x):
    if kind == "ret":
        return f"{math.expm1(x) * 100:+.1f}%"
    if kind == "ann":
        return f"{x * math.sqrt(252) * 100:.0f}%"
    if kind == "pct":
        return f"{x * 100:.1f}%"
    if kind == "days":
        return f"{x:.0f}"
    return f"{x:.2f}"


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def context(market, sym, rankings=None, news=None):
    """Everything the prompt and the page need for `sym` at the latest close. Raises NotForecast if v7 has no row."""
    d = _market_dirs(market)
    meta = json.load(open(os.path.join(d["gpu"], "meta.json")))
    feats = json.load(open(os.path.join(d["gpu"], "features.json")))["features"]
    T, N, asof = meta["T"], meta["N"], meta["dates"][-1]
    if sym not in meta["symbols"]:
        raise NotForecast(f"{sym} is not in the {market.upper()} data")
    j = meta["symbols"].index(sym)
    rec = _shadow_record(d["v7"], asof)
    row = next((r for r in rec["rows"] if r[0] == sym), None) if rec else None
    if row is None:
        raise NotForecast(f"{sym} has no price forecast for the close of {asof} (outside the forecast universe)")
    _, m, sd, close = row
    hz = {5: {"m": float(m), "sd": float(sd), "R": np.array(rec["R_v7"])}}
    for h in (1, 20):                                                      # v7 amendment A: shown only if the gate passed
        hd = os.path.join(d["v7"], f"h{h}")
        g = os.path.join(hd, "gate.json")
        if not (os.path.exists(g) and json.load(open(g)).get("pass")):
            continue
        rh = _shadow_record(hd, asof)
        rw = next((r for r in rh["rows"] if r[0] == sym), None) if rh else None
        if rw:
            hz[h] = {"m": float(rw[1]), "sd": float(rw[2]), "R": np.array(rh["R_v7"])}
    raw = np.memmap(os.path.join(d["gpu"], "raw.f32"), np.float32, "r", shape=(len(feats), T, N))
    rk = np.memmap(os.path.join(d["gpu"], "ranks.f32"), np.float32, "r", shape=(len(feats), T, N))
    facts = []
    for f, label, kind in FACTS:
        if f not in feats:
            continue
        i = feats.index(f)
        x, r = float(raw[i, T - 1, j]), float(rk[i, T - 1, j])
        if not math.isfinite(x):
            continue
        facts.append({"feature": f, "label": label, "value": _fmt(kind, x), "pct": round(r * 100) if math.isfinite(r) else None})
    lc = np.memmap(os.path.join(d["gpu"], "logclose.f32"), np.float32, "r", shape=(T, N))
    hist = lc[T - HISTORY_DAYS:, j].astype(np.float64)
    history = [[meta["dates"][T - HISTORY_DAYS + k], round(float(np.exp(v)), 4)] for k, v in enumerate(hist) if np.isfinite(v)]
    if rankings is None:
        rp = os.path.join(d["v6"], "rankings.json")
        rankings = json.load(open(rp)) if os.path.exists(rp) else {"rankings": []}
    rr = next((r for r in rankings.get("rankings", []) if r["symbol"] == sym), None)
    names = {a["symbol"]: a.get("name") or "" for a in json.load(open(d["names"]))}
    industry, peers, mkt = None, [], None
    ip = os.path.join(d["gpu"], "industry.f32")
    if market == "nasdaq":
        if news is None:
            news = json.load(open(llm_news.TODAY)) if os.path.exists(llm_news.TODAY) else None
        if news and news.get("asof") != asof:
            news = None
        if os.path.exists(ip):
            ind = np.fromfile(ip, np.float32)
            code = ind[j]
            if np.isfinite(code) and code > 0:
                industry = SIC2.get(int(code), f"SIC {int(code):02d}")
                if news:
                    same = {meta["symbols"][k] for k in np.nonzero(ind == code)[0]} - {sym}
                    cand = [(s, v) for s, v in news["stocks"].items() if s in same and v["n"] >= 1]
                    cand.sort(key=lambda x: -abs(x[1]["news_dir"]))
                    for s, v in cand[:MAX_PEERS]:
                        top = v["top"][0] if v["top"] else None
                        peers.append({"symbol": s, "name": names.get(s, ""), "news_dir": round(v["news_dir"], 2), "n": v["n"],
                                      "headline": top["headline"] if top else None,
                                      "read": None if not top else ("good" if top["dir"] > 0.33 else "bad" if top["dir"] < -0.33 else "neutral")})
        if news:
            mkt = {"S": news["market"]["S"], "n": news["market"]["n"], "ai_anxiety": news["market"]["ai_anxiety"]}
    return {"market": market, "symbol": sym, "name": names.get(sym, rr["name"] if rr else ""), "asof": asof, "close": float(close),
            "m": float(m), "sd": float(sd), "R": np.array(rec["R_v7"]), "facts": facts, "universe": int(len(rec["rows"])),
            "rank": {"score_pct": rr["score_pct"], "decile": rr["decile"]} if rr else None, "industry": industry,
            "peers": peers, "market_news": mkt, "history": history, "hz": hz}


def own_headlines(sym, asof, now=None):
    """The stock's own headlines (Alpaca / Benzinga), last NEWS_DAYS days up to the next open after `asof`; templated
    recaps removed (amendment A list). Newest first."""
    now = now or datetime.now(timezone.utc)
    end = min(now, next_open_utc(asof))
    start = end - timedelta(days=NEWS_DAYS)
    creds = alpaca.credentials()
    params = {"symbols": sym, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "limit": 50, "sort": "desc", "include_content": "false"}
    items, seen, n_recap = [], set(), 0
    for _ in range(3):
        doc = alpaca._get("/v1beta1/news", params, creds)
        for n in doc.get("news", []):
            h = (n.get("headline") or "").strip()
            if not h or h.lower() in seen:
                continue
            seen.add(h.lower())
            if llm_news.RECAP.search(h):
                n_recap += 1
                continue
            items.append({"t": n["created_at"], "headline": h[:240]})
        token = doc.get("next_page_token")
        if not token or len(items) >= MAX_HEADLINES:
            break
        params["page_token"] = token
    return {"source": "Alpaca (Benzinga)", "from": params["start"], "cutoff": params["end"], "n_recap_removed": n_recap,
            "headlines": items[:MAX_HEADLINES]}


# ---------------------------------------------------------------- forecast arithmetic

def dist(close, m, sd, R):
    """Price quantiles and P(up) of the v7 Student-t with recalibration levels R."""
    q = v7.quantiles(np.array([m]), np.array([sd]), R)[0]
    px = lambda i: float(close * math.exp(q[i]))
    return {"median": px(v7.I50), "lo50": px(I25), "hi50": px(I75), "lo80": px(v7.I10), "hi80": px(v7.I90), "lo95": px(v7.I025),
            "hi95": px(v7.I975), "p_up": p_up(m, sd, R), "exp_ret": float(math.expm1(q[v7.I50]))}


def p_up(m, sd, R):
    return float(1 - v7.p_le(0.0, np.array([m]), np.array([sd]), R)[0])


def m_for_p_up(p, sd, R):
    """The centre m at which the calibrated P(up) equals p (P(up) increases with m; bisection)."""
    lo, hi = -6 * sd, 6 * sd
    for _ in range(60):
        mid = (lo + hi) / 2
        if p_up(mid, sd, R) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def combine(p_model, p_qwen, w=None):
    """Amendment E: Qwen first, post-processed by our models."""
    w = W_QWEN if w is None else w
    return float(np.clip(p_model + w * (p_qwen - 0.5), *P_CLIP))


def signal(p):
    return "buy" if p >= BUY_AT else "sell" if p <= SELL_AT else "none"


def cone(close, asof, m, sd, R, days=5, market="nasdaq"):
    """Daily points from the close to the 5-day horizon for the chart. Only the day-5 values are the forecast; the
    days in between scale the centre by h/5 and the spread by sqrt(h/5) (illustrative). NASDAQ dates skip US market
    holidays; SGX dates skip weekends only (no SGX holiday calendar)."""
    from .holidays import is_session
    q = v7.quantiles(np.array([m]), np.array([sd]), R)[0]
    pts, d = [{"date": asof, **{k: close for k in ("median", "lo50", "hi50", "lo80", "hi80", "lo95", "hi95")}}], datetime.fromisoformat(asof)
    for h in range(1, days + 1):
        d += timedelta(days=1)
        while d.weekday() >= 5 or (market == "nasdaq" and not is_session(d.date())):
            d += timedelta(days=1)
        a, b = h / days, math.sqrt(h / days)
        f = lambda i: float(close * math.exp(a * q[v7.I50] + b * (q[i] - q[v7.I50])))
        pts.append({"date": d.strftime("%Y-%m-%d"), "median": f(v7.I50), "lo50": f(I25), "hi50": f(I75), "lo80": f(v7.I10),
                    "hi80": f(v7.I90), "lo95": f(v7.I025), "hi95": f(v7.I975)})
    return pts


# ---------------------------------------------------------------- the prompt and Qwen's verdict

def build_prompt(ctx, news, extra=""):
    sym, mk = ctx["symbol"], ctx["market"]
    base = dist(ctx["close"], ctx["m"], ctx["sd"], ctx["R"])
    L = [f"Stock: {sym} ({ctx['name'] or 'name unknown'}), listed on {'NASDAQ' if mk == 'nasdaq' else 'SGX (Singapore)'}. "
         f"Last close {ctx['close']:.4g} on {ctx['asof']}." + (f" Industry: {ctx['industry']}." if ctx["industry"] else ""), "",
         f"Quantitative facts (percentile within the {ctx['universe']} stocks we forecast; 100th = highest):"]
    for f in ctx["facts"]:
        L.append(f"- {f['label']}: {f['value']}" + (f" ({_ordinal(f['pct'])} percentile)" if f["pct"] is not None else ""))
    if ctx["rank"]:
        L.append(f"- cross-stock model rank: {_ordinal(round(ctx['rank']['score_pct']))} percentile (decile {ctx['rank']['decile']} of 10; "
                 "this model's edge is small)")
    L.append(f"- statistical 5-day forecast without news: median {base['median']:.4g} ({base['exp_ret']:+.1%}), "
             f"80% range {base['lo80']:.4g} to {base['hi80']:.4g}, chance of a rise {base['p_up']:.0%}")
    if ctx["market_news"]:
        mn = ctx["market_news"]
        L.append(f"- market news index over the last 5 sessions: {mn['S']:+.2f} on a -1 (all bad) to +1 (all good) scale, from {mn['n']} headlines")
    if ctx["peers"]:
        L += ["", "Same-industry stocks with the strongest news in the last 5 sessions (news score -2 bad to +2 good):"]
        for p in ctx["peers"]:
            L.append(f"- {p['symbol']} ({p['name'][:40]}): news score {p['news_dir']:+.2f} from {p['n']} headlines"
                     + (f"; strongest: \"{p['headline']}\" (read as {p['read']})" if p["headline"] else ""))
    if extra:
        L += ["", extra]
    L.append("")
    if news is None:
        L.append("No news source is available for this market: judge from the facts only.")
    elif not news["headlines"]:
        L.append(f"No headlines about {sym} in the last {NEWS_DAYS} days.")
    else:
        L.append(f"Headlines about {sym} in the last {NEWS_DAYS} days, newest first:")
        for h in news["headlines"]:
            L.append(f"- {h['t'][:10]}: {h['headline']}")
    L += ["", f"Task: judge how {sym}'s price is likely to move over the next 5 trading days compared with the statistical "
          f"forecast above, which ignores the news. First list up to 3 positives and up to 3 negatives for {sym}. Write each "
          f"one as one sentence in your own words that says what happened and why it matters for {sym}'s price (for example "
          f"\"A recall of its main product could cut next quarter's sales.\"). Do not copy headlines. Peers' news counts "
          f"only if it clearly affects {sym}. Weigh recent days more than older ones. Then give lean (strong_down, down, "
          "neutral, up or strong_up), confidence (low, medium or high) and risk (elevated if an event such as earnings, a "
          "court or regulatory decision, a deal or a trial result is due within the next 5 trading days, or the news is "
          "unusually turbulent; otherwise normal). Pick a direction when the recent news clearly leans one way; use neutral "
          "only when the evidence is balanced or there is almost no news."]
    user = "\n".join(L)
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", user


def parse_verdict(text):
    """Schema-checked verdict; anything unparsable or off-schema counts as neutral / low / normal (amendment C)."""
    try:
        v = json.loads(text)
        out = {"lean": v["lean"], "confidence": v["confidence"], "risk": v["risk"],
               "positives": [str(x)[:240] for x in v.get("positives", [])][:3], "negatives": [str(x)[:240] for x in v.get("negatives", [])][:3]}
        if out["lean"] in LEAN_K and out["confidence"] in CONF_MULT and out["risk"] in RISK_SD:
            return out, True
    except (ValueError, KeyError, TypeError):
        pass
    return {**FALLBACK, "positives": [], "negatives": []}, False


def ask_probability(prompt, json_text, sym, h=5, slot=0, direction="higher"):
    """Amendment E: after its verdict, Qwen's own chance (0-100) that the price is higher in 5 trading days. Qwen writes
    numbers digit by digit, so the first answer token is the tens digit: P_qwen is the expectation over that digit's
    probabilities (digit d counts as 10d + 5; "1" may also start 100, which is rare). This is graded, unlike a one-word
    rise/fall answer, which simply repeated the verdict with ~100% certainty in the hand check.
    Returns (P_qwen, probability mass on digit tokens)."""
    follow = (f"{prompt}{json_text}<|im_end|>\n<|im_start|>user\nGiven all of this, what is the chance (0 to 100) that {sym}'s "
              f"price is {direction} in {h} trading day{'s' if h > 1 else ''} than at the last close? Answer with a whole number only.<|im_end|>\n"
              "<|im_start|>assistant\n")
    body = {"prompt": follow, "n_predict": 1, "temperature": 0, "n_probs": 30, "cache_prompt": True, "id_slot": slot}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    pr = np.zeros(10)
    for x in r["completion_probabilities"][0]["top_logprobs"]:
        w = x["token"].strip()
        if len(w) == 1 and w.isdigit():
            pr[int(w)] += math.exp(x["logprob"])
    mass = float(pr.sum())
    return (float(np.dot(pr / mass, np.arange(10) * 10 + 5)) / 100 if mass > 0 else 0.5), mass


def views(ctx, p_qwen, risk, h=5):
    """The three logged views at horizon h: total (Qwen post-processed by our models), Qwen only, model only."""
    x = ctx["hz"][h] if "hz" in ctx else {"m": ctx["m"], "sd": ctx["sd"], "R": ctx["R"]}
    close, m, sd, R, asof = ctx["close"], x["m"], x["sd"], x["R"], ctx["asof"]
    p_model = p_up(m, sd, R)
    p_tot = combine(p_model, p_qwen)
    sd_t = sd * RISK_SD[risk]
    m_t = m_for_p_up(p_tot, sd_t, R)
    m_q = m_for_p_up(float(np.clip(p_qwen, 0.02, 0.98)), sd, R)
    mk = lambda mm, ss: {**dist(close, mm, ss, R), "cone": cone(close, asof, mm, ss, R, days=h, market=ctx["market"]), "m": mm, "sd": ss}
    return {"total": {**mk(m_t, sd_t), "signal": signal(p_tot), "w": W_QWEN, "sd_mult": RISK_SD[risk]},
            "qwen": {**mk(m_q, sd), "signal": signal(float(np.clip(p_qwen, *P_CLIP)))},
            "model": {**mk(m, sd), "signal": signal(p_model)}}


def generate(prompt, slot=0):
    body = {"prompt": prompt, "n_predict": 700, "temperature": 0, "cache_prompt": True, "id_slot": slot, "json_schema": SCHEMA}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    t = r.get("timings", {})
    return r.get("content", ""), {"prompt_tokens": t.get("prompt_n"), "output_tokens": t.get("predicted_n"),
                                  "tok_per_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None}


# ---------------------------------------------------------------- cache, shadow log, index

def cache_path(market, sym, asof, kind="json"):
    return os.path.join(EVALS, market, f"{sym}_{asof}.{kind}")


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def partial(ctx):
    """What the page can show before Qwen answers: price history and the model-only cone."""
    base = dist(ctx["close"], ctx["m"], ctx["sd"], ctx["R"])
    return {"status": "evaluating", "market": ctx["market"], "symbol": ctx["symbol"], "name": ctx["name"], "asof": ctx["asof"],
            "close": ctx["close"], "history": ctx["history"], "model": {**base, "cone": cone(ctx["close"], ctx["asof"], ctx["m"], ctx["sd"], ctx["R"], market=ctx["market"])},
            "facts": ctx["facts"], "rank": ctx["rank"], "industry": ctx["industry"], "peers": ctx["peers"], "market_news": ctx["market_news"],
            "horizons": {str(h): {"model": {**dist(ctx["close"], x["m"], x["sd"], x["R"]), "cone": cone(ctx["close"], ctx["asof"], x["m"], x["sd"], x["R"], days=h, market=ctx["market"])}}
                         for h, x in sorted(ctx.get("hz", {}).items())}}


def evaluate(market, sym, device="GPU", log=print):
    """Full evaluation of one stock (the llama-server on PORT must be running). Writes the cache, the index entry and
    the shadow-log line; returns the result."""
    t0 = time.time()
    ctx = context(market, sym)
    _dump(partial(ctx), cache_path(market, sym, ctx["asof"], "partial.json"))
    news = own_headlines(sym, ctx["asof"]) if market == "nasdaq" else None
    news_cut = min(datetime.now(timezone.utc), next_open_utc(ctx["asof"]))

    def run(prompt):
        text, timing = generate(prompt)
        verdict, ok = parse_verdict(text)
        pq, lean = {}, {}
        for h in sorted(ctx["hz"]):                                          # v8 amendments F and H
            hi, m_hi = ask_probability(prompt, text, sym, h)
            lo, m_lo = ask_probability(prompt, text, sym, h, direction="lower")
            raw = hi / (hi + lo) if hi + lo > 0 else 0.5                      # symmetric: no lean from the question's wording
            shift = shifts.get(h, 0.0)
            pq[h] = (float(np.clip(raw - shift, 0.02, 0.98)), min(m_hi, m_lo))
            lean[h] = {"p_higher": hi, "p_lower": lo, "p_raw": raw, "shift": shift}
        return {"verdict": verdict, "ok": ok, "pq": pq, "lean": lean, "timing": timing,
                "vws": {h: views(ctx, pq[h][0], verdict["risk"], h) for h in pq}}

    def pack(r):
        return {"ai": {**r["verdict"], "parse_ok": r["ok"], "p_qwen": r["pq"][5][0], "p_mass": round(r["pq"][5][1], 3),
                       "p_qwen_h": {str(h): v[0] for h, v in r["pq"].items()}, **r["timing"]},
                "horizons": {str(h): {"views": v, "signal": v["total"]["signal"], "p_qwen": r["pq"][h][0], "p_mass": round(r["pq"][h][1], 3),
                                      **r["lean"][h]} for h, v in r["vws"].items()},
                "signal": r["vws"][5]["total"]["signal"]}

    shifts = {h: lean_shift(market, h) for h in ctx["hz"]}                   # amendment H, from earlier evaluations only
    A = run(build_prompt(ctx, news)[0])
    # v8 amendment G, variant B: the same analysis with the company's product profile and its related companies' news
    prof, rel, B = None, [], None
    try:
        prof = profiles.load(market, sym, ctx["asof"]) or profiles.build(market, sym, ctx["asof"], name=ctx["name"], port=PORT)
        if prof and "named_by" not in prof:
            prof["named_by"] = profiles.reverse_links(market, sym)
        if prof:
            try:
                rel = profiles.related_news(prof, news_cut)
            except Exception as e:  # noqa: BLE001 - related news is optional
                log(f"[eval] related news for {sym} skipped: {type(e).__name__}: {e}", flush=True)
            moves = peer_moves(market, prof) if market == "sgx" else None
            B = run(build_prompt(ctx, news, extra=profiles.prompt_block(prof, rel, moves))[0])
    except Exception as e:  # noqa: BLE001 - the profile variant must never block the main analysis
        log(f"[eval] product-profile variant for {sym} skipped: {type(e).__name__}: {e}", flush=True)
    main = B or A
    pm, pa = pack(main), pack(A)
    vw = main["vws"][5]
    out = partial(ctx)
    out.update({"status": "done", "version": 3, "generated": datetime.now().isoformat(timespec="seconds"),
                "views": vw, "advised": vw["total"], "model": vw["model"], "signal": pm["signal"], "horizons": pm["horizons"],
                "ai": {**pm["ai"], "model": MODEL_NAME, "device": device, "seconds": round(time.time() - t0, 1)},
                "variant": "B" if B else "A", "variant_a": pa if B else None,
                "profile": prof, "related": rel,
                "news": news, "status_note": "Qwen + models: unproven (forward shadow test, v8 amendments E-G)"})
    _dump(out, cache_path(market, sym, ctx["asof"]))
    try:
        os.remove(cache_path(market, sym, ctx["asof"], "partial.json"))
    except OSError:
        pass
    for variant, r in (("A", A), ("B", B)):
        if not r:
            continue
        for h, v in r["vws"].items():
            cth = {"asof": ctx["asof"], "symbol": sym, "close": ctx["close"], "m": ctx["hz"][h]["m"], "sd": ctx["hz"][h]["sd"]}
            _shadow(market, cth, v["total"]["m"], v["total"]["sd"], r["verdict"], r["ok"], h=h, variant=variant,
                    extra={"p_qwen": r["pq"][h][0], "p_total": v["total"]["p_up"], "m_qwen": v["qwen"]["m"], "signal": v["total"]["signal"], "w": W_QWEN,
                           **r["lean"][h]})
    _index_add(market, ctx["asof"], sym, out)
    desc = lambda r: (f"{r['verdict']['lean']}/{r['verdict']['confidence']}/{r['verdict']['risk']}"
                      + "".join(f" {h}d Qwen {r['pq'][h][0]:.0%} total {r['vws'][h]['total']['p_up']:.0%} [{r['vws'][h]['total']['signal']}]" for h in sorted(r["vws"])))
    log(f"[eval] {market} {sym} {ctx['asof']}: A {desc(A)}" + (f" | B (profile, {len(rel)} related) {desc(B)}" if B else " | no profile")
        + f"; {len(news['headlines']) if news else 0} headlines; {time.time() - t0:.1f}s", flush=True)
    return out


def peer_moves(market, prof, sessions=5):
    """Amendment H: SGX peers' share price moves over the last 5 sessions, from our own SGX data (no SGX news source)."""
    d = _market_dirs(market)
    meta = json.load(open(os.path.join(d["gpu"], "meta.json")))
    T, N = meta["T"], meta["N"]
    lc = np.memmap(os.path.join(d["gpu"], "logclose.f32"), np.float32, "r", shape=(T, N))
    out = {}
    for e in prof.get("competitors", []):
        t = e.get("ticker")
        if t and t in meta["symbols"]:
            j = meta["symbols"].index(t)
            a, b = float(lc[T - 1 - sessions, j]), float(lc[T - 1, j])
            if np.isfinite(a) and np.isfinite(b):
                out[t] = math.expm1(b - a)
    return out


def shadow_path(market, h=5, variant="A"):
    v = "" if variant == "A" else "_prof"                                  # amendment G: B (product profile) has its own logs
    return os.path.join(EVALS, market, f"advice_shadow{v}.jsonl" if h == 5 else f"advice_shadow{v}_h{h}.jsonl")


def _shadow(market, ctx, m2, sd2, verdict, ok, extra=None, h=5, variant="A"):
    path = shadow_path(market, h, variant)
    key = f'"asof": "{ctx["asof"]}", "symbol": "{ctx["symbol"]}"'
    if os.path.exists(path) and key in open(path, encoding="utf-8").read():
        return                                                             # one advised forecast per stock and close
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"asof": ctx["asof"], "symbol": ctx["symbol"], "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "close": ctx["close"], "m": ctx["m"], "sd": ctx["sd"], "m_adv": m2, "sd_adv": sd2,
                            "lean": verdict["lean"], "confidence": verdict["confidence"], "risk": verdict["risk"], "parse_ok": ok,
                            **(extra or {})}) + "\n")


LEAN_WINDOW, LEAN_MIN = 200, 30


def lean_shift(market, h):
    """Amendment H: Qwen's average lean over its last LEAN_WINDOW evaluations at horizon h (raw Qwen chance minus the
    model's), this market's if it has >= LEAN_MIN, else all markets pooled, else 0. Uses no outcomes."""
    def pairs(mk):
        out = []
        for f in sorted(glob.glob(os.path.join(EVALS, mk, "index_*.json")), reverse=True):
            for e in json.load(open(f)).values():
                x = (e.get("h") or {}).get(str(h))
                if x and x.get("sym_q") and x.get("p_model") is not None:      # symmetric-question evaluations only (amendment H)
                    out.append(x["p_raw"] - x["p_model"])
            if len(out) >= LEAN_WINDOW:
                break
        return out[:LEAN_WINDOW]
    own = pairs(market)
    if len(own) >= LEAN_MIN:
        return float(np.mean(own))
    pooled = (own + [d for mk in ("nasdaq", "sgx") if mk != market for d in pairs(mk)])[:LEAN_WINDOW]
    return float(np.mean(pooled)) if len(pooled) >= LEAN_MIN else 0.0


def _index_add(market, asof, sym, out):
    path = os.path.join(EVALS, market, f"index_{asof}.json")
    idx = json.load(open(path)) if os.path.exists(path) else {}
    idx[sym] = {"p_up": out["advised"]["p_up"], "median": out["advised"]["median"], "lo80": out["advised"]["lo80"],
                "hi80": out["advised"]["hi80"], "lean": out["ai"]["lean"], "confidence": out["ai"]["confidence"], "risk": out["ai"]["risk"],
                "signal": out["signal"], "p_qwen": out["ai"]["p_qwen"], "p_model": out["model"]["p_up"],
                "h": {k: {"p_up": v["views"]["total"]["p_up"], "signal": v["signal"], "p_qwen": v["p_qwen"], "p_model": v["views"]["model"]["p_up"],
                          "p_raw": v.get("p_raw", v["p_qwen"]), "sym_q": "p_lower" in v}
                      for k, v in out.get("horizons", {}).items()}}
    _dump(idx, path)


# ---------------------------------------------------------------- the queue and the background evaluator

def enqueue(market, sym, prio):
    os.makedirs(QUEUE, exist_ok=True)
    if any(f.endswith(f"-{market}-{sym}.req") and f.startswith(f"{prio}-") for f in os.listdir(QUEUE)):
        return
    open(os.path.join(QUEUE, f"{prio}-{int(time.time() * 1000)}-{market}-{sym}.req"), "w").close()


def queued():
    if not os.path.isdir(QUEUE):
        return []
    out = []
    for f in sorted(os.listdir(QUEUE)):
        if f.endswith(".req"):
            prio, _, market, sym = f[:-4].split("-", 3)
            out.append((f, int(prio), market, sym))
    return out


def _state(**kw):
    _dump({"pid": os.getpid(), "heartbeat": time.time(), "time": datetime.now().isoformat(timespec="seconds"), **kw}, STATE)


def _alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259                                              # STILL_ACTIVE


def _news_reading():
    """True while the app-start refresh is reading the news with its own model (never two models on the GPU)."""
    try:
        s = json.load(open(REFRESH_STATUS))
        return s.get("state") == "running" and "news" in s.get("step", "") and time.time() - os.path.getmtime(REFRESH_STATUS) < 3600
    except (OSError, ValueError):
        return False


def _app_open(url):
    try:
        urllib.request.urlopen(url + "/api/rankings/status", timeout=3).read()
        return True
    except Exception:
        return False


def _tdr_reports():
    try:
        return set(os.listdir(r"C:\Windows\LiveKernelReports\WATCHDOG"))
    except OSError:
        return set()


def serve(a):
    if not (a.allow_gpu or a.cpu):
        raise SystemExit("choose --allow-gpu (only while you are at the PC) or --cpu")
    os.makedirs(EVALS, exist_ok=True)
    if a.log:                                                              # started hidden by the app: no console
        sys.stdout = sys.stderr = open(a.log, "a", buffering=1, encoding="utf-8")
    lockf = open(LOCK, "a+")
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(lockf.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("[eval] another evaluator is running", flush=True)
            return
    # a llama-server left on PORT by an evaluator that died is ours to stop
    try:
        old = json.load(open(STATE)).get("server_pid")
        if old and _alive(old):
            subprocess.run(["taskkill", "/PID", str(old), "/F"], capture_output=True)
    except (OSError, ValueError):
        pass
    gpu = a.allow_gpu and not a.cpu
    device = "GPU (Vulkan0)" if gpu else "CPU"
    tdr0, srv, idle_since, misses, done = _tdr_reports(), None, time.time(), 0, 0

    def stop_server():
        nonlocal srv
        if srv:
            srv.close()
            srv = None

    try:
        while True:
            if gpu and _tdr_reports() - tdr0:
                stop_server()
                _state(state="stopped", note="a GPU driver reset was reported: evaluation stopped", done=done)
                print("[eval] WARNING: new GPU driver reset report; stopping", flush=True)
                return
            if not _app_open(a.app_url):
                misses += 1
                if misses >= 2:
                    _state(state="stopped", note="the app was closed", done=done)
                    return
            else:
                misses = 0
            jobs = queued()
            for f, prio, market, sym in jobs:                              # clicked stocks: chart data straight away
                if prio == 0:
                    try:
                        asof = latest_asof(market)
                        if asof and not os.path.exists(cache_path(market, sym, asof)) and not os.path.exists(cache_path(market, sym, asof, "partial.json")):
                            _dump(partial(context(market, sym)), cache_path(market, sym, asof, "partial.json"))
                    except Exception as e:  # noqa: BLE001 - reported when the job runs
                        print(f"[eval] partial {market} {sym}: {type(e).__name__}: {e}", flush=True)
            if not jobs:
                if time.time() - idle_since > IDLE_EXIT_S:
                    _state(state="stopped", note="idle: model unloaded", done=done)
                    return
                _state(state="idle", queue=0, done=done, server_pid=srv.proc.pid if srv and srv.proc else None)
                time.sleep(1)
                continue
            if _news_reading():
                stop_server()
                _state(state="waiting", note="waiting for the news reading to finish", queue=len(jobs), done=done)
                time.sleep(3)
                continue
            f, prio, market, sym = jobs[0]
            try:
                asof = latest_asof(market)
                if asof and not os.path.exists(cache_path(market, sym, asof)):
                    if srv is None:
                        _state(state="loading", note="loading the AI model", current={"market": market, "symbol": sym}, queue=len(jobs), done=done)
                        srv = llm_news.Server(gpu, ctx_per_slot=CTX, slots=SLOTS, port=PORT)
                    _state(state="running", current={"market": market, "symbol": sym}, queue=len(jobs), done=done,
                           server_pid=srv.proc.pid if srv.proc else None)
                    evaluate(market, sym, device=device)
                    done += 1
            except NotForecast as e:
                _dump({"status": "error", "error": str(e), "market": market, "symbol": sym, "asof": asof},
                      cache_path(market, sym, asof or "none", "error.json"))
            except Exception as e:  # noqa: BLE001 - one bad stock must not stop the queue
                print(f"[eval] {market} {sym} failed: {type(e).__name__}: {e}", flush=True)
                _dump({"status": "error", "error": f"{type(e).__name__}: {e}", "market": market, "symbol": sym, "asof": asof},
                      cache_path(market, sym, asof or "none", "error.json"))
            for g, _, m2, s2 in jobs:                                      # drop this stock's duplicates too
                if m2 == market and s2 == sym:
                    try:
                        os.remove(os.path.join(QUEUE, g))
                    except OSError:
                        pass
            idle_since = time.time()
    finally:
        stop_server()
        if os.path.exists(STATE):
            try:
                s = json.load(open(STATE))
                if s.get("state") not in ("stopped",):
                    _state(state="stopped", note="evaluator exited", done=done)
            except (OSError, ValueError):
                pass


def batch(a=None):
    """App-start queue: first the NASDAQ start-up set (top/bottom 25 by rank score, top/bottom 10 by model P(up)) at
    priority 1, then every other ranked NASDAQ and SGX stock at priority 2; only stocks not yet evaluated for the
    latest close. Clicked stocks (priority 0) always go first. Runs only while the app is open (see serve)."""
    n = 0
    for market in ("nasdaq", "sgx"):
        d = _market_dirs(market)
        rk = json.load(open(os.path.join(d["v6"], "rankings.json")))
        asof = latest_asof(market)
        rows = rk["rankings"]
        first = []
        if market == "nasdaq":
            first = [r["symbol"] for r in rows[:25] + rows[-25:]]
            fc = sorted([r for r in rows if r.get("fc")], key=lambda r: r["fc"]["p_up"])
            first += [r["symbol"] for r in fc[:10] + fc[-10:]]
        first = list(dict.fromkeys(first))
        rest = [r["symbol"] for r in rows if r["symbol"] not in set(first)]
        for prio, syms in ((1, first), (2, rest)):
            for s in syms:
                if not os.path.exists(cache_path(market, s, asof)) and not os.path.exists(cache_path(market, s, asof, "error.json")):
                    enqueue(market, s, prio)
                    n += 1
        print(f"[eval] app-start queue: {market} {asof}", flush=True)
    _market_dirs("nasdaq")
    print(f"[eval] app-start queue: {n} stocks queued", flush=True)
    return n


def start_daemon(app_url="http://127.0.0.1:5080"):
    """Start the background evaluator (hidden) unless one is running."""
    try:
        s = json.load(open(STATE))
        if s.get("state") != "stopped" and _alive(s["pid"]) and time.time() - s["heartbeat"] < 60:
            return False
    except (OSError, ValueError, KeyError):
        pass
    subprocess.Popen([sys.executable, "-W", "ignore", "-m", "trademaster_ml.stock_eval", "serve", "--allow-gpu", "--app-url", app_url,
                      "--log", os.path.join("private", "reports", "jobs", "stock_eval.log")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return True


# ---------------------------------------------------------------- forward test

def score(a):
    market = getattr(a, "market", "nasdaq") if a else "nasdaq"
    H = getattr(a, "horizon", 5) if a else 5
    variant = getattr(a, "variant", "A") if a else "A"
    d = _market_dirs(market)
    path = shadow_path(market, H, variant)
    v7dir = d["v7"] if H == 5 else os.path.join(d["v7"], f"h{H}")
    if not os.path.exists(path):
        print(f"[eval] {market}: no advised forecasts yet", flush=True)
        return None
    meta = json.load(open(os.path.join(d["gpu"], "meta.json")))
    T, N = meta["T"], meta["N"]
    di = {x: i for i, x in enumerate(meta["dates"])}
    si = {x: i for i, x in enumerate(meta["symbols"])}
    lc = np.memmap(os.path.join(d["gpu"], "logclose.f32"), np.float32, "r", shape=(T, N))
    by_date = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_date.setdefault(r["asof"], []).append(r)
    if variant == "B" and os.path.exists(shadow_path(market, H)):           # amendment G: pair B with A on the same stock and close
        pa = {(x["asof"], x["symbol"]): x for x in map(json.loads, open(shadow_path(market, H), encoding="utf-8"))}
        for rows in by_date.values():
            for r in rows:
                x = pa.get((r["asof"], r["symbol"]))
                r["m_a"], r["sd_a"] = (x["m_adv"], x["sd_adv"]) if x else (np.nan, np.nan)
    days = []
    for asof, rows in sorted(by_date.items()):
        t = di.get(asof)
        if t is None or t + H > T - 1:
            continue
        rec = _shadow_record(v7dir, asof)
        if not rec:
            continue
        R = np.array(rec["R_v7"])
        rows = [r for r in rows if r["symbol"] in si]
        j = np.array([si[r["symbol"]] for r in rows])
        y = np.full(len(j), np.nan)
        for k in range(1, H + 1):
            v = lc[t + k, j].astype(np.float64)
            y = np.where(np.isfinite(v), v - lc[t, j], y)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        g = lambda key: np.array([r.get(key, np.nan) for r in rows], float)[ok]
        m, sd, ma, sa, mq, y = g("m"), g("sd"), g("m_adv"), g("sd_adv"), g("m_qwen"), y[ok]
        sig = np.array([r.get("signal", "none") for r in rows])[ok]
        qa = v7.quantiles(ma, sa, R)
        hit = lambda mm: float(np.mean(np.sign(mm[(mm != 0) & (y != 0)]) == np.sign(y[(mm != 0) & (y != 0)]))) if ((mm != 0) & (y != 0)).any() else float("nan")
        days.append({"asof": asof, "n": int(ok.sum()), "crps_adv": float(v7.crps_rows(qa, y).mean()),
                     "crps_model": float(v7.crps_rows(v7.quantiles(m, sd, R), y).mean()), "hit_adv": hit(ma), "hit_model": hit(m),
                     "cov80_adv": float(np.mean((y >= qa[:, v7.I10]) & (y <= qa[:, v7.I90]))),
                     "crps_qwen": float(v7.crps_rows(v7.quantiles(mq, sd, R), y).mean()) if np.isfinite(mq).all() else None,
                     "hit_qwen": hit(mq) if np.isfinite(mq).all() else None,
                     "buy_n": int((sig == "buy").sum()), "buy_right": int(((sig == "buy") & (y > 0)).sum()),
                     "sell_n": int((sig == "sell").sum()), "sell_right": int(((sig == "sell") & (y < 0)).sum())})
        if variant == "B":
            m_a, s_a = g("m_a"), g("sd_a")
            pr = np.isfinite(m_a)
            if pr.any():
                yy = y[pr]
                hp = lambda mm: float(np.mean(np.sign(mm[(mm != 0) & (yy != 0)]) == np.sign(yy[(mm != 0) & (yy != 0)]))) if ((mm != 0) & (yy != 0)).any() else float("nan")
                days[-1].update({"n_pair": int(pr.sum()), "crps_b_pair": float(v7.crps_rows(v7.quantiles(ma[pr], sa[pr], R), yy).mean()),
                                 "crps_a_pair": float(v7.crps_rows(v7.quantiles(m_a[pr], s_a[pr], R), yy).mean()),
                                 "hit_b_pair": hp(ma[pr]), "hit_a_pair": hp(m_a[pr])})
    n_rows = sum(x["n"] for x in days)
    out = {"market": market, "horizon": H, "matured_dates": len(days), "matured_forecasts": n_rows, "needed_dates": 60, "needed_forecasts": 300,
           "logged": sum(len(v) for v in by_date.values()), "days": days, "status": "collecting"}
    if len(days) >= 5:
        from .daily_v4 import block_bootstrap_diff
        out["crps_adv_minus_model"] = block_bootstrap_diff([x["crps_adv"] for x in days], [x["crps_model"] for x in days], block=max(5, H))
        w = np.array([x["n"] for x in days], float)
        for k in ("hit_adv", "hit_model", "cov80_adv"):
            v = np.array([x[k] for x in days])
            out[k] = float(np.nansum(v * w) / w[np.isfinite(v)].sum())
    for k in ("buy", "sell"):
        n, r = sum(x.get(f"{k}_n", 0) for x in days), sum(x.get(f"{k}_right", 0) for x in days)
        if n:
            ph, z = r / n, 1.96                                            # Wilson 95% interval
            c, h = (ph + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            out[f"{k}_signals"] = {"n": n, "right": r, "rate": ph, "ci95": [c - h, c + h]}
    if len(days) >= 60 and n_rows >= 300:
        promoted = out["crps_adv_minus_model"][2] < 0 and out["hit_adv"] >= out["hit_model"] and 0.75 <= out["cov80_adv"] <= 0.85
        out["status"] = "promoted" if promoted else "not promoted: advice weights set to 0"
    if variant == "B":                                                     # amendment G decision: B vs A on paired forecasts
        pdays = [x for x in days if "n_pair" in x]
        ab = {"paired_dates": len(pdays), "paired_forecasts": sum(x["n_pair"] for x in pdays), "status": "collecting"}
        if len(pdays) >= 5:
            from .daily_v4 import block_bootstrap_diff
            ab["crps_b_minus_a"] = block_bootstrap_diff([x["crps_b_pair"] for x in pdays], [x["crps_a_pair"] for x in pdays], block=max(5, H))
            w = np.array([x["n_pair"] for x in pdays], float)
            for k in ("hit_b_pair", "hit_a_pair"):
                v = np.array([x[k] for x in pdays])
                ab[k] = float(np.nansum(v * w) / w[np.isfinite(v)].sum())
        if len(pdays) >= 60 and ab["paired_forecasts"] >= 300:
            keep_b = ab["crps_b_minus_a"][2] < 0 and ab["hit_b_pair"] >= ab["hit_a_pair"]
            ab["status"] = "B kept: replaces A" if keep_b else "B dropped: the page goes back to A"
        out["b_vs_a"] = ab
    suffix = ("" if variant == "A" else "_prof") + ("" if H == 5 else f"_h{H}")
    _dump(out, os.path.join(EVALS, market, f"advice_score{suffix}.json"))
    print(f"[eval] {market} advice forward test: {len(days)} of 60 dates, {n_rows} of 300 forecasts matured ({out['status']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "serve", "batch", "score", "prompt"])
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--market", choices=["nasdaq", "sgx"], default="nasdaq")
    ap.add_argument("--horizon", type=int, choices=[1, 5, 20], default=5)
    ap.add_argument("--variant", choices=["A", "B"], default="A")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--app-url", default="http://127.0.0.1:5080")
    ap.add_argument("--log", help="serve: append output to this file")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a)
    elif a.cmd == "batch":
        batch(a)
    elif a.cmd == "score":
        score(a)
    elif a.cmd == "prompt":                                                # show the exact prompt (no model needed)
        for s in a.symbols:
            ctx = context(a.market, s)
            print(build_prompt(ctx, own_headlines(s, ctx["asof"]) if a.market == "nasdaq" else None)[1])
    else:
        if not (a.allow_gpu or a.cpu):
            raise SystemExit("choose --allow-gpu (only while you are at the PC) or --cpu")
        gpu = a.allow_gpu and not a.cpu
        srv = llm_news.Server(gpu, ctx_per_slot=CTX, slots=SLOTS, port=PORT)
        try:
            for s in a.symbols:
                r = evaluate(a.market, s, device="GPU (Vulkan0)" if gpu else "CPU")
                for x in r["ai"]["positives"]:
                    print(f"    + {x}")
                for x in r["ai"]["negatives"]:
                    print(f"    - {x}")
        finally:
            srv.close()


if __name__ == "__main__":
    sys.exit(main())
