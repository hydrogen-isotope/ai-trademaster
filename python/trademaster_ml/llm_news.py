"""
v8 stage 1: a local LLM (Qwen2.5-7B-Instruct via llama.cpp) reads the last few days of news headlines.
Research only; protocol: docs/FORECAST-V8-STAGES-PROTOCOL.md (amendment B: rolling news only, judged by a forward
shadow test). Each headline is read by Qwen once; afterwards only Qwen's compact verdict is kept (the raw item shrinks to
id + time), and everything older than KEEP_DAYS is deleted on every run.

  rolling   fetch the last 7 days of headlines (top 300 stocks by dollar volume + market ETFs), score the new ones,
            prune old ones, and write today's features -> private/data/news/news_today.json
            (--allow-gpu only while the user is at the PC, e.g. when the app is open; otherwise --cpu)
  sample    score 200 recent headlines and write them out for a hand check
  bench     llama-bench speed on this PC

Usage: python -m trademaster_ml.llm_news rolling --allow-gpu | rolling --cpu | sample --allow-gpu | bench --allow-gpu
"""
import argparse, glob, json, math, os, random, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import alpaca

NEWS = os.path.join("private", "data", "news")
ITEMS = os.path.join(NEWS, "rolling_items.jsonl")
SCORES = os.path.join(NEWS, "rolling_scores.jsonl")
TODAY = os.path.join(NEWS, "news_today.json")
LLM = os.path.join("private", "models", "llm")
NY = ZoneInfo("America/New_York")
MARKET_ETFS = ["QQQ", "SPY", "DIA", "IWM", "SMH", "SOXX", "XLK", "TLT"]
TOP_K = 300
FETCH_DAYS, KEEP_DAYS, WINDOW_SESSIONS = 7, 10, 5
PORT = 8765
KINDS = ("topic", "dir", "mdir")
THREADS = 10
SLOTS = THREADS * len(KINDS)                  # one server slot per (worker, question kind): instructions stay cached

SYSTEM = "You are a careful equity analyst. Answer each question about a news headline with exactly one word."
QUESTIONS = {
    "dir": ("Is the headline below good, neutral or bad news for the stock price of the stock named below it? Answer good, neutral or bad.",
            ["good", "neutral", "bad"]),
    "topic": ("Is the headline below mainly about artificial intelligence (AI chips, AI models, AI spending or demand), the overall stock "
              "market or economy (interest rates, inflation, the Fed, tariffs, recession, broad sector moves), or other? "
              "Answer ai, market or other.", ["ai", "market", "other"]),
    "mdir": ("Is the headline below good, neutral or bad news for the overall US stock market? Answer good, neutral or bad.",
             ["good", "neutral", "bad"]),
}
# amendment A: templated recaps / lists / previews are not scored (weight 0)
RECAP = re.compile(r"(^What'?s Going On With|^Why .{1,60}(Shares|Stock)s? (Is|Are) (Trading|Falling|Rising|Moving|Sliding|Soaring|Diving|"
                   r"Up|Down|Higher|Lower)|Stocks? Moving|Whale (Activity|Alert)|^Earnings Scheduled For|Here'?s How Much|Options? Activity|"
                   r"Stocks? To Watch|Preview|On Investors'? Radars?|52-Week (High|Low)|Circuit Breaker|Price Over Earnings|P/E Ratio|"
                   r"Short Interest|Peer Comparison|A Look At|Insights|Analyst Ratings|Trading (Higher|Lower)|Shares Are|Stock Is (Up|Down)|"
                   r"Market Cap|Top \d+|^\d+ .{0,40}Stocks)", re.I)
MAX_TICKERS = 2


# ---------------------------------------------------------------- llama.cpp server

def model_path():
    m = sorted(glob.glob(os.path.join(LLM, "qwen2.5-7b", "*00001-of-*.gguf")))
    if not m:
        raise SystemExit("Qwen model not found under private/models/llm/qwen2.5-7b")
    return m[0]


def server_exe(name="llama-server.exe"):
    m = sorted(glob.glob(os.path.join(LLM, "llama-b*", name)))
    if not m:
        raise SystemExit("llama.cpp not found under private/models/llm")
    return m[-1]


def _health(port=PORT):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok"
    except Exception:
        return False


class Server:
    """llama-server on the RX 7800 XT only (Vulkan0), or CPU-only. Reuses a running server; stops one it started."""

    def __init__(self, gpu, ctx_per_slot=1024, slots=SLOTS, port=PORT):
        self.proc = None
        if _health(port):
            return
        args = [server_exe(), "-m", model_path(), "-np", str(slots), "-c", str(ctx_per_slot * slots), "--port", str(port), "--host", "127.0.0.1"]
        args += ["-ngl", "99", "--device", "Vulkan0"] if gpu else ["-ngl", "0", "--device", "none", "--no-op-offload"]   # "none": no GPU at all (with -ngl 0 alone, llama.cpp still offloads prompt batches to the GPU)
        os.makedirs(os.path.join("private", "reports", "jobs"), exist_ok=True)
        log = open(os.path.join("private", "reports", "jobs", "llama_server.log"), "a")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(240):
            if _health(port):
                return
            time.sleep(1)
        raise RuntimeError("llama-server did not start (see private/reports/jobs/llama_server.log)")

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)


def _start(a):
    if not (a.allow_gpu or a.cpu):
        raise SystemExit("choose --allow-gpu (only while you are at the PC) or --cpu")
    return Server(gpu=a.allow_gpu and not a.cpu)


_slot = threading.local()
_slot_lock = threading.Lock()
_slot_next = [0]


def _worker():
    if not hasattr(_slot, "id"):
        with _slot_lock:
            _slot.id = _slot_next[0] % THREADS
            _slot_next[0] += 1
    return _slot.id


def ask(headline, kind, stock=None):
    """Probabilities of the one-word answers, read from the model's first answer token (renormalised over the options).
    The fixed instructions come first and each slot only ever asks one kind of question, so they stay cached."""
    question, options = QUESTIONS[kind]
    tail = f"Headline: {headline}" + (f"\nStock: {stock}" if stock else "")
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}\n\n{tail}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 20, "cache_prompt": True,
            "id_slot": KINDS.index(kind) * THREADS + _worker()}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    p = {o: 0.0 for o in options}
    for x in r["completion_probabilities"][0]["top_logprobs"]:
        w = x["token"].strip().lower()
        if w in p:
            p[w] += math.exp(x["logprob"])
    s = sum(p.values())
    return ({o: v / s for o, v in p.items()} if s > 0 else {o: 1 / len(options) for o in options}), s


def score_item(item, wanted):
    h = item["headline"].strip()[:300]
    out = {"id": item["id"], "t": item["t"], "headline": h[:140], "dir": {}}
    cov = 1.0
    for t in [t for t in item.get("symbols", []) if t in wanted][:MAX_TICKERS]:
        p, c = ask(h, "dir", stock=t)
        out["dir"][t] = round(p["good"] - p["bad"], 4)
        cov = min(cov, c)
    p, c = ask(h, "topic")
    out["ai"], out["mkt"] = round(p["ai"], 4), round(p["market"], 4)
    cov = min(cov, c)
    if out["ai"] + out["mkt"] >= 0.5:
        p, c = ask(h, "mdir")
        out["mdir"] = round(p["good"] - p["bad"], 4)
        cov = min(cov, c)
    out["cov"] = round(cov, 4)
    return out


# ---------------------------------------------------------------- rolling store

def top_symbols():
    """The current top TOP_K universe stocks by 20-day dollar volume (latest day of the v6 feature panel)."""
    gdir = os.path.join("private", "data", "us", "gpu")
    meta = json.load(open(os.path.join(gdir, "meta.json")))
    fmeta = json.load(open(os.path.join(gdir, "features.json")))
    T, N = meta["T"], meta["N"]
    raw = np.memmap(os.path.join(gdir, "raw.f32"), np.float32, "r", shape=(len(fmeta["features"]), T, N))
    dv = np.array(raw[fmeta["features"].index("log_dollar_vol20")][T - 1])
    mask = np.fromfile(os.path.join(gdir, "mask.f32"), np.float32).reshape(T, N)[T - 1] > 0.5
    x = np.where(mask & np.isfinite(dv), dv, -np.inf)
    return [meta["symbols"][j] for j in np.argsort(-x)[:TOP_K] if np.isfinite(x[j])], meta["dates"][-1]


def _read(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def fetch_recent(syms):
    creds = alpaca.credentials()
    old = _read(ITEMS)
    have = {x["id"] for x in old}
    start_dt = datetime.now(timezone.utc) - timedelta(days=FETCH_DAYS)
    if old:                                     # stream only what arrived since the newest stored headline (1 h overlap)
        newest = max(datetime.fromisoformat(x["t"].replace("Z", "+00:00")) for x in old)
        start_dt = max(start_dt, newest - timedelta(hours=1))
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    new = []
    for g in [syms[i:i + 100] for i in range(0, len(syms), 100)] + [MARKET_ETFS]:
        params = {"symbols": ",".join(g), "start": start, "limit": 50, "sort": "asc", "include_content": "false"}
        token = None
        while True:
            if token:
                params["page_token"] = token
            doc = alpaca._get("/v1beta1/news", params, creds)
            for n in doc.get("news", []):
                if n["id"] not in have and n.get("headline"):
                    have.add(n["id"])
                    new.append({"id": n["id"], "t": n["created_at"], "symbols": n.get("symbols", []), "headline": n["headline"]})
            token = doc.get("next_page_token")
            time.sleep(0.35)
            if not token:
                break
    keep_from = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = [x for x in old + new if x["t"] >= keep_from]
    _write(ITEMS, items)
    return items, len(new)


def session_of(ts, sessions):
    """Trading session a headline counts for: up to 16:00 New York time it counts for that day's close, later for the
    next session. Returns an index into `sessions` (len(sessions) = after the last known close)."""
    t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY)
    day = t.strftime("%Y-%m-%d")
    if t.hour >= 16:
        day = day + "~"                                                      # after the close -> next session
    return int(np.searchsorted(np.array(sessions), day, side="left"))


def features(scores, sessions, asof, wanted_stocks, now=None):
    """Today's point-in-time news features for forecasts made from the close `asof`.
    Window: the WINDOW_SESSIONS sessions ending at `asof`, plus headlines after that close but before the next market
    open (09:30 New York on the next weekday) if the forecast is made before it; never anything later."""
    now = now or datetime.now(timezone.utc)
    t_asof = sessions.index(asof)
    first = max(0, t_asof - WINDOW_SESSIONS + 1)
    nxt = datetime.fromisoformat(asof).replace(tzinfo=NY) + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    cutoff = min(now, nxt.replace(hour=9, minute=30).astimezone(timezone.utc))
    stock = {}
    mkt_w = mkt_s = ai_w = ai_s = 0.0
    n_mkt = 0
    headlines = []
    for r in scores:
        ts = datetime.fromisoformat(r["t"].replace("Z", "+00:00"))
        k = session_of(r["t"], sessions)
        if k < first or ts > cutoff:
            continue
        for sym, d in r.get("dir", {}).items():
            if sym in wanted_stocks:
                s = stock.setdefault(sym, {"sum": 0.0, "n": 0, "items": []})
                s["sum"] += d
                s["n"] += 1
                s["items"].append((d, r["headline"], r["t"]))
        if "mdir" in r:
            w = r["ai"] + r["mkt"]
            mkt_w += w
            mkt_s += w * r["mdir"]
            n_mkt += 1
            if r["ai"] > 0.5:
                ai_w += r["ai"]
                ai_s += r["ai"] * r["mdir"]
            headlines.append({"t": r["t"], "headline": r["headline"], "mdir": r["mdir"], "ai": r["ai"], "mkt": r["mkt"]})
    out_st = {}
    for sym, s in stock.items():
        s["items"].sort(key=lambda x: -abs(x[0]))
        out_st[sym] = {"news_dir": float(np.clip(s["sum"] / math.sqrt(s["n"]), -2, 2)), "n": s["n"],
                       "top": [{"dir": d, "headline": h, "t": t} for d, h, t in s["items"][:3]]}
    S = mkt_s / mkt_w if n_mkt >= 10 and mkt_w > 0 else 0.0
    headlines.sort(key=lambda x: x["mdir"])
    return {"asof": asof, "window": [sessions[first], cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")], "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "market": {"S": float(S), "n": n_mkt, "ai_anxiety": float(-ai_s / ai_w) if ai_w > 0 else None,
                       "most_negative": headlines[:6], "most_positive": headlines[::-1][:6]},
            "stocks": out_st}


def rolling(a):
    t0 = time.time()
    os.makedirs(NEWS, exist_ok=True)
    syms, asof = top_symbols()
    items, n_new = fetch_recent(syms)
    wanted = set(syms) | set(MARKET_ETFS)
    have = {r["id"]: r for r in _read(SCORES)}
    # compacted items (id + time only) were already processed: scored, or skipped as templated recaps
    todo = [x for x in items if x["id"] not in have and x.get("headline") and not RECAP.search(x["headline"])]
    print(f"news: {len(items)} headlines in the last {KEEP_DAYS} days ({n_new} new), {len(todo)} to score", flush=True)
    if todo:
        srv = _start(a)
        try:
            with ThreadPoolExecutor(THREADS) as ex:
                for r in ex.map(lambda x: score_item(x, wanted), todo):
                    have[r["id"]] = r
        finally:
            srv.close()
    keep_ids = {x["id"] for x in items}
    scores = [r for i, r in have.items() if i in keep_ids]                   # prune with the items
    _write(SCORES, scores)
    # compress: once Qwen has read a headline, keep only its compact verdict (scores file, with a short headline for
    # display); the raw item shrinks to id + time, which is all that dedupe and pruning need
    done = set(have) | {x["id"] for x in items if x.get("headline") and RECAP.search(x["headline"])}
    _write(ITEMS, [{"id": x["id"], "t": x["t"]} if x["id"] in done else x for x in items])
    sessions = json.load(open(os.path.join("private", "data", "us", "gpu", "meta.json")))["dates"]
    f = features(scores, sessions, asof, set(syms))
    tmp = TODAY + ".tmp"
    json.dump(f, open(tmp, "w"))
    os.replace(tmp, TODAY)
    m = f["market"]
    print(f"news features for close {asof}: {len(f['stocks'])} stocks with news, market index {m['S']:+.3f} from {m['n']} "
          f"market/AI headlines, AI-anxiety {m['ai_anxiety'] if m['ai_anxiety'] is None else round(m['ai_anxiety'], 3)} "
          f"({time.time() - t0:.0f} s)", flush=True)


def sample(a):
    syms, _ = top_symbols()
    items, _ = fetch_recent(syms)
    items = [x for x in items if not RECAP.search(x["headline"])]
    random.Random(7).shuffle(items)
    wanted = set(syms) | set(MARKET_ETFS)
    srv = _start(a)
    t0 = time.time()
    try:
        with ThreadPoolExecutor(THREADS) as ex:
            res = list(ex.map(lambda x: score_item(x, wanted), items[:200]))
    finally:
        srv.close()
    os.makedirs(os.path.join("private", "reports", "v8"), exist_ok=True)
    with open(os.path.join("private", "reports", "v8", "llm_sample.txt"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(f"{r['t'][:16]} | {r['headline'][:110]}\n    dir {r['dir']}  ai {r['ai']:.2f}  mkt {r['mkt']:.2f}"
                    f"{'  market-dir %+.2f' % r['mdir'] if 'mdir' in r else ''}  cov {r['cov']:.2f}\n")
    print(f"scored {len(res)} headlines in {time.time() - t0:.0f} s", flush=True)


def bench(a):
    args = [server_exe("llama-bench.exe"), "-m", model_path(), "-p", "512", "-n", "32"]
    args += ["-ngl", "99", "-dev", "Vulkan0"] if a.allow_gpu else ["-ngl", "0", "-dev", "none"]
    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rolling", "sample", "bench"])
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
