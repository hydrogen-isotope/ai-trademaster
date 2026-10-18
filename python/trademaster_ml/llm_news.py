"""
v8 stage 1: a local LLM (Qwen2.5-7B-Instruct, llama.cpp) reads news headlines. Research only; protocol:
docs/FORECAST-V8-STAGES-PROTOCOL.md. Look-ahead control: only headlines from 2024-07-01 (after the model's training
data) are scored or used.

  fetch      Alpaca/Benzinga headlines since 2024-07-01 for every stock that was in the top 300 by dollar volume at a
             month start in that window, plus market ETFs -> private/data/news/v8_items.jsonl (resumable)
  bench      llama-bench speed on this PC (GPU needs --allow-gpu; the GPU is only used while the user is present)
  sample     score 200 random headlines and print them for a hand check
  score      score unscored headlines -> private/data/news/v8_llm.jsonl (resumable; --allow-gpu or --cpu; --max-minutes 120)
  aggregate  point-in-time daily features -> private/data/news/v8_news.npz (aligned to the v6 panel)

Usage: python -m trademaster_ml.llm_news fetch | bench | sample | score | aggregate  [--allow-gpu | --cpu]
"""
import argparse, glob, json, math, os, random, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import alpaca

NEWS = os.path.join("private", "data", "news")
ITEMS = os.path.join(NEWS, "v8_items.jsonl")
STATE = os.path.join(NEWS, "v8_fetch_state.json")
SCORES = os.path.join(NEWS, "v8_llm.jsonl")
AGG = os.path.join(NEWS, "v8_news.npz")
LLM = os.path.join("private", "models", "llm")
START = "2024-07-01"                          # model training data reportedly ends 2024-06-30
NY = ZoneInfo("America/New_York")
MARKET_ETFS = ["QQQ", "SPY", "DIA", "IWM", "SMH", "SOXX", "XLK", "TLT"]
TOP_K = 300
PORT = 8765
SLOTS = 32


# ---------------------------------------------------------------- fetch

def universe_window_symbols():
    """Symbols in the top TOP_K by 20-day dollar volume (inside the v6 universe) at any month start since START."""
    gdir = os.path.join("private", "data", "us", "gpu")
    meta = json.load(open(os.path.join(gdir, "meta.json")))
    fmeta = json.load(open(os.path.join(gdir, "features.json")))
    T, N = meta["T"], meta["N"]
    F = len(fmeta["features"])
    raw = np.memmap(os.path.join(gdir, "raw.f32"), np.float32, "r", shape=(F, T, N))
    dv = raw[fmeta["features"].index("log_dollar_vol20")]
    mask = np.fromfile(os.path.join(gdir, "mask.f32"), np.float32).reshape(T, N) > 0.5
    dates = meta["dates"]
    keep = set()
    last_month = None
    for t, d in enumerate(dates):
        if d < START or d[:7] == last_month:
            continue
        last_month = d[:7]
        x = np.where(mask[t] & np.isfinite(dv[t]), dv[t], -np.inf)
        top = np.argsort(-x)[:TOP_K]
        keep.update(meta["symbols"][j] for j in top if np.isfinite(x[j]))
    return sorted(keep)


def fetch():
    os.makedirs(NEWS, exist_ok=True)
    syms = universe_window_symbols()
    groups = [syms[i:i + 100] for i in range(0, len(syms), 100)] + [MARKET_ETFS]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    seen = set()
    if os.path.exists(ITEMS):
        for line in open(ITEMS, encoding="utf-8"):
            try:
                seen.add(json.loads(line)["id"])
            except ValueError:
                pass
    months = []
    d = datetime.fromisoformat(START)
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    while d < end:
        nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        months.append((d.strftime("%Y-%m-%d"), min(nxt, end + timedelta(days=1)).strftime("%Y-%m-%d")))
        d = nxt
    jobs = [(g, gi, m) for gi, g in enumerate(groups) for m in months]
    creds = alpaca.credentials()
    lock = threading.Lock()
    out = open(ITEMS, "a", encoding="utf-8")
    print(f"news for {len(syms)} stocks + {len(MARKET_ETFS)} market ETFs, {len(months)} months, {len(seen)} items already stored", flush=True)
    t0, done = time.time(), [0]

    def one(job):
        g, gi, (a, b) = job
        key = f"{gi}:{a}"
        current_month = b > datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if state.get(key) == "done" and not current_month:
            return
        params = {"symbols": ",".join(g), "start": a + "T00:00:00Z", "end": b + "T00:00:00Z", "limit": 50, "sort": "asc",
                  "include_content": "false"}
        token = None
        while True:
            if token:
                params["page_token"] = token
            t1 = time.time()
            doc = alpaca._get("/v1beta1/news", params, creds)
            with lock:
                for n in doc.get("news", []):
                    if n["id"] in seen:
                        continue
                    seen.add(n["id"])
                    out.write(json.dumps({"id": n["id"], "t": n["created_at"], "symbols": n.get("symbols", []),
                                          "headline": n.get("headline", "")}) + "\n")
            token = doc.get("next_page_token")
            time.sleep(max(0.0, 1.05 - (time.time() - t1)))              # 3 workers x < 1 request/s
            if not token:
                break
        with lock:
            state[key] = "done"
            out.flush()
            json.dump(state, open(STATE, "w"))
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  {done[0]}/{len(jobs)} group-months, {len(seen)} items, {(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one, jobs))
    out.close()
    print(f"fetch complete: {len(seen)} headlines", flush=True)


def load_items(skip_recaps=True):
    """Headlines since START; templated recaps are dropped (amendment A)."""
    items = {}
    for line in open(ITEMS, encoding="utf-8"):
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x["t"][:10] >= START and x.get("headline") and not (skip_recaps and RECAP.search(x["headline"])):
            items[x["id"]] = x
    return items


# ---------------------------------------------------------------- LLM scoring (prompt fixed before any v8 result)

SYSTEM = "You are a careful equity analyst. Answer each question about a news headline with exactly one word."
QUESTIONS = {
    "dir": ("Is the headline below good, neutral or bad news for the stock price of the stock named below it? Answer good, neutral or bad.",
            ["good", "neutral", "bad"]),
    "topic": ("Is the headline below mainly about artificial intelligence (AI chips, AI models, AI spending or demand), the overall stock "
              "market or economy (interest rates, inflation, the Fed, tariffs, recession, broad sector moves), or other? "
              "Answer ai, market or other.", ["ai", "market", "other"]),
}
# amendment A: templated recaps / lists / previews are not scored (weight 0); the LLM's own new-vs-old answer did not discriminate
RECAP = re.compile(r"(^What'?s Going On With|^Why .{1,60}(Shares|Stock)s? (Is|Are) (Trading|Falling|Rising|Moving|Sliding|Soaring|Diving|"
                   r"Up|Down|Higher|Lower)|Stocks? Moving|Whale (Activity|Alert)|^Earnings Scheduled For|Here'?s How Much|Options? Activity|"
                   r"Stocks? To Watch|Preview|On Investors'? Radars?|52-Week (High|Low)|Circuit Breaker|Price Over Earnings|P/E Ratio|"
                   r"Short Interest|Peer Comparison|A Look At|Insights|Analyst Ratings|Trading (Higher|Lower)|Shares Are|Stock Is (Up|Down)|"
                   r"Market Cap|Top \d+|^\d+ .{0,40}Stocks)", re.I)
MAX_TICKERS = 2


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


def _health():
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok"
    except Exception:
        return False


class Server:
    """llama-server on the RX 7800 XT only (Vulkan0), or CPU-only. Reuses a running server; stops one it started."""

    def __init__(self, gpu):
        self.proc = None
        if _health():
            return
        args = [server_exe(), "-m", model_path(), "-np", str(SLOTS), "-c", str(1024 * SLOTS), "--port", str(PORT), "--host", "127.0.0.1"]
        args += ["-ngl", "99", "--device", "Vulkan0"] if gpu else ["-ngl", "0"]
        log = open(os.path.join("private", "reports", "jobs", "llama_server.log"), "a")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(180):
            if _health():
                return
            time.sleep(1)
        raise RuntimeError("llama-server did not start (see private/reports/jobs/llama_server.log)")

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)


_slot = threading.local()
_slot_lock = threading.Lock()
_slot_next = [0]


def _my_slot():
    """Each worker thread always uses the same server slot, so a headline's questions share the cached prompt prefix."""
    if not hasattr(_slot, "id"):
        with _slot_lock:
            _slot.id = _slot_next[0] % (SLOTS // 2)
            _slot_next[0] += 1
    return _slot.id


def ask(headline, question, options, kind, stock=None):
    """Probabilities of the one-word answers, from the model's first answer token (renormalised over the options)."""
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nHeadline: {headline}\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 20, "cache_prompt": True, "id_slot": _my_slot()}
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
    tick = [t for t in item.get("symbols", []) if t in wanted][:MAX_TICKERS]
    out = {"id": item["id"], "t": item["t"], "dir": {}}
    cov = 1.0
    for t in tick:
        p, c = ask(h, QUESTIONS["dir"][0], QUESTIONS["dir"][1], "dir", stock=t)
        out["dir"][t] = round(p["good"] - p["bad"], 4)
        cov = min(cov, c)
    p, c = ask(h, QUESTIONS["topic"][0], QUESTIONS["topic"][1], "topic")
    out["ai"], out["mkt"] = round(p["ai"], 4), round(p["market"], 4)
    cov = min(cov, c)
    out["cov"] = round(cov, 4)
    return out


def _wanted():
    return set(universe_window_symbols()) | set(MARKET_ETFS)


def _start(a):
    if not (a.allow_gpu or a.cpu):
        raise SystemExit("choose --allow-gpu (only while you are at the PC) or --cpu")
    return Server(gpu=a.allow_gpu and not a.cpu)


def bench(a):
    args = [server_exe("llama-bench.exe"), "-m", model_path(), "-p", "512", "-n", "32"]
    args += ["-ngl", "99", "-dev", "Vulkan0"] if a.allow_gpu else ["-ngl", "0"]
    subprocess.run(args, check=True)


def sample(a):
    items = list(load_items().values())
    random.Random(7).shuffle(items)
    wanted = _wanted()
    srv = _start(a)
    t0 = time.time()
    try:
        with ThreadPoolExecutor(SLOTS // 2) as ex:
            res = list(ex.map(lambda x: score_item(x, wanted), items[:200]))
    finally:
        srv.close()
    dt = time.time() - t0
    os.makedirs(os.path.join("private", "reports", "v8"), exist_ok=True)
    with open(os.path.join("private", "reports", "v8", "llm_sample.txt"), "w", encoding="utf-8") as f:
        for it, r in zip(items, res):
            f.write(f"{it['t'][:16]} | {it['headline'][:110]}\n    dir {r['dir']}  ai {r['ai']:.2f}  mkt {r['mkt']:.2f}  cov {r['cov']:.2f}\n")
    print(f"scored 200 headlines in {dt:.0f} s ({dt / 200:.2f} s each with {SLOTS} parallel slots); "
          f"low-coverage answers: {sum(r['cov'] < 0.5 for r in res)}; written to private/reports/v8/llm_sample.txt", flush=True)


def score(a):
    items = load_items()
    done = set()
    if os.path.exists(SCORES):
        for line in open(SCORES, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except ValueError:
                pass
    todo = [x for i, x in sorted(items.items(), key=lambda kv: kv[1]["t"]) if i not in done]
    print(f"{len(items)} headlines since {START}, {len(done)} already scored, {len(todo)} to score", flush=True)
    if not todo:
        return
    wanted = _wanted()
    srv = _start(a)
    t0, n = time.time(), 0
    try:
        with open(SCORES, "a", encoding="utf-8") as out, ThreadPoolExecutor(SLOTS // 2) as ex:
            for k in range(0, len(todo), 400):
                if time.time() - t0 > a.max_minutes * 60:                 # GPU runs are supervised: stop on time
                    print(f"  stopping after {a.max_minutes} min (resumable; run score again to continue)", flush=True)
                    break
                for r in ex.map(lambda x: score_item(x, wanted), todo[k:k + 400]):
                    out.write(json.dumps(r) + "\n")
                out.flush()
                n += min(400, len(todo) - k)
                el = time.time() - t0
                print(f"  {n}/{len(todo)} scored, {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)
    finally:
        srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "bench", "sample", "score", "aggregate"])
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=120, help="score: stop after this long (resumable)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        return fetch()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
