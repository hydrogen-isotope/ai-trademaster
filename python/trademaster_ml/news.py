"""
News for the v4 forecaster (research only; CPU only).

  fetch   Alpaca news API (Benzinga) for the universe, 2016 onward -> private/data/news/items.jsonl (resumable)
  score   FinBERT (ProsusAI/finbert) sentiment for every item -> private/data/news/scores.jsonl (resumable)

Each item is scored on "headline. summary" truncated to 128 tokens; sentiment = P(positive) - P(negative).
"""
import argparse, json, os, sys, time

from . import alpaca

DIR = os.path.join("private", "data", "news")
ITEMS, SCORES, STATE = (os.path.join(DIR, f) for f in ("items.jsonl", "scores.jsonl", "fetch_state.json"))


def _fetch_year(year, syms, creds, state, lock, out, seen):
    key = str(year)
    if state.get(key) == "done":
        return 0
    params = {"symbols": ",".join(syms), "start": f"{year}-01-01T00:00:00Z", "end": f"{year + 1}-01-01T00:00:00Z",
              "limit": 50, "sort": "asc", "include_content": "false"}
    token, pages, new = state.get(key), 0, 0
    while True:
        if token:
            params["page_token"] = token
        t0 = time.time()
        doc = alpaca._get("/v1beta1/news", params, creds)
        with lock:
            for n in doc.get("news", []):
                if n["id"] in seen:
                    continue
                seen.add(n["id"])
                out.write(json.dumps({"id": n["id"], "t": n["created_at"], "symbols": n.get("symbols", []), "headline": n.get("headline", ""),
                                      "summary": (n.get("summary") or "")[:400], "source": n.get("source", "")}) + "\n")
                new += 1
            token, pages = doc.get("next_page_token"), pages + 1
            state[key] = token or "done"
            if pages % 25 == 0 or not token:
                out.flush()
                json.dump(state, open(STATE, "w"))
                last = doc["news"][-1]["created_at"][:10] if doc.get("news") else "?"
                print(f"{year}: {pages} pages, {new} new, reached {last}, total {len(seen)}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.time() - t0)))           # 3 workers x <1 request/s stays under 200/min
        if not token:
            return new


def fetch():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(DIR, exist_ok=True)
    u = json.load(open(os.path.join("private", "data", "daily_universe", "universe.json")))
    syms = u["stocks"] + u["etfs"]
    seen = set()
    if os.path.exists(ITEMS):
        for line in open(ITEMS, encoding="utf-8"):
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                pass
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    state = {k: v for k, v in state.items() if k.isdigit()}         # drop the old per-group state
    creds, lock = alpaca.credentials(), threading.Lock()
    out = open(ITEMS, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda y: _fetch_year(y, syms, creds, state, lock, out, seen), range(2016, 2027)))
    out.flush()
    print(f"fetch complete: {len(seen)} items", flush=True)


def score(batch=64, threads=8):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(threads)
    os.environ.setdefault("HF_HOME", os.path.join("private", "models", "hf"))
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    m = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    lab = {v: k for k, v in m.config.id2label.items()}
    done = set()
    if os.path.exists(SCORES):
        for line in open(SCORES, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = []
    for line in open(ITEMS, encoding="utf-8"):
        n = json.loads(line)
        if n["id"] not in done:
            todo.append(n)
    print(f"scoring {len(todo)} items ({len(done)} already scored)", flush=True)
    out = open(SCORES, "a", encoding="utf-8")
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            text = [(n["headline"] + ". " + n["summary"]).strip() for n in chunk]
            x = tok(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            p = torch.softmax(m(**x).logits, -1).numpy()
            for n, pr in zip(chunk, p):
                out.write(json.dumps({"id": n["id"], "pos": round(float(pr[lab["positive"]]), 4), "neg": round(float(pr[lab["negative"]]), 4)}) + "\n")
            if (i // batch) % 100 == 0:
                out.flush()
                rate = (i + len(chunk)) / max(1e-9, time.time() - t0)
                left = (len(todo) - i - len(chunk)) / max(rate, 1e-9) / 60
                print(f"scored {i + len(chunk)}/{len(todo)}  {rate:.0f}/s  ~{left:.0f} min left", flush=True)
    out.flush()
    print("score complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "score"])
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch()
    else:
        score()


if __name__ == "__main__":
    main()
