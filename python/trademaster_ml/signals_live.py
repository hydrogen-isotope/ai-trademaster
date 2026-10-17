"""
Live buy/sell indicators for the dashboard's forward-test page (research only: no orders, not trading advice).

  prepare  pull the latest 1-minute bars (merged into private/data/hist_1m_30d), then freeze on the newest
           complete week if it isn't frozen yet. `live` runs this first unless --no-refresh.
  freeze   fit the 30-minute model + signal thresholds on the most recent full week and save them, hashed
  live     wait for the next NASDAQ session; at each 30-minute checkpoint fetch today's 1-minute bars (Alpaca IEX,
           real time), compute
           signals from bars up to that minute only, write them to disk, and score each one 30 minutes later
  replay   run the identical code over a past session from private/data/hist_1m_30d (pipeline rehearsal)

Output: private/data/forward/signals-{live|replay}-<date>/signals.json (read by the dashboard at /forward).

Usage:
  python -m trademaster_ml.signals_live prepare
  python -m trademaster_ml.signals_live freeze
  python -m trademaster_ml.signals_live live
  python -m trademaster_ml.signals_live replay --date 2026-10-16 --frozen private/models/forward/signals-fit-2026-10-09
"""
import argparse, datetime as dt, hashlib, json, os, time, urllib.request
from zoneinfo import ZoneInfo

import numpy as np

from .weektest import load
from .rolling import weeks_of, fit, features, FEATS, SKIP
from .signals import H, STEP, COST_ONE_WAY, predict_rel, checkpoints
from .forward import keep_awake, next_session
from .fetch1m import fetch_all
from . import alpaca

NY = ZoneInfo("America/New_York")
HIST = "private/data/hist_1m_30d"
MINUTES = 390


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- freeze
def freeze(hist=HIST, week_end=None, force=False):
    syms, ts, days, close, vol, ref, stocks = load(hist)
    today = dt.datetime.now(NY).date()
    wk = [w for w in weeks_of(sorted(set(days))) if max(w) < today]      # complete weeks only
    if week_end:
        wk = [w for w in wk if max(w) <= dt.date.fromisoformat(week_end)]
    train = wk[-1]
    from .weektest import session
    sess = {d: session(close, vol, days, d) for d in train}
    beta, w = fit(sess, train, ref, stocks, H)
    tr = np.concatenate([predict_rel(sess, d, beta, w, ref, stocks)[checkpoints(len(sess[d][0]))].ravel() for d in train])
    tr = tr[np.isfinite(tr)]
    q = dict(zip(["p1", "p5", "p95", "p99"], map(float, np.percentile(tr, [1, 5, 95, 99]))))
    out = os.path.join("private", "models", "forward", f"signals-fit-{train[-1]}")
    if os.path.exists(os.path.join(out, "model.json")) and not force:
        print(f"already frozen: {out}")
        return out
    os.makedirs(out, exist_ok=True)
    m = dict(kind="signals-30m", symbols=syms, ref=ref.tolist(), stocks=stocks.tolist(), beta=beta.tolist(),
             weights=w.tolist(), features=FEATS, thresholds=q, horizon_min=H, step_min=STEP, skip=SKIP,
             cost_one_way_bps=COST_ONE_WAY, train=[str(train[0]), str(train[-1])],
             frozen_at=dt.datetime.now(dt.timezone.utc).isoformat())
    json.dump(m, open(os.path.join(out, "model.json"), "w"), indent=1)
    print(f"frozen {out}  (trained {train[0]}..{train[-1]}, sha256 {sha256(os.path.join(out, 'model.json'))[:16]})")
    return out


class Frozen:
    def __init__(self, path):
        self.path = path
        self.m = json.load(open(os.path.join(path, "model.json")))
        self.sha = sha256(os.path.join(path, "model.json"))
        self.symbols = self.m["symbols"]
        self.ref = np.array(self.m["ref"]); self.stocks = np.array(self.m["stocks"])
        self.beta = np.array(self.m["beta"]); self.w = np.array(self.m["weights"]); self.q = self.m["thresholds"]
        self.names = [s for s, k in zip(self.symbols, self.stocks) if k]

    def signals_at(self, c, v, t):
        """Signals for checkpoint t from bars 0..t only. Returns list of (stock index, kind, pred_bps)."""
        c, v = c[:t + 1], v[:t + 1]                                  # hard cut: nothing after t exists here
        mu = (features(c, v, self.beta, self.ref) @ self.w)[t, self.stocks]
        mu = mu - np.nanmean(mu)
        floor = 2 * self.m["cost_one_way_bps"]
        out = []
        for i, x in enumerate(mu):
            if not np.isfinite(x) or not np.isfinite(c[t, self.stocks][i]):
                continue
            if x >= max(self.q["p95"], floor):
                out.append((i, "STRONG BUY" if x >= self.q["p99"] else "BUY", float(x)))
            elif x <= min(self.q["p5"], -floor):
                out.append((i, "STRONG SELL" if x <= self.q["p1"] else "SELL", float(x)))
        return out


# ---------------------------------------------------------------- session arrays
def ffill(c):
    c = c.copy()
    for t in range(1, len(c)):
        m = ~np.isfinite(c[t])
        c[t, m] = c[t - 1, m]
    return c


def arrays_from_bars(symbols, bars, open_epoch):
    """bars: {symbol: (ts list, close list, volume list)} -> c, v [390, N] indexed by minute since 09:30 ET."""
    c = np.full((MINUTES, len(symbols)), np.nan)
    v = np.zeros((MINUTES, len(symbols)))
    for j, s in enumerate(symbols):
        for t, cl, vo in zip(*bars.get(s, ([], [], []))):
            k = (int(t) - open_epoch) // 60
            if 0 <= k < MINUTES and cl is not None:
                c[k, j], v[k, j] = cl, vo or 0
    return ffill(c), v


def fetch_today(symbols):
    """Today's regular-session 1-minute bars from Alpaca's real-time IEX feed. The model's only volume input is a
    ratio to the stock's own recent volume, so the single-exchange volume scale cancels out."""
    now = dt.datetime.now(NY)
    open_epoch = int(dt.datetime.combine(now.date(), dt.time(9, 30), NY).timestamp())
    try:
        data = alpaca.get_bars(list(symbols), "1Min", open_epoch, None, feed="iex", pause=0.1)
    except Exception as e:
        return {}, [f"download failed: {e}"]
    bars = {}
    for s in symbols:
        rows = alpaca.regular_session(data.get(s, []))
        bars[s] = ([r[0] for r in rows], [r[4] for r in rows], [r[5] for r in rows])
    return bars, []


# ---------------------------------------------------------------- run log
class Run:
    def __init__(self, f, mode, date):
        self.f, self.mode, self.date = f, mode, date
        self.dir = os.path.join("private", "data", "forward", f"signals-{mode}-{date}")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "signals.jsonl")
        self.sigs = [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []
        self.done = {s["t"] for s in self.sigs} | set(json.load(open(self._cp())) if os.path.exists(self._cp()) else [])
        self.status, self.errors = "starting", []

    def _cp(self):
        return os.path.join(self.dir, "checkpoints.json")

    def log_checkpoint(self, t, c, v, open_epoch):
        if t in self.done:
            return 0
        new = []
        for i, kind, pred in self.f.signals_at(c, v, t):
            new.append(dict(t=int(t), ts=int(open_epoch + 60 * t), time=f"{9 + (30 + t) // 60:02d}:{(30 + t) % 60:02d}",
                            symbol=self.f.names[i], signal=kind, pred_bps=round(pred, 2),
                            price=round(float(c[t, self.f.stocks][i]), 4), logged_at=dt.datetime.now(dt.timezone.utc).isoformat()))
        with open(self.path, "a") as fh:                              # written before any outcome exists
            for s in new:
                fh.write(json.dumps(s) + "\n")
        self.sigs += new
        self.done.add(t)
        json.dump(sorted(self.done), open(self._cp(), "w"))
        return len(new)

    def score(self, c, last_complete):
        """Fill outcomes for signals whose 30-minute window has completed (bars <= last_complete)."""
        cs = c[:, self.f.stocks]
        for s in self.sigs:
            t = s["t"]
            if "actual_rel_bps" in s or t + H > last_complete:
                continue
            y = np.log(cs[t + H] / cs[t]) * 1e4
            i = self.f.names.index(s["symbol"])
            if np.isfinite(y[i]):
                s["actual_bps"] = round(float(y[i]), 2)
                s["actual_rel_bps"] = round(float(y[i] - np.nanmean(y)), 2)
                side = 1 if "BUY" in s["signal"] else -1
                s["right"] = bool(side * s["actual_rel_bps"] > 0)
                s["net_bps"] = round(side * s["actual_rel_bps"] - 2 * self.f.m["cost_one_way_bps"], 2)

    def write_summary(self, last_poll=None):
        board = {}
        for kind in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            s = [x for x in self.sigs if x["signal"] == kind and "net_bps" in x]
            board[kind] = dict(count=sum(1 for x in self.sigs if x["signal"] == kind), scored=len(s),
                               hit_rate=float(np.mean([x["right"] for x in s])) if s else None,
                               avg_net_bps=float(np.mean([x["net_bps"] for x in s])) if s else None)
        doc = dict(mode=self.mode, session_date=self.date, status=self.status, last_poll_utc=last_poll,
                   horizon_min=H, step_min=STEP, cost_one_way_bps=self.f.m["cost_one_way_bps"],
                   frozen=dict(path=self.f.path, sha256=self.f.sha, train=self.f.m["train"], frozen_at=self.f.m["frozen_at"]),
                   checkpoints_logged=len(self.done), signals=self.sigs, scoreboard=board, errors=self.errors[-10:],
                   data_source="Alpaca Market Data API v2 (IEX real-time feed)")
        tmp = os.path.join(self.dir, "signals.json.tmp")
        json.dump(doc, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, "signals.json"))


def latest_frozen():
    root = os.path.join("private", "models", "forward")
    c = sorted(d for d in os.listdir(root) if d.startswith("signals-fit-"))
    if not c:
        raise SystemExit("no frozen signal model: run  python -m trademaster_ml.signals_live freeze")
    return os.path.join(root, c[-1])


def prepare():
    """Pull data first, then make sure the newest complete week is frozen. Returns the frozen model path."""
    print("pulling latest 1-minute bars ...")
    try:
        saved, errors = fetch_all(HIST, log=lambda m: None)
        print(f"  {len(saved)} symbols updated" + (f", {len(errors)} fetch errors (kept previous data)" if errors else ""))
    except Exception as e:
        print(f"  data pull failed ({e}); using data already on disk")
    try:
        return freeze()
    except Exception as e:
        print(f"  freeze failed ({e}); using the newest model already frozen")
        return latest_frozen()


# ---------------------------------------------------------------- modes
def replay(date_s, frozen):
    f = Frozen(frozen)
    d = dt.date.fromisoformat(date_s)
    if dt.date.fromisoformat(f.m["train"][1]) >= d:
        print("note: this session is inside or before the training week -> not out of sample")
    syms, ts, days, close, vol, ref, stocks = load(HIST)
    assert syms == f.symbols, "symbol order mismatch"
    open_epoch = int(dt.datetime.combine(d, dt.time(9, 30), NY).timestamp())
    sel = days == d
    bars = {s: (ts[sel].tolist(), close[sel, j].tolist(), vol[sel, j].tolist()) for j, s in enumerate(syms)}
    c, v = arrays_from_bars(syms, bars, open_epoch)
    run = Run(f, "replay", date_s)
    for t in range(SKIP, MINUTES):                                  # minute by minute, as live would see it
        if (t - SKIP) % STEP == 0 and t + H < MINUTES:
            run.log_checkpoint(t, c, v, open_epoch)
        run.score(c, t)
    run.status = "replay complete"
    run.write_summary(dt.datetime.now(dt.timezone.utc).isoformat())
    s = json.load(open(os.path.join(run.dir, "signals.json")))
    print(f"replay {date_s}: {len(s['signals'])} signals over {s['checkpoints_logged']} checkpoints -> {run.dir}")
    for k, b in s["scoreboard"].items():
        print(f"  {k:11s} {b['count']:3d}  right {b['hit_rate'] if b['hit_rate'] is None else round(b['hit_rate'], 3)}  net {b['avg_net_bps']}")


def live(frozen, poll_offset_s=25):
    try:
        alpaca.credentials()
    except alpaca.AlpacaConfigError as e:
        raise SystemExit(str(e))
    f = Frozen(frozen)
    keep_awake(True)
    try:
        d = next_session(dt.datetime.now(NY))
        open_dt = dt.datetime.combine(d, dt.time(9, 30), NY)
        open_epoch = int(open_dt.timestamp())
        run = Run(f, "live", d.isoformat())
        print(f"signals: next session {d}, frozen {f.path} (trained {f.m['train'][0]}..{f.m['train'][1]})")
        cps = [t for t in range(SKIP, MINUTES) if (t - SKIP) % STEP == 0 and t + H < MINUTES]
        # poll right after each checkpoint bar completes, plus once after the last outcome window closes
        polls = sorted(set(cps + [t + H for t in cps]))
        for t in polls:
            due = open_dt + dt.timedelta(minutes=t + 1, seconds=poll_offset_s)
            while dt.datetime.now(NY) < due:
                run.status = "waiting for the open (09:30 ET)" if dt.datetime.now(NY) < open_dt else "live"
                run.write_summary(dt.datetime.now(dt.timezone.utc).isoformat())
                time.sleep(min(30, max(1, (due - dt.datetime.now(NY)).total_seconds())))
            run.status = "live"
            bars, errs = fetch_today(f.symbols)
            run.errors += [f"{dt.datetime.now(NY):%H:%M:%S} {e}" for e in errs]
            c, v = arrays_from_bars(f.symbols, bars, open_epoch)
            last_complete = min(MINUTES - 1, int((time.time() - open_epoch) // 60) - 1)
            if t in cps and t <= last_complete:
                n = run.log_checkpoint(t, c, v, open_epoch)
                print(f"  {9 + (30 + t) // 60:02d}:{(30 + t) % 60:02d} ET  {n} signals logged")
            run.score(c, last_complete)
            run.write_summary(dt.datetime.now(dt.timezone.utc).isoformat())
        run.status = "session closed"
        run.write_summary(dt.datetime.now(dt.timezone.utc).isoformat())
        print(f"done: {run.dir}")
    finally:
        keep_awake(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "freeze", "live", "replay"])
    ap.add_argument("--no-refresh", action="store_true", help="live: skip pulling data / refreezing first")
    ap.add_argument("--force", action="store_true", help="freeze: refit even if this week is already frozen")
    ap.add_argument("--frozen")
    ap.add_argument("--date")
    ap.add_argument("--week-end", help="freeze: use the last full week ending on/before this date")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "freeze":
        freeze(week_end=a.week_end, force=a.force)
    elif a.cmd == "replay":
        replay(a.date, a.frozen or latest_frozen())
    else:
        live(a.frozen or (latest_frozen() if a.no_refresh else prepare()))


if __name__ == "__main__":
    main()
