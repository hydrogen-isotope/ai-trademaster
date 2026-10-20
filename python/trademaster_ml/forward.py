"""
Frozen-model forward test (predictions only; no orders; CPU only).

  freeze  copies the chosen artifacts into models/forward/<id>/ with SHA-256 hashes. A frozen set is never overwritten
          or retrained; the logger refuses to run if any hash changes.
  live    polls Alpaca 5-minute bars (IEX feed, real time) during the NASDAQ regular session. After each completed bar it logs every model's
          prediction for the next 30 minutes (6 bars) BEFORE the outcome exists, then scores each prediction once its
          outcome bar has completed. Waits for the open, stops after the close. Keeps the PC awake while running.
  replay  runs the identical pipeline over a past session from data/hist2 (pipeline rehearsal, not evidence).

Outputs (data/forward/<run>/): predictions.jsonl, outcomes.jsonl, summary.json (read by the dashboard).
Target: forward 6-bar return minus the cross-sectional mean of stocks; scored by rank IC and a long/short quintile
portfolio held 6 bars, gross and net of an assumed 1.5 bps one-way cost.

Usage:
  python -m trademaster_ml.forward freeze
  python -m trademaster_ml.forward live   [--frozen models/forward/frozen-2026-10-17]
  python -m trademaster_ml.forward replay --date 2026-10-16
"""
import argparse, ctypes, datetime as dt, hashlib, json, os, shutil, sys, time, urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from . import alpaca
from .dataset import FEATURES, build_panel

NY = ZoneInfo("America/New_York")
HORIZON = 6
COST_BPS = 1.5
FROZEN_DEFAULT = "private/models/forward/frozen-2026-10-17"
UNIVERSE_SRC = "private/data/hist2/universe.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- freeze
def freeze(out=FROZEN_DEFAULT, b1_dir="private/models/b1/v2-h6", research="private/models/research/5m-h6/research.json"):
    if os.path.exists(os.path.join(out, "manifest.json")):
        print(f"{out} already frozen; not touching it")
        return
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(os.path.join(b1_dir, "model.spec.json"), encoding="utf-8"))
    res = json.load(open(research, encoding="utf-8"))
    assert spec["symbols"] == res["symbols"], "symbol order differs between artifacts"
    shutil.copy(os.path.join(b1_dir, "model.onnx"), os.path.join(out, "b1.onnx"))
    json.dump({"mean": spec["mean"], "std": spec["std"], "clip": spec.get("clip", 5.0), "source": b1_dir},
              open(os.path.join(out, "b1.norm.json"), "w"), indent=1)
    json.dump({"weights": res["ridge_weights_final"], "mean": res["mean"], "std": res["std"], "source": research},
              open(os.path.join(out, "ridge.json"), "w"), indent=1)
    files = ["b1.onnx", "b1.norm.json", "ridge.json"]
    manifest = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(), "symbols": spec["symbols"], "is_etf": spec["is_etf"],
        "features": FEATURES, "interval": "5m", "horizon_bars": HORIZON, "target": spec["target"], "cost_bps_one_way": COST_BPS,
        "models": {"b0": "rule: -0.5 x 30-min residual vs reference ETF", "ridge": "ridge.json", "b1": "b1.onnx (3-seed ensemble)"},
        "sha256": {f: sha256(os.path.join(out, f)) for f in files},
        "note": "Frozen for forward testing. Never retrain or overwrite; create a new frozen set instead.",
    }
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"frozen {out}: " + ", ".join(f"{k} {v[:12]}" for k, v in manifest["sha256"].items()))


class Frozen:
    def __init__(self, path):
        self.path = path
        self.m = json.load(open(os.path.join(path, "manifest.json"), encoding="utf-8"))
        for f, h in self.m["sha256"].items():
            if sha256(os.path.join(path, f)) != h:
                raise SystemExit(f"FROZEN ARTIFACT CHANGED: {f} hash mismatch — refusing to run")
        self.norm = json.load(open(os.path.join(path, "b1.norm.json")))
        self.ridge = json.load(open(os.path.join(path, "ridge.json")))
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(os.path.join(path, "b1.onnx"), so, providers=["CPUExecutionProvider"])
        self.symbols = self.m["symbols"]
        self.stock = ~np.array(self.m["is_etf"])

    @staticmethod
    def _z(X, mean, std, clip=5.0):
        Z = np.clip((X - np.array(mean)) / np.array(std), -clip, clip)
        valid = np.isfinite(Z).all(-1)
        return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32), valid

    def predict(self, x):
        """x: [N, F] raw features at one bar -> dict of [N] predictions (bps) and validity mask."""
        zb, vb = self._z(x, self.norm["mean"], self.norm["std"], self.norm.get("clip", 5.0))
        out = self.sess.run(None, {"x": zb[None], "mask": vb.astype(np.float32)[None]})[0][0]
        zr, vr = self._z(x, self.ridge["mean"], self.ridge["std"])
        ridge = np.c_[zr, np.ones(len(zr))] @ np.array(self.ridge["weights"])
        b0 = -0.5 * np.nan_to_num(x[:, FEATURES.index("resid")]) * 1e4
        valid = vb & vr & self.stock
        return {"b0": b0, "ridge": ridge, "b1": out[:, 0], "b1_sigma": np.exp(0.5 * out[:, 1])}, valid


# ---------------------------------------------------------------- scoring
def score(pred_rows, outcomes):
    """Rank IC per bar and non-overlapping L/S quintile P&L per model."""
    models = ("b0", "ridge", "b1")
    out = {m: {"ic": [], "ls_gross": [], "ls_net": [], "hit": []} for m in models}
    series, cum = [], {m: 0.0 for m in models}
    prev = {m: (set(), set()) for m in models}
    last_rebal = None
    for p in pred_rows:
        o = outcomes.get(p["t"])
        if o is None:
            continue
        y = np.array([np.nan if v is None else v for v in o["realized_bps"]])
        valid = np.array(p["valid"]) & np.isfinite(y)
        if valid.sum() < 10:
            continue
        point = {"t": p["t"], "label": p["label"]}
        rebalance = last_rebal is None or p["bar_index"] - last_rebal >= HORIZON
        for m in models:
            pr = np.array(p[m])[valid]; yy = y[valid]
            ic = float(np.corrcoef(pr.argsort().argsort(), yy.argsort().argsort())[0, 1]) if pr.std() > 0 else 0.0
            out[m]["ic"].append(ic); point[m] = round(ic, 4)
            nz = (pr != 0) & (yy != 0)
            out[m]["hit"].append(float(((pr[nz] > 0) == (yy[nz] > 0)).mean()) if nz.any() else 0.5)
            if rebalance:
                idx = np.where(valid)[0]; q = max(1, len(idx) // 5); order = idx[np.array(p[m])[idx].argsort()]
                longs, shorts = set(order[-q:].tolist()), set(order[:q].tolist())
                gross = 0.5 * (y[list(longs)].mean() - y[list(shorts)].mean())   # gross exposure 1 (0.5 long, 0.5 short)
                pl, ps = prev[m]
                changed = (len(longs - pl) + len(shorts - ps)) / (2 * q) if pl else 1.0
                net = gross - 2 * changed * COST_BPS                              # in and out, per unit gross traded
                out[m]["ls_gross"].append(gross); out[m]["ls_net"].append(net)
                cum[m] += net; prev[m] = (longs, shorts)
        if rebalance:
            last_rebal = p["bar_index"]
            point["cum_net"] = {m: round(cum[m], 2) for m in models}
        series.append(point)
    summ = {}
    for m in models:
        if out[m]["ls_net"]:
            out[m]["ls_net"][-1] -= COST_BPS  # charge exiting the final position (1 unit of gross)
        ic = np.array(out[m]["ic"]); g = np.array(out[m]["ls_gross"]); n = np.array(out[m]["ls_net"])
        summ[m] = {
            "bars_scored": int(len(ic)), "rank_ic_mean": float(ic.mean()) if len(ic) else None,
            "rank_ic_t_eff": float(ic.mean() / ic.std() * np.sqrt(len(ic) / HORIZON)) if len(ic) > 2 and ic.std() > 0 else None,
            "hit_rate": float(np.mean(out[m]["hit"])) if out[m]["hit"] else None,
            "rebalances": int(len(n)), "ls_gross_bps_per_rebalance": float(g.mean()) if len(g) else None,
            "ls_net_bps_per_rebalance": float(n.mean()) if len(n) else None, "ls_cum_net_bps": float(n.sum()) if len(n) else 0.0,
        }
    return summ, series


# ---------------------------------------------------------------- logging
class Run:
    def __init__(self, out_dir, frozen, mode, session_date):
        self.dir, self.f, self.mode, self.date = out_dir, frozen, mode, session_date
        os.makedirs(out_dir, exist_ok=True)
        self.pred_path = os.path.join(out_dir, "predictions.jsonl")
        self.out_path = os.path.join(out_dir, "outcomes.jsonl")
        self.preds = [json.loads(l) for l in open(self.pred_path)] if os.path.exists(self.pred_path) else []
        self.outs = {o["t"]: o for o in (json.loads(l) for l in open(self.out_path))} if os.path.exists(self.out_path) else {}
        self.status, self.last_poll, self.errors = "starting", None, []

    def log_prediction(self, t, bar_index, x, close):
        if any(p["t"] == t for p in self.preds):
            return False
        pr, valid = self.f.predict(x)
        rec = {"t": int(t), "label": dt.datetime.fromtimestamp(t, NY).strftime("%H:%M"), "bar_index": int(bar_index),
               "logged_at": dt.datetime.now(dt.timezone.utc).isoformat(), "valid": valid.tolist(),
               "close": [None if not np.isfinite(c) else round(float(c), 4) for c in close]}
        for k, v in pr.items():
            rec[k] = [round(float(a), 3) for a in v]
        with open(self.pred_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.preds.append(rec)
        return True

    def log_outcome(self, t, realized):
        if t in self.outs:
            return
        rec = {"t": int(t), "realized_bps": [None if not np.isfinite(v) else round(float(v), 3) for v in realized],
               "scored_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.outs[t] = rec

    def write_summary(self):
        summ, series = score(self.preds, self.outs)
        latest = self.preds[-1] if self.preds else None
        rows = []
        if latest:
            for i, s in enumerate(self.f.symbols):
                if not latest["valid"][i]:
                    continue
                rows.append({"symbol": s, "close": latest["close"][i], "b0": latest["b0"][i], "ridge": latest["ridge"][i],
                             "b1": latest["b1"][i], "b1_sigma": latest["b1_sigma"][i]})
        doc = {
            "mode": self.mode, "session_date": self.date, "status": self.status, "last_poll_utc": self.last_poll,
            "frozen": {"path": self.f.path, "frozen_at": self.f.m["frozen_at"], "sha256": self.f.m["sha256"]},
            "horizon_bars": HORIZON, "interval": "5m", "cost_bps_one_way": COST_BPS, "target": self.f.m["target"],
            "bars_logged": len(self.preds), "bars_scored": sum(1 for p in self.preds if p["t"] in self.outs),
            "models": summ, "series": series, "latest": {"t": latest["t"], "label": latest["label"], "rows": rows} if latest else None,
            "errors": self.errors[-10:],
            "data_source": "Alpaca Market Data API v2 (IEX real-time feed)",
        }
        tmp = os.path.join(self.dir, "summary.json.tmp")
        json.dump(doc, open(tmp, "w"), default=float)
        os.replace(tmp, os.path.join(self.dir, "summary.json"))


def session_bars(panel, date):
    days = np.array([dt.datetime.fromtimestamp(int(t), NY).date() for t in panel.ts])
    return np.where(days == date)[0]


def realized(panel, t_idx, stock):
    """Forward 6-bar log return (bps) minus the cross-sectional mean of stocks; NaN if not yet complete."""
    y = panel.y[t_idx] * 1e4
    m = np.nanmean(np.where(stock, y, np.nan)) if np.isfinite(y[stock]).any() else np.nan
    return y - m


def replay(date_str, frozen_path, hist="private/data/hist2"):
    f = Frozen(frozen_path)
    date = dt.date.fromisoformat(date_str)
    run = Run(os.path.join("private", "data", "forward", f"replay-{date_str}"), f, "replay (pipeline rehearsal, not evidence)", date_str)
    P = build_panel(hist, horizon=HORIZON)
    assert P.symbols == f.symbols, "symbol order mismatch"
    idx = session_bars(P, date)
    for k, t in enumerate(idx):
        run.log_prediction(int(P.ts[t]), k, P.X[t], P.close[t])
        if np.isfinite(P.y[t]).any():
            run.log_outcome(int(P.ts[t]), realized(P, t, f.stock))
    run.status = "replay complete"
    run.last_poll = dt.datetime.now(dt.timezone.utc).isoformat()
    run.write_summary()
    s = json.load(open(os.path.join(run.dir, "summary.json")))
    print(f"replay {date_str}: {s['bars_logged']} bars logged, {s['bars_scored']} scored -> {run.dir}")
    for m, v in s["models"].items():
        print(f"  {m:6s} rank IC {v['rank_ic_mean'] or 0:+.4f}  hit {v['hit_rate'] or 0:.3f}  L/S net {v['ls_net_bps_per_rebalance'] or 0:+.2f} bps/rebalance, cum {v['ls_cum_net_bps']:+.1f}")


def fetch_all(symbols, out_dir):
    """Last ~7 days of 5-minute bars from Alpaca's real-time IEX feed (one consistent feed for the whole window, so
    volume-based features compare like with like), written in the chart format build_panel() reads."""
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(UNIVERSE_SRC, os.path.join(out_dir, "universe.json"))
    try:
        data = alpaca.get_bars(list(symbols), "5Min", int(time.time()) - 7 * 86400, None, feed="iex", pause=0.1)
    except Exception as e:  # keep the last good files
        return [f"download failed: {e}"]
    errors = []
    for s in symbols:
        rows = alpaca.regular_session(data.get(s, []))
        if not rows:
            errors.append(f"{s}: no bars")
            continue
        tmp = os.path.join(out_dir, f"{s}-5m.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(alpaca.chart_json(rows), fh)
        os.replace(tmp, os.path.join(out_dir, f"{s}-5m.json"))
    return errors


def keep_awake(on):
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: the PC won't idle-sleep while this runs (the screen may still turn off).
    # Released automatically when the process exits. No settings are changed.
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))




def next_session(now_ny):
    """The trading date whose regular session has not yet ended (weekends and US market holidays skipped)."""
    from .holidays import is_session
    d = now_ny.date()
    if now_ny.time() >= dt.time(16, 0):
        d += dt.timedelta(days=1)
    while not is_session(d):
        d += dt.timedelta(days=1)
    return d


def live(frozen_path, poll_offset_s=25):
    try:
        alpaca.credentials()
    except alpaca.AlpacaConfigError as e:
        raise SystemExit(str(e))
    f = Frozen(frozen_path)
    keep_awake(True)
    try:
        date = next_session(dt.datetime.now(NY))
        open_t = dt.datetime.combine(date, dt.time(9, 30), NY)
        print(f"next NASDAQ session: {date} — waiting for 09:30 ET ({open_t.astimezone():%a %d %b %H:%M} local)")
        close_t = dt.datetime.combine(date, dt.time(16, 0), NY)
        end_t = close_t + dt.timedelta(minutes=10)
        run = Run(os.path.join("private", "data", "forward", f"live-{date.isoformat()}"), f, "live", date.isoformat())
        live_dir = os.path.join("private", "data", "forward", "live-cache")
        while dt.datetime.now(NY) < open_t + dt.timedelta(minutes=5, seconds=poll_offset_s):
            run.status = f"waiting for the open (09:30 ET); first bar completes 09:35 ET"
            run.last_poll = dt.datetime.now(dt.timezone.utc).isoformat()
            run.write_summary()
            time.sleep(30)
        print(f"live forward test {date}: polling every completed 5-minute bar until {end_t:%H:%M} ET")
        while dt.datetime.now(NY) < end_t:
            run.status = "live"
            errs = fetch_all(f.symbols, live_dir)
            run.errors += [f"{dt.datetime.now(NY):%H:%M:%S} {e}" for e in errs]
            P = build_panel(live_dir, horizon=HORIZON)
            if P.symbols != f.symbols:
                raise SystemExit("symbol order mismatch in live data")
            idx = session_bars(P, date)
            now_s = time.time()
            done = [t for t in idx if int(P.ts[t]) + 300 <= now_s]  # completed bars only
            for k, t in enumerate(done):
                if run.log_prediction(int(P.ts[t]), k, P.X[t], P.close[t]):
                    print(f"  {dt.datetime.fromtimestamp(int(P.ts[t]), NY):%H:%M} logged prediction")
                if np.isfinite(P.y[t]).any() and k + HORIZON < len(done):
                    run.log_outcome(int(P.ts[t]), realized(P, t, f.stock))
            run.last_poll = dt.datetime.now(dt.timezone.utc).isoformat()
            run.write_summary()
            nxt = (int(time.time()) // 300 + 1) * 300 + poll_offset_s
            time.sleep(max(5, nxt - time.time()))
        run.status = "session closed"
        run.write_summary()
        print(f"done: {run.dir}")
    finally:
        keep_awake(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "live", "replay"])
    ap.add_argument("--frozen", default=FROZEN_DEFAULT)
    ap.add_argument("--date")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.frozen)
    elif a.cmd == "replay":
        replay(a.date, a.frozen)
    else:
        live(a.frozen)


if __name__ == "__main__":
    main()
