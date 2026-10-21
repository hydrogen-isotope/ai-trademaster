"""
Larger 5-minute model under docs/OPTIONS-ALARM-PROTOCOL.md amendment A: the 100 most-traded NASDAQ stocks of each year
(point in time), 5-minute bars 2019-2026, several network sizes trained on the GPU (DirectML). Research only.

Usage: python -m trademaster_ml.intraday_big universe | fetch | build
       (training runs in python/.venv-gpu: python -m trademaster_ml.intraday_big train)
"""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np

DIR = os.path.join("private", "data", "intraday_big")
OUT = os.path.join("private", "reports", "intraday_big")
YEARS = range(2019, 2027)
TOP = 100
END = "2026-10-20"
EXTRA = ("NVDA", "MU", "QQQ")


def universe(log=print):
    """{year: [symbols]}: top 100 NASDAQ stocks by 20-day dollar volume before the year's first trading day."""
    p = np.load(os.path.join("private", "data", "us", "panel.npz"))
    dates = [str(d) for d in p["dates"]]
    syms = [str(s) for s in p["symbols"]]
    exch = [str(e) for e in p["exchange"]]
    dv = (p["close"].astype(float) * p["volume"].astype(float))
    out = {}
    for y in YEARS:
        i = next(k for k, d in enumerate(dates) if d >= f"{y}-01-01")
        m = np.nanmean(dv[i - 20:i], axis=0)
        ok = np.array([e.upper().startswith("NASDAQ") for e in exch]) & np.isfinite(m)
        ok &= np.array([s != "QQQ" and not s.endswith(("W", "U")) for s in syms])
        idx = np.argsort(np.where(ok, -m, np.inf))[:TOP]
        out[str(y)] = [syms[j] for j in idx]
    os.makedirs(DIR, exist_ok=True)
    json.dump(out, open(os.path.join(DIR, "universe.json"), "w"), indent=0)
    allsyms = sorted({s for v in out.values() for s in v} | set(EXTRA))
    log(f"universe: {len(allsyms)} symbols over {len(out)} years; 2019 top 5 {out['2019'][:5]}, 2026 top 5 {out['2026'][:5]}")
    return out


def _fetch_one(sym, start, creds):
    from . import alpaca
    path = os.path.join(DIR, "bars", f"{sym}.npz")
    if os.path.exists(path):
        return sym, -1
    rows, token = [], None
    while True:
        p = {"timeframe": "5Min", "feed": "sip", "adjustment": "split", "start": f"{start}T13:00:00Z",
             "end": f"{END}T21:00:00Z", "limit": 10000}
        if token:
            p["page_token"] = token
        for wait in range(12):                                     # shared rate limit: back off on HTTP 429
            try:
                d = alpaca._get(f"/v2/stocks/{sym}/bars", p, creds) or {}
                break
            except RuntimeError as e:
                if "429" not in str(e) or wait == 11:
                    raise
                time.sleep(15 + 5 * wait)
        for x in d.get("bars") or []:
            t = datetime.fromisoformat(x["t"].replace("Z", "+00:00")).astimezone(alpaca.NY)
            m = t.hour * 60 + t.minute
            if 570 <= m < 960:
                rows.append((int(t.strftime("%Y%m%d")), m, x["o"], x["h"], x["l"], x["c"], x["v"]))
        token = d.get("next_page_token")
        if not token:
            break
    if not rows:
        return sym, 0
    a = np.array(rows, dtype=np.float64)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, day=a[:, 0].astype(np.int32), minute=a[:, 1].astype(np.int16), o=a[:, 2], h=a[:, 3],
                        l=a[:, 4], c=a[:, 5], v=a[:, 6])
    os.replace(tmp, path)
    return sym, len(rows)


def fetch(log=print):
    from . import alpaca
    uni = json.load(open(os.path.join(DIR, "universe.json")))
    first = {}
    for y, ss in uni.items():
        for s in ss:
            first.setdefault(s, int(y))
    for s in EXTRA:
        first[s] = 2019
    os.makedirs(os.path.join(DIR, "bars"), exist_ok=True)
    creds = alpaca.credentials()
    jobs = sorted(first.items())
    t0 = time.time()
    with ThreadPoolExecutor(12) as ex:
        for k, (sym, n) in enumerate(ex.map(lambda j: _fetch_one(j[0], f"{j[1] - 1}-11-15", creds), jobs), 1):
            if n >= 0:
                log(f"  {k}/{len(jobs)} {sym}: {n} bars ({(time.time() - t0) / 60:.1f} min)", flush=True)
    log("fetch done", flush=True)


# ---------------------------------------------------------------- features (CPU)

SLOTS, LOOK, H, TRAIN_STRIDE = 78, 48, 6, 2      # 9:30..15:55 bars; 48-bar history; 30-minute target
LAST_SLOT = 69                                   # bar starting 15:15 (closes 15:20): its target ends 15:50
FIT_END, VAL = 20241231, (20250102, 20250630)
FINAL = (20250701, 20261020)
NQ = 12


def _grid(sym, days, di, z=None):
    z = z if z is not None else np.load(os.path.join(DIR, "bars", f"{sym}.npz"))
    T = len(days)
    O, C, V = (np.full((T, SLOTS), np.nan) for _ in range(3))
    r = np.array([di.get(int(d), -1) for d in z["day"]])
    s = (z["minute"].astype(int) - 570) // 5
    ok = (r >= 0) & (s >= 0) & (s < SLOTS)
    O[r[ok], s[ok]], C[r[ok], s[ok]], V[r[ok], s[ok]] = z["o"][ok], z["c"][ok], z["v"][ok]
    V = np.nan_to_num(V)
    flat = C.reshape(-1).copy()                                    # carry the last close over missing bars
    idx = np.where(np.isfinite(flat), np.arange(flat.size), 0)
    np.maximum.accumulate(idx, out=idx)
    flat = flat[idx]
    flat[~np.isfinite(C.reshape(-1)) & (idx == 0)] = np.nan
    C = flat.reshape(T, SLOTS)
    O0 = np.where(np.isfinite(O[:, 0]), O[:, 0], C[:, 0])
    return O0, C, V


def _features(O0, C, V, Q):
    """Per-bar arrays for one symbol: normalised returns and volume ratios (flattened), and per-day scales."""
    T = C.shape[0]
    R = np.empty_like(C)
    R[:, 1:] = np.log(C[:, 1:] / C[:, :-1])
    R[:, 0] = np.log(C[:, 0] / O0)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.full(T, np.nan)
    vprof = np.full((T, SLOTS), np.nan)
    cs = np.cumsum(np.r_[0.0, (R ** 2).sum(1)])
    vc = np.cumsum(np.vstack([np.zeros((1, SLOTS)), V]), axis=0)
    for d in range(20, T):
        vol[d] = math.sqrt((cs[d] - cs[d - 20]) / (20 * SLOTS))
        vprof[d] = (vc[d] - vc[d - 20]) / 20
    vol = np.maximum(vol, 1e-4)
    VR = np.log((V + 1) / (vprof + 1))
    gap = np.full(T, np.nan)
    gap[1:] = np.log(O0[1:] / C[:-1, -1])
    return R, VR, vol, gap


def build(log=print):
    """Feature matrices for fit / validation / final -> private/data/intraday_big/{fit,val,final}_{X,y,meta}.npy."""
    from numpy.lib.stride_tricks import sliding_window_view as win
    uni = {int(y): set(v) for y, v in json.load(open(os.path.join(DIR, "universe.json"))).items()}
    zq = np.load(os.path.join(DIR, "bars", "QQQ.npz"))
    days = np.unique(zq["day"])
    days = days[days >= 20181115]
    di = {int(d): i for i, d in enumerate(days)}
    qO, qC, qV = _grid("QQQ", days, di)
    qR, _, qvol, _ = _features(qO, qC, qV, None)
    qRn = (qR / qvol[:, None]).reshape(-1)
    qopen = np.log(qC / qO[:, None]) / (qvol[:, None] * np.sqrt(np.arange(1, SLOTS + 1)))
    dow = np.array([date(int(str(d)[:4]), int(str(d)[4:6]), int(str(d)[6:])).weekday() for d in days])
    syms = sorted(f[:-4] for f in os.listdir(os.path.join(DIR, "bars")) if f.endswith(".npz") and f[:-4] != "QQQ")
    parts = {k: [] for k in ("fit", "val", "final")}
    for si, sym in enumerate(syms):
        try:
            O0, C, V = _grid(sym, days, di)
        except Exception as e:  # noqa: BLE001
            log(f"  {sym}: skipped ({e})")
            continue
        R, VR, vol, gap = _features(O0, C, V, None)
        Rn = np.clip(R / vol[:, None], -20, 20).reshape(-1)
        VRf = np.clip(np.nan_to_num(VR, nan=0.0), -5, 5).reshape(-1)
        sinceo = np.log(C / O0[:, None]) / (vol[:, None] * np.sqrt(np.arange(1, SLOTS + 1)))
        y = np.full((len(days), SLOTS), np.nan)
        y[:, :SLOTS - H] = np.log(C[:, H:] / C[:, :-H]) / vol[:, None]
        for name, (a, b, stride) in {"fit": (0, FIT_END, TRAIN_STRIDE), "val": (VAL[0], VAL[1], 1),
                                     "final": (FINAL[0], FINAL[1], 1)}.items():
            rows = []
            for d in range(25, len(days)):
                dd = int(days[d])
                if not (a <= dd <= b) or not (sym in uni.get(dd // 10000, ()) or sym in EXTRA):
                    continue
                if not np.isfinite(C[d]).all() or not np.isfinite(vol[d]) or not np.isfinite(gap[d]) or (V[d] > 0).sum() < 60:
                    continue
                for s in range(0, LAST_SLOT + 1, stride):
                    if np.isfinite(y[d, s]):
                        rows.append((d, s))
            if not rows:
                continue
            rr = np.array(rows)
            flat = rr[:, 0] * SLOTS + rr[:, 1]
            X = np.hstack([win(Rn, LOOK)[flat - LOOK + 1], win(VRf, LOOK)[flat - LOOK + 1],
                           np.c_[np.clip(sinceo[rr[:, 0], rr[:, 1]], -10, 10), np.clip(gap[rr[:, 0]] / (vol[rr[:, 0]] * math.sqrt(SLOTS)), -10, 10)],
                           np.clip(win(qRn, NQ)[flat - NQ + 1], -20, 20), np.clip(qopen[rr[:, 0], rr[:, 1]], -10, 10)[:, None],
                           (rr[:, 1] / (SLOTS - 1))[:, None], np.eye(5)[dow[rr[:, 0]]]]).astype(np.float32)
            meta = np.c_[np.full(len(rr), si), days[rr[:, 0]], rr[:, 1]].astype(np.int32)
            parts[name].append((np.nan_to_num(X), np.clip(y[rr[:, 0], rr[:, 1]], -10, 10).astype(np.float32), meta))
        if si % 20 == 0:
            log(f"  {si + 1}/{len(syms)} {sym}: fit {sum(len(p[1]) for p in parts['fit']):,} samples", flush=True)
    for name, ps in parts.items():
        np.save(os.path.join(DIR, f"{name}_X.npy"), np.concatenate([p[0] for p in ps]))
        np.save(os.path.join(DIR, f"{name}_y.npy"), np.concatenate([p[1] for p in ps]))
        np.save(os.path.join(DIR, f"{name}_meta.npy"), np.concatenate([p[2] for p in ps]))
        log(f"{name}: {sum(len(p[1]) for p in ps):,} samples, {ps[0][0].shape[1]} inputs", flush=True)
    json.dump(syms, open(os.path.join(DIR, "symbols.json"), "w"))


def vector(zs, zq, day, slot):
    """The model's 117 inputs for one stock at the close of bar `slot` on `day` (YYYYMMDD), from bar dicts
    {day, minute, o, h, l, c, v} for the stock (zs) and QQQ (zq). Same formulas as build()."""
    days = np.unique(zq["day"])
    days = days[days <= day]
    di = {int(x): i for i, x in enumerate(days)}
    d = di[int(day)]
    qO, qC, qV = _grid("QQQ", days, di, zq)
    qR, _, qvol, _ = _features(qO, qC, qV, None)
    qRn = (qR / qvol[:, None]).reshape(-1)
    O0, C, V = _grid("", days, di, zs)
    R, VR, vol, gap = _features(O0, C, V, None)
    Rn = np.clip(R / vol[:, None], -20, 20).reshape(-1)
    VRf = np.clip(np.nan_to_num(VR, nan=0.0), -5, 5).reshape(-1)
    f = d * SLOTS + slot
    so = np.log(C[d, slot] / O0[d]) / (vol[d] * math.sqrt(slot + 1))
    qo = np.log(qC[d, slot] / qO[d]) / (qvol[d] * math.sqrt(slot + 1))
    dw = date(int(str(day)[:4]), int(str(day)[4:6]), int(str(day)[6:])).weekday()
    x = np.r_[Rn[f - LOOK + 1:f + 1], VRf[f - LOOK + 1:f + 1], np.clip(so, -10, 10),
              np.clip(gap[d] / (vol[d] * math.sqrt(SLOTS)), -10, 10), np.clip(qRn[f - NQ + 1:f + 1], -20, 20),
              np.clip(qo, -10, 10), slot / (SLOTS - 1), np.eye(5)[dw]].astype(np.float32)
    return np.nan_to_num(x)


def live_features(sym, creds=None):
    """The model's inputs for `sym` at the latest complete 5-minute bar today (IEX feed, the live data the account can
    read), saved with each manual trade so the user's decisions can be compared with what the model saw."""
    from . import alpaca
    creds = creds or alpaca.credentials()
    now = datetime.now(alpaca.NY)
    start = (now - timedelta(days=45)).strftime("%Y-%m-%dT00:00:00Z")

    def get(s):
        rows, token = [], None
        while True:
            p = {"timeframe": "5Min", "feed": "iex", "adjustment": "split", "start": start, "limit": 10000}
            if token:
                p["page_token"] = token
            d = alpaca._get(f"/v2/stocks/{s}/bars", p, creds) or {}
            for x in d.get("bars") or []:
                t = datetime.fromisoformat(x["t"].replace("Z", "+00:00")).astimezone(alpaca.NY)
                m = t.hour * 60 + t.minute
                if 570 <= m < 960 and t + timedelta(minutes=5) <= now:
                    rows.append((int(t.strftime("%Y%m%d")), m, x["o"], x["h"], x["l"], x["c"], x["v"]))
            token = d.get("next_page_token")
            if not token:
                break
        a = np.array(rows, dtype=np.float64)
        return {"day": a[:, 0].astype(np.int64), "minute": a[:, 1].astype(int), "o": a[:, 2], "h": a[:, 3], "l": a[:, 4],
                "c": a[:, 5], "v": a[:, 6]}

    zs, zq = get(sym), get("QQQ")
    day = int(zq["day"].max())
    slot = int((zq["minute"][zq["day"] == day].max() - 570) // 5)
    return {"feed": "iex", "day": day, "bar": f"{(570 + 5 * slot) // 60}:{(570 + 5 * slot) % 60:02d}",
            "inputs": [round(float(v), 5) for v in vector(zs, zq, day, slot)]}


# ---------------------------------------------------------------- training (GPU, python/.venv-gpu)

SIZES = {"small": ("mlp", 2, 64), "medium": ("mlp", 3, 256), "large": ("mlp", 4, 1024), "conv": ("conv", 4, 256)}


def daily_ic(pred, y, meta):
    """Cross-sectional Spearman at each (day, bar), averaged per day -> (mean, t over days)."""
    def rankdata(x):                                               # plain NumPy (the GPU venv has no SciPy); ties are rare
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(len(x))
        return r
    key = meta[:, 1].astype(np.int64) * 100 + meta[:, 2]
    order = np.argsort(key, kind="stable")
    k, p, t = key[order], pred[order], y[order]
    cuts = np.flatnonzero(np.diff(k)) + 1
    by_day = {}
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(k)]):
        if b - a < 10:
            continue
        ic = np.corrcoef(rankdata(p[a:b]), rankdata(t[a:b]))[0, 1]
        if np.isfinite(ic):
            by_day.setdefault(int(k[a] // 100), []).append(ic)
    v = np.array([np.mean(x) for x in by_day.values()])
    return float(v.mean()), float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))), len(v)


def train(log=print):
    import torch
    import torch_directml
    dev = torch_directml.device(1)                                  # the RX 7800 XT (device 0 is the integrated GPU)
    log(f"device: {torch_directml.device_name(1).strip(chr(0))}", flush=True)
    torch.manual_seed(7)
    np.random.seed(7)
    Xf, yf = np.load(os.path.join(DIR, "fit_X.npy")), np.load(os.path.join(DIR, "fit_y.npy"))
    Xv, yv, mv = (np.load(os.path.join(DIR, f"val_{k}.npy")) for k in ("X", "y", "meta"))
    Xt = np.load(os.path.join(DIR, "final_X.npy"))
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    nin = Xf.shape[1]
    os.makedirs(OUT, exist_ok=True)

    class Conv(torch.nn.Module):
        def __init__(self, layers, ch):
            super().__init__()
            mods, c = [], 2
            for _ in range(layers):
                mods += [torch.nn.Conv1d(c, ch, 5, padding=2), torch.nn.GELU()]
                c = ch
            self.conv = torch.nn.Sequential(*mods)
            self.head = torch.nn.Sequential(torch.nn.Linear(ch * 2 + nin - 2 * LOOK, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

        def forward(self, x):
            seq = torch.stack([x[:, :LOOK], x[:, LOOK:2 * LOOK]], 1)
            h = self.conv(seq)
            return self.head(torch.cat([h.mean(2), h[:, :, -1], x[:, 2 * LOOK:]], 1)).squeeze(1)

    def mlp(layers, width):
        mods, c = [], nin
        for _ in range(layers):
            mods += [torch.nn.Linear(c, width), torch.nn.GELU(), torch.nn.Dropout(0.1)]
            c = width
        return torch.nn.Sequential(*mods, torch.nn.Linear(c, 1), torch.nn.Flatten(0))

    def predict(model, X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.from_numpy((X[i:i + 8192] - mu) / sd).to(dev)
                out.append(model(xb).cpu().numpy())
        model.train()
        return np.concatenate(out)

    summary = {}
    for name, (kind, layers, width) in SIZES.items():
        if os.path.exists(os.path.join(OUT, f"pred_final_{name}.npy")):
            summary[name] = json.load(open(os.path.join(OUT, f"train_{name}.json")))
            continue
        torch.manual_seed(7)
        model = (Conv(layers, width) if kind == "conv" else mlp(layers, width)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3 if width <= 256 else 3e-4, weight_decay=1e-5)
        best, best_ic, bad, hist = None, -1.0, 0, []
        for ep in range(8):
            t0 = time.time()
            perm = np.random.permutation(len(Xf))
            tot = 0.0
            for b in range(0, len(perm), 8192):
                i = np.sort(perm[b:b + 8192])
                xb = torch.from_numpy((Xf[i] - mu) / sd).to(dev)
                yb = torch.from_numpy(yf[i]).to(dev)
                loss = torch.nn.functional.mse_loss(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(i)
            pv = predict(model, Xv)
            ic, t, nd = daily_ic(pv, yv, mv)
            hist.append({"epoch": ep + 1, "loss": tot / len(perm), "val_ic": ic, "val_t": t, "s": time.time() - t0})
            log(f"  {name} epoch {ep + 1}: loss {tot / len(perm):.4f}, validation IC {ic:+.4f} (t {t:+.2f}), {time.time() - t0:.0f}s", flush=True)
            if ic > best_ic:
                best_ic, bad = ic, 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        model.load_state_dict(best)
        torch.save(best, os.path.join(OUT, f"model_{name}.pt"))
        np.save(os.path.join(OUT, f"pred_val_{name}.npy"), predict(model, Xv))
        np.save(os.path.join(OUT, f"pred_final_{name}.npy"), predict(model, Xt))      # not read until the final check
        summary[name] = {"best_val_ic": best_ic, "history": hist, "params": sum(p.numel() for p in model.parameters())}
        json.dump(summary[name], open(os.path.join(OUT, f"train_{name}.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "scaler.npy"), np.vstack([mu, sd]))
    log("training done: " + ", ".join(f"{k} IC {v['best_val_ic']:+.4f}" for k, v in summary.items()), flush=True)


# ---------------------------------------------------------------- selection and the final check (CPU)

def _lookup(part, name):
    """{(sym, 'YYYY-MM-DD', minute): prediction} for NVDA and MU."""
    syms = json.load(open(os.path.join(DIR, "symbols.json")))
    meta = np.load(os.path.join(DIR, f"{part}_meta.npy"))
    pred = np.load(os.path.join(OUT, f"pred_{part}_{name}.npy"))
    out = {}
    for s in ("NVDA", "MU"):
        k = syms.index(s)
        for j in np.flatnonzero(meta[:, 0] == k):
            d = str(meta[j, 1])
            out[(s, f"{d[:4]}-{d[4:6]}-{d[6:]}", 570 + 5 * int(meta[j, 2]))] = float(pred[j])
    return out


def _rules(P, thr):
    from . import options_alarm as A
    D = {s: A.load_sym(s) for s in A.SYMS}

    def signal(sym, day, i):
        m = D[sym][day]["m"][i]
        v = P.get((sym, day, m))
        return v is not None and v < thr

    def entry_ok(sym, day):
        v = P.get((sym, day, 570))                                  # the 9:30 bar's close, just before the 9:35 entry
        return v is not None and v > 0
    return signal, entry_ok


def select(log=print):
    from . import options_alarm as A
    yv, mv = np.load(os.path.join(DIR, "val_y.npy")), np.load(os.path.join(DIR, "val_meta.npy"))
    ics = {}
    for name in SIZES:
        p = np.load(os.path.join(OUT, f"pred_val_{name}.npy"))
        ics[name] = daily_ic(p, yv, mv)
        log(f"  {name}: validation IC {ics[name][0]:+.4f} (t {ics[name][1]:+.2f}, {ics[name][2]} days)")
    best = max(ics, key=lambda k: ics[k][0])
    pv = np.load(os.path.join(OUT, f"pred_val_{best}.npy"))
    P = _lookup("val", best)
    v0, v1 = "2025-01-02", "2025-06-30"
    thr = {}
    for q in (10, 20, 30):
        t = float(np.percentile(pv, q))
        sig, _ = _rules(P, t)
        r = A.evaluate(v0, v1, {"C": None}, signal=sig)["C"]
        thr[q] = {"threshold": t, **r}
        log(f"  exit rule C at the {q}th percentile ({t:+.3f}): C-A {r['mean_diff_pct']:+.3f}% per trade (t {r['t']:+.2f}), {r['trades']} trades")
    q = max(thr, key=lambda k: thr[k]["mean_diff_pct"])
    _, ent = _rules(P, 0)
    d = A.evaluate(v0, v1, {"D": None}, entry_ok=ent)["D"]
    log(f"  entry rule D on validation: D-A {d['mean_diff_pct']:+.3f}% per trade (t {d['t']:+.2f})")
    out = {"ics": ics, "selected": best, "threshold_pct": q, "threshold": thr[q]["threshold"], "val_C": thr, "val_D": d}
    json.dump(out, open(os.path.join(OUT, "select.json"), "w"), indent=1)
    log(f"selected: {best}, exit threshold = {q}th percentile")
    return out


def final(log=print):
    from . import options_alarm as A
    if os.path.exists(os.path.join(OUT, "final.json")):
        raise SystemExit("the final check was already run once")
    s = json.load(open(os.path.join(OUT, "select.json")))
    name = s["selected"]
    yt, mt = np.load(os.path.join(DIR, "final_y.npy")), np.load(os.path.join(DIR, "final_meta.npy"))
    ic = daily_ic(np.load(os.path.join(OUT, f"pred_final_{name}.npy")), yt, mt)
    P = _lookup("final", name)
    sig, ent = _rules(P, s["threshold"])
    f0, f1 = A.FINAL_FROM, str(A.END)
    c = A.evaluate(f0, f1, {"C": None}, signal=sig)["C"]
    d = A.evaluate(f0, f1, {"D": None}, entry_ok=ent)["D"]
    out = {"model": name, "ic": ic, "C": c, "D": d,
           "pass_C": bool(c["mean_diff_pct"] > 0 and c["t"] >= 2.24), "pass_D": bool(d["mean_diff_pct"] > 0 and d["t"] >= 2.24)}
    json.dump(out, open(os.path.join(OUT, "final.json"), "w"), indent=1)
    log(f"final ({name}): IC {ic[0]:+.4f} (t {ic[1]:+.2f}, {ic[2]} days)")
    log(f"  exit rule C: C-A {c['mean_diff_pct']:+.3f}% per trade (t {c['t']:+.2f}) -> {'PASS' if out['pass_C'] else 'fails'}")
    log(f"  entry rule D: D-A {d['mean_diff_pct']:+.3f}% per trade (t {d['t']:+.2f}) -> {'PASS' if out['pass_D'] else 'fails'}")
    return out


if __name__ == "__main__":
    {"universe": universe, "fetch": fetch, "build": build, "train": train, "select": select, "final": final}[sys.argv[1]]()
