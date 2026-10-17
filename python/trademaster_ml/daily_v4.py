"""
5-day forecast v4 (research only; CPU only). Rules fixed before any result: docs/FORECAST-V4-NEWS-PROTOCOL.md
(incl. Amendment A).

  inputs   trend + acceleration (split down/up), stock vs sector ETF, leveraged-ETF rebalancing pressure, breadth
           stress, Treasury yields, bonds/fear ETFs, FinBERT news sentiment + "priced in", market news themes
  model    small neural net (2 x 32, weight decay), 5 seeds averaged, early-stopped; pooled over the universe;
           predicts each stock's 1..5-day return relative to the average stock (in volatility units)
  ranges   v3: GJR-GARCH-t + symmetric self-correcting ranges, centred on the predicted mean

Usage:
  python -m trademaster_ml.daily_v4 select    # development only: weight decay by pooled IC (cheaper settings)
  python -m trademaster_ml.daily_v4 run       # development + final test vs v3, all reports
  python -m trademaster_ml.daily_v4 forecast  # next 5 trading days (only if v4 was adopted)
"""
import argparse, json, math, os, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import torch

from . import daily_v2 as v2
from .daily_forecast import H, OUT, FC_DIR, QS, Garch, next_trading_days

torch.set_num_threads(8)
NY = ZoneInfo("America/New_York")
EXTRA = os.path.join("private", "data", "daily_extra")
NEWS = os.path.join("private", "data", "news")
DEV_START, DEV_END = "2017-01-03", "2025-10-10"
WDS = [1e-4, 1e-3, 1e-2]
SEEDS = 5
SEEDS_OFFSET = 0                                             # changed only by the seed-stability check
THEMES = {
    "ai": ["ai bubble", "ai spending", "ai capex", "ai fear", "ai stocks"],
    "rates": ["rate hike", "fed", "treasury yield", "bond yield", "inflation"],
    "delever": ["margin call", "deleverag", "liquidation", "forced selling", "leveraged etf"],
    "export": ["export control", "chip ban", "tariff"],
}


# ---------------------------------------------------------------- helpers
def series(name, dates, key="close"):
    d = json.load(open(os.path.join(EXTRA, f"{name}.json")))
    m = dict(zip(d["dates"], d[key]))
    return np.array([m.get(x, np.nan) for x in dates], float)


def ffill(a):
    a = a.copy()
    for t in range(1, len(a)):
        if not np.isfinite(a[t]):
            a[t] = a[t - 1]
    return a


def lagd(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def expanding_z(x, minp=250):
    """z-score of a market-level series using only data up to each day; clipped to +-4."""
    out = np.full_like(x, np.nan)
    s = s2 = n = 0.0
    for t in range(len(x)):
        if np.isfinite(x[t]):
            if n >= minp:
                sd = math.sqrt(max(s2 / n - (s / n) ** 2, 1e-12))
                out[t] = max(-4.0, min(4.0, (x[t] - s / n) / sd))
            s += x[t]; s2 += x[t] * x[t]; n += 1
    return out


def roll_mean(a, w):
    out = np.full_like(a, np.nan)
    c = np.nancumsum(np.nan_to_num(a), axis=0)
    out[w - 1:] = (c[w - 1:] - np.vstack([np.zeros((1,) + a.shape[1:]), c[:-w]])) / w if a.ndim > 1 else (c[w - 1:] - np.concatenate([[0], c[:-w]])) / w
    return out


# ---------------------------------------------------------------- news -> daily per-stock and market features
def news_daily(dates, syms, is_stock):
    """Assign each item to the trading close it belongs to (published after the previous 16:00 ET close and at or
    before this day's 16:00 ET close; weekends/holidays roll forward)."""
    T, N = len(dates), len(syms)
    idx = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(syms)}
    score = {}
    sp = os.path.join(NEWS, "scores.jsonl")
    if not os.path.exists(sp) or os.environ.get("V4_SMOKE"):
        print("  WARNING: no news scores used (smoke test or scores missing); news features are all zero", flush=True)
        sp = os.devnull
    for line in open(sp, encoding="utf-8"):
        r = json.loads(line)
        score[r["id"]] = r["pos"] - r["neg"]
    cnt, ssum = np.zeros((T, N)), np.zeros((T, N))
    th_cnt = {k: np.zeros(T) for k in THEMES}
    th_sum = {k: np.zeros(T) for k in THEMES}
    mk_cnt, mk_sum = np.zeros(T), np.zeros(T)
    first = datetime.fromisoformat(dates[0]).date()
    for line in open(os.path.join(NEWS, "items.jsonl"), encoding="utf-8"):
        n = json.loads(line)
        sc = score.get(n["id"])
        if sc is None:
            continue
        ts = datetime.fromisoformat(n["t"].replace("Z", "+00:00")).astimezone(NY)
        d = ts.date() + (timedelta(days=1) if ts.hour >= 16 else timedelta(0))
        if d < first:
            continue
        for _ in range(10):                                  # roll to the next trading close
            if str(d) in idx:
                break
            d += timedelta(days=1)
        t = idx.get(str(d))
        if t is None:
            continue
        mk_cnt[t] += 1; mk_sum[t] += sc
        text = (n["headline"] + " " + n["summary"]).lower()
        for k, words in THEMES.items():
            if any(w in text for w in words):
                th_cnt[k][t] += 1; th_sum[k][t] += sc
        for s in n.get("symbols", []):
            j = col.get(s)
            if j is not None and is_stock[j]:
                cnt[t, j] += 1; ssum[t, j] += sc
    return cnt, ssum, th_cnt, th_sum, mk_cnt, mk_sum


# ---------------------------------------------------------------- features
STOCK_FEATS = ["slope3", "slope5", "slope10", "accel", "curve",
               "slope3_dn", "slope3_up", "slope10_dn", "slope10_up", "accel_dn", "accel_up",
               "rel1", "rel5", "rel20", "lev1", "lev5",
               "news_n1", "news_s1", "news_s5", "news_pi1", "news_pi5", "news_gap1"]
MKT_FEATS = ["breadth_low", "breadth_volsell", "y2_lvl", "y10_lvl", "y2_d5", "y2_d20", "y10_d5", "y10_d20", "curve_2s10s",
             "tlt5", "tlt20", "vixy5", "spy_vol20", "spy_dd", "news_mkt",
             "th_ai_n", "th_ai_s", "th_rates_n", "th_rates_s", "th_delever_n", "th_delever_s", "th_export_n", "th_export_s"]


def build(dates, syms, is_stock, C, V, ref, spy, log=print):
    T, N = C.shape
    L = np.log(C)
    r1 = L - lagd(L, 1)
    vol20 = v2.roll(r1, 20, np.nanstd)
    vz = lambda k: (L - lagd(L, k)) / (vol20 * math.sqrt(k))
    z3, z5, z10 = vz(3), vz(5), vz(10)
    z3prev = (lagd(L, 3) - lagd(L, 6)) / (vol20 * math.sqrt(3))
    accel = z3 - z3prev
    curve = z5 - z10
    raw = {"slope3": z3, "slope5": z5, "slope10": z10, "accel": accel, "curve": curve,
           "slope3_dn": np.minimum(z3, 0), "slope3_up": np.maximum(z3, 0), "slope10_dn": np.minimum(z10, 0), "slope10_up": np.maximum(z10, 0),
           "accel_dn": np.where(z3 < 0, accel, 0.0), "accel_up": np.where(z3 > 0, accel, 0.0)}
    for k in (1, 5, 20):
        raw[f"rel{k}"] = (L - lagd(L, k)) - (L - lagd(L, k))[:, ref]
    # leveraged-ETF rebalancing pressure: lev*(lev-1)*underlying return, scaled by the fund's dollar-volume surprise
    def fund(name):
        c, v = ffill(series(name, dates)), series(name, dates, "volume")
        dv = c * v
        surprise = dv / v2.roll(dv[:, None], 60, np.nanmean)[:, 0]
        return np.nan_to_num(surprise, nan=0.0)
    under_r = {"SMH": r1[:, syms.index("SMH")], "QQQ": r1[:, syms.index("QQQ")]}
    sector_p = {}
    for etf, (bull, bear) in {"SMH": ("SOXL", "SOXS"), "QQQ": ("TQQQ", "SQQQ")}.items():
        r = np.nan_to_num(under_r[etf])
        sector_p[etf] = 6 * r * fund(bull) + 12 * r * fund(bear)          # 3x: 3*2 ; -3x: (-3)*(-4)
    lev1 = np.zeros((T, N))
    semis = [j for j, s in enumerate(syms) if is_stock[j] and syms[ref[j]] == "SMH"]
    for j in range(N):
        if is_stock[j]:
            lev1[:, j] = sector_p["SMH"] if j in semis else sector_p["QQQ"]
    for stock, f in (("MU", "MUU"), ("NVDA", "NVDL")):                    # 2x single-stock funds: 2*1
        j = syms.index(stock)
        lev1[:, j] += 2 * np.nan_to_num(r1[:, j]) * fund(f)
    lev1 = lev1 / np.where(np.isfinite(vol20) & (vol20 > 0), vol20, np.nan)
    raw["lev1"] = lev1
    raw["lev5"] = sum(lagd(lev1, k) if k else lev1 for k in range(5))
    # news
    if log:
        log("  loading scored news ...")
    cnt, ssum, th_cnt, th_sum, mk_cnt, mk_sum = news_daily(dates, syms, is_stock)
    s1 = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
    c5 = sum(lagd(cnt, k) if k else cnt for k in range(5)); s5n = sum(lagd(ssum, k) if k else ssum for k in range(5))
    s5 = np.where(np.nan_to_num(c5) > 0, np.nan_to_num(s5n) / np.maximum(np.nan_to_num(c5), 1), 0.0)
    zr1 = r1 / vol20
    raw.update({"news_n1": np.log1p(cnt), "news_s1": s1, "news_s5": s5, "news_pi1": s1 * np.nan_to_num(zr1), "news_pi5": s5 * np.nan_to_num(z5),
                "news_gap1": s1 - np.tanh(np.nan_to_num(zr1))})
    F_stock = np.stack([v2.rank_signed(raw[k], is_stock) for k in STOCK_FEATS], axis=2)
    # market-level
    st = is_stock
    low20 = (C <= v2.roll(C, 20, np.nanmin))
    vol_ratio = V / v2.roll(V, 60, np.nanmean)
    mk = {"breadth_low": np.nanmean(np.where(st, low20, np.nan), axis=1),
          "breadth_volsell": np.nanmean(np.where(st, (vol_ratio >= 2) & (r1 < 0), np.nan), axis=1)}
    y = json.load(open(os.path.join(EXTRA, "yields.json")))
    ym = dict(zip(y["dates"], zip(y["dgs2"], y["dgs10"])))
    y2 = ffill(np.array([ym.get(d, (np.nan, np.nan))[0] for d in dates])); y10 = ffill(np.array([ym.get(d, (np.nan, np.nan))[1] for d in dates]))
    mk.update({"y2_lvl": y2, "y10_lvl": y10, "y2_d5": y2 - lagd(y2, 5), "y2_d20": y2 - lagd(y2, 20), "y10_d5": y10 - lagd(y10, 5),
               "y10_d20": y10 - lagd(y10, 20), "curve_2s10s": y10 - y2})
    lt = np.log(ffill(series("TLT", dates))); lv = np.log(ffill(series("VIXY", dates)))
    mk.update({"tlt5": lt - lagd(lt, 5), "tlt20": lt - lagd(lt, 20), "vixy5": lv - lagd(lv, 5)})
    ls = L[:, spy]
    mk["spy_vol20"] = vol20[:, spy]
    hi = np.array([np.nanmax(ls[max(0, t - 249):t + 1]) for t in range(T)])
    mk["spy_dd"] = ls - hi
    mk["news_mkt"] = np.where(mk_cnt > 0, mk_sum / np.maximum(mk_cnt, 1), 0.0)
    for k in THEMES:
        mk[f"th_{k}_n"] = np.log1p(th_cnt[k]); mk[f"th_{k}_s"] = np.where(th_cnt[k] > 0, th_sum[k] / np.maximum(th_cnt[k], 1), 0.0)
    M = np.stack([np.nan_to_num(expanding_z(mk[k]), nan=0.0) for k in MKT_FEATS], axis=1)
    F = np.concatenate([F_stock, np.repeat(M[:, None, :], N, axis=1)], axis=2)
    F[:, ~is_stock, :] = np.nan
    # stress regime (data up to each day only): SPY >= 8% below 250-day high, or SPY 20-day vol in its top 20% so far
    vq = np.array([np.nanquantile(mk["spy_vol20"][:t + 1], 0.8) if t > 250 else np.inf for t in range(T)])
    stress = (mk["spy_dd"] <= math.log(0.92)) | (mk["spy_vol20"] >= vq)
    return L, r1, vol20, F, stress, raw


# ---------------------------------------------------------------- network
class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.GELU(), torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, H))

    def forward(self, x):
        return self.f(x)


def train_rows(F, Y, t_last):
    """Rows for training at an origin: day s is used for horizon h only if s + h <= origin (outcome known)."""
    X = F[:t_last + 1]
    T_, N_, D = X.shape
    Yt = np.stack([Y[h][:t_last + 1] for h in range(1, H + 1)], axis=2)       # [T, N, H]
    for h in range(1, H + 1):                                                    # hide outcomes not yet known
        Yt[max(0, t_last + 1 - h):, :, h - 1] = np.nan
    X = X.reshape(-1, D); Yt = Yt.reshape(-1, H)
    ok = np.isfinite(X).all(1) & np.isfinite(Yt).any(1)
    return X[ok], np.clip(Yt[ok], -6, 6)


def fit_net(X, Y, wd, seed, epochs=40, batch=4096):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X)
    split = int(n * 0.8)                                                          # rows are time-ordered: last 20% = validation
    Xt, Yt = torch.tensor(X[:split], dtype=torch.float32), torch.tensor(Y[:split], dtype=torch.float32)
    Xv, Yv = torch.tensor(X[split:], dtype=torch.float32), torch.tensor(Y[split:], dtype=torch.float32)
    net = Net(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    def loss_fn(p, y):
        m = torch.isfinite(y)
        return ((p - torch.nan_to_num(y)) ** 2)[m].mean()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xv), Yv))
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def walk_forward(F, Y, first, last, wd, seeds=SEEDS, retrain=21, log=print):
    P = {h: np.full(F.shape[:2], np.nan) for h in range(1, H + 1)}
    nets, trained = None, None
    t0 = time.time()
    for t in range(first, last + 1):
        if trained is None or t - trained >= retrain:
            X, Yr = train_rows(F, Y, t)
            nets = [fit_net(X, Yr, wd, SEEDS_OFFSET + seed) for seed in range(seeds)]
            trained = t
            if log:
                done = (t - first) / max(1, last - first)
                el = (time.time() - t0) / 60
                log(f"  wd={wd:g}: retrained at origin {t - first + 1}/{last - first + 1} on {len(X)} rows, "
                    f"{el:.0f} min elapsed, ~{el / max(done, 1e-3) - el:.0f} min left")
        x = F[t]
        ok = np.isfinite(x).all(1)
        if ok.any():
            with torch.no_grad():
                p = np.mean([n(torch.tensor(x[ok], dtype=torch.float32)).numpy() for n in nets], axis=0)
            for h in range(1, H + 1):
                P[h][t, ok] = p[:, h - 1]
    return P


def regime_metrics(P, Y, first, last, is_stock, stress):
    out = {}
    for name, keep in (("all", lambda t: True), ("stress", lambda t: bool(stress[t])), ("calm", lambda t: not stress[t])):
        ics, hits = [], []
        for t in range(first, last + 1, H):
            if not keep(t):
                continue
            p, y = P[H][t], Y[H][t]
            ok = np.isfinite(p) & np.isfinite(y) & is_stock
            if ok.sum() < 20:
                continue
            ics.append(v2.spearman(p[ok], y[ok]))
            hits.append(np.mean(np.sign(p[ok]) == np.sign(y[ok])))
        ics, hits = np.array(ics), np.array(hits)
        if len(ics) < 5:
            out[name] = {"windows": int(len(ics))}
            continue
        t_ic = ics.mean() / ics.std(ddof=1) * math.sqrt(len(ics))
        t_hit = (hits.mean() - 0.5) / (hits.std(ddof=1) / math.sqrt(len(hits)))
        out[name] = dict(windows=int(len(ics)), ic=float(ics.mean()), ic_t=float(t_ic), hit=float(hits.mean()),
                         hit_p_one_sided=float(0.5 * math.erfc(t_hit / math.sqrt(2))))
    return out


def block_bootstrap_diff(a, b, block=5, n=2000, seed=0):
    """CI for mean(a - b) with a moving-block bootstrap over origins."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(len(d) / block))
    starts = np.arange(0, len(d) - block + 1)
    ms = [np.mean(np.concatenate([d[s:s + block] for s in rng.choice(starts, nb)])[:len(d)]) for _ in range(n)]
    return float(np.mean(d)), float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))


# ---------------------------------------------------------------- commands
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "forecast", "smoke", "ablate", "explain"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    dates, syms, is_stock, C, V, ref, spy = v2.load_panel()
    print(f"panel: {len(dates)} days x {len(syms)} symbols ({dates[0]} .. {dates[-1]})", flush=True)
    L, r1, vol20, F, stress, raw = build(dates, syms, is_stock, C, V, ref, spy)
    Y = v2.rel_targets(L, vol20, is_stock)
    print(f"features: {F.shape[2]} ({len(STOCK_FEATS)} per stock + {len(MKT_FEATS)} market-wide); stress days {int(np.nansum(stress))}/{len(dates)}"
          f"; built in {(time.time() - t_start) / 60:.1f} min", flush=True)
    T = len(dates)
    first_dev = next(i for i, d in enumerate(dates) if d >= DEV_START)
    dev_last = max(i for i, d in enumerate(dates) if d <= DEV_END)
    test_last = T - 1 - H
    test_first = test_last - v2.TEST_N + 1
    sel_path = os.path.join(OUT, "v4_selection.json")

    if a.cmd == "explain":
        # Human-readable snapshot of the latest day's inputs for the dashboard's "Why it's moving" panel.
        t = T - 1
        def pct_rank(series_, v):
            x = series_[:t + 1]; x = x[np.isfinite(x)]
            return float(np.mean(x <= v)) if len(x) > 50 and np.isfinite(v) else None
        y = json.load(open(os.path.join(EXTRA, "yields.json")))
        ym = dict(zip(y["dates"], zip(y["dgs2"], y["dgs10"])))
        y2 = ffill(np.array([ym.get(d, (np.nan, np.nan))[0] for d in dates])); y10 = ffill(np.array([ym.get(d, (np.nan, np.nan))[1] for d in dates]))
        lt = np.log(ffill(series("TLT", dates))); lv = np.log(ffill(series("VIXY", dates)))
        ls = L[:, spy]; hi250 = np.nanmax(ls[max(0, t - 249):t + 1])
        cnt, ssum, th_cnt, th_sum, mk_cnt, mk_sum = news_daily(dates, syms, is_stock)
        themes = {k: {"items_5d": int(th_cnt[k][t - 4:t + 1].sum()),
                      "sentiment_5d": float(th_sum[k][t - 4:t + 1].sum() / max(1, th_cnt[k][t - 4:t + 1].sum())),
                      "vs_normal": pct_rank(np.convolve(th_cnt[k], np.ones(5), "full")[:T], th_cnt[k][t - 4:t + 1].sum())} for k in THEMES}
        low20 = (C <= v2.roll(C, 20, np.nanmin)); r1_ = r1
        market = {"asof": dates[t],
                  "yield_2y": float(y2[t]), "yield_10y": float(y10[t]), "yield_2y_change_20d_bp": float((y2[t] - y2[t - 20]) * 100),
                  "yield_10y_change_20d_bp": float((y10[t] - y10[t - 20]) * 100),
                  "bonds_tlt_20d_pct": float((math.exp(lt[t] - lt[t - 20]) - 1) * 100), "fear_vixy_5d_pct": float((math.exp(lv[t] - lv[t - 5]) - 1) * 100),
                  "spy_below_1y_high_pct": float((math.exp(ls[t] - hi250) - 1) * 100), "stress_regime": bool(stress[t]),
                  "stocks_at_20d_low_pct": float(np.nanmean(np.where(is_stock, low20[t], np.nan)) * 100),
                  "news_items_today": int(mk_cnt[t]), "news_sentiment_today": float(mk_sum[t] / max(1, mk_cnt[t])), "themes": themes}
        stocks = {}
        for sym in v2.TARGETS:
            j = syms.index(sym)
            lev_series = raw["lev5"][:, j]
            stocks[sym] = {
                "move_3d_pct": float((math.exp(L[t, j] - L[t - 3, j]) - 1) * 100), "move_prev3d_pct": float((math.exp(L[t - 3, j] - L[t - 6, j]) - 1) * 100),
                "move_10d_pct": float((math.exp(L[t, j] - L[t - 10, j]) - 1) * 100),
                "slope3_sigma": float(raw["slope3"][t, j]), "slope10_sigma": float(raw["slope10"][t, j]), "accel_sigma": float(raw["accel"][t, j]),
                "vs_sector_5d_pct": float(raw["rel5"][t, j] * 100), "sector_etf": syms[ref[j]],
                "leveraged_etf_pressure_5d": float(lev_series[t]), "leveraged_etf_pressure_pctile": pct_rank(lev_series, lev_series[t]),
                "news_today": int(cnt[t, j]), "news_sentiment_today": float(raw["news_s1"][t, j]), "news_sentiment_5d": float(raw["news_s5"][t, j]),
                "news_priced_in_5d": float(raw["news_pi5"][t, j]), "news_gap_today": float(raw["news_gap1"][t, j])}
        json.dump({"asof": dates[t], "market": market, "stocks": stocks, "created_at": datetime.now(timezone.utc).isoformat()},
                  open(os.path.join(OUT, "v4_explain.json"), "w"), indent=1)
        print(json.dumps({"market": {k: v for k, v in market.items() if k != "themes"}, "MU": stocks["MU"]}, indent=1))
        return

    if a.cmd == "ablate":
        # Extra checks, reported only (they do not change adoption): which layer carries signal, and are results stable
        # across independent seed sets. Same settings as the registered run (5 seeds, retrain every 21 days).
        wd = json.load(open(sel_path))["chosen_wd"]
        names = STOCK_FEATS + MKT_FEATS
        groups = {
            "trend only": [n for n in names if n.startswith(("slope", "accel", "curve"))],
            "ETF + macro only": [n for n in names if n.startswith(("rel", "lev", "breadth", "y2", "y10", "curve_2s10s", "tlt", "vixy", "spy_"))],
            "news only": [n for n in names if n.startswith(("news", "th_"))],
        }
        out = {}
        for g, keep in groups.items():
            Fg = F.copy()
            drop = [i for i, n in enumerate(names) if n not in keep]
            Fg[:, :, drop] = np.where(np.isfinite(Fg[:, :, drop]), 0.0, np.nan)
            P = walk_forward(Fg, Y, first_dev, test_last, wd, log=None)
            out[g] = {"features": keep, "dev": regime_metrics(P, Y, first_dev, dev_last, is_stock, stress),
                      "test": regime_metrics(P, Y, test_first, test_last, is_stock, stress)}
            for per in ("dev", "test"):
                m = out[g][per]
                print(f"{g:17s} {per:4s}: " + "  ".join(f"{reg} IC {m[reg]['ic']:+.4f} (t {m[reg]['ic_t']:+.1f}) hit {m[reg]['hit'] * 100:.1f}%"
                                                         for reg in ("all", "stress", "calm") if "ic" in m[reg]), flush=True)
        for rep_i in (1, 2):                                   # independent seed sets for the full model
            torch_seed_base = 100 * rep_i
            P = {h: np.full(F.shape[:2], np.nan) for h in range(1, H + 1)}
            global SEEDS_OFFSET
            SEEDS_OFFSET = torch_seed_base
            P = walk_forward(F, Y, first_dev, test_last, wd, log=None)
            SEEDS_OFFSET = 0
            m = {"dev": regime_metrics(P, Y, first_dev, dev_last, is_stock, stress)["all"], "test": regime_metrics(P, Y, test_first, test_last, is_stock, stress)["all"]}
            out[f"full model, seed set {rep_i + 1}"] = m
            print(f"full model seed set {rep_i + 1}: dev IC {m['dev']['ic']:+.4f} (t {m['dev']['ic_t']:+.1f})  test IC {m['test']['ic']:+.4f} (t {m['test']['ic_t']:+.1f}) "
                  f"hit {m['test']['hit'] * 100:.1f}%", flush=True)
        json.dump(out, open(os.path.join(OUT, "v4_ablation.json"), "w"), indent=1)
        return

    if a.cmd == "smoke":                                      # quick end-to-end check on a few weeks; no results kept
        t0 = time.time()
        P = walk_forward(F, Y, dev_last - 60, dev_last, 1e-3, seeds=1, retrain=21)
        print(f"  walk-forward ok: {np.isfinite(P[5][dev_last - 60:dev_last + 1]).sum()} predictions, {(time.time() - t0):.0f}s", flush=True)
        print("  regime metrics:", regime_metrics(P, Y, dev_last - 60, dev_last, is_stock, stress)["all"], flush=True)
        j = syms.index("MU")
        recs, _ = v2.eval_symbol("MU", j, L, r1, vol20, P, spy, dev_last - 30, dev_last, dev_last, log=None, variants=("v3sym", "v4sym"))
        print(f"  evaluator ok: {len(recs)} origins, v4sym 5-day 80% hit "
              f"{np.mean([r['v']['v4sym'][4]['in80'] for r in recs if r['v']['v4sym'][4]]):.2f}; total {(time.time() - t_start) / 60:.1f} min", flush=True)
        return

    if a.cmd == "select":
        # development only. Cheaper settings for the search (2 seeds, retrain every 63 days); the final run uses the
        # registered settings (5 seeds, every 21 days).
        res = {}
        for wd in WDS:
            P = walk_forward(F, Y, first_dev, dev_last, wd, seeds=2, retrain=63)
            res[wd] = regime_metrics(P, Y, first_dev, dev_last, is_stock, stress)["all"]
            print(f"weight decay {wd:g}: dev IC {res[wd]['ic']:+.4f} (t {res[wd]['ic_t']:+.2f}) direction hit {res[wd]['hit'] * 100:.2f}%", flush=True)
        best = max(res, key=lambda k: res[k]["ic"])
        json.dump({"chosen_wd": best, "dev": {str(k): v for k, v in res.items()}, "search_settings": "2 seeds, retrain every 63 days"},
                  open(sel_path, "w"), indent=1)
        print(f"chosen weight decay: {best}", flush=True)
        return

    wd = json.load(open(sel_path))["chosen_wd"]
    if a.cmd == "run":
        P = walk_forward(F, Y, first_dev, test_last, wd)
        np.savez_compressed(os.path.join(OUT, "v4_predictions.npz"), **{f"h{h}": P[h] for h in range(1, H + 1)})
        mm = {"dev": regime_metrics(P, Y, first_dev, dev_last, is_stock, stress),
              "test": regime_metrics(P, Y, test_first, test_last, is_stock, stress)}
        for per in ("dev", "test"):
            for reg in ("all", "stress", "calm"):
                m = mm[per][reg]
                if "ic" in m:
                    print(f"{per:4s} {reg:6s}: IC {m['ic']:+.4f} (t {m['ic_t']:+.2f})  direction hit {m['hit'] * 100:.2f}% (p={m['hit_p_one_sided']:.3f})  "
                          f"windows {m['windows']}", flush=True)
        results, recs_all, pits_store = {}, {}, {}
        for sym in v2.TARGETS:
            j = syms.index(sym)
            recs, pits = v2.eval_symbol(sym, j, L, r1, vol20, P, spy, first_dev, test_last, dev_last, variants=("v3sym", "v4sym"))
            results[sym] = {v: {"dev": v2.summarize(recs, first_dev, dev_last, v), "test": v2.summarize(recs, test_first, test_last, v)}
                            for v in ("v3sym", "v4sym")}
            recs_all[sym], pits_store[sym] = recs, pits
        pooled = lambda v, per: float(np.mean([np.mean([x["crps"] for x in results[s][v][per]]) for s in v2.TARGETS]))
        adopt = mm["dev"]["all"]["ic_t"] > 2 and pooled("v4sym", "dev") < pooled("v3sym", "dev")
        chosen = "v4sym" if adopt else "v3sym"
        # test-period comparison with a block-bootstrap CI (per origin, pooled over the 4 stocks and 5 horizons)
        per_origin = defaultdict(lambda: {"v3sym": [], "v4sym": []})
        for s in v2.TARGETS:
            for r in recs_all[s]:
                if test_first <= r["t"] <= test_last:
                    for v in ("v3sym", "v4sym"):
                        per_origin[r["t"]][v].append(np.mean([x["crps"] for x in r["v"][v] if x]))
        ts_ = sorted(per_origin)
        diff = block_bootstrap_diff([np.mean(per_origin[t]["v4sym"]) for t in ts_], [np.mean(per_origin[t]["v3sym"]) for t in ts_])
        base = float(np.mean([np.mean([x["crps_base"] for x in results[s][chosen]["test"]]) for s in v2.TARGETS]))
        test_skill = 1 - pooled(chosen, "test") / base
        # stress vs calm CRPS on the test year, per variant
        sc = {}
        for v in ("v3sym", "v4sym"):
            for reg, keep in (("stress", True), ("calm", False)):
                vals = [np.mean([x["crps"] for x in r["v"][v] if x]) for s in v2.TARGETS for r in recs_all[s]
                        if test_first <= r["t"] <= test_last and bool(stress[r["t"]]) == keep]
                sc[f"{v}_{reg}"] = float(np.mean(vals)) if vals else None
        rules = {"calibration": {s: v2.passes_calibration(results[s][chosen]["test"]) for s in v2.TARGETS},
                 "crps_skill_vs_random_walk": float(test_skill), "crps_rule_pass": bool(test_skill > 0),
                 "direction": mm["test"]["all"], "direction_rule_pass": bool(mm["test"]["all"].get("hit", 0) > 0.5 and mm["test"]["all"].get("hit_p_one_sided", 1) < 0.05)}
        doc = {"protocol": "docs/FORECAST-V4-NEWS-PROTOCOL.md", "weight_decay": wd, "adopted": bool(adopt), "chosen_variant": chosen,
               "dev_pooled_crps": {v: pooled(v, "dev") for v in ("v3sym", "v4sym")}, "test_pooled_crps": {v: pooled(v, "test") for v in ("v3sym", "v4sym")},
               "test_crps_diff_v4_minus_v3": {"mean": diff[0], "ci95": [diff[1], diff[2]]}, "test_crps_by_regime": sc,
               "mean_model": mm, "results": results, "rules": rules, "stress_days_test": int(np.sum(stress[test_first:test_last + 1])),
               "dev_period": [dates[first_dev], dates[dev_last]], "test_period": [dates[test_first], dates[test_last]],
               "runtime_min": (time.time() - t_start) / 60, "created_at": datetime.now(timezone.utc).isoformat()}
        json.dump(doc, open(os.path.join(OUT, "v4_results.json"), "w"), indent=1)
        json.dump({"pits": pits_store, "variant": chosen}, open(os.path.join(OUT, "v4_pits.json"), "w"))
        print(f"\ndevelopment pooled CRPS x1e4: v3sym {doc['dev_pooled_crps']['v3sym'] * 1e4:.3f}  v4sym {doc['dev_pooled_crps']['v4sym'] * 1e4:.3f}")
        print(f"ADOPTED v4: {adopt} -> {chosen}")
        print(f"test pooled CRPS x1e4: v3sym {doc['test_pooled_crps']['v3sym'] * 1e4:.3f}  v4sym {doc['test_pooled_crps']['v4sym'] * 1e4:.3f}  "
              f"(v4-v3 {diff[0] * 1e4:+.3f}, 95% CI {diff[1] * 1e4:+.3f}..{diff[2] * 1e4:+.3f})")
        print("test CRPS by regime x1e4:", {k: round(v * 1e4, 3) if v else None for k, v in sc.items()})
        for s in v2.TARGETS:
            for v in ("v3sym", "v4sym"):
                t_ = results[s][v]["test"]
                print(f"{s:5s} {v:6s} TEST 80%: " + " ".join(f"{x['coverage']['in80']:.2f}" for x in t_) + "  95%: " + " ".join(f"{x['coverage']['in95']:.2f}" for x in t_)
                      + f"  skill {np.mean([x['crps_skill_vs_baseline'] for x in t_]) * 100:+.1f}%  dir {np.mean([x['dir_hit'] for x in t_]) * 100:.1f}%")
        print("RULES:", json.dumps(rules["calibration"]), f"crps skill {test_skill * 100:+.2f}% pass={rules['crps_rule_pass']}",
              f"direction pass={rules['direction_rule_pass']}", f"| runtime {(time.time() - t_start) / 60:.0f} min", flush=True)
        # hold-out week, information only
        t_as = dates.index("2026-10-09")
        for s in v2.TARGETS:
            r = next(x for x in recs_all[s] if x["t"] == t_as)
            print(f"  hold-out 12-16 Oct {s}: " + "  ".join(f"{v} {sum(1 for x in r['v'][v] if x and x['in80'])}/5" for v in ("v3sym", "v4sym")))
        return

    # forecast (only meaningful if adopted)
    res = json.load(open(os.path.join(OUT, "v4_results.json")))
    if not res["adopted"]:
        print("v4 was not adopted under the protocol; v3 stays live. Nothing written.")
        return
    raise SystemExit("v4 forecast writer: to be enabled after review of the adopted model")


if __name__ == "__main__":
    main()
