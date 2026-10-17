"""
Point-in-time cross-sectional dataset from intraday bars.

Every feature at bar t uses closes/volumes at bars <= t of the same session only. The target is the forward
log return from t to t+H bars, inside the same session (no overnight). Sessions are split chronologically.
"""
import json, os
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")  # handles EST/EDT across multi-year histories

# Window lengths are in bars and depend on the bar interval (see PROFILES).
FEATURES = ["r1", "r_s", "r_m", "r_l", "vol_l", "resid", "volz", "tod", "rel", "dopen", "rpos"]

PROFILES = {
    # short/mid/long return windows, volatility window, volume-surprise lookback, residual and relative windows,
    # beta half-life (bars), and whether trailing windows may span sessions (overnight returns are always excluded).
    "5m": dict(short=3, mid=6, long=12, vol=12, volz=78, resid=6, rel=6, beta_halflife=390, cross_session=False),
    "1h": dict(short=2, mid=3, long=7, vol=7, volz=70, resid=3, rel=3, beta_halflife=140, cross_session=True),
}


@dataclass
class Panel:
    symbols: list          # N
    is_etf: np.ndarray     # [N] bool
    ref_idx: np.ndarray    # [N] index of reference ETF (-1 for ETFs)
    ts: np.ndarray         # [T] unix seconds
    session: np.ndarray    # [T] int session index
    X: np.ndarray          # [T, N, F] raw features (nan where unavailable)
    y: np.ndarray          # [T, N] forward return over the horizon (nan where unavailable)
    y1: np.ndarray         # [T, N] forward 1-bar return, same session (for portfolio simulation)
    close: np.ndarray      # [T, N]


def _load(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    return r["timestamp"], q["close"], q["volume"]


def build_panel(hist_dir, interval=None, horizon=2):
    uni = json.load(open(os.path.join(hist_dir, "universe.json"), encoding="utf-8"))
    interval = interval or uni.get("interval", "5m")
    prof = PROFILES[interval]
    beta_halflife = prof["beta_halflife"]
    symbols = uni["stocks"] + uni["etfs"]
    data = {s: _load(os.path.join(hist_dir, f"{s}-{interval}.json")) for s in symbols}
    ts = sorted(set().union(*[set(d[0]) for d in data.values()]))
    tix = {t: i for i, t in enumerate(ts)}
    T, N = len(ts), len(symbols)
    close = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    for j, s in enumerate(symbols):
        for t, c, v in zip(*data[s]):
            if c is not None and c > 0:
                close[tix[t], j] = c
                vol[tix[t], j] = v if v is not None else np.nan
    days = [datetime.fromtimestamp(t, NY).date() for t in ts]
    uniq = sorted(set(days))
    session = np.array([uniq.index(d) for d in days])
    mins = np.array([(datetime.fromtimestamp(t, NY).hour * 60 + datetime.fromtimestamp(t, NY).minute) - 570 for t in ts])

    # 1-bar log returns within a session (first bar of each session has no return).
    r1 = np.full((T, N), np.nan)
    same = np.r_[False, session[1:] == session[:-1]]
    r1[1:] = np.log(close[1:] / close[:-1])
    r1[~same] = np.nan

    cross = prof["cross_session"]
    first_bar = ~same

    def windowed(a, k, fn):
        # Cross-session mode: the (missing) overnight return at each session's first bar counts as 0.
        src = np.where(first_bar[:, None] & np.isfinite(close), 0.0, a) if cross else a
        out = np.full_like(a, np.nan)
        for t in range(T):
            lo = t - k + 1
            if lo < 0 or (not cross and session[lo] != session[t]):
                continue
            w = src[lo:t + 1]
            out[t] = np.where(np.isnan(w).any(0), np.nan, fn(w))
        return out

    trailing_sum = lambda a, k: windowed(a, k, lambda w: w.sum(0))
    trailing_std = lambda a, k: windowed(a, k, lambda w: w.std(0))

    is_etf = np.array([s in uni["etfs"] for s in symbols])
    ref_idx = np.array([symbols.index(uni["reference_etf"][s]) if s in uni["reference_etf"] else -1 for s in symbols])

    # Causal EW beta of each stock's r1 to its reference ETF (state carries across sessions; uses data <= t-1 for bar t).
    alpha = 1 - 0.5 ** (1 / beta_halflife)
    cov = np.zeros(N); var = np.zeros(N); beta = np.ones(N)
    resid = np.full((T, N), np.nan)
    for t in range(T):
        for j in range(N):
            k = ref_idx[j]
            if k < 0 or np.isnan(r1[t, j]) or np.isnan(r1[t, k]):
                continue
            resid[t, j] = r1[t, j] - beta[j] * r1[t, k]
            cov[j] = (1 - alpha) * cov[j] + alpha * r1[t, j] * r1[t, k]
            var[j] = (1 - alpha) * var[j] + alpha * r1[t, k] ** 2
            if var[j] > 1e-14:
                beta[j] = np.clip(cov[j] / var[j], -3, 3)

    # Volume surprise: log volume vs trailing mean of the previous `volz` bars (any session).
    lv = np.log1p(vol)
    volz = np.full((T, N), np.nan)
    L = prof["volz"]
    with np.errstate(all="ignore"):
        for t in range(L, T):
            w = lv[t - L:t]
            mu, sd = np.nanmean(w, 0), np.nanstd(w, 0)
            volz[t] = (lv[t] - mu) / np.where(sd > 1e-9, sd, np.nan)

    r_rel = trailing_sum(r1, prof["rel"])
    # Sector-relative momentum: own trailing return minus the reference ETF's (ETFs: 0).
    rel6 = np.zeros((T, N))
    for j in range(N):
        rel6[:, j] = r_rel[:, j] - r_rel[:, ref_idx[j]] if ref_idx[j] >= 0 else 0.0
    # Distance from the session's first close, and position inside the session's running close range (causal).
    dopen = np.full((T, N), np.nan)
    rpos = np.full((T, N), np.nan)
    for sidx in np.unique(session):
        rows = np.where(session == sidx)[0]
        c = close[rows]
        first = c[0]
        run_max = np.fmax.accumulate(c, axis=0)
        run_min = np.fmin.accumulate(c, axis=0)
        dopen[rows] = np.log(c / first)
        span = run_max - run_min
        rpos[rows] = np.where(span > 0, (c - run_min) / np.where(span > 0, span, 1), 0.5)

    X = np.stack([r1, trailing_sum(r1, prof["short"]), trailing_sum(r1, prof["mid"]), trailing_sum(r1, prof["long"]),
                  trailing_std(r1, prof["vol"]), trailing_sum(resid, prof["resid"]), volz,
                  np.repeat((mins / 390.0)[:, None], N, 1), rel6, dopen, rpos], axis=-1)

    y = np.full((T, N), np.nan)
    y1 = np.full((T, N), np.nan)
    for t in range(T - 1):
        if session[t + 1] == session[t]:
            y1[t] = np.log(close[t + 1] / close[t])
        if t + horizon < T and session[t + horizon] == session[t]:
            y[t] = np.log(close[t + horizon] / close[t])
    return Panel(symbols, is_etf, ref_idx, np.array(ts), session, X, y, y1, close)


def split_sessions(panel, train=40, val=10):
    s = panel.session
    n = s.max() + 1
    test_start = min(train + val, n - 1)
    return (s < train), (s >= train) & (s < train + val), (s >= test_start)
