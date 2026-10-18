"""
Full US equity history for v6 (research only; read-only market data).

  assets  list every US equity on the main exchanges from Alpaca, active AND delisted (avoids survivorship bias)
  bars    10+ years of daily bars (SIP, split-adjusted) for all of them, in resumable chunks
  panel   assemble one float32 panel: private/data/us/panel.npz  (dates x symbols: open, high, low, close, volume)
  export  raw float32 [T, N] files for the HLSL feature job (trademaster gpu-features)

Usage: python -m trademaster_ml.universe_us assets | bars | panel | export
"""
import argparse, json, os, re, time, urllib.parse, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from . import alpaca

DIR = os.path.join("private", "data", "us")
CHUNKS = os.path.join(DIR, "chunks")
NY = ZoneInfo("America/New_York")
EXCHANGES = {"NASDAQ"}                                  # NASDAQ-listed only, for now
START = "2016-01-01T00:00:00Z"


def assets():
    kid, sec, _ = alpaca.credentials()
    out = []
    for status in ("active", "inactive"):
        url = "https://paper-api.alpaca.markets/v2/assets?" + urllib.parse.urlencode({"status": status, "asset_class": "us_equity"})
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})
        for a in json.load(urllib.request.urlopen(req, timeout=180)):
            if a["exchange"] in EXCHANGES and re.fullmatch(r"[A-Z]{1,5}", a["symbol"]):
                out.append({k: a.get(k) for k in ("symbol", "name", "exchange", "status", "shortable", "easy_to_borrow", "tradable")})
    os.makedirs(DIR, exist_ok=True)
    json.dump(out, open(os.path.join(DIR, "assets.json"), "w"), indent=0)
    print(f"{len(out)} assets ({sum(a['status'] == 'active' for a in out)} active, {sum(a['status'] != 'active' for a in out)} delisted)", flush=True)


def bars(chunk_size=200):
    os.makedirs(CHUNKS, exist_ok=True)
    syms = sorted({a["symbol"] for a in json.load(open(os.path.join(DIR, "assets.json")))})
    creds = alpaca.credentials()
    groups = [syms[i:i + chunk_size] for i in range(0, len(syms), chunk_size)]
    end = int(time.time()) - 16 * 60
    t0 = time.time()
    for g, group in enumerate(groups):
        path = os.path.join(CHUNKS, f"c{g:04d}.npz")
        if os.path.exists(path):
            continue
        data = alpaca.get_bars(group, "1Day", int(datetime.fromisoformat(START.replace("Z", "+00:00")).timestamp()), end, feed="sip", pause=0.31)
        s_list, t_list, o, h, l, c, v = [], [], [], [], [], [], []
        for s, rows in data.items():
            for t, oo, hh, ll, cc, vv in rows:
                s_list.append(s); t_list.append(t); o.append(oo); h.append(hh); l.append(ll); c.append(cc); v.append(vv)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, sym=np.array(s_list), t=np.array(t_list, np.int64), o=np.array(o, np.float32), h=np.array(h, np.float32),
                            l=np.array(l, np.float32), c=np.array(c, np.float32), v=np.array(v, np.float64))
        os.replace(tmp, path)
        el = (time.time() - t0) / 60
        print(f"chunk {g + 1}/{len(groups)}: {len(set(s_list))} symbols, {len(s_list):,} bars, {el:.1f} min elapsed, "
              f"~{el / (g + 1) * (len(groups) - g - 1):.0f} min left", flush=True)
    print("bars complete", flush=True)


def panel():
    files = sorted(f for f in os.listdir(CHUNKS) if f.endswith(".npz") and ".tmp" not in f)
    parts = [np.load(os.path.join(CHUNKS, f)) for f in files]
    days = sorted({datetime.fromtimestamp(int(t), NY).strftime("%Y-%m-%d") for p in parts for t in np.unique(p["t"])})
    di = {d: i for i, d in enumerate(days)}
    syms = sorted({s for p in parts for s in np.unique(p["sym"])})
    si = {s: i for i, s in enumerate(syms)}
    T, N = len(days), len(syms)
    arr = {k: np.full((T, N), np.nan, np.float32) for k in ("open", "high", "low", "close", "volume")}
    for p in parts:
        ti = np.array([di[datetime.fromtimestamp(int(t), NY).strftime("%Y-%m-%d")] for t in p["t"]])
        sj = np.array([si[s] for s in p["sym"]])
        for k, src in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
            arr[k][ti, sj] = p[src]
    meta = {a["symbol"]: a for a in json.load(open(os.path.join(DIR, "assets.json")))}
    np.savez(os.path.join(DIR, "panel.npz"), dates=np.array(days), symbols=np.array(syms),
             exchange=np.array([meta.get(s, {}).get("exchange", "") for s in syms]),
             active=np.array([meta.get(s, {}).get("status") == "active" for s in syms]), **arr)
    print(f"panel: {T} days x {N} symbols ({days[0]} .. {days[-1]}), {sum(np.isfinite(arr['close']).sum(0) > 0)} with data", flush=True)


def export():
    """Raw float32 [T, N] files for the GPU feature job (trademaster gpu-features --in private/data/us/gpu)."""
    p = np.load(os.path.join(DIR, "panel.npz"))
    out = os.path.join(DIR, "gpu")
    os.makedirs(out, exist_ok=True)
    c = p["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.where(c > 0, np.log(c), np.nan).astype(np.float32)
    np.ascontiguousarray(lc).tofile(os.path.join(out, "logclose.f32"))
    np.ascontiguousarray(p["volume"].astype(np.float32)).tofile(os.path.join(out, "volume.f32"))
    T, N = c.shape
    json.dump({"T": T, "N": N, "dates": [str(d) for d in p["dates"]], "symbols": [str(s) for s in p["symbols"]]},
              open(os.path.join(out, "meta.json"), "w"))
    print(f"exported {T} x {N} to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assets", "bars", "panel", "export"])
    a = ap.parse_args()
    {"assets": assets, "bars": bars, "panel": panel, "export": export}[a.cmd]()


if __name__ == "__main__":
    main()
