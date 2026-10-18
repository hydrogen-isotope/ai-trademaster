"""
Full US equity history for v6 (research only; read-only market data).

  assets  list every US equity on the main exchanges from Alpaca, active AND delisted (avoids survivorship bias)
  bars    10+ years of daily bars (SIP, split-adjusted) for all of them, in resumable chunks
  panel   assemble one float32 panel: private/data/us/panel.npz  (dates x symbols: open, high, low, close, volume)
  update  stream only new days (and re-fetch stocks that split since the last run)
  export  raw float32 [T, N] files for the HLSL feature job (trademaster gpu-features)

Usage: python -m trademaster_ml.universe_us assets | bars | panel | update | export
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


def _day(t):
    return datetime.fromtimestamp(int(t), NY).strftime("%Y-%m-%d")


def update():
    """Stream only what is new since the panel's last day: the latest bars for active stocks, the full history of
    newly listed ones, and a full re-fetch of any stock whose recent closes no longer match (a split since last run)."""
    path = os.path.join(DIR, "panel.npz")
    p = dict(np.load(path))
    dates, syms = [str(d) for d in p["dates"]], [str(s) for s in p["symbols"]]
    assets()                                                        # refresh listings / delistings
    meta = {a["symbol"]: a for a in json.load(open(os.path.join(DIR, "assets.json")))}
    active = sorted(s for s, a in meta.items() if a["status"] == "active")
    new = [s for s in active if s not in set(syms)]
    known = [s for s in active if s in set(syms)]
    creds = alpaca.credentials()
    end = int(time.time()) - 16 * 60
    look = int(datetime.fromisoformat(dates[-1] + "T00:00:00+00:00").timestamp()) - 10 * 86400
    recent = alpaca.get_bars(known, "1Day", look, end, feed="sip", creds=creds, pause=0.31)
    si, di = {s: i for i, s in enumerate(syms)}, {d: i for i, d in enumerate(dates)}
    resplit = []
    for s, rows in recent.items():
        j = si[s]
        old = [(di[_day(r[0])], r[4]) for r in rows if _day(r[0]) in di and np.isfinite(p["close"][di[_day(r[0])], j])]
        if any(abs(c / p["close"][t, j] - 1) > 0.01 for t, c in old):
            resplit.append(s)
    full_start = int(datetime.fromisoformat(START.replace("Z", "+00:00")).timestamp())
    full = alpaca.get_bars(new + resplit, "1Day", full_start, end, feed="sip", creds=creds, pause=0.31) if new + resplit else {}
    rows_by_sym = {**{s: r for s, r in recent.items() if s not in resplit}, **full}
    add_days = sorted({_day(r[0]) for rows in rows_by_sym.values() for r in rows} - set(dates))
    dates2, syms2 = dates + add_days, syms + new
    T2, N2 = len(dates2), len(syms2)
    for k in ("open", "high", "low", "close", "volume"):
        a = np.full((T2, N2), np.nan, np.float32)
        a[:len(dates), :len(syms)] = p[k]
        p[k] = a
    di, si = {d: i for i, d in enumerate(dates2)}, {s: i for i, s in enumerate(syms2)}
    for s in resplit:
        for k in ("open", "high", "low", "close", "volume"):
            p[k][:, si[s]] = np.nan
    for s, rows in rows_by_sym.items():
        for t, o, h, l, c, v in rows:
            i, j = di[_day(t)], si[s]
            for k, x in (("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)):
                p[k][i, j] = x
    p["dates"], p["symbols"] = np.array(dates2), np.array(syms2)
    p["exchange"] = np.array([meta.get(s, {}).get("exchange", "") for s in syms2])
    p["active"] = np.array([meta.get(s, {}).get("status") == "active" for s in syms2])
    tmp = path + ".tmp.npz"
    np.savez(tmp, **p)
    os.replace(tmp, path)
    print(f"update: +{len(add_days)} day(s) to {dates2[-1]}, {len(new)} new listing(s), {len(resplit)} re-fetched after splits "
          f"({', '.join(resplit[:10])}{'...' if len(resplit) > 10 else ''})", flush=True)


# Names that are not operating companies: ETFs/ETNs and funds, leveraged products, warrants, rights, units, notes,
# preferreds and SPAC shells ("Acquisition Corp"). They never enter the v6 universe.
NOT_COMPANY = re.compile(r"\b(ETFs?|ETNs?|Fund|iShares|ProShares|Direxion|GraniteShares|Invesco|Leverage Shares|Daily|2X|3X|"
                         r"Warrants?|Rights?|Units?|Preferred|Notes?|Bonds?|Index|Acquisition|Capital Trust|Series \d+|Trust Series|"
                         r"Depositary Units?|Debentures?|Subordinated|Perpetual|Contingent Value)\b", re.I)
MARKET = "QQQ"


def export():
    """Raw float32 files for the GPU feature job (trademaster gpu-features --in private/data/us/gpu):
    logclose/volume [T, N], market [T] (QQQ log close), eligible [N] (operating companies), and, when
    fundamentals.npz exists, industry [N] (2-digit SIC) plus its point-in-time fields as extra_<name> [T, N]."""
    p = np.load(os.path.join(DIR, "panel.npz"))
    out = os.path.join(DIR, "gpu")
    os.makedirs(out, exist_ok=True)
    syms = [str(s) for s in p["symbols"]]
    dates = [str(d) for d in p["dates"]]
    c = p["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.where(c > 0, np.log(c), np.nan).astype(np.float32)
    T, N = c.shape
    if MARKET not in syms:
        raise SystemExit(f"{MARKET} is not in the panel")
    market = lc[:, syms.index(MARKET)]
    names = {a["symbol"]: a.get("name") or "" for a in json.load(open(os.path.join(DIR, "assets.json")))}
    eligible = np.array([0.0 if NOT_COMPANY.search(names.get(s, "")) else 1.0 for s in syms], np.float32)
    meta = {"T": T, "N": N, "dates": dates, "symbols": syms, "market": MARKET, "extras": []}
    for f in ("logclose.f32", "volume.f32", "market.f32", "eligible.f32", "industry.f32"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    np.ascontiguousarray(lc).tofile(os.path.join(out, "logclose.f32"))
    np.ascontiguousarray(p["volume"].astype(np.float32)).tofile(os.path.join(out, "volume.f32"))
    np.ascontiguousarray(market.astype(np.float32)).tofile(os.path.join(out, "market.f32"))
    eligible.tofile(os.path.join(out, "eligible.f32"))
    fpath = os.path.join(DIR, "fundamentals.npz")
    if os.path.exists(fpath):
        fz = np.load(fpath)
        fsyms = [str(s) for s in fz["symbols"]] if "symbols" in fz.files else syms
        if fsyms != syms or ("dates" in fz.files and [str(d) for d in fz["dates"]] != dates):
            print("fundamentals.npz is not aligned with the current panel: rebuild it (fundamentals panel); skipping it", flush=True)
        else:
            if "sic2" in fz.files:
                np.ascontiguousarray(fz["sic2"].astype(np.float32)).tofile(os.path.join(out, "industry.f32"))
            for k in fz.files:
                if fz[k].shape == (T, N):
                    np.ascontiguousarray(fz[k].astype(np.float32)).tofile(os.path.join(out, f"extra_{k}.f32"))
                    meta["extras"].append(k)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print(f"exported {T} x {N} to {out}: {int(eligible.sum())} operating companies, market {MARKET}, "
          f"industries {'yes' if os.path.exists(os.path.join(out, 'industry.f32')) else 'no'}, extras {meta['extras']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assets", "bars", "panel", "update", "export"])
    a = ap.parse_args()
    {"assets": assets, "bars": bars, "panel": panel, "update": update, "export": export}[a.cmd]()


if __name__ == "__main__":
    main()
