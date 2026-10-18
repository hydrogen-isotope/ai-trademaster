"""
5-day forecast v5 (research only; CPU only). Rules fixed before any result: docs/FORECAST-V5-OPTIONS-PROTOCOL.md
(incl. Amendment B). v5 = v4 inputs + options-market features + earnings-event features, same network and ranges.

Usage:
  python -m trademaster_ml.daily_v5 run       # development (2024-04-01 .. 2025-10-10) + final test vs v3, all reports
"""
import argparse, json, math, os, re, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from . import daily_v2 as v2
from . import daily_v4 as v4
from .daily_forecast import H, OUT

DEV_START, DEV_END = "2024-04-01", "2025-10-10"
WD = 0.01                                                     # the v4 choice, fixed by the protocol
OPT_FEATS = ["iv_atm", "iv_rv", "iv_chg5", "skew", "pc_vol", "pc_vol_z"]
EARN_FEATS = ["earn_day", "earn_age", "earn_react", "earn_sent", "earn_drift"]
EARN = re.compile(r"\bq[1-4]\b.*\b(earnings|results)\b|\breports q[1-4]\b|earnings beat|earnings miss|beats estimates|misses estimates|\beps of\b|quarterly results", re.I)
PREVIEW = re.compile(r"ahead of|likely to|preview|expect|what to watch|\bcould\b|will report|set to report", re.I)


def options_raw(dates, syms, vol20):
    z = np.load(os.path.join("private", "data", "options", "features.npz"))
    assert list(z["dates"]) == list(dates) and list(z["syms"]) == list(syms), "options features out of date: rerun options features"
    iv, sk, pc = z["iv_atm"], z["skew"], z["pc_vol"]
    rv = vol20 * math.sqrt(252)
    ivc = np.full_like(iv, np.nan); ivc[5:] = iv[5:] - iv[:-5]
    pcm = np.full_like(pc, np.nan)
    for t in range(len(pc)):
        w = pc[max(0, t - 59):t + 1]
        if np.isfinite(w).sum() >= 10:
            pcm[t] = np.nanmean(w, axis=0)
    return {"iv_atm": iv, "iv_rv": iv - rv, "iv_chg5": ivc, "skew": sk, "pc_vol": pc, "pc_vol_z": pc - pcm}


def earnings_raw(dates, syms, is_stock, r1, ref, vol20):
    T, N = len(dates), len(syms)
    idx = {d: i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(syms)}
    score = {}
    for line in open(os.path.join(v4.NEWS, "scores.jsonl"), encoding="utf-8"):
        r = json.loads(line); score[r["id"]] = r["pos"] - r["neg"]
    ecount, esum = np.zeros((T, N)), np.zeros((T, N))
    first = datetime.fromisoformat(dates[0]).date()
    for line in open(os.path.join(v4.NEWS, "items.jsonl"), encoding="utf-8"):
        n = json.loads(line)
        if not EARN.search(n["headline"]) or PREVIEW.search(n["headline"]) or len(n.get("symbols", [])) > 3:
            continue                                         # amendment B (previews) and C (broad multi-ticker items)
        ts = datetime.fromisoformat(n["t"].replace("Z", "+00:00")).astimezone(v4.NY)
        d = ts.date() + (timedelta(days=1) if ts.hour >= 16 else timedelta(0))
        if d < first:
            continue
        for _ in range(10):
            if str(d) in idx:
                break
            d += timedelta(days=1)
        t = idx.get(str(d))
        if t is None:
            continue
        for s in n.get("symbols", []):
            j = col.get(s)
            if j is not None and is_stock[j]:
                ecount[t, j] += 1; esum[t, j] += score.get(n["id"], 0.0)
    day = (ecount > 0).astype(float)
    rel = np.nan_to_num(r1 - r1[:, ref]) / np.where(vol20 > 0, vol20, np.nan)
    age, react, sent = np.full((T, N), 60.0), np.zeros((T, N)), np.zeros((T, N))
    for j in range(N):
        last_t = None
        for t in range(T):
            if day[t, j]:
                last_t = t
                react[t, j] = np.nan_to_num(rel[t, j]); sent[t, j] = esum[t, j] / ecount[t, j]
            elif last_t is not None:
                react[t, j] = react[last_t, j]; sent[t, j] = sent[last_t, j]
            if last_t is not None:
                age[t, j] = min(60, t - last_t)
    drift = react * np.maximum(0, 1 - age / 20)
    return {"earn_day": day, "earn_age": age, "earn_react": react, "earn_sent": sent, "earn_drift": drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    t_start = time.time()
    dates, syms, is_stock, C, V, ref, spy = v2.load_panel()
    L, r1, vol20, F4, stress, raw = v4.build(dates, syms, is_stock, C, V, ref, spy)
    o = options_raw(dates, syms, vol20)
    e = earnings_raw(dates, syms, is_stock, r1, ref, vol20)
    new = np.stack([np.nan_to_num(v2.rank_signed(o[k], is_stock), nan=0.0) for k in OPT_FEATS] +
                   [np.nan_to_num(v2.rank_signed(e[k], is_stock), nan=0.0) for k in EARN_FEATS], axis=2)
    new[:, ~is_stock, :] = np.nan
    F = np.concatenate([F4, new], axis=2)
    names = v4.STOCK_FEATS + v4.MKT_FEATS + OPT_FEATS + EARN_FEATS
    Y = v2.rel_targets(L, vol20, is_stock)
    T = len(dates)
    first_dev = next(i for i, d in enumerate(dates) if d >= DEV_START)
    dev_last = max(i for i, d in enumerate(dates) if d <= DEV_END)
    test_last = T - 1 - H
    test_first = test_last - v2.TEST_N + 1
    cov = np.isfinite(o["iv_atm"][test_first:test_last + 1][:, is_stock]).mean()
    print(f"features: {F.shape[2]} (v4 45 + options {len(OPT_FEATS)} + earnings {len(EARN_FEATS)}); "
          f"options coverage in test {cov * 100:.0f}% of stock-days; earnings days {int(e['earn_day'].sum())}", flush=True)

    P = v4.walk_forward(F, Y, first_dev, test_last, WD)
    mm = {"dev": v4.regime_metrics(P, Y, first_dev, dev_last, is_stock, stress), "test": v4.regime_metrics(P, Y, test_first, test_last, is_stock, stress)}
    for per in ("dev", "test"):
        for reg in ("all", "stress", "calm"):
            m = mm[per][reg]
            if "ic" in m:
                print(f"{per:4s} {reg:6s}: IC {m['ic']:+.4f} (t {m['ic_t']:+.2f})  direction hit {m['hit'] * 100:.2f}% (p={m['hit_p_one_sided']:.3f})  windows {m['windows']}", flush=True)
    results, recs_all = {}, {}
    par = v2.eval_symbols_parallel(v2.TARGETS, {x: syms.index(x) for x in v2.TARGETS}, L, r1, vol20, P, spy, first_dev, test_last, dev_last,
                                   ("v3sym", "v4sym"))
    for sym in v2.TARGETS:
        recs, _ = par[sym]
        results[sym] = {v: {"dev": v2.summarize(recs, first_dev, dev_last, v), "test": v2.summarize(recs, test_first, test_last, v)} for v in ("v3sym", "v4sym")}
        recs_all[sym] = recs
    pooled = lambda v, per: float(np.mean([np.mean([x["crps"] for x in results[s][v][per]]) for s in v2.TARGETS]))
    adopt = mm["dev"]["all"]["ic_t"] > 2 and pooled("v4sym", "dev") < pooled("v3sym", "dev")
    per_origin = defaultdict(lambda: {"v3sym": [], "v4sym": []})
    for s in v2.TARGETS:
        for r in recs_all[s]:
            if test_first <= r["t"] <= test_last:
                for v in ("v3sym", "v4sym"):
                    per_origin[r["t"]][v].append(np.mean([x["crps"] for x in r["v"][v] if x]))
    ts_ = sorted(per_origin)
    diff = v4.block_bootstrap_diff([np.mean(per_origin[t]["v4sym"]) for t in ts_], [np.mean(per_origin[t]["v3sym"]) for t in ts_])
    chosen = "v4sym" if adopt else "v3sym"
    base = float(np.mean([np.mean([x["crps_base"] for x in results[s][chosen]["test"]]) for s in v2.TARGETS]))
    skill = 1 - pooled(chosen, "test") / base
    # extra reports (not used for adoption): options-only, earnings-only, and earnings-only over the full history
    extra = {}
    for g, keep in (("options only", OPT_FEATS), ("earnings only", EARN_FEATS)):
        Fg = F.copy()
        drop = [i for i, n in enumerate(names) if n not in keep]
        Fg[:, :, drop] = np.where(np.isfinite(Fg[:, :, drop]), 0.0, np.nan)
        print(f"  extra check: {g} ...", flush=True)
        Pg = v4.walk_forward(Fg, Y, first_dev, test_last, WD, log=print)
        extra[g] = {"dev": v4.regime_metrics(Pg, Y, first_dev, dev_last, is_stock, stress)["all"], "test": v4.regime_metrics(Pg, Y, test_first, test_last, is_stock, stress)["all"]}
        if g == "earnings only":
            f17 = next(i for i, d in enumerate(dates) if d >= "2017-01-03")
            print("  extra check: earnings only over 2017-2026 ...", flush=True)
            Pf = v4.walk_forward(Fg, Y, f17, test_last, WD, log=print)
            extra["earnings only, 2017-2026"] = v4.regime_metrics(Pf, Y, f17, test_last, is_stock, stress)["all"]
    rules = {"calibration": {s: v2.passes_calibration(results[s][chosen]["test"]) for s in v2.TARGETS}, "crps_skill_vs_random_walk": float(skill),
             "crps_rule_pass": bool(skill > 0), "direction": mm["test"]["all"],
             "direction_rule_pass": bool(mm["test"]["all"].get("hit", 0) > 0.5 and mm["test"]["all"].get("hit_p_one_sided", 1) < 0.05)}
    doc = {"protocol": "docs/FORECAST-V5-OPTIONS-PROTOCOL.md", "adopted": bool(adopt), "chosen_variant": "v5" if adopt else "v3sym",
           "dev_pooled_crps": {"v3": pooled("v3sym", "dev"), "v5": pooled("v4sym", "dev")}, "test_pooled_crps": {"v3": pooled("v3sym", "test"), "v5": pooled("v4sym", "test")},
           "test_crps_diff_v5_minus_v3": {"mean": diff[0], "ci95": [diff[1], diff[2]]}, "mean_model": mm, "extra": extra, "rules": rules,
           "results": results, "runtime_min": (time.time() - t_start) / 60, "created_at": datetime.now(timezone.utc).isoformat()}
    json.dump(doc, open(os.path.join(OUT, "v5_results.json"), "w"), indent=1)
    print(f"\ndev pooled CRPS x1e4: v3 {doc['dev_pooled_crps']['v3'] * 1e4:.3f}  v5 {doc['dev_pooled_crps']['v5'] * 1e4:.3f}")
    print(f"ADOPTED v5: {adopt}")
    print(f"test pooled CRPS x1e4: v3 {doc['test_pooled_crps']['v3'] * 1e4:.3f}  v5 {doc['test_pooled_crps']['v5'] * 1e4:.3f}  (v5-v3 {diff[0] * 1e4:+.3f}, 95% CI {diff[1] * 1e4:+.3f}..{diff[2] * 1e4:+.3f})")
    for g, m in extra.items():
        mm_ = m if "ic" in m else m.get("test", {})
        print(f"{g:28s}: " + (f"dev IC {m['dev']['ic']:+.4f} (t {m['dev']['ic_t']:+.1f})  test IC {m['test']['ic']:+.4f} (t {m['test']['ic_t']:+.1f}) hit {m['test']['hit'] * 100:.1f}%"
                               if "dev" in m else f"IC {m['ic']:+.4f} (t {m['ic_t']:+.1f}) hit {m['hit'] * 100:.1f}%"), flush=True)
    for s in v2.TARGETS:
        t_ = results[s]["v4sym"]["test"]
        print(f"{s:5s} v5 TEST 80%: " + " ".join(f"{x['coverage']['in80']:.2f}" for x in t_) + f"  dir {np.mean([x['dir_hit'] for x in t_]) * 100:.1f}%")
    t_as = dates.index("2026-10-09")
    for s in v2.TARGETS:
        r = next(x for x in recs_all[s] if x["t"] == t_as)
        print(f"  hold-out 12-16 Oct {s}: v3 {sum(1 for x in r['v']['v3sym'] if x and x['in80'])}/5  v5 {sum(1 for x in r['v']['v4sym'] if x and x['in80'])}/5")
    print("RULES:", json.dumps(rules["calibration"]), f"crps skill {skill * 100:+.2f}% pass={rules['crps_rule_pass']} direction pass={rules['direction_rule_pass']} | {(time.time() - t_start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
