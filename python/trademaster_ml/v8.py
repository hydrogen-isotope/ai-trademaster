"""
v8: specialised stage models + a combiner, under docs/FORECAST-V8-STAGES-PROTOCOL.md (research only, CPU).

  S2 price, S3 fundamentals, S4 industry: one ridge per feature family (walk-forward, monthly refits)
  S1 news (LLM; see llm_news.py): per stock-day news features, used from 2024-07 only
  S6 combiner: ridge on the cross-sectional ranks of the stages' out-of-sample scores

  testA   staging vs the single v6 ridge (development 2021-24, and the seen 2025-26 period)
  testB   does LLM news add to the ranking (2025-01 onward, combiner fitted on 2024-07+ stage outputs)
  testC   do the AI-anxiety / market-news indices predict QQQ's next 5 days

Usage: python -m trademaster_ml.v8 testA | testB | testC
"""
import argparse, json, os, time

import numpy as np

from . import v6

OUT = os.path.join("private", "reports", "v8")
OOS_START = "2019-01-02"            # stage scores out of sample from here, so the combiner has history by 2021
COMB_ALPHA = 1.0


def families(names):
    ind = [n for n in names if n.startswith("ind_") or n == "mom21_minus_ind"]
    price = [n for n in v6.__dict__.get("PRICE", [])] or [n for n in names if n in (
        "mom5", "mom21", "mom63", "mom126", "mom12_1", "vol20", "vol60", "dist_252_high", "volume_surprise",
        "log_dollar_vol20", "beta252", "resid_mom12_1", "max_ret21", "idio_vol60")]
    fund = [n for n in names if n not in price and n not in ind]
    return {"S2 price": price, "S3 fundamentals": fund, "S4 industry": ind}


def stage_scores(P, d0, d1, extra=None, log=print):
    """Out-of-sample walk-forward score of each stage for days d0..d1 (each stage = ridge on its own family)."""
    fi = {n: i for i, n in enumerate(P.names)}
    out = {}
    for name, cols in families(P.names).items():
        idx = [fi[c] for c in cols]
        if log:
            log(f"  {name}: {len(idx)} features", flush=True)
        out[name] = v6.walk_forward(P, d0, d1, v6.ridge(), X=np.ascontiguousarray(P.X[:, idx]), log=None)
    for name, X in (extra or {}).items():
        out[name] = v6.walk_forward(P, d0, d1, v6.ridge(), X=X, log=None)
    return out


def combine(P, stages, c0, d0, d1, alpha=COMB_ALPHA):
    """S6: monthly walk-forward ridge on the cross-sectional ranks of the stage scores. At a refit on day r it is fitted
    on rows from day c0 whose 5-day target ended before r; stage scores there are themselves out of sample."""
    names = list(stages)
    Z = np.stack([v6.xs_rank_pred(P, stages[n]) for n in names], 1)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    pred = np.full(len(P.t), np.nan)
    weights = []
    refits = v6.month_starts(P, d0, d1)
    for k, r in enumerate(refits):
        nxt = refits[k + 1] if k + 1 < len(refits) else d1 + 1
        tr = slice(P.day_start[c0], P.day_start[max(r - v6.H, c0)])
        ok = np.isfinite(P.yrank[tr]) & np.isfinite(stages[names[0]][tr])
        A = Z[tr][ok]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(len(names)), A.T @ P.yrank[tr][ok])
        te = slice(P.day_start[r], P.day_start[nxt])
        pred[te] = Z[te] @ w
        weights.append([P.dates[r]] + [float(x) for x in w])
    return pred, names, weights


def show(res):
    for x in res:
        print(f"  {x['model']:<34} IC {x['ic']:+.4f} (t {x['ic_t_nw']:+.1f})  top>bottom {x['top_beats_bottom']:.0%}  "
              f"L/S Sharpe {x['ls_sharpe']:+.2f} [10% borrow {x['ls_sharpe_borrow10']:+.2f}]  long-only vs QQQ {x['longonly_vs_qqq_ann']:+.1%}/yr",
              flush=True)


def testA(args):
    t0 = time.time()
    P = v6.Panel()
    o0 = P.idx(OOS_START)
    dv0, dv1 = v6.period_bounds(P, "dev")
    ts0, ts1 = v6.period_bounds(P, "test")
    print(f"stage walk-forward {P.dates[o0]} .. {P.dates[ts1]}", flush=True)
    st = stage_scores(P, o0, ts1)
    r_single = v6.walk_forward(P, dv0, ts1, v6.ridge(), log=None)
    comb, names, w = combine(P, st, o0, dv0, ts1)
    out = {"periods": {}, "combiner_weights": {"stages": names, "by_month": w}}
    for label, (a, b) in (("development 2021-24", (dv0, dv1)), ("seen 2025-26", (ts0, ts1))):
        res = [v6.evaluate(P, r_single, a, b, "v6 R: one ridge, all features")]
        res += [v6.evaluate(P, st[n], a, b, n + " alone") for n in names]
        res.append(v6.evaluate(P, comb, a, b, "S6 combiner of S2-S4"))
        print(label, flush=True)
        show(res)
        out["periods"][label] = res
    dev = out["periods"]["development 2021-24"]
    seen = out["periods"]["seen 2025-26"]
    out["pass"] = bool(dev[-1]["ic"] > dev[0]["ic"] and seen[-1]["ic"] >= seen[0]["ic"] - 0.005)
    print(f"combiner weights at the last refit: " + ", ".join(f"{n} {x:+.3f}" for n, x in zip(names, w[-1][1:])), flush=True)
    print(f"test A: {'PASS' if out['pass'] else 'FAIL'} (combiner dev IC {dev[-1]['ic']:+.4f} vs single ridge {dev[0]['ic']:+.4f}; "
          f"seen {seen[-1]['ic']:+.4f} vs {seen[0]['ic']:+.4f})  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "testA.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["testA", "testB", "testC"])
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
