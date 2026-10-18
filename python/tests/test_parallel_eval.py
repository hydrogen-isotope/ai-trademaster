"""
Checks that the parallel range simulation gives exactly the same results as running one stock at a time, and reports
the speedup. Needs the private daily data. Run: python python/tests/test_parallel_eval.py
"""
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from trademaster_ml import daily_v2 as v2  # noqa: E402


def main():
    dates, syms, is_stock, C, V, ref, spy = v2.load_panel()
    L, r1, vol20, F = v2.build(C, V, is_stock, ref, spy)
    T = len(dates)
    last = T - 1 - v2.H
    first = last - 39
    P = {h: np.full(L.shape, np.nan) for h in range(1, v2.H + 1)}
    idx = {s: syms.index(s) for s in v2.TARGETS}
    t0 = time.time()
    seq = {s: v2.eval_symbol(s, idx[s], L, r1, vol20, P, spy, first, last, last - 20, log=None, variants=("v3sym",)) for s in v2.TARGETS}
    t_seq = time.time() - t0
    t0 = time.time()
    par = v2.eval_symbols_parallel(v2.TARGETS, idx, L, r1, vol20, P, spy, first, last, last - 20, ("v3sym",))
    t_par = time.time() - t0
    for s in v2.TARGETS:
        a, b = seq[s][0], par[s][0]
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            for h in range(v2.H):
                xa, xb = ra["v"]["v3sym"][h], rb["v"]["v3sym"][h]
                assert (xa is None) == (xb is None)
                if xa:
                    assert xa["crps"] == xb["crps"] and xa["in80"] == xb["in80"], (s, ra["t"], h)
    print(f"identical results for {len(v2.TARGETS)} stocks x 40 origins; one-at-a-time {t_seq:.0f}s, parallel {t_par:.0f}s "
          f"({t_seq / t_par:.1f}x faster)")


if __name__ == "__main__":
    main()
