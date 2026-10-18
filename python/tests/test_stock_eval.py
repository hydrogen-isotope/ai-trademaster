"""
Checks for trademaster_ml.stock_eval (v8 amendment C mapping, parsing, cone, queue, shadow log; no model, no network).
Run: python python/tests/test_stock_eval.py
"""
import json, math, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trademaster_ml import stock_eval as se  # noqa: E402
from trademaster_ml.daily_v2 import GRID  # noqa: E402

close = lambda a, b, tol=1e-9: math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_advise_mapping_follows_amendment_c():
    m, sd = 0.01, 0.05
    assert se.advise(m, sd, {"lean": "neutral", "confidence": "high", "risk": "normal"}) == (m, sd, 0.0)
    m2, sd2, k = se.advise(m, sd, {"lean": "strong_up", "confidence": "high", "risk": "elevated"})
    assert close(k, 0.375) and close(m2, m + 0.375 * sd) and close(sd2, sd * 1.25)
    m2, _, k = se.advise(m, sd, {"lean": "down", "confidence": "low", "risk": "normal"})
    assert close(k, -0.06) and m2 < m


def test_p_up_moves_with_lean_but_stays_bounded():
    R = GRID.copy()
    assert abs(se.dist(100.0, 0.0, 0.05, R)["p_up"] - 0.5) < 0.01
    hi = se.dist(100.0, *se.advise(0.0, 0.05, {"lean": "strong_up", "confidence": "high", "risk": "normal"})[:2], R)
    lo = se.dist(100.0, *se.advise(0.0, 0.05, {"lean": "strong_down", "confidence": "high", "risk": "normal"})[:2], R)
    assert 0.6 < hi["p_up"] < 0.7 and 0.3 < lo["p_up"] < 0.4
    assert lo["lo95"] < lo["lo80"] < lo["median"] < lo["hi80"] < lo["hi95"]


def test_unparsable_or_off_schema_reply_is_neutral():
    for bad in ["", "not json", json.dumps({"lean": "very_up", "confidence": "high", "risk": "normal"}), json.dumps({"lean": "up"})]:
        v, ok = se.parse_verdict(bad)
        assert not ok and (v["lean"], v["confidence"], v["risk"]) == ("neutral", "low", "normal")
    v, ok = se.parse_verdict(json.dumps({"positives": ["a"], "negatives": [], "lean": "up", "confidence": "medium", "risk": "elevated"}))
    assert ok and v["lean"] == "up" and v["positives"] == ["a"]


def test_cone_starts_at_close_and_only_goes_forward_on_weekdays():
    R = GRID.copy()
    pts = se.cone(50.0, "2026-10-16", 0.01, 0.04, R)                   # a Friday
    assert pts[0]["date"] == "2026-10-16" and pts[0]["median"] == 50.0
    assert [p["date"] for p in pts[1:]] == ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
    d = se.dist(50.0, 0.01, 0.04, R)
    assert all(close(pts[-1][k], d[k], 1e-9) for k in ("median", "lo80", "hi80", "lo95", "hi95"))


def test_news_cutoff_is_next_open_after_the_close():
    assert se.next_open_utc("2026-10-16").strftime("%Y-%m-%d %H:%M") == "2026-10-19 13:30"   # Mon 09:30 New York (EDT)


def test_queue_order_and_symbols_with_dots():
    with tempfile.TemporaryDirectory() as tmp:
        se.QUEUE = os.path.join(tmp, "queue")
        se.enqueue("nasdaq", "AAPL", 1)
        se.enqueue("sgx", "D05.SI", 0)
        se.enqueue("sgx", "D05.SI", 0)                                 # once only
        assert [(p, m, s) for _, p, m, s in se.queued()] == [(0, "sgx", "D05.SI"), (1, "nasdaq", "AAPL")]


def test_shadow_log_written_once_per_stock_and_close():
    with tempfile.TemporaryDirectory() as tmp:
        se.EVALS = tmp
        os.makedirs(os.path.join(tmp, "nasdaq"))
        ctx = {"asof": "2026-10-16", "symbol": "MU", "close": 100.0, "m": 0.01, "sd": 0.05}
        v = {"lean": "down", "confidence": "medium", "risk": "normal"}
        se._shadow("nasdaq", ctx, 0.004, 0.05, v, True)
        se._shadow("nasdaq", ctx, 0.004, 0.05, v, True)
        lines = open(os.path.join(tmp, "nasdaq", "advice_shadow.jsonl")).read().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["m_adv"] == 0.004


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
