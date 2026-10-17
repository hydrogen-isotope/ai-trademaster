"""
Compact chart data for today's session from <dir>/<SYM>-eval.json files (written by eval_bars.py).
Prints one JSON object: per symbol, 5-minute-sampled labels / predicted / actual (bps), and summary stats.
Usage: py -I tools/chart_data.py data/live MU NVDA AMD AAPL
"""
import json, os, sys
from datetime import datetime, timezone, timedelta

NY = timezone(timedelta(hours=-4))  # EDT (valid until 2026-11-01)


def main():
    d, syms = sys.argv[1], sys.argv[2:]
    out = {}
    for s in syms:
        with open(os.path.join(d, f"{s}-eval.json"), encoding="utf-8") as f:
            e = json.load(f)
        rows = e["rows"]
        today = max(datetime.fromtimestamp(r["t"], NY).date() for r in rows)
        tr = [r for r in rows if datetime.fromtimestamp(r["t"], NY).date() == today]
        tr = [r for r in tr if datetime.fromtimestamp(r["t"], NY).minute % 5 == 0]
        sm = e["summary"]
        out[s] = {
            "ref": sm["reference"],
            "last": round(sm["last_price"], 2),
            "lastTime": sm["last"][11:16],
            "all": {k: round(v, 3) if isinstance(v, float) else v for k, v in sm["all_days"].items()},
            "today": {k: round(v, 3) if isinstance(v, float) else v for k, v in sm["today"].items()},
            "L": [datetime.fromtimestamp(r["t"], NY).strftime("%H:%M") for r in tr],
            "P": [round(r["mu"] * 1e4, 1) for r in tr],
            "A": [None if r["actual"] is None else round(r["actual"] * 1e4, 1) for r in tr],
        }
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
