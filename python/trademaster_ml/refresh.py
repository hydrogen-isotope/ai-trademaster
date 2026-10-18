"""
Bring the v6 rankings and v7 price forecasts up to date (research only, CPU only; the GPU is never used here).

Runs in the background when the app opens, and from the daily job. It streams only what is new and stops early when
the rankings already cover the latest completed trading day. Progress goes to private/reports/v6/refresh_status.json
(shown on the Rankings page); a lock file prevents two refreshes at once.

Usage: python -m trademaster_ml.refresh [--force]
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime

STATUS = os.path.join("private", "reports", "v6", "refresh_status.json")
LOCK = os.path.join("private", "reports", "v6", "refresh.lock")
EXE = os.path.join("src", "Trademaster.Cli", "bin", "Release", "net10.0", "trademaster.exe")


def status(state, step, **kw):
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    tmp = STATUS + ".tmp"
    json.dump({"state": state, "step": step, "time": datetime.now().isoformat(timespec="seconds"), **kw}, open(tmp, "w"))
    os.replace(tmp, STATUS)
    print(f"[refresh] {state}: {step}", flush=True)


def rankings_asof():
    p = os.path.join("private", "reports", "v6", "rankings.json")
    try:
        return json.load(open(p)).get("asof")
    except (OSError, ValueError):
        return None


def run(args):
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3600:
        print("[refresh] another refresh is running; skipping", flush=True)
        return
    open(LOCK, "w").write(str(os.getpid()))
    t0 = time.time()
    try:
        from . import universe_us, fundamentals, v6, v7
        status("running", "new daily prices")
        last = universe_us.update()
        if not args.force and rankings_asof() == last:
            status("done", f"already up to date (close {last})", asof=last, seconds=round(time.time() - t0))
            return
        status("running", "SEC filings older than 7 days")
        fundamentals.fetch(max_age_days=7)
        status("running", "company financials")
        fundamentals.panel()
        universe_us.export()
        status("running", "features")
        subprocess.run([EXE, "gpu-features", "--cpu"], check=True)
        status("running", "price forecasts")
        v7.score(None)
        v7.forecast(None)
        status("running", "rankings")
        v6.rank(None)
        status("done", f"updated to close {last}", asof=last, seconds=round(time.time() - t0))
    except Exception as e:  # noqa: BLE001 - report any failure on the dashboard, keep the old rankings
        status("error", f"{type(e).__name__}: {e}", seconds=round(time.time() - t0))
        raise
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the rankings already cover the latest close")
    run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
