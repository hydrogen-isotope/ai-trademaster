"""
Bring the v6 rankings, v7 price forecasts and the rolling LLM news up to date (research only). Everything runs on the
CPU except the news reading, which uses the GPU only with --gpu-news (passed by the app launcher, i.e. while the user
is at the PC); without it the news is read on the CPU.

Runs in the background when the app opens, and from the daily job. It streams only what is new and stops early when
the rankings already cover the latest completed trading day (then only the news and forecasts are refreshed). Progress goes to private/reports/v6/refresh_status.json
(shown on the Rankings page); a lock file prevents two refreshes at once.

Usage: python -m trademaster_ml.refresh [--force] [--gpu-news]
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


def news_step(args):
    """v8 stage 1: score the last days of headlines with the local LLM (GPU only when started from the open app)."""
    from . import llm_news
    status("running", "reading the news (local AI model)")
    try:
        llm_news.rolling(argparse.Namespace(allow_gpu=args.gpu_news, cpu=not args.gpu_news))
    except Exception as e:  # noqa: BLE001 - news is optional: keep forecasting without it
        print(f"[refresh] news step skipped: {type(e).__name__}: {e}", flush=True)


def sgx_step():
    """SGX (Eulerpool): stream recent days, rebuild features, price forecasts and rankings when a new close exists.
    Optional: any failure is reported and NASDAQ results are kept."""
    from . import sgx, v6, v7
    try:
        status("running", "SGX prices")
        last = sgx.update()
        try:                                                    # SGX momentum forward test (amendment B): log + score
            from . import sgx_momentum
            sgx_momentum.update()
        except Exception as e:  # noqa: BLE001
            print(f"[refresh] SGX momentum step failed: {type(e).__name__}: {e}", flush=True)
        p = os.path.join("private", "reports", "sgx", "v6", "rankings.json")
        if os.path.exists(p) and json.load(open(p)).get("asof") == last:
            return
        sgx.export()
        status("running", "SGX features")
        subprocess.run([EXE, "gpu-features", "--cpu", "--in", os.path.join("private", "data", "sgx", "gpu"),
                        "--top", "150", "--min-price", "0.2"], check=True)
        status("running", "SGX price forecasts and rankings")
        v7.use_market("sgx")
        v7.score(None)
        v7.forecast(None)
        v6.rank(None)
    except Exception as e:  # noqa: BLE001
        print(f"[refresh] SGX step failed: {type(e).__name__}: {e}", flush=True)
    finally:
        v7.use_market("nasdaq")


def run(args):
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 3600:
        print("[refresh] another refresh is running; skipping", flush=True)
        return
    open(LOCK, "w").write(str(os.getpid()))
    t0 = time.time()
    try:
        from . import universe_us, fundamentals, v6, v7
        v7.use_market("nasdaq")
        status("running", "new daily prices")
        last = universe_us.update()
        if not args.force and rankings_asof() == last:
            # no new close: only the rolling news can have changed (e.g. weekend or pre-market headlines)
            news_step(args)
            status("running", "price forecasts")
            v7.forecast(None)
            status("running", "rankings")
            v6.rank(None)
            sgx_step()
            status("done", f"news refreshed; prices already up to date (close {last})", asof=last, seconds=round(time.time() - t0))
            return
        status("running", "SEC filings older than 7 days")
        fundamentals.fetch(max_age_days=7)
        status("running", "company financials")
        fundamentals.panel()
        universe_us.export()
        status("running", "features")
        subprocess.run([EXE, "gpu-features", "--cpu"], check=True)
        news_step(args)
        status("running", "price forecasts")
        v7.score(None)
        v7.forecast(None)
        status("running", "rankings")
        v6.rank(None)
        sgx_step()
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
    ap.add_argument("--gpu-news", action="store_true", help="score news on the GPU (only when the user is at the PC, e.g. app start)")
    run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
