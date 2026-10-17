"""
Watchdog for long background jobs. Runs a command, and stops watching (exit code 2) with a clear WARNING line if:
  - the job prints nothing for --stall-min minutes,
  - the job exits with an error,
  - free disk space on the job's drive drops below --min-disk-gb,
  - the job's memory use goes above --max-ram-gb,
  - a new GPU driver reset (TDR) report appears while it runs (this job should never touch the GPU).
Everything the job prints is copied to --log, followed by the watchdog's own status lines.

Usage: python tools/watchdog.py --log private/reports/jobs/news.log --stall-min 10 -- python -m trademaster_ml.news fetch
"""
import argparse, os, shutil, subprocess, sys, threading, time

WATCHDOG_DIR = r"C:\Windows\LiveKernelReports\WATCHDOG"


def tdr_reports():
    try:
        return set(os.listdir(WATCHDOG_DIR))
    except OSError:
        return set()


def rss_gb(pid):
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
            kb = out.strip().split('","')[-1].replace('"', "").replace(",", "").replace(" K", "").strip()
            return int(kb) / 1e6 if kb.isdigit() else 0.0
        return int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--stall-min", type=float, default=10)
    ap.add_argument("--min-disk-gb", type=float, default=5)
    ap.add_argument("--max-ram-gb", type=float, default=12)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    log = open(a.log, "a", encoding="utf-8", buffering=1)
    say = lambda m: (log.write(m + "\n"), print(m, flush=True))
    start, last_out = time.time(), [time.time()]
    tdr0 = tdr_reports()
    say(f"[watchdog] {time.strftime('%H:%M:%S')} start: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env=dict(os.environ, PYTHONUNBUFFERED="1"))

    def pump():
        for line in p.stdout:
            last_out[0] = time.time()
            log.write(line)
    threading.Thread(target=pump, daemon=True).start()

    status = 0
    while p.poll() is None:
        time.sleep(15)
        idle = (time.time() - last_out[0]) / 60
        disk = shutil.disk_usage(os.path.abspath(a.log)).free / 1e9
        ram = rss_gb(p.pid)
        new_tdr = tdr_reports() - tdr0
        problem = None
        if idle > a.stall_min:
            problem = f"STALLED: no output for {idle:.0f} min"
        elif disk < a.min_disk_gb:
            problem = f"LOW DISK: {disk:.1f} GB free"
        elif ram > a.max_ram_gb:
            problem = f"HIGH MEMORY: job uses {ram:.1f} GB"
        elif new_tdr:
            problem = f"GPU DRIVER RESET during the job: {sorted(new_tdr)} (job is CPU-only; check what else is running)"
        if problem:
            say(f"[watchdog] {time.strftime('%H:%M:%S')} WARNING {problem}")
            if problem.startswith(("STALLED", "LOW DISK", "HIGH MEMORY")):
                p.terminate()
                status = 2
                break
    rc = p.wait()
    took = (time.time() - start) / 60
    if status:
        say(f"[watchdog] {time.strftime('%H:%M:%S')} STOPPED the job after {took:.0f} min")
    elif rc != 0:
        say(f"[watchdog] {time.strftime('%H:%M:%S')} WARNING job FAILED with exit code {rc} after {took:.0f} min")
        status = 1
    else:
        say(f"[watchdog] {time.strftime('%H:%M:%S')} OK finished in {took:.1f} min")
    sys.exit(status)


if __name__ == "__main__":
    main()
