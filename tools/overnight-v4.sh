#!/usr/bin/env bash
# Overnight v4 pipeline (CPU only). Every step runs under tools/watchdog.py; the first failure stops the pipeline.
# Steps resume where they left off (news fetch/scoring are incremental). Keeps the PC from idle-sleeping while running.
# Progress: private/reports/jobs/overnight.log (+ one log per step).
set -u
cd "$(dirname "$0")/.."
PY="$(cygpath -w "$PWD")\\python\\.venv\\Scripts\\python.exe"
export PYTHONPATH=python HF_HOME="$(cygpath -w "$PWD")\\private\\models\\hf"
J=private/reports/jobs; mkdir -p $J
say() { echo "[overnight] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a $J/overnight.log; }

# keep awake (no settings changed; released when this script exits)
"$PY" -c "import ctypes,time,os; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); pid=$$
import subprocess
while subprocess.run(['tasklist','/FI',f'PID eq {os.getppid()}'],capture_output=True,text=True).stdout.count('\n')>3: time.sleep(60)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null' EXIT

step() {  # log-name, stall-minutes, max-ram-gb, module, args...
  local name=$1 stall=$2 ram=$3 mod=$4; shift 4
  say "start $name"
  if "$PY" tools/watchdog.py --log $J/$name.log --stall-min "$stall" --max-ram-gb "$ram" -- "$PY" -u -W ignore -m "$mod" "$@"; then
    say "done  $name"
  else
    say "STOP: $name failed (see $J/$name.log)"; exit 1
  fi
}
step news_fetch 10 4  trademaster_ml.news fetch
step news_score 15 8  trademaster_ml.news score
step v4_select  30 14 trademaster_ml.daily_v4 select
step v4_run     30 14 trademaster_ml.daily_v4 run
step v4_ablate  30 14 trademaster_ml.daily_v4 ablate
step v4_explain 20 8  trademaster_ml.daily_v4 explain
say "ALL DONE"
