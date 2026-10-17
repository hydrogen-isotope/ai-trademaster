#!/usr/bin/env bash
# Overnight v4 pipeline (CPU only). Each step runs under tools/watchdog.py; the first failure stops the pipeline.
# Progress: private/reports/jobs/overnight.log (+ one log per step).
set -u
cd "$(dirname "$0")/.."
PY="$(cygpath -w "$PWD")\\python\\.venv\\Scripts\\python.exe"
export PYTHONPATH=python HF_HOME="$(cygpath -w "$PWD")\\private\\models\\hf"
J=private/reports/jobs; mkdir -p $J
say() { echo "[overnight] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a $J/overnight.log; }

say "waiting for news download + FinBERT scoring"
until grep -q "score complete" $J/news_score.log 2>/dev/null; do
  if grep -q "WARNING\|STOPPED" $J/news_fetch.log $J/news_score.log 2>/dev/null; then say "STOP: news step reported a problem"; exit 1; fi
  sleep 30
done
say "news ready: $(wc -l < private/data/news/items.jsonl) items, $(wc -l < private/data/news/scores.jsonl) scored"

step() {  # name, stall-minutes, args...
  local name=$1 stall=$2; shift 2
  say "start $name"
  if "$PY" tools/watchdog.py --log $J/v4_$name.log --stall-min "$stall" --max-ram-gb 14 -- "$PY" -u -W ignore -m trademaster_ml.daily_v4 "$@"; then
    say "done  $name"
  else
    say "STOP: $name failed (see $J/v4_$name.log)"; exit 1
  fi
}
step select 30 select
step run    30 run
step ablate 30 ablate
say "ALL DONE"
