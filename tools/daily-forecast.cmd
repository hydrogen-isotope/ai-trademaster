@echo off
rem End-of-day 5-trading-day forecasts for MU, NVDA, AMD, AAPL, Gold (GC) and Silver (SI). Research only, CPU only.
rem Run after the US close (e.g. 05:30 SGT). Steps: refresh futures settlements (if a Databento key exists, capped
rem spend), refresh daily closes, score earlier forecasts against what happened, then write the next forecast.
cd /d "%~dp0.."
set PYTHONPATH=python
set PY=python\.venv\Scripts\python.exe -W ignore
%PY% -m trademaster_ml.databento_client --roots GC SI --start 2019-01-01 --download --incremental --max-cost-usd 1
%PY% -m trademaster_ml.daily_data
%PY% -m trademaster_ml.daily_forecast score
%PY% -m trademaster_ml.daily_forecast forecast
