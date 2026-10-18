@echo off
rem End-of-day v6 NASDAQ cross-stock rankings. Research only, CPU only (the GPU is never used unattended).
rem Streams only what is new: the latest daily bars (plus full history for new listings and stocks that split),
rem SEC filings for companies not refreshed in 7 days; then rebuilds features and writes today's rankings.
rem The MLP weights come from the last supervised GPU fit (trademaster gpu-mlp --period latest).
cd /d "%~dp0.."
set PYTHONPATH=python
set PY=python\.venv\Scripts\python.exe -W ignore
%PY% -m trademaster_ml.universe_us update || exit /b 1
%PY% -m trademaster_ml.fundamentals fetch --max-age-days 7
%PY% -m trademaster_ml.fundamentals panel
%PY% -m trademaster_ml.universe_us export || exit /b 1
src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe gpu-features --cpu || exit /b 1
%PY% -m trademaster_ml.v6 rank
