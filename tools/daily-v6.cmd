@echo off
rem End-of-day v6 NASDAQ cross-stock rankings. Research only, CPU only (the GPU is never used unattended).
rem Streams only what is new: the latest daily bars (plus full history for new listings and stocks that split),
rem SEC filings for companies not refreshed in 7 days; then rebuilds features and writes today's rankings.
rem Rankings use the adopted model (ridge, see docs/FORECAST-V6-CROSS-SECTION-PROTOCOL.md); no GPU step is needed.
cd /d "%~dp0.."
set PYTHONPATH=python
set PY=python\.venv\Scripts\python.exe -W ignore
%PY% -m trademaster_ml.universe_us update || exit /b 1
%PY% -m trademaster_ml.fundamentals fetch --max-age-days 7
%PY% -m trademaster_ml.fundamentals panel
%PY% -m trademaster_ml.universe_us export || exit /b 1
src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe gpu-features --cpu || exit /b 1
rem v7 price forecasts (shadow test, docs/FORECAST-V7-PRICE-PROTOCOL.md): score matured ones, then forecast today
%PY% -m trademaster_ml.v7 score
%PY% -m trademaster_ml.v7 forecast
%PY% -m trademaster_ml.v6 rank
