@echo off
rem Starts the frozen-model forward test for the next NASDAQ session, the buy/sell indicator logger, and the
rem dashboard in its own window. Safe to start any time: data is pulled first, the loggers wait for 09:30 ET,
rem keep the PC from idle-sleeping while they run, and exit after the 16:00 ET close.
rem CPU only, predictions and research indicators only, no orders.
cd /d "%~dp0.."
set PYTHONPATH=python
echo Pulling latest data and checking the frozen signal model...
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.signals_live prepare
start "AI Trademaster dashboard" src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe app --page forward
start "AI Trademaster buy/sell indicators" python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.signals_live live --no-refresh
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.forward live
pause
