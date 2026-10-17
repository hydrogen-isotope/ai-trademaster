@echo off
rem Starts the frozen-model forward test for the next NASDAQ session, the buy/sell indicator logger, and the local dashboard.
rem Safe to start any time: the loggers wait for 09:30 ET, keep the PC from idle-sleeping while they run,
rem and exit after the 16:00 ET close. CPU only, predictions and research indicators only, no orders.
cd /d "%~dp0.."
start "AI Trademaster dashboard" src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe serve --port 5080
set PYTHONPATH=python
start "AI Trademaster buy/sell indicators" python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.signals_live live
echo Dashboard: http://127.0.0.1:5080/forward
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.forward live
pause
