@echo off
rem Starts the frozen-model forward test for the next NASDAQ session plus the local dashboard.
rem Safe to start any time: the logger waits for 09:30 ET, keeps the PC from idle-sleeping while it runs,
rem and exits after the 16:00 ET close. CPU only, predictions only, no orders.
cd /d "%~dp0.."
start "AI Trademaster dashboard" src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe serve --port 5080
echo Dashboard: http://127.0.0.1:5080/forward
set PYTHONPATH=python
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.forward live
pause
