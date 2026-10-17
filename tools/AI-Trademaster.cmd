@echo off
rem AI Trademaster in its own window (no browser tab needed).
rem 1) pulls the latest 1-minute data and refreshes the frozen signal model if a newer full week exists
rem 2) opens the dashboard window on the forward-test page; closing the window stops the dashboard.
rem CPU only, research only, no orders.
cd /d "%~dp0.."
set PYTHONPATH=python
echo Pulling latest data...
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.signals_live prepare
start "" /b src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe app --page market
