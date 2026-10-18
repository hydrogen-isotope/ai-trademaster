@echo off
rem End-of-day v6 NASDAQ rankings + v7 price forecasts. Research only, CPU only (the GPU is never used unattended).
rem Streams only what is new (latest bars, SEC filings older than 7 days) and skips the rebuild when the rankings
rem already cover the latest close. The app also runs this in the background each time it opens.
cd /d "%~dp0.."
set PYTHONPATH=python
python\.venv\Scripts\python.exe -W ignore -m trademaster_ml.refresh
