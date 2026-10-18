' Starts AI Trademaster as a desktop app: no console window, just the app window.
' Closing the app window stops the local dashboard. Research only: no orders are ever placed.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
exe = root & "\src\Trademaster.Cli\bin\Release\net10.0\trademaster.exe"
If Not fso.FileExists(exe) Then
  MsgBox "AI Trademaster is not built yet." & vbCrLf & "Run: dotnet build -c Release" & vbCrLf & "in " & root, vbExclamation, "AI Trademaster"
  WScript.Quit 1
End If
sh.CurrentDirectory = root
sh.Run """" & exe & """ app --page market --hide-console", 0, False
' Bring the NASDAQ rankings and price forecasts up to date in the background (hidden, CPU only; does nothing when
' they already cover the latest close). Progress shows on the Rankings page.
py = root & "\python\.venv\Scripts\python.exe"
If fso.FileExists(py) Then
  sh.Environment("Process")("PYTHONPATH") = "python"
  sh.Run """" & py & """ -W ignore -m trademaster_ml.refresh", 0, False
End If
