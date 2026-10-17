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
