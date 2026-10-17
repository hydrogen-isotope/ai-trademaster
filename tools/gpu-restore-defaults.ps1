# Restores Windows' default GPU hang-recovery settings and removes a non-default AMD power tweak.
# Run it YOURSELF from an elevated PowerShell (Run as administrator), then reboot:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\gpu-restore-defaults.ps1
# What it changes (a .reg backup of both keys is written first, so it can be undone by double-clicking the backup):
#   - GraphicsDrivers\TdrDelay / TdrDdiDelay (currently 120 s): deleted -> Windows default of 2 s / 5 s.
#     With 120 s, a GPU hang leaves the screen black and the desktop frozen for 2+ minutes (remote control drops),
#     which looks like a crash. With defaults Windows resets the driver within seconds.
#     Side effect: GPU jobs with single dispatches longer than ~2 s (heavy compute/shader tests) will now be reset
#     instead of waited on. That is intended - a reset is recoverable, a 2-minute hang often isn't.
#   - RX 7800 XT PP_SclkDeepSleepDisable=1: deleted -> driver default power management.
#   - TdrLevel is left alone (3 = recover, which is the default).
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

$gd = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupDir = Join-Path $env:USERPROFILE "Documents\gpu-settings-backup-$stamp"
New-Item -ItemType Directory -Force $backupDir | Out-Null
reg export 'HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' (Join-Path $backupDir 'GraphicsDrivers.reg') /y | Out-Null
reg export 'HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' (Join-Path $backupDir 'DisplayClass.reg') /y | Out-Null
Write-Host "Backup written to $backupDir"

foreach ($name in 'TdrDelay', 'TdrDdiDelay') {
    if ($null -ne (Get-ItemProperty $gd -Name $name -ErrorAction SilentlyContinue)) {
        Remove-ItemProperty $gd -Name $name
        Write-Host "Removed $name (back to Windows default)"
    }
}

Get-ChildItem $cls | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if ($p.DriverDesc -like '*RX 7800 XT*' -and $null -ne $p.PP_SclkDeepSleepDisable) {
        Remove-ItemProperty $_.PSPath -Name 'PP_SclkDeepSleepDisable'
        Write-Host "Removed PP_SclkDeepSleepDisable from $($p.DriverDesc)"
    }
}

Write-Host ""
Write-Host "Done. Reboot to apply. Then run:  dotnet run -c Release --project src/Trademaster.Cli -- doctor"
