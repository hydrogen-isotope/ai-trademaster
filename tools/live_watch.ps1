# Refreshes 1-minute bars and B0 prediction-vs-actual evaluations every N minutes until the NASDAQ close.
# CPU only. Keeps the PC (and display) awake while running via SetThreadExecutionState; no settings are changed,
# and the request is released automatically when this script exits.
# Usage: powershell -NoProfile -File tools/live_watch.ps1 [-IntervalMin 10] [-Dir data/live]
param([int]$IntervalMin = 10, [string]$Dir = "private/data/live")

$pairs = [ordered]@{ MU = 'SMH'; NVDA = 'SMH'; AMD = 'SMH'; AAPL = 'QQQ' }
$symbols = @($pairs.Keys) + @($pairs.Values | Select-Object -Unique)

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM_REQUIRED = [uint32]1; $ES_DISPLAY_REQUIRED = [uint32]2
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)

function NyNow { [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Eastern Standard Time') }
New-Item -ItemType Directory -Force $Dir | Out-Null
$headers = @{ 'User-Agent' = 'Mozilla/5.0' }

try {
    while ($true) {
        $ny = NyNow
        $final = $ny.TimeOfDay -ge [TimeSpan]'16:01'
        $ok = $true
        foreach ($s in $symbols) {
            $u = "https://query1.finance.yahoo.com/v8/finance/chart/$($s)?interval=1m&range=5d&includePrePost=false"
            try { Invoke-WebRequest -Uri $u -Headers $headers -OutFile "$Dir/$s-1m.json" -UseBasicParsing -TimeoutSec 30 }
            catch { $ok = $false; Write-Output "ERROR fetch $s $($_.Exception.Message)" }
        }
        foreach ($k in $pairs.Keys) {
            & py -I tools/eval_bars.py $Dir $k $pairs[$k] *> "$Dir/$k-summary.json"
            if ($LASTEXITCODE -ne 0) { $ok = $false; Write-Output "ERROR eval $k" }
        }
        Write-Output ("{0} {1:HH:mm} ET ok={2}" -f ($(if ($final) { 'DONE' } else { 'REFRESH' })), $ny, $ok)
        if ($final) { break }
        $next = $ny.Date.AddMinutes([Math]::Floor($ny.TimeOfDay.TotalMinutes / $IntervalMin + 1) * $IntervalMin).AddSeconds(20)
        $close = $ny.Date.Add([TimeSpan]'16:01:30')
        if ($next -gt $close) { $next = $close }
        Start-Sleep -Seconds ([Math]::Max(5, ($next - (NyNow)).TotalSeconds))
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
