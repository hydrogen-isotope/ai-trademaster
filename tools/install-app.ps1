# Installs AI Trademaster as a desktop app for the current user:
#   - renders the app icon from src/Trademaster.Cli/ui/favicon.svg (Edge headless, GPU off) into
#     %LOCALAPPDATA%\AITrademaster\app.ico
#   - creates "AI Trademaster" shortcuts on the Desktop and in the Start menu that run tools\AI-Trademaster.vbs
# No admin rights, no system settings changed. Remove by deleting the two shortcuts.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $env:LOCALAPPDATA 'AITrademaster'
New-Item -ItemType Directory -Force $appDir | Out-Null

$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
$ico = Join-Path $appDir 'app.ico'
if ($edge) {
    $svg = (Join-Path $repo 'src\Trademaster.Cli\ui\favicon.svg') -replace '\\', '/'
    $tmp = Join-Path $appDir 'icon-build'
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $sizes = 256, 48, 32, 16
    $pngs = @()
    foreach ($n in $sizes) {
        $html = Join-Path $tmp "i$n.html"
        Set-Content -Path $html -Encoding ASCII -Value "<html><body style='margin:0;background:transparent'><img src='file:///$svg' width='$n' height='$n'></body></html>"
        $png = Join-Path $tmp "i$n.png"
        $edgeArgs = @('--headless=new', '--disable-gpu', '--hide-scrollbars', '--default-background-color=00000000',
            '--allow-file-access-from-files', "--window-size=$n,$n", "--screenshot=`"$png`"", "file:///$($html -replace '\\','/')")
        Start-Process -FilePath $edge -ArgumentList $edgeArgs -Wait -WindowStyle Hidden -RedirectStandardError (Join-Path $tmp 'edge.log')
        for ($i = 0; $i -lt 20 -and -not (Test-Path $png); $i++) { Start-Sleep -Milliseconds 250 }
        $pngs += , [IO.File]::ReadAllBytes($png)
    }
    # ICO container with embedded PNG images
    $ms = New-Object IO.MemoryStream
    $w = New-Object IO.BinaryWriter $ms
    $w.Write([UInt16]0); $w.Write([UInt16]1); $w.Write([UInt16]$sizes.Count)
    $offset = 6 + 16 * $sizes.Count
    for ($k = 0; $k -lt $sizes.Count; $k++) {
        $n = $sizes[$k]; $d = if ($n -ge 256) { 0 } else { $n }
        $w.Write([Byte]$d); $w.Write([Byte]$d); $w.Write([Byte]0); $w.Write([Byte]0)
        $w.Write([UInt16]1); $w.Write([UInt16]32); $w.Write([UInt32]$pngs[$k].Length); $w.Write([UInt32]$offset)
        $offset += $pngs[$k].Length
    }
    foreach ($p in $pngs) { $w.Write($p) }
    $w.Flush(); [IO.File]::WriteAllBytes($ico, $ms.ToArray())
    Remove-Item -Recurse -Force $tmp
}

$sh = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $lnk = $sh.CreateShortcut((Join-Path $dir 'AI Trademaster.lnk'))
    $lnk.TargetPath = "$env:WINDIR\System32\wscript.exe"
    $lnk.Arguments = '"' + (Join-Path $repo 'tools\AI-Trademaster.vbs') + '"'
    $lnk.WorkingDirectory = $repo
    if (Test-Path $ico) { $lnk.IconLocation = "$ico,0" }
    $lnk.Description = 'AI Trademaster - markets, AI signals and 5-day forecasts (research only, no orders)'
    $lnk.Save()
    Write-Output "shortcut: $(Join-Path $dir 'AI Trademaster.lnk')"
}
