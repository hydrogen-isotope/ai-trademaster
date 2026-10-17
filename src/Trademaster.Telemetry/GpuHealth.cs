namespace Trademaster.Telemetry;

/// <summary>
/// Read-only check for recent GPU driver resets (TDR). Windows writes a WATCHDOG live-kernel dump each time
/// the display driver hangs and is reset (LiveKernelEvent 0x141 / 0x117). On this machine those resets have
/// escalated into full system hangs (black screen, no remote access), so GPU work is gated on this check.
/// </summary>
public sealed record GpuHealth(int ResetsLast7Days, DateTime? LastReset, bool CouldRead)
{
    public const string WatchdogDir = @"C:\Windows\LiveKernelReports\WATCHDOG";

    public bool Healthy => CouldRead && ResetsLast7Days == 0;

    public static GpuHealth Check(DateTime? nowUtc = null)
    {
        var now = nowUtc ?? DateTime.UtcNow;
        if (!OperatingSystem.IsWindows()) return new GpuHealth(0, null, false);
        try
        {
            if (!Directory.Exists(WatchdogDir)) return new GpuHealth(0, null, true);
            var stamps = new DirectoryInfo(WatchdogDir).EnumerateFiles("WATCHDOG-*.dmp")
                .Select(f => f.LastWriteTimeUtc)
                .Where(t => now - t <= TimeSpan.FromDays(7))
                .ToList();
            return new GpuHealth(stamps.Count, stamps.Count > 0 ? stamps.Max() : null, true);
        }
        catch (Exception e) when (e is UnauthorizedAccessException or IOException)
        {
            return new GpuHealth(0, null, false);
        }
    }

    public string Describe() =>
        !CouldRead ? "GPU health: unknown (cannot read driver reset history) — treat GPU as unsafe"
        : ResetsLast7Days == 0 ? "GPU health: no display-driver resets recorded in the last 7 days"
        : $"GPU health: {ResetsLast7Days} display-driver reset(s) (TDR) in the last 7 days, last {LastReset:yyyy-MM-dd HH:mm} UTC — GPU work is BLOCKED";
}

/// <summary>
/// Read-only check of Windows/driver settings known to turn a GPU hang into a full freeze on this machine
/// (see docs/GPU-STABILITY.md). Reports findings; never changes anything.
/// </summary>
public static class GpuConfigCheck
{
    private const string GraphicsDrivers = @"SYSTEM\CurrentControlSet\Control\GraphicsDrivers";
    private const string DisplayClass = @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    public static IReadOnlyList<string> Findings()
    {
        var f = new List<string>();
        if (!OperatingSystem.IsWindows()) return f;
        using (var gd = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(GraphicsDrivers))
        {
            foreach (var (name, def) in new[] { ("TdrDelay", 2), ("TdrDdiDelay", 5) })
                if (gd?.GetValue(name) is int v && v != def)
                    f.Add($"{name} = {v} s (Windows default {def} s): a GPU hang freezes the screen this long before recovery");
        }
        var amd = new List<(string Name, string Version)>();
        using (var cls = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(DisplayClass))
        {
            if (cls is not null)
                foreach (var sub in cls.GetSubKeyNames().Where(s => s.Length == 4 && s.All(char.IsDigit)))
                {
                    using var k = cls.OpenSubKey(sub);
                    if (k?.GetValue("DriverDesc") is not string desc) continue;
                    if (k.GetValue("ProviderName") is string prov && !prov.Contains("Advanced Micro Devices", StringComparison.OrdinalIgnoreCase)
                        && !desc.Contains("AMD", StringComparison.OrdinalIgnoreCase)) continue;
                    amd.Add((desc, k.GetValue("DriverVersion") as string ?? "?"));
                    if (k.GetValue("PP_SclkDeepSleepDisable") is int pp && pp != 0)
                        f.Add($"{desc}: PP_SclkDeepSleepDisable = {pp} (non-default power tweak)");
                }
        }
        if (amd.Select(a => a.Version).Distinct().Count() > 1)
            f.Add("AMD adapters run different driver versions (" + string.Join(", ", amd.Select(a => $"{a.Name} {a.Version}")) +
                  "): install one Adrenalin package for both");
        return f;
    }
}

/// <summary>Single gate every GPU code path must pass before touching the device.</summary>
public static class GpuGate
{
    /// <summary>
    /// Throws unless GPU use is explicitly allowed in config AND the driver has been stable for 7 days
    /// (or the operator explicitly overrides the stability check).
    /// </summary>
    public static void Require(bool allowGpu, bool overrideStabilityCheck, GpuHealth health)
    {
        if (!allowGpu)
            throw new InvalidOperationException("GPU use is disabled (compute.allow_gpu = false). This run must use the CPU backend.");
        if (!health.Healthy && !overrideStabilityCheck)
            throw new InvalidOperationException(health.Describe() +
                ". Fix the driver/hardware first (see docs/GPU-STABILITY.md), or set compute.override_gpu_stability_check = true at your own risk.");
    }
}
