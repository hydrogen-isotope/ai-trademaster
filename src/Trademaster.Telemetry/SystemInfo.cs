using System.Diagnostics;
using System.Runtime;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace Trademaster.Telemetry;

public sealed record GpuInfo(string Name, double VramGb);

/// <summary>Machine and runtime inventory, read from the OS. Inventory only — no performance claims.</summary>
public sealed record SystemInfo(
    string Os, string Cpu, int LogicalCores, double RamGb, string Runtime, string Arch,
    bool ServerGc, string GcLatencyMode, long StopwatchFrequency, IReadOnlyList<GpuInfo> Gpus)
{
    public static SystemInfo Collect()
    {
        string cpu = RuntimeInformation.ProcessArchitecture.ToString();
        var gpus = new List<GpuInfo>();
        if (OperatingSystem.IsWindows())
        {
            cpu = (Registry.GetValue(@"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString", null) as string)?.Trim() ?? cpu;
            using var cls = Registry.LocalMachine.OpenSubKey(@"SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}");
            if (cls is not null)
                foreach (var sub in cls.GetSubKeyNames().Where(s => s.Length == 4 && s.All(char.IsDigit)).Order())
                {
                    using var k = cls.OpenSubKey(sub);
                    if (k?.GetValue("DriverDesc") is not string name) continue;
                    double vram = k.GetValue("HardwareInformation.qwMemorySize") is long q ? q / 1073741824.0 : 0;
                    gpus.Add(new GpuInfo(name, Math.Round(vram, 1)));
                }
        }
        return new SystemInfo(
            RuntimeInformation.OSDescription, cpu, Environment.ProcessorCount,
            Math.Round(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1073741824.0, 1),
            RuntimeInformation.FrameworkDescription, RuntimeInformation.ProcessArchitecture.ToString(),
            GCSettings.IsServerGC, GCSettings.LatencyMode.ToString(), Stopwatch.Frequency, gpus);
    }

    public IEnumerable<string> Lines()
    {
        yield return $"OS:          {Os}";
        yield return $"CPU:         {Cpu} ({LogicalCores} logical cores)";
        yield return $"RAM:         {RamGb} GB available to process";
        yield return $"Runtime:     {Runtime} {Arch}, server GC={ServerGc}, GC latency={GcLatencyMode}, tiered JIT={TieredJit}";
        yield return $"Timer:       Stopwatch {StopwatchFrequency:N0} Hz ({1e9 / StopwatchFrequency:0.#} ns resolution)";
        foreach (var g in Gpus) yield return $"GPU:         {g.Name} ({g.VramGb} GB)";
        if (Gpus.Count == 0) yield return "GPU:         (none detected / not Windows)";
    }

    public static long WorkingSetBytes() => Process.GetCurrentProcess().WorkingSet64;

    public static string TieredJit =>
        Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") is "0" ? "off (env)"
        : AppContext.GetData("System.Runtime.TieredCompilation") switch
        {
            bool b => b ? "on" : "off",
            string s when bool.TryParse(s, out var b) => b ? "on" : "off",
            _ => "on (default)",
        };
}
