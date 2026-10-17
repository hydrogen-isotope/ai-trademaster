using Trademaster.Core;
using Trademaster.Telemetry;

namespace Trademaster.Tests;

public class GpuGateTests
{
    private static readonly GpuHealth Stable = new(0, null, true);
    private static readonly GpuHealth Resets = new(3, DateTime.UtcNow, true);
    private static readonly GpuHealth Unknown = new(0, null, false);

    [Fact]
    public void Gpu_is_off_by_default_in_config()
    {
        Assert.False(new EngineConfig().Compute.AllowGpu);
        Assert.False(EngineConfig.Load(FindRepoFile("config/default.json")).Compute.AllowGpu);
    }

    [Fact]
    public void Gate_blocks_when_disabled_unstable_or_unknown()
    {
        Assert.Throws<InvalidOperationException>(() => GpuGate.Require(false, false, Stable));
        Assert.Throws<InvalidOperationException>(() => GpuGate.Require(true, false, Resets));
        Assert.Throws<InvalidOperationException>(() => GpuGate.Require(true, false, Unknown));
        GpuGate.Require(true, false, Stable);
        GpuGate.Require(true, true, Resets);
    }

    private static string FindRepoFile(string rel)
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Trademaster.slnx"))) return Path.Combine(d.FullName, rel);
        throw new FileNotFoundException(rel);
    }
}
