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

    [Fact]
    public void Onnx_gpu_backend_is_refused_before_any_session_is_created()
    {
        // The model path doesn't exist: if the gate didn't run first we'd get FileNotFound/OnnxRuntimeException instead.
        var noGpu = new Trademaster.Engine.Inference.OnnxBackend("dml", AllowGpu: false);
        var ex = Assert.Throws<InvalidOperationException>(() => new Trademaster.Engine.Inference.OnnxRunner("does-not-exist.onnx", 30, 8, 1, 1, noGpu));
        Assert.Contains("GPU", ex.Message);
        Assert.Throws<ArgumentException>(() => new Trademaster.Engine.Inference.OnnxRunner("x.onnx", 30, 8, 1, 1, new("opencl")));
    }

    [Fact]
    public void Backend_comes_from_config_and_defaults_to_cpu()
    {
        var b = Trademaster.Engine.Inference.OnnxBackend.From(new ComputeSection());
        Assert.Equal("cpu", b.Name);
        Assert.False(b.IsGpu);
        Assert.False(b.AllowGpu);
    }

    [Fact]
    public void Adapter_enumeration_is_read_only_and_picks_largest_vram_hardware_gpu()
    {
        var list = GpuAdapters.Enumerate();
        if (!OperatingSystem.IsWindows()) { Assert.Empty(list); return; }
        Assert.NotEmpty(list);
        var fake = new List<GpuAdapter> { new(0, "iGPU", 0x1002, 2, false), new(1, "dGPU", 0x1002, 16, false), new(2, "Basic Render", 0x1414, 0, true) };
        Assert.Equal(1, GpuAdapters.PickDiscrete(fake));
        Assert.Equal(-1, GpuAdapters.PickDiscrete([new(0, "sw", 0x1414, 0, true)]));
    }

    private static string FindRepoFile(string rel)
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Trademaster.slnx"))) return Path.Combine(d.FullName, rel);
        throw new FileNotFoundException(rel);
    }
}
