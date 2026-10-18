using Trademaster.Gpu;

namespace Trademaster.Tests;

public class GpuMlpTests
{
    private static (float[] x, float[] y) Data(int rows, int f, int seed)
    {
        var rng = new Random(seed);
        var x = new float[rows * f];
        var y = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            for (int k = 0; k < f; k++) x[r * f + k] = (float)(rng.NextDouble() - 0.5);
            // nonlinear target the network can learn, plus noise; a few missing targets
            y[r] = rng.NextDouble() < 0.02 ? float.NaN
                : (float)(0.3 * x[r * f] + 0.4 * Math.Abs(x[r * f + 1]) - 0.2 * x[r * f + 2] * x[r * f + 3] + 0.05 * (rng.NextDouble() - 0.5));
        }
        return (x, y);
    }

    [Fact]
    public void Cpu_LearnsANonlinearTarget()
    {
        const int rows = 6000, f = 6;
        var (x, y) = Data(rows, f, 1);
        var o = new GpuMlp.Options(Hidden: 16, Seeds: 2, Batch: 128, MaxEpochs: 15, Patience: 3, Lr: 3e-3f);
        using var be = new GpuMlp.Cpu(x, y, f, o);
        var pred = GpuMlp.FitPredict(be, y, f, (0, 4000), (4000, 5000), (5000, 6000), o, seedBase: 1);
        double mse = 0, var0 = 0, mean = 0; int c = 0;
        for (int i = 0; i < 1000; i++) if (float.IsFinite(y[5000 + i])) { mean += y[5000 + i]; c++; }
        mean /= c;
        for (int i = 0; i < 1000; i++)
        {
            float t = y[5000 + i];
            if (!float.IsFinite(t)) continue;
            mse += (pred[i] - t) * (pred[i] - t);
            var0 += (t - mean) * (t - mean);
        }
        Assert.True(mse < 0.5 * var0, $"MSE {mse / c:F5} vs variance {var0 / c:F5}");
    }

    /// <summary>GPU vs CPU training parity. Touches the GPU, so it only runs when TM_GPU_TESTS=1 (supervised).</summary>
    [Fact]
    public void Gpu_MatchesCpuReference()
    {
        if (!OperatingSystem.IsWindows() || Environment.GetEnvironmentVariable("TM_GPU_TESTS") != "1") return;
        const int rows = 6000, f = 6;
        var (x, y) = Data(rows, f, 2);
        var o = new GpuMlp.Options(Hidden: 16, Seeds: 3, Batch: 256, MaxEpochs: 3, Patience: 3);
        using var c = new GpuMlp.Cpu(x, y, f, o);
        using var g = new GpuMlp.Gpu(x, y, f, o);
        var pc = GpuMlp.FitPredict(c, y, f, (0, 4000), (4000, 5000), (5000, 6000), o, seedBase: 5);
        var pg = GpuMlp.FitPredict(g, y, f, (0, 4000), (4000, 5000), (5000, 6000), o, seedBase: 5);
        double maxDiff = pc.Zip(pg).Max(p => Math.Abs(p.First - p.Second));
        double scale = pc.Select(Math.Abs).Max();
        Assert.True(maxDiff < 1e-3 * Math.Max(1, scale), $"max |GPU - CPU| = {maxDiff:E2}");
    }
}
