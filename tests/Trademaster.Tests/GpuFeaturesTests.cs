using Trademaster.Gpu;

namespace Trademaster.Tests;

public class GpuFeaturesTests
{
    private static (float[] lc, float[] v) Panel(int T, int N, int seed)
    {
        var rng = new Random(seed);
        var lc = new float[T * N];
        var v = new float[T * N];
        for (int n = 0; n < N; n++)
        {
            double p = Math.Log(5 + 100 * rng.NextDouble());
            int start = rng.Next(0, T / 3);             // late listings
            for (int t = 0; t < T; t++)
            {
                p += 0.02 * (rng.NextDouble() - 0.5);
                bool missing = t < start || rng.NextDouble() < 0.01;
                lc[t * N + n] = missing ? float.NaN : (float)p;
                v[t * N + n] = missing ? float.NaN : (float)(1e5 * (0.5 + rng.NextDouble()));
            }
        }
        return (lc, v);
    }

    [Fact]
    public void CpuRank_HandlesTiesMaskAndMissing()
    {
        float[] values = [3, 1, 3, float.NaN, 2];
        float[] mask = [1, 1, 1, 1, 0];
        var r = GpuFeatures.CpuRank(values, mask, 1, 5);
        Assert.Equal(0f, r[1]);
        Assert.Equal(0.75f, r[0]);   // less = 1, one equal other -> (1 + 0.5) / 2
        Assert.Equal(0.75f, r[2]);
        Assert.True(float.IsNaN(r[3]));
        Assert.True(float.IsNaN(r[4]));
    }

    [Fact]
    public void CpuFeatures_MomentumIsLogPriceChange()
    {
        var (lc, v) = Panel(300, 4, 1);
        int t = 299, n = 2;
        if (float.IsFinite(lc[t * 4 + n]) && float.IsFinite(lc[(t - 21) * 4 + n]))
            Assert.Equal(lc[t * 4 + n] - lc[(t - 21) * 4 + n], GpuFeatures.CpuFeature(1, t, n, lc, v, 4));
    }

    /// <summary>GPU vs CPU parity. Touches the GPU, so it only runs when TM_GPU_TESTS=1 is set on a supervised Windows machine.</summary>
    [Fact]
    public void Gpu_MatchesCpuReference()
    {
        if (!OperatingSystem.IsWindows() || Environment.GetEnvironmentVariable("TM_GPU_TESTS") != "1") return;
        const int T = 400, N = 700;
        var (lc, v) = Panel(T, N, 7);
        var g = GpuFeatures.RunGpu(lc, v, T, N, 300, 5);
        var c = GpuFeatures.RunCpu(lc, v, T, N, 300, 5);
        // float (GPU) vs double (CPU) dollar volume can swap near-tied stocks at the universe cut-off
        Assert.True(c.Mask.Zip(g.Mask).Count(p => p.First != p.Second) <= T * N / 1000);
        for (int f = 0; f < GpuFeatures.Names.Length; f++)
            for (int i = 0; i < T * N; i++)
            {
                Assert.Equal(float.IsFinite(c.Raw[f][i]), float.IsFinite(g.Raw[f][i]));
                if (float.IsFinite(c.Raw[f][i])) Assert.True(Math.Abs(c.Raw[f][i] - g.Raw[f][i]) < 2e-3, $"{GpuFeatures.Names[f]} at {i}");
            }
    }
}
