using Trademaster.Gpu;

namespace Trademaster.Tests;

public class GpuFeaturesTests
{
    private static GpuFeatures.Inputs Inputs(int T, int N, int seed, int topK)
    {
        var (lc, v, m) = Panel(T, N, seed);
        var rng = new Random(seed + 1);
        var eligible = Enumerable.Range(0, N).Select(_ => rng.NextDouble() < 0.95 ? 1f : 0f).ToArray();
        var industry = Enumerable.Range(0, N).Select(_ => rng.NextDouble() < 0.9 ? (float)rng.Next(0, 12) : -1f).ToArray();
        var extra = new float[T * N];
        for (int i = 0; i < extra.Length; i++) extra[i] = rng.NextDouble() < 0.1 ? float.NaN : (float)rng.NextDouble();
        return new GpuFeatures.Inputs(T, N, lc, v, m, eligible, industry, [("value", extra)], topK, 5);
    }

    private static (float[] lc, float[] v, float[] m) Panel(int T, int N, int seed)
    {
        var rng = new Random(seed);
        var lc = new float[T * N];
        var v = new float[T * N];
        var m = new float[T];
        double mp = 0;
        var mr = new double[T];
        for (int t = 0; t < T; t++) { mr[t] = 0.01 * (rng.NextDouble() - 0.5); mp += mr[t]; m[t] = (float)mp; }
        for (int n = 0; n < N; n++)
        {
            double p = Math.Log(5 + 100 * rng.NextDouble()), beta = 2 * rng.NextDouble();
            int start = rng.Next(0, T / 3);             // late listings
            for (int t = 0; t < T; t++)
            {
                p += beta * mr[t] + 0.02 * (rng.NextDouble() - 0.5);
                bool missing = t < start || rng.NextDouble() < 0.01;
                lc[t * N + n] = missing ? float.NaN : (float)p;
                v[t * N + n] = missing ? float.NaN : (float)(1e5 * (0.5 + rng.NextDouble()));
            }
        }
        return (lc, v, m);
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
        var (lc, v, m) = Panel(300, 4, 1);
        int t = 299, n = 2;
        if (float.IsFinite(lc[t * 4 + n]) && float.IsFinite(lc[(t - 21) * 4 + n]))
            Assert.Equal(lc[t * 4 + n] - lc[(t - 21) * 4 + n], GpuFeatures.CpuFeature(1, t, n, lc, v, m, 4));
    }

    [Fact]
    public void CpuFeatures_BetaRecoversTheSimulatedBeta()
    {
        const int T = 600, N = 50;
        var (lc, v, m) = Panel(T, N, 3);
        var betas = Enumerable.Range(0, N).Select(n => GpuFeatures.CpuFeature(10, T - 1, n, lc, v, m, N)).Where(float.IsFinite).ToArray();
        Assert.True(betas.Length > 30);
        Assert.InRange(betas.Average(), 0.7, 1.3);   // simulated betas are uniform on [0, 2]
    }

    [Fact]
    public void CpuIndustry_MeansBreadthAndLeadersExcludeSelf()
    {
        // one day, 5 stocks: industry 0 = stocks 0..3, stock 4 has none
        float[] values = [0.1f, -0.2f, 0.3f, 0.4f, 9f];
        float[] mask = [1, 1, 1, 1, 1];
        float[] leader = [1, 0, 1, 0, 1];
        float[] industry = [0, 0, 0, 0, -1];
        var mean = GpuFeatures.CpuIndustry(0, values, mask, leader, industry, 1, 5);
        Assert.Equal(0.15f, mean[1], 5);
        Assert.True(float.IsNaN(mean[4]));
        var breadth = GpuFeatures.CpuIndustry(1, values, mask, leader, industry, 1, 5);
        Assert.Equal(0.75f, breadth[0]);
        var lead = GpuFeatures.CpuIndustry(2, values, mask, leader, industry, 1, 5);
        Assert.Equal(0.3f, lead[0], 5);   // leaders 0 and 2, excluding itself
        Assert.Equal(0.2f, lead[1], 5);   // not a leader: mean of 0.1 and 0.3
    }

    private static void Close(float[] cpu, float[] gpu, string what, double tol = 2e-3)
    {
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(float.IsFinite(cpu[i]) == float.IsFinite(gpu[i]), $"{what} at {i}: {cpu[i]} vs {gpu[i]}");
            if (float.IsFinite(cpu[i])) Assert.True(Math.Abs(cpu[i] - gpu[i]) < tol * Math.Max(1, Math.Abs(cpu[i])), $"{what} at {i}: {cpu[i]} vs {gpu[i]}");
        }
    }

    /// <summary>GPU vs CPU parity. Touches the GPU, so it only runs when TM_GPU_TESTS=1 is set on a supervised Windows machine.</summary>
    [Fact]
    public void Gpu_MatchesCpuReference()
    {
        if (!OperatingSystem.IsWindows() || Environment.GetEnvironmentVariable("TM_GPU_TESTS") != "1") return;
        const int T = 400, N = 700;
        var x = Inputs(T, N, 7, 300);
        using var gb = new GpuFeatures.Gpu(x);
        using var cb = new GpuFeatures.Cpu(x);

        // 1. every kernel on identical inputs
        for (int f = 0; f < GpuFeatures.StockFeatures.Length; f++) Close(cb.Feature(f), gb.Feature(f), GpuFeatures.StockFeatures[f]);
        var values = x.Extras[0].Values;
        var rng = new Random(11);
        var mask = values.Select(_ => rng.NextDouble() < 0.6 ? 1f : 0f).ToArray();
        var leader = values.Select(_ => rng.NextDouble() < 0.3 ? 1f : 0f).ToArray();
        Close(cb.Rank(values, mask), gb.Rank(values, mask), "rank", 1e-5);
        for (int op = 0; op < 3; op++) Close(cb.Industry(op, values, mask, leader), gb.Industry(op, values, mask, leader), $"industry op {op}", 1e-4);

        // 2. whole pipeline: float (GPU) vs double (CPU) dollar volume may swap near-tied stocks at the universe and
        //    industry-leader cut-offs, which moves a few downstream values; require that to stay rare.
        var g = GpuFeatures.Build(x, gb);
        var c = GpuFeatures.Build(x, cb);
        Assert.Equal(c.Names, g.Names);
        int maskOff = c.Mask.Zip(g.Mask).Count(p => p.First != p.Second);
        Console.WriteLine($"universe mask: {maskOff} of {T * N} cells differ");
        Assert.True(maskOff <= T * N / 1000);
        for (int f = 0; f < c.Names.Length; f++)
        {
            int off = 0;
            for (int i = 0; i < T * N; i++)
            {
                bool fc = float.IsFinite(c.Ranks[f][i]), fg = float.IsFinite(g.Ranks[f][i]);
                if (fc != fg || (fc && Math.Abs(c.Ranks[f][i] - g.Ranks[f][i]) > 0.01)) off++;
            }
            Console.WriteLine($"{c.Names[f]}: {off} ranks differ by > 0.01");
            // an industry value is shared by all members, so one flipped member moves the whole industry's rank
            bool industry = GpuFeatures.IndustryFeatures.Contains(c.Names[f]);
            Assert.True(off <= T * N / (industry ? 40 : 200), $"{c.Names[f]}: {off} ranks differ by > 0.01");
        }
    }
}
