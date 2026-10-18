namespace Trademaster.Gpu;

/// <summary>
/// The v6 feature panel: per-stock features on a [T, N] grid (Shaders/features.hlsl), then cross-sectional
/// percentile ranks inside a point-in-time liquid universe. GPU path plus an exact-logic CPU reference.
/// </summary>
public static class GpuFeatures
{
    public static readonly string[] Names =
        ["mom5", "mom21", "mom63", "mom126", "mom12_1", "vol20", "vol60", "dist_252_high", "volume_surprise", "log_dollar_vol20"];

    public const int DollarVolume = 9;

    /// <summary>Raw features [F][T*N], the universe mask [T*N] and ranks [F][T*N].</summary>
    public sealed record Result(float[][] Raw, float[] Mask, float[][] Ranks, string Device, double Seconds);

    /// <summary>
    /// Universe at day t: price ≥ minPrice and among the <paramref name="topK"/> largest 20-day average dollar volumes,
    /// using only data up to day t (no look-ahead).
    /// </summary>
    public static float[] UniverseMask(float[] logClose, float[] dollarVolRank, float[] dollarVol, int T, int N, int topK, float minPrice)
    {
        var mask = new float[T * N];
        float minLog = MathF.Log(minPrice);
        for (int t = 0; t < T; t++)
        {
            int count = 0;
            for (int n = 0; n < N; n++) if (float.IsFinite(dollarVol[t * N + n])) count++;
            if (count < 2) continue;
            // rank r in [0,1]: number of stocks above = (1 - r) * (count - 1)
            for (int n = 0; n < N; n++)
            {
                int i = t * N + n;
                float r = dollarVolRank[i];
                if (float.IsFinite(r) && (1 - r) * (count - 1) < topK && logClose[i] >= minLog) mask[i] = 1;
            }
        }
        return mask;
    }

    /// <summary>Runs on the GPU. Each dispatch covers a slice of days sized to stay far below the 2 s driver limit.</summary>
    public static Result RunGpu(float[] logClose, float[] volume, int T, int N, int topK, float minPrice, int adapter = -1, TextWriter? log = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var gpu = new D3D11Compute(adapter);
        log?.WriteLine($"GPU: {gpu.AdapterName}");
        var featureCs = gpu.CompileShader("features.hlsl", "FeatureKernel");
        var rankCs = gpu.CompileShader("features.hlsl", "RankKernel");
        var lc = gpu.CreateBuffer<float>(logClose);
        var vol = gpu.CreateBuffer<float>(volume);
        var outBuf = gpu.CreateBuffer(T * N);
        var cb = gpu.CreateConstants(new Params());
        uint groupsX = (uint)((N + 255) / 256);

        // Day-slices: feature kernels do ≤ 252 steps per thread; the rank kernel does N steps per thread.
        int featDays = Math.Max(1, (int)Math.Min(D3D11Compute.MaxGroupsPerDispatch, 40_000_000L / (N * 252L)));
        int rankDays = Math.Max(1, (int)Math.Min(D3D11Compute.MaxGroupsPerDispatch, 2_000_000_000L / ((long)N * N)));

        void Run(nint shader, int feature, D3D11Compute.GpuBuffer a, D3D11Compute.GpuBuffer b, int days)
        {
            for (int t0 = 0; t0 < T; t0 += days)
            {
                gpu.UpdateConstants(cb, new Params { N = (uint)N, T = (uint)T, T0 = (uint)t0, Feature = (uint)feature });
                gpu.Dispatch(shader, cb, groupsX, (uint)Math.Min(days, T - t0), a, b, outBuf);
                gpu.Flush();
            }
        }

        var raw = new float[Names.Length][];
        for (int f = 0; f < Names.Length; f++)
        {
            Run(featureCs, f, lc, vol, featDays);
            raw[f] = new float[T * N];
            gpu.Read<float>(outBuf, raw[f]);
        }

        // Rank dollar volume among all stocks with data, then build the universe on the CPU (cheap, sequential logic).
        var all = new float[T * N];
        for (int i = 0; i < all.Length; i++) all[i] = 1;
        var allBuf = gpu.CreateBuffer<float>(all);
        var dvBuf = gpu.CreateBuffer<float>(raw[DollarVolume]);
        Run(rankCs, 0, dvBuf, allBuf, rankDays);
        var dvRank = new float[T * N];
        gpu.Read<float>(outBuf, dvRank);
        var mask = UniverseMask(logClose, dvRank, raw[DollarVolume], T, N, topK, minPrice);

        var maskGpu = gpu.CreateBuffer<float>(mask);
        var ranks = new float[Names.Length][];
        for (int f = 0; f < Names.Length; f++)
        {
            var v = gpu.CreateBuffer<float>(raw[f]);
            Run(rankCs, 0, v, maskGpu, rankDays);
            ranks[f] = new float[T * N];
            gpu.Read<float>(outBuf, ranks[f]);
        }
        return new Result(raw, mask, ranks, gpu.AdapterName, sw.Elapsed.TotalSeconds);
    }

    /// <summary>CPU reference (same logic, double accumulation); also the fallback where Direct3D is unavailable.</summary>
    public static Result RunCpu(float[] logClose, float[] volume, int T, int N, int topK, float minPrice)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var raw = new float[Names.Length][];
        for (int f = 0; f < Names.Length; f++)
        {
            var o = raw[f] = new float[T * N];
            int ff = f;
            Parallel.For(0, T, t => { for (int n = 0; n < N; n++) o[t * N + n] = CpuFeature(ff, t, n, logClose, volume, N); });
        }
        var ones = new float[T * N];
        Array.Fill(ones, 1f);
        var dvRank = CpuRank(raw[DollarVolume], ones, T, N);
        var mask = UniverseMask(logClose, dvRank, raw[DollarVolume], T, N, topK, minPrice);
        var ranks = raw.Select(r => CpuRank(r, mask, T, N)).ToArray();
        return new Result(raw, mask, ranks, "CPU", sw.Elapsed.TotalSeconds);
    }

    private static float Lc(float[] lc, int t, int n, int N) => t < 0 ? float.NaN : lc[t * N + n];

    public static float CpuFeature(int f, int t, int n, float[] lc, float[] v, int N)
    {
        float c = Lc(lc, t, n, N);
        if (!float.IsFinite(c)) return float.NaN;
        switch (f)
        {
            case <= 3:
            {
                int k = f switch { 0 => 5, 1 => 21, 2 => 63, _ => 126 };
                float p = Lc(lc, t - k, n, N);
                return float.IsFinite(p) ? c - p : float.NaN;
            }
            case 4:
            {
                float a = Lc(lc, t - 21, n, N), b = Lc(lc, t - 252, n, N);
                return float.IsFinite(a) && float.IsFinite(b) ? a - b : float.NaN;
            }
            case 5 or 6:
            {
                int k = f == 5 ? 20 : 60;
                double s = 0, s2 = 0;
                for (int j = 0; j < k; j++)
                {
                    float r = Lc(lc, t - j, n, N) - Lc(lc, t - j - 1, n, N);
                    if (!float.IsFinite(r)) return float.NaN;
                    s += r; s2 += (double)r * r;
                }
                double m = s / k;
                return (float)Math.Sqrt(Math.Max(s2 / k - m * m, 0));
            }
            case 7:
            {
                if (t < 251) return float.NaN;
                float hi = c;
                for (int j = 1; j < 252; j++) { float p = Lc(lc, t - j, n, N); if (float.IsFinite(p)) hi = Math.Max(hi, p); }
                return c - hi;
            }
            case 8:
            {
                float x = v[t * N + n];
                if (!float.IsFinite(x) || x <= 0 || t < 60) return float.NaN;
                double s = 0; int cnt = 0;
                for (int j = 1; j <= 60; j++) { float p = v[(t - j) * N + n]; if (float.IsFinite(p) && p > 0) { s += Math.Log(p); cnt++; } }
                return cnt < 40 ? float.NaN : (float)(Math.Log(x) - s / cnt);
            }
            case 9:
            {
                if (t < 19) return float.NaN;
                double s = 0;
                for (int j = 0; j < 20; j++)
                {
                    float p = Lc(lc, t - j, n, N), x = v[(t - j) * N + n];
                    if (!float.IsFinite(p) || !float.IsFinite(x)) return float.NaN;
                    s += Math.Exp(p) * x;
                }
                return s > 0 ? (float)Math.Log(s / 20) : float.NaN;
            }
            default: return float.NaN;
        }
    }

    /// <summary>Same rank definition as the HLSL kernel, by sorting (O(N log N) per day).</summary>
    public static float[] CpuRank(float[] values, float[] mask, int T, int N)
    {
        var o = new float[T * N];
        Array.Fill(o, float.NaN);
        Parallel.For(0, T, t =>
        {
            var idx = new List<int>(N);
            for (int n = 0; n < N; n++) { int i = t * N + n; if (mask[i] > 0.5f && float.IsFinite(values[i])) idx.Add(i); }
            int cnt = idx.Count;
            if (cnt < 2) return;
            idx.Sort((a, b) => values[a].CompareTo(values[b]));
            for (int s = 0; s < cnt;)
            {
                int e = s;
                while (e + 1 < cnt && values[idx[e + 1]] == values[idx[s]]) e++;
                float r = (s + 0.5f * (e - s)) / (cnt - 1);   // less = s, equal-others = e - s
                for (int k = s; k <= e; k++) o[idx[k]] = r;
                s = e + 1;
            }
        });
        return o;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct Params { public uint N, T, T0, Feature; }
}
