namespace Trademaster.Gpu;

/// <summary>
/// The v6 feature panel: per-stock features on a [T, N] grid (Shaders/features.hlsl), a point-in-time liquid
/// universe, industry (cross-stock) aggregates, and cross-sectional percentile ranks inside the universe.
/// One pipeline (<see cref="Build"/>) runs on either backend: the GPU (HLSL) or the exact-logic CPU reference.
/// </summary>
public static class GpuFeatures
{
    /// <summary>Per-stock features computed by FeatureKernel, in kernel id order.</summary>
    public static readonly string[] StockFeatures =
        ["mom5", "mom21", "mom63", "mom126", "mom12_1", "vol20", "vol60", "dist_252_high", "volume_surprise", "log_dollar_vol20",
         "beta252", "resid_mom12_1", "max_ret21", "idio_vol60"];

    /// <summary>Industry features (need an industry code per stock).</summary>
    public static readonly string[] IndustryFeatures = ["ind_mom21", "ind_mom126", "ind_breadth21", "ind_leader_mom5", "mom21_minus_ind"];

    public const int Mom5 = 0, Mom21 = 1, Mom126 = 3, DollarVolume = 9;

    /// <summary>Panel inputs. Arrays are [T, N] row-major unless noted.</summary>
    public sealed record Inputs(int T, int N, float[] LogClose, float[] Volume, float[] MarketLogClose /* [T] */,
        float[] Eligible /* [N], 1 = operating company */, float[]? Industry /* [N], < 0 = unknown */,
        IReadOnlyList<(string Name, float[] Values)> Extras, int TopK, float MinPrice);

    public sealed record Result(string[] Names, float[][] Raw, float[] Mask, float[][] Ranks, string Device, double Seconds);

    /// <summary>The operations a backend provides. Each returns a new [T, N] array.</summary>
    public interface IBackend : IDisposable
    {
        string Device { get; }
        float[] Feature(int id);
        float[] Rank(float[] values, float[] mask);
        float[] Industry(int op, float[] values, float[] mask, float[] leader);
    }

    public static Result Build(Inputs x, IBackend be, TextWriter? log = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int T = x.T, N = x.N;
        var names = new List<string>();
        var raw = new List<float[]>();
        for (int f = 0; f < StockFeatures.Length; f++)
        {
            names.Add(StockFeatures[f]);
            raw.Add(be.Feature(f));
        }
        log?.WriteLine($"  {StockFeatures.Length} stock features: {sw.Elapsed.TotalSeconds:F1} s");

        // Universe: eligible stocks, price >= MinPrice, top-K by 20-day dollar volume, using data up to day t only.
        var eligible = new float[T * N];
        for (int t = 0; t < T; t++) Array.Copy(x.Eligible, 0, eligible, t * N, N);
        var dvAll = be.Rank(raw[DollarVolume], eligible);
        var mask = UniverseMask(x.LogClose, dvAll, T, N, x.TopK, x.MinPrice);

        if (x.Industry is not null)
        {
            var dvUni = be.Rank(raw[DollarVolume], mask);
            var leader = new float[T * N];
            for (int i = 0; i < leader.Length; i++) leader[i] = dvUni[i] >= 2f / 3f ? 1 : 0;   // NaN compares false
            var indMom21 = be.Industry(0, raw[Mom21], mask, leader);
            names.AddRange(IndustryFeatures);
            raw.Add(indMom21);
            raw.Add(be.Industry(0, raw[Mom126], mask, leader));
            raw.Add(be.Industry(1, raw[Mom21], mask, leader));
            raw.Add(be.Industry(2, raw[Mom5], mask, leader));
            var rel = new float[T * N];
            for (int i = 0; i < rel.Length; i++) rel[i] = raw[Mom21][i] - indMom21[i];
            raw.Add(rel);
            log?.WriteLine($"  industry features: {sw.Elapsed.TotalSeconds:F1} s");
        }
        foreach (var (name, values) in x.Extras)
        {
            names.Add(name);
            raw.Add(values);
        }
        var ranks = raw.Select(r => be.Rank(r, mask)).ToArray();
        log?.WriteLine($"  {ranks.Length} cross-sectional ranks: {sw.Elapsed.TotalSeconds:F1} s");
        return new Result(names.ToArray(), raw.ToArray(), mask, ranks, be.Device, sw.Elapsed.TotalSeconds);
    }

    /// <summary>
    /// Universe at day t: rank (among eligible stocks with data) puts the stock in the top <paramref name="topK"/> by
    /// 20-day average dollar volume, and price ≥ minPrice.
    /// </summary>
    public static float[] UniverseMask(float[] logClose, float[] dollarVolRank, int T, int N, int topK, float minPrice)
    {
        var mask = new float[T * N];
        float minLog = MathF.Log(minPrice);
        for (int t = 0; t < T; t++)
        {
            int count = 0;
            for (int n = 0; n < N; n++) if (float.IsFinite(dollarVolRank[t * N + n])) count++;
            if (count < 2) continue;
            for (int n = 0; n < N; n++)
            {
                int i = t * N + n;
                float r = dollarVolRank[i];   // stocks above = (1 - r) * (count - 1)
                if (float.IsFinite(r) && (1 - r) * (count - 1) < topK - 0.5f && logClose[i] >= minLog) mask[i] = 1;
            }
        }
        return mask;
    }

    // ------------------------------------------------------------------ GPU backend

    /// <summary>HLSL backend. Every dispatch covers a slice of days sized to stay far below the 2 s driver limit.</summary>
    public sealed class Gpu : IBackend
    {
        private readonly D3D11Compute _gpu;
        private readonly Inputs _x;
        private readonly nint _featureCs, _rankCs, _industryCs, _cb;
        private readonly D3D11Compute.GpuBuffer _lc, _vol, _market, _industry, _out, _a, _b, _d;
        private readonly int _featDays, _rankDays;
        public string Device => _gpu.AdapterName;

        public Gpu(Inputs x, int adapter = -1)
        {
            _x = x;
            int T = x.T, N = x.N;
            _gpu = new D3D11Compute(adapter);
            _featureCs = _gpu.CompileShader("features.hlsl", "FeatureKernel");
            _rankCs = _gpu.CompileShader("features.hlsl", "RankKernel");
            _industryCs = _gpu.CompileShader("features.hlsl", "IndustryKernel");
            _lc = _gpu.CreateBuffer<float>(x.LogClose);
            _vol = _gpu.CreateBuffer<float>(x.Volume);
            _market = _gpu.CreateBuffer<float>(x.MarketLogClose);
            var ind = x.Industry ?? Enumerable.Repeat(-1f, N).ToArray();
            _industry = _gpu.CreateBuffer<float>(ind);
            _out = _gpu.CreateBuffer(T * N);
            _a = _gpu.CreateBuffer(T * N);
            _b = _gpu.CreateBuffer(T * N);
            _d = _gpu.CreateBuffer(T * N);
            _cb = _gpu.CreateConstants(new Params());
            // Feature threads do up to ~3 x 252 steps; rank/industry threads do N steps.
            _featDays = Math.Max(1, (int)Math.Min(D3D11Compute.MaxGroupsPerDispatch, 20_000_000L / (N * 252L)));
            _rankDays = Math.Max(1, (int)Math.Min(D3D11Compute.MaxGroupsPerDispatch, 2_000_000_000L / ((long)N * N)));
        }

        private float[] Run(nint shader, int id, int days, params D3D11Compute.GpuBuffer[] uavs)
        {
            int T = _x.T, N = _x.N;
            for (int t0 = 0; t0 < T; t0 += days)
            {
                _gpu.UpdateConstants(_cb, new Params { N = (uint)N, T = (uint)T, T0 = (uint)t0, Feature = (uint)id });
                _gpu.Dispatch(shader, _cb, (uint)((N + 255) / 256), (uint)Math.Min(days, T - t0), uavs);
                _gpu.Flush();
            }
            var o = new float[T * N];
            _gpu.Read<float>(_out, o);
            return o;
        }

        public float[] Feature(int id) => Run(_featureCs, id, _featDays, _lc, _vol, _out, _market);

        public float[] Rank(float[] values, float[] mask)
        {
            _gpu.Write(_a, values);
            _gpu.Write(_b, mask);
            return Run(_rankCs, 0, _rankDays, _a, _b, _out);
        }

        public float[] Industry(int op, float[] values, float[] mask, float[] leader)
        {
            _gpu.Write(_a, values);
            _gpu.Write(_b, mask);
            _gpu.Write(_d, leader);
            return Run(_industryCs, op, _rankDays, _a, _b, _out, _industry, _d);
        }

        public void Dispose() => _gpu.Dispose();
    }

    // ------------------------------------------------------------------ CPU reference backend

    /// <summary>Same logic as the HLSL kernels with double accumulation; also the fallback without Direct3D.</summary>
    public sealed class Cpu(Inputs x) : IBackend
    {
        public string Device => "CPU";
        public float[] Feature(int id)
        {
            var o = new float[x.T * x.N];
            Parallel.For(0, x.T, t => { for (int n = 0; n < x.N; n++) o[t * x.N + n] = CpuFeature(id, t, n, x.LogClose, x.Volume, x.MarketLogClose, x.N); });
            return o;
        }
        public float[] Rank(float[] values, float[] mask) => CpuRank(values, mask, x.T, x.N);
        public float[] Industry(int op, float[] values, float[] mask, float[] leader) =>
            CpuIndustry(op, values, mask, leader, x.Industry ?? Enumerable.Repeat(-1f, x.N).ToArray(), x.T, x.N);
        public void Dispose() { }
    }

    private static float Lc(float[] lc, int t, int n, int N) => t < 0 ? float.NaN : lc[t * N + n];
    private static float Mk(float[] m, int t) => t < 0 ? float.NaN : m[t];

    private static double Beta252(int t, int n, float[] lc, float[] m, int N)
    {
        double sr = 0, sm = 0, srm = 0, smm = 0; int cnt = 0;
        for (int j = 0; j < 252; j++)
        {
            float r = Lc(lc, t - j, n, N) - Lc(lc, t - j - 1, n, N), mr = Mk(m, t - j) - Mk(m, t - j - 1);
            if (!float.IsFinite(r) || !float.IsFinite(mr)) continue;
            sr += r; sm += mr; srm += (double)r * mr; smm += (double)mr * mr; cnt++;
        }
        if (cnt < 200) return double.NaN;
        double vm = smm / cnt - sm / cnt * (sm / cnt);
        return vm > 0 ? (srm / cnt - sr / cnt * (sm / cnt)) / vm : double.NaN;
    }

    private static (double Mean, double Sd, int Count) Resid(int t, int n, double b, int j0, int j1, float[] lc, float[] m, int N)
    {
        double s = 0, s2 = 0; int cnt = 0;
        for (int j = j0; j < j1; j++)
        {
            float r = Lc(lc, t - j, n, N) - Lc(lc, t - j - 1, n, N), mr = Mk(m, t - j) - Mk(m, t - j - 1);
            if (!float.IsFinite(r) || !float.IsFinite(mr)) continue;
            double e = r - b * mr;
            s += e; s2 += e * e; cnt++;
        }
        double mean = cnt > 0 ? s / cnt : double.NaN;
        return (mean, cnt > 1 ? Math.Sqrt(Math.Max(s2 / cnt - mean * mean, 0)) : double.NaN, cnt);
    }

    public static float CpuFeature(int f, int t, int n, float[] lc, float[] v, float[] market, int N)
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
            case 10: return (float)Beta252(t, n, lc, market, N);
            case 11 or 13:
            {
                double b = Beta252(t, n, lc, market, N);
                if (double.IsNaN(b)) return float.NaN;
                if (f == 11)
                {
                    var (m, sd, cnt) = Resid(t, n, b, 21, 252, lc, market, N);
                    return cnt < 150 || !(sd > 0) ? float.NaN : (float)(m / sd);
                }
                var (_, sd2, cnt2) = Resid(t, n, b, 0, 60, lc, market, N);
                return cnt2 < 40 ? float.NaN : (float)sd2;
            }
            case 12:
            {
                float hi = float.MinValue;
                for (int j = 0; j < 21; j++)
                {
                    float r = Lc(lc, t - j, n, N) - Lc(lc, t - j - 1, n, N);
                    if (!float.IsFinite(r)) return float.NaN;
                    hi = Math.Max(hi, r);
                }
                return hi;
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

    /// <summary>Same as IndustryKernel: per-day industry mean (op 0), share above zero (1), leaders' mean excluding self (2).</summary>
    public static float[] CpuIndustry(int op, float[] values, float[] mask, float[] leader, float[] industry, int T, int N)
    {
        var o = new float[T * N];
        Parallel.For(0, T, t =>
        {
            var sum = new Dictionary<float, (double S, int C)>();
            for (int j = 0; j < N; j++)
            {
                int i = t * N + j;
                float z = values[i], g = industry[j];
                if (mask[i] <= 0.5f || g < 0 || !float.IsFinite(z) || (op == 2 && leader[i] < 0.5f)) continue;
                sum.TryGetValue(g, out var a);
                sum[g] = (a.S + (op == 1 ? (z > 0 ? 1 : 0) : z), a.C + 1);
            }
            for (int n = 0; n < N; n++)
            {
                int i = t * N + n;
                float g = industry[n];
                if (g < 0 || !sum.TryGetValue(g, out var a)) { o[i] = float.NaN; continue; }
                // op 2 excludes the stock itself if it counted
                if (op == 2 && mask[i] > 0.5f && leader[i] >= 0.5f && float.IsFinite(values[i])) a = (a.S - values[i], a.C - 1);
                o[i] = a.C < (op == 2 ? 1 : 3) ? float.NaN : op == 1 ? MathF.Round((float)(a.S / a.C) * 4096) / 4096 : (float)(a.S / a.C);
            }
        });
        return o;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct Params { public uint N, T, T0, Feature; }
}
