namespace Trademaster.Gpu;

/// <summary>
/// v6 model M: an F → H → H → 1 ReLU network trained with MSE + AdamW for several seeds at once on the GPU
/// (Shaders/mlp.hlsl), early-stopped per seed on a validation range, predictions averaged over seeds.
/// <see cref="Cpu"/> runs the identical arithmetic on the CPU (reference for the parity test, and fallback).
/// </summary>
public static class GpuMlp
{
    public sealed record Options(int Hidden = 64, int Seeds = 10, int Batch = 4096, int MaxEpochs = 20, int Patience = 3,
        float Lr = 1e-3f, float WeightDecay = 1e-4f);

    public static int ParamCount(int f, int h) => f * h + h * h + 3 * h + 1;

    /// <summary>He-normal weights, zero biases; deterministic per seed.</summary>
    public static float[] Init(int f, int h, int seed)
    {
        var rng = new Random(seed);
        var p = new float[ParamCount(f, h)];
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        void Fill(int off, int n, int fanIn) { for (int i = 0; i < n; i++) p[off + i] = (float)(Normal() * Math.Sqrt(2.0 / fanIn)); }
        Fill(0, f * h, f);                       // W1
        Fill(f * h + h, h * h, h);               // W2
        Fill(f * h + 2 * h + h * h, h, h);       // W3
        return p;
    }

    /// <summary>The operations a backend provides (all seeds at once).</summary>
    public interface IBackend : IDisposable
    {
        string Device { get; }
        /// <summary>Uploads parameters [S][NP] and resets the Adam state.</summary>
        void Reset(float[] parameters);
        /// <summary>Per-seed shuffled row lists, [S][nIdx] flattened.</summary>
        void SetIndices(uint[] idx, int nIdx);
        /// <summary>One AdamW step on the batch at offset <paramref name="off"/> of every seed's index list.</summary>
        void Step(int off, int t);
        float[] ReadParameters();
        /// <summary>Outputs [S][rn] for rows r0..r0+rn-1 using the current parameters.</summary>
        float[] Predict(int r0, int rn);
    }

    /// <summary>
    /// Trains on rows [train0, train1) with a finite target, early-stops each seed on rows [val0, val1),
    /// then returns the seed-averaged predictions for rows [pred0, pred1).
    /// </summary>
    public static float[] FitPredict(IBackend be, float[] y, int f, (int, int) train, (int, int) val, (int, int) pred,
        Options o, int seedBase, TextWriter? log = null)
    {
        int S = o.Seeds, H = o.Hidden, NP = ParamCount(f, H);
        var init = new float[S * NP];
        for (int s = 0; s < S; s++) Array.Copy(Init(f, H, seedBase + s), 0, init, s * NP, NP);
        be.Reset(init);
        var rows = Enumerable.Range(train.Item1, train.Item2 - train.Item1).Where(r => float.IsFinite(y[r])).ToArray();
        int nIdx = rows.Length, batches = nIdx / o.Batch;
        if (batches == 0) throw new InvalidOperationException("fewer training rows than one batch");
        var best = (float[])init.Clone();
        var bestLoss = Enumerable.Repeat(double.PositiveInfinity, S).ToArray();
        var since = new int[S];
        int t = 0, epoch = 0;
        var idx = new uint[S * nIdx];
        for (; epoch < o.MaxEpochs && since.Any(x => x < o.Patience); epoch++)
        {
            for (int s = 0; s < S; s++)
            {
                var rng = new Random(seedBase * 7919 + s * 101 + epoch);
                var perm = (int[])rows.Clone();
                for (int i = perm.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (perm[i], perm[j]) = (perm[j], perm[i]); }
                for (int i = 0; i < nIdx; i++) idx[s * nIdx + i] = (uint)perm[i];
            }
            be.SetIndices(idx, nIdx);
            for (int b = 0; b < batches; b++) be.Step(b * o.Batch, ++t);
            var loss = ValLoss(be, y, val, S);
            var p = be.ReadParameters();
            for (int s = 0; s < S; s++)
            {
                if (loss[s] < bestLoss[s]) { bestLoss[s] = loss[s]; since[s] = 0; Array.Copy(p, s * NP, best, s * NP, NP); }
                else since[s]++;
            }
        }
        be.Reset(best);
        var outp = new float[pred.Item2 - pred.Item1];
        var all = be.Predict(pred.Item1, outp.Length);
        for (int i = 0; i < outp.Length; i++)
        {
            double sum = 0;
            for (int s = 0; s < S; s++) sum += all[s * outp.Length + i];
            outp[i] = (float)(sum / S);
        }
        log?.WriteLine($"    {nIdx:N0} training rows, {epoch} epochs, best validation MSE {bestLoss.Average():F5} (seed range {bestLoss.Min():F5}..{bestLoss.Max():F5})");
        return outp;
    }

    private static double[] ValLoss(IBackend be, float[] y, (int, int) val, int S)
    {
        int n = val.Item2 - val.Item1;
        var p = be.Predict(val.Item1, n);
        var loss = new double[S];
        for (int s = 0; s < S; s++)
        {
            double sum = 0; int c = 0;
            for (int i = 0; i < n; i++)
            {
                float yy = y[val.Item1 + i];
                if (!float.IsFinite(yy)) continue;
                double d = p[s * n + i] - yy;
                sum += d * d; c++;
            }
            loss[s] = c > 0 ? sum / c : double.PositiveInfinity;
        }
        return loss;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct Params
    {
        public uint S, B, F, H, NP, Off, NIdx, R0, RN;
        public float Lr, Wd, C1, C2;
    }

    private static Params Make(int S, int B, int F, int H, Options o) => new()
    {
        S = (uint)S, B = (uint)B, F = (uint)F, H = (uint)H, NP = (uint)ParamCount(F, H), Lr = o.Lr, Wd = o.WeightDecay,
    };

    // ------------------------------------------------------------------ GPU backend

    public sealed class Gpu : IBackend
    {
        private const int PredictChunk = 65536;
        private readonly D3D11Compute _gpu;
        private readonly Options _o;
        private readonly int _f;
        private Params _p;
        private readonly nint _cb, _fwd1, _fwd2, _outK, _back2, _back1, _grad, _predict;
        private readonly D3D11Compute.GpuBuffer _x, _y, _par, _act, _dlt, _mv, _pout;
        private readonly D3D11Compute.GpuBuffer _idx;
        public string Device => _gpu.AdapterName;

        public Gpu(float[] x, float[] y, int f, Options o, int adapter = -1)
        {
            if (o.Hidden > 64) throw new ArgumentException("hidden width is limited to 64 by the Predict kernel");
            _o = o; _f = f;
            _gpu = new D3D11Compute(adapter);
            nint K(string e) => _gpu.CompileShader("mlp.hlsl", e);
            (_fwd1, _fwd2, _outK, _back2, _back1, _grad, _predict) = (K("Forward1"), K("Forward2"), K("Output"), K("Back2"), K("Back1"), K("GradAdam"), K("Predict"));
            int S = o.Seeds, B = o.Batch, H = o.Hidden, NP = ParamCount(f, H);
            _p = Make(S, B, f, H, o);
            _cb = _gpu.CreateConstants(_p);
            _x = _gpu.CreateBuffer<float>(x);
            _y = _gpu.CreateBuffer<float>(y.Select(v => float.IsFinite(v) ? v : 0f).ToArray());
            _par = _gpu.CreateBuffer(S * NP);
            _act = _gpu.CreateBuffer(2 * S * B * H);
            _dlt = _gpu.CreateBuffer(S * B * (2 * H + 1));
            _mv = _gpu.CreateBuffer(S * NP * 2);
            _pout = _gpu.CreateBuffer(S * PredictChunk);
            _idx = _gpu.CreateBuffer(S * (x.Length / f));             // room for every row, per seed
        }

        public void Reset(float[] parameters)
        {
            _gpu.Write<float>(_par, parameters);
            _gpu.Write<float>(_mv, new float[_mv.Length]);
        }

        public void SetIndices(uint[] idx, int nIdx)
        {
            _gpu.Write<uint>(_idx, idx);
            _p.NIdx = (uint)nIdx;
        }

        private D3D11Compute.GpuBuffer[] Train => [_x, _y, _idx, _par, _act, _dlt, _mv];

        public void Step(int off, int t)
        {
            int S = _o.Seeds, B = _o.Batch, H = _o.Hidden, NP = ParamCount(_f, H);
            _p.Off = (uint)off;
            _p.C1 = 1 - MathF.Pow(0.9f, t);
            _p.C2 = 1 - MathF.Pow(0.999f, t);
            _gpu.UpdateConstants(_cb, _p);
            var u = Train;
            uint bh = (uint)((B * H + 255) / 256), bb = (uint)((B + 255) / 256);
            _gpu.Dispatch(_fwd1, _cb, bh, (uint)S, u);
            _gpu.Dispatch(_fwd2, _cb, bh, (uint)S, u);
            _gpu.Dispatch(_outK, _cb, bb, (uint)S, u);
            _gpu.Dispatch(_back2, _cb, bh, (uint)S, u);
            _gpu.Dispatch(_back1, _cb, bh, (uint)S, u);
            _gpu.Dispatch(_grad, _cb, (uint)((NP + 63) / 64), (uint)S, u);
            if (t % 16 == 0) _gpu.Flush();
        }

        public float[] ReadParameters()
        {
            var p = new float[_par.Length];
            _gpu.Read<float>(_par, p);
            return p;
        }

        public float[] Predict(int r0, int rn)
        {
            int S = _o.Seeds;
            var all = new float[S * rn];
            var chunk = new float[S * PredictChunk];
            for (int c0 = 0; c0 < rn; c0 += PredictChunk)
            {
                int n = Math.Min(PredictChunk, rn - c0);
                _p.R0 = (uint)(r0 + c0); _p.RN = (uint)n;
                _gpu.UpdateConstants(_cb, _p);
                _gpu.Dispatch(_predict, _cb, (uint)((n + 255) / 256), (uint)S, _x, _pout, _idx, _par);
                _gpu.Read<float>(_pout, chunk);
                for (int s = 0; s < S; s++) Array.Copy(chunk, s * n, all, s * rn + c0, n);
            }
            return all;
        }

        public void Dispose() => _gpu.Dispose();
    }

    // ------------------------------------------------------------------ CPU reference backend

    /// <summary>Same arithmetic as mlp.hlsl in float, same summation order; single-threaded per seed.</summary>
    public sealed class Cpu(float[] x, float[] y, int f, Options o) : IBackend
    {
        private readonly int S = o.Seeds, B = o.Batch, H = o.Hidden, NP = ParamCount(f, o.Hidden);
        private float[] _p = [], _m = [], _v = [];
        private uint[] _idx = [];
        private int _nIdx;
        public string Device => "CPU";

        public void Reset(float[] parameters) { _p = (float[])parameters.Clone(); _m = new float[_p.Length]; _v = new float[_p.Length]; }
        public void SetIndices(uint[] idx, int nIdx) { _idx = idx; _nIdx = nIdx; }
        public float[] ReadParameters() => (float[])_p.Clone();

        public void Step(int off, int t)
        {
            float c1 = 1 - MathF.Pow(0.9f, t), c2 = 1 - MathF.Pow(0.999f, t);
            Parallel.For(0, S, s =>
            {
                int p = s * NP, oB1 = f * H, oW2 = oB1 + H, oB2 = oW2 + H * H, oW3 = oB2 + H, oB3 = oW3 + H;
                var a1 = new float[B * H]; var a2 = new float[B * H]; var d1 = new float[B * H]; var d2 = new float[B * H]; var dO = new float[B];
                var rows = new int[B];
                for (int b = 0; b < B; b++) rows[b] = (int)_idx[s * _nIdx + off + b];
                for (int b = 0; b < B; b++)
                    for (int j = 0; j < H; j++)
                    {
                        float z = _p[p + oB1 + j];
                        for (int k = 0; k < f; k++) z += x[rows[b] * f + k] * _p[p + k * H + j];
                        a1[b * H + j] = MathF.Max(z, 0);
                    }
                for (int b = 0; b < B; b++)
                    for (int k = 0; k < H; k++)
                    {
                        float z = _p[p + oB2 + k];
                        for (int j = 0; j < H; j++) z += a1[b * H + j] * _p[p + oW2 + j * H + k];
                        a2[b * H + k] = MathF.Max(z, 0);
                    }
                for (int b = 0; b < B; b++)
                {
                    float z = _p[p + oB3];
                    for (int k = 0; k < H; k++) z += a2[b * H + k] * _p[p + oW3 + k];
                    dO[b] = 2f * (z - y[rows[b]]) / B;
                }
                for (int b = 0; b < B; b++)
                    for (int k = 0; k < H; k++)
                    {
                        float d = dO[b] * _p[p + oW3 + k];
                        d2[b * H + k] = a2[b * H + k] > 0 ? d : 0;
                    }
                for (int b = 0; b < B; b++)
                    for (int j = 0; j < H; j++)
                    {
                        float d = 0;
                        for (int k = 0; k < H; k++) d += d2[b * H + k] * _p[p + oW2 + j * H + k];
                        d1[b * H + j] = a1[b * H + j] > 0 ? d : 0;
                    }
                var grad = new float[NP];
                var isWeight = new bool[NP];
                for (int q = 0; q < NP; q++)
                {
                    float gr = 0; bool w = true;
                    if (q < oB1) { int k = q / H, j = q % H; for (int b = 0; b < B; b++) gr += x[rows[b] * f + k] * d1[b * H + j]; }
                    else if (q < oW2) { int j = q - oB1; w = false; for (int b = 0; b < B; b++) gr += d1[b * H + j]; }
                    else if (q < oB2) { int j = (q - oW2) / H, k = (q - oW2) % H; for (int b = 0; b < B; b++) gr += a1[b * H + j] * d2[b * H + k]; }
                    else if (q < oW3) { int k = q - oB2; w = false; for (int b = 0; b < B; b++) gr += d2[b * H + k]; }
                    else if (q < oB3) { int k = q - oW3; for (int b = 0; b < B; b++) gr += a2[b * H + k] * dO[b]; }
                    else { w = false; for (int b = 0; b < B; b++) gr += dO[b]; }
                    grad[q] = gr; isWeight[q] = w;
                }
                for (int q = 0; q < NP; q++)
                {
                    int i = p + q;
                    float m = 0.9f * _m[i] + 0.1f * grad[q];
                    float v = 0.999f * _v[i] + 0.001f * grad[q] * grad[q];
                    _m[i] = m; _v[i] = v;
                    float w = _p[i];
                    if (isWeight[q]) w -= o.Lr * o.WeightDecay * w;
                    _p[i] = w - o.Lr * (m / c1) / (MathF.Sqrt(v / c2) + 1e-8f);
                }
            });
        }

        public float[] Predict(int r0, int rn)
        {
            var all = new float[S * rn];
            Parallel.For(0, S, s =>
            {
                int p = s * NP, oB1 = f * H, oW2 = oB1 + H, oB2 = oW2 + H * H, oW3 = oB2 + H, oB3 = oW3 + H;
                var h1 = new float[H];
                for (int i = 0; i < rn; i++)
                {
                    int r = r0 + i;
                    for (int j = 0; j < H; j++)
                    {
                        float z = _p[p + oB1 + j];
                        for (int k = 0; k < f; k++) z += x[r * f + k] * _p[p + k * H + j];
                        h1[j] = MathF.Max(z, 0);
                    }
                    float outv = _p[p + oB3];
                    for (int k = 0; k < H; k++)
                    {
                        float z = _p[p + oB2 + k];
                        for (int j = 0; j < H; j++) z += h1[j] * _p[p + oW2 + j * H + k];
                        outv += MathF.Max(z, 0) * _p[p + oW3 + k];
                    }
                    all[s * rn + i] = outv;
                }
            });
            return all;
        }

        public void Dispose() { }
    }
}
