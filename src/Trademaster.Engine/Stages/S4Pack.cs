using System.IO.Hashing;
using System.Runtime.InteropServices;
using Trademaster.Core;

namespace Trademaster.Engine.Stages;

/// <summary>Feature contract a model was trained with. Inference refuses a model whose spec does not match.</summary>
public sealed class FeatureSpec
{
    public required string[] Names { get; init; }
    public required float[] Mean { get; init; }
    public required float[] Std { get; init; }
    public required int Window { get; init; }

    public static FeatureSpec Identity(int window) => new()
    {
        Names = F.Names, Mean = new float[F.Count], Std = Enumerable.Repeat(1f, F.Count).ToArray(), Window = window,
    };

    public void CheckCompatible()
    {
        if (!Names.SequenceEqual(F.Names))
            throw new InvalidOperationException($"Model feature list [{string.Join(",", Names)}] does not match engine [{string.Join(",", F.Names)}].");
        if (Mean.Length != F.Count || Std.Length != F.Count || Std.Any(s => !(s > 0)))
            throw new InvalidOperationException("Model normalization stats are malformed.");
    }
}

public enum Device { Cpu }

/// <summary>
/// Model input owned by stage 4. Layout: ring buffer X[slot, instrument, feature] with <see cref="Head"/>
/// the newest slot. Slot s = (Head - lag + W) % W holds the cutoff <c>lag</c> cycles ago.
/// </summary>
public sealed class ModelInputBatch
{
    public required int N { get; init; }
    public required int W { get; init; }
    public required int Fdim { get; init; }
    public required float[] X { get; init; }
    public required byte[] Mask { get; init; }
    public required float[] Quality { get; init; }
    public Device Device { get; init; } = Device.Cpu;
    public long CycleId;
    public long CutoffTs;
    public int Head = -1;
    public int Filled;
    public ulong InputHash;

    public int Slot(int lag) => (Head - lag + W) % W;
    public int Offset(int slot, int instrument) => (slot * N + instrument) * Fdim;
}

/// <summary>
/// Cached cross-sectional statistics estimated from observed returns (provenance: "estimated").
/// Exponentially weighted beta of each stock's one-cycle return to its reference ETF.
/// </summary>
public sealed class ObservedStats
{
    private readonly double _alpha;
    public readonly int[] RefEtf;
    public readonly double[] Cov, VarRef, Beta;

    public ObservedStats(int[] refEtf, int halflifeCycles)
    {
        RefEtf = refEtf;
        _alpha = 1 - Math.Pow(0.5, 1.0 / Math.Max(1, halflifeCycles));
        Cov = new double[refEtf.Length]; VarRef = new double[refEtf.Length];
        Beta = Enumerable.Repeat(1.0, refEtf.Length).ToArray();
    }

    public void Update(ModelInputBatch b)
    {
        int slot = b.Slot(0);
        for (int i = 0; i < b.N; i++)
        {
            int j = RefEtf[i];
            if (j < 0 || b.Mask[i] == 0 || b.Mask[j] == 0) continue;
            double ri = b.X[b.Offset(slot, i) + F.R1], rj = b.X[b.Offset(slot, j) + F.R1];
            Cov[i] = (1 - _alpha) * Cov[i] + _alpha * ri * rj;
            VarRef[i] = (1 - _alpha) * VarRef[i] + _alpha * rj * rj;
            if (VarRef[i] > 1e-14) Beta[i] = Math.Clamp(Cov[i] / VarRef[i], -3, 3);
        }
    }
}

/// <summary>
/// Stage 4 — cross-sectional assembly and packing. Writes only the newest time slice into the ring
/// buffer (no shifting), applies frozen normalization from the model's feature spec, builds the mask,
/// updates the observed-statistics cache and hashes the batch for the audit record.
/// </summary>
public sealed class S4Pack
{
    private readonly S2Evaluate _s2;
    private readonly S3Features _s3;
    private readonly FeatureSpec _spec;
    private readonly float[] _scratch = new float[F.Count];

    public ModelInputBatch Batch { get; }
    public ObservedStats Stats { get; }

    public S4Pack(int instruments, FeatureSpec spec, S2Evaluate s2, S3Features s3, int[] refEtf, int betaHalflifeCycles)
    {
        spec.CheckCompatible();
        _spec = spec;
        _s2 = s2;
        _s3 = s3;
        Batch = new ModelInputBatch
        {
            N = instruments, W = spec.Window, Fdim = F.Count,
            X = new float[spec.Window * instruments * F.Count],
            Mask = new byte[instruments], Quality = new float[instruments],
        };
        Stats = new ObservedStats(refEtf, betaHalflifeCycles);
    }

    public ModelInputBatch Pack(long cycleId, long cutoff, bool marketOpen)
    {
        var b = Batch;
        b.CycleId = cycleId;
        b.CutoffTs = cutoff;
        b.Head = (b.Head + 1) % b.W;
        b.Filled = Math.Min(b.W, b.Filled + 1);
        int slot = b.Head;
        for (int i = 0; i < b.N; i++)
        {
            var dst = b.X.AsSpan(b.Offset(slot, i), F.Count);
            bool valid = _s2.Mask[i] == MaskReason.Valid;
            b.Mask[i] = valid ? (byte)1 : (byte)0;
            b.Quality[i] = _s2.Quality[i];
            if (!valid) { dst.Clear(); continue; }
            _s3.Read(i, cutoff, _scratch);
            for (int f = 0; f < F.Count; f++) dst[f] = (_scratch[f] - _spec.Mean[f]) / _spec.Std[f];
        }
        _s3.EndCycle();
        if (marketOpen) Stats.Update(b);

        var h = new XxHash3();
        h.Append(MemoryMarshal.AsBytes(b.X.AsSpan()));
        h.Append(b.Mask);
        Span<byte> meta = stackalloc byte[8];
        BitConverter.TryWriteBytes(meta, b.Head);
        BitConverter.TryWriteBytes(meta[4..], b.Filled);
        h.Append(meta);
        b.InputHash = h.GetCurrentHashAsUInt64();
        return b;
    }
}
