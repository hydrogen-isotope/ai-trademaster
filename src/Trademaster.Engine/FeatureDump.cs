using System.Text.Json;
using Trademaster.Core;
using Trademaster.Engine.Stages;

namespace Trademaster.Engine;

/// <summary>
/// Training-data export: per cycle, the raw (un-normalized) engine features of every instrument, its good mid
/// price and validity mask, exactly as stage 3 computed them at the cutoff. The Python trainer reads this, so the
/// network learns on the same features it will later see live (no train/serve skew).
///
/// features.bin (little-endian): magic "TMFD", int32 version=1, int32 N, int32 F, then per cycle:
///   int64 cutoff_ns, int32 phase (0 = closed, 1 = open), float32 mid[N] (NaN if none), uint8 mask[N], float32 x[N*F]
/// features.json: feature names, symbols, instrument types, cadence, data version.
/// </summary>
public sealed class FeatureDump : IDisposable
{
    private readonly BinaryWriter _w;
    private readonly float[] _scratch = new float[F.Count];
    private readonly int _n;

    public FeatureDump(string dir, Universe u, DataManifest manifest, int cadenceMs)
    {
        _n = u.Count;
        var fs = new FileStream(Path.Combine(dir, "features.bin"), FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
        _w = new BinaryWriter(fs);
        _w.Write("TMFD"u8);
        _w.Write(1);
        _w.Write(_n);
        _w.Write(F.Count);
        var meta = new
        {
            format = "TMFD v1", features = F.Names, n = _n, f = F.Count, cadence_ms = cadenceMs,
            symbols = u.Instruments.Select(i => i.Symbol), types = u.Instruments.Select(i => i.Type.ToString()),
            data_version = manifest.DataVersion, synthetic = manifest.Synthetic,
            note = "Contains market-derived data. Keep under private/ (never commit).",
        };
        File.WriteAllText(Path.Combine(dir, "features.json"), JsonSerializer.Serialize(meta, Json.Options));
    }

    public void Write(long cutoff, bool open, S2Evaluate s2, S3Features s3)
    {
        _w.Write(cutoff);
        _w.Write(open ? 1 : 0);
        for (int i = 0; i < _n; i++) _w.Write(s2.GoodBid[i] > 0 ? (float)s2.GoodMid(i) : float.NaN);
        for (int i = 0; i < _n; i++) _w.Write((byte)(s2.Mask[i] == MaskReason.Valid ? 1 : 0));
        for (int i = 0; i < _n; i++)
        {
            s3.Read(i, cutoff, _scratch);
            foreach (var v in _scratch) _w.Write(v);
        }
    }

    public void Dispose() => _w.Dispose();
}
