using System.Security.Cryptography;

namespace Trademaster.Core;

/// <summary>
/// Binary replay file: header (magic, version, count) followed by fixed-size little-endian records.
/// Records are stored in the order the stream is meant to be consumed (feed: RecvTs, exchange: ExchangeTs).
/// </summary>
public static class ReplayFile
{
    public const uint Magic = 0x50524D54; // "TMRP"
    public const int Version = 1;
    public const int RecordSize = 8 + 8 + 8 + 4 + 1 + 8 + 8 + 8 + 8 + 1;
    private const int HeaderSize = 4 + 4 + 8;

    public static void Write(string path, IReadOnlyList<MarketEvent> events)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
        using var w = new BinaryWriter(fs);
        w.Write(Magic);
        w.Write(Version);
        w.Write((long)events.Count);
        foreach (var e in events)
        {
            w.Write(e.Seq); w.Write(e.ExchangeTs); w.Write(e.RecvTs); w.Write(e.Instrument); w.Write((byte)e.Kind);
            w.Write(e.Px1); w.Write(e.Px2); w.Write(e.Sz1); w.Write(e.Sz2); w.Write((byte)e.Status);
        }
    }

    public static string Sha256(string path)
    {
        using var fs = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(fs));
    }
}

/// <summary>Streaming reader with one-record look-ahead, so the engine can stop exactly at a cutoff time.</summary>
public sealed class ReplayReader : IDisposable
{
    private readonly BinaryReader _r;
    private readonly long _count;
    private long _read;
    private MarketEvent _peek;
    private bool _hasPeek;

    public ReplayReader(string path)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        _r = new BinaryReader(fs);
        if (_r.ReadUInt32() != ReplayFile.Magic) throw new InvalidDataException($"{path} is not a replay file.");
        int version = _r.ReadInt32();
        if (version != ReplayFile.Version) throw new InvalidDataException($"{path}: unsupported replay version {version}.");
        _count = _r.ReadInt64();
    }

    public long Count => _count;
    public long Consumed => _read - (_hasPeek ? 1 : 0);

    public bool TryPeek(out MarketEvent ev)
    {
        if (!_hasPeek)
        {
            if (_read >= _count) { ev = default; return false; }
            _peek = ReadOne();
            _read++;
            _hasPeek = true;
        }
        ev = _peek;
        return true;
    }

    public bool TryRead(out MarketEvent ev)
    {
        if (!TryPeek(out ev)) return false;
        _hasPeek = false;
        return true;
    }

    private MarketEvent ReadOne() => new(
        Seq: _r.ReadInt64(), ExchangeTs: _r.ReadInt64(), RecvTs: _r.ReadInt64(), Instrument: _r.ReadInt32(),
        Kind: (EventKind)_r.ReadByte(), Px1: _r.ReadInt64(), Px2: _r.ReadInt64(), Sz1: _r.ReadInt64(), Sz2: _r.ReadInt64(),
        Status: (TradingStatus)_r.ReadByte());

    public static List<MarketEvent> ReadAll(string path)
    {
        using var r = new ReplayReader(path);
        var list = new List<MarketEvent>((int)Math.Min(r.Count, int.MaxValue));
        while (r.TryRead(out var e)) list.Add(e);
        return list;
    }

    public void Dispose() => _r.Dispose();
}

/// <summary>Describes a generated or imported data set. Engine verifies file hashes before a run.</summary>
public sealed class DataManifest
{
    public required string Kind { get; init; }              // "synthetic" | "historical"
    public required bool Synthetic { get; init; }
    public required string Exchange { get; init; }
    public required long StartTs { get; init; }
    public required long EndTs { get; init; }
    public required string FeedFile { get; init; }
    public required string FeedSha256 { get; init; }
    public required long FeedEvents { get; init; }
    public required string ExchangeFile { get; init; }
    public required string ExchangeSha256 { get; init; }
    public required long ExchangeEvents { get; init; }
    public required string UniverseFile { get; init; }
    public string? GroundTruthFile { get; init; }
    public Dictionary<string, string> Parameters { get; init; } = [];

    public string DataVersion => $"{Kind}:{FeedSha256[..16]}";

    public static DataManifest Load(string dir) => Json.Load<DataManifest>(Path.Combine(dir, "manifest.json"));

    public void Verify(string dir)
    {
        if (ReplayFile.Sha256(Path.Combine(dir, FeedFile)) != FeedSha256)
            throw new InvalidDataException("Feed file hash does not match manifest.");
        if (ReplayFile.Sha256(Path.Combine(dir, ExchangeFile)) != ExchangeSha256)
            throw new InvalidDataException("Exchange file hash does not match manifest.");
    }
}
