using System.Buffers;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Trademaster.Core;
using Trademaster.Engine.Risk;
using Trademaster.Engine.Stages;
using Trademaster.Telemetry;

namespace Trademaster.Engine;

/// <summary>Everything the audit record needs about one cycle. Contains no wall-clock measurements in virtual mode.</summary>
public sealed class CycleDecision
{
    public long CycleId;
    public long CutoffTs;
    public long DecidedTs;
    public CycleStatus Status;
    public MarketPhase Phase;
    public ulong InputHash;
    public int Valid;
    public int[] MaskCounts = new int[4];
    public bool FeedDown;
    public int PredictionsValid;
    public readonly List<(long OrderId, OrderIntent Intent)> Submitted = [];
    public readonly List<OrderIntent> NotSent = [];
    public int Kept;
    public readonly List<(long OrderId, string Why)> Cancels = [];
    public readonly List<(OrderIntent Intent, RiskReason Reason)> RiskRejects = [];
    public int ReportsApplied;
    public int FillsApplied;
    public int OpenPositions;
    public decimal Equity;
    public bool KillSwitch;

    public void Reset()
    {
        Submitted.Clear(); NotSent.Clear(); Cancels.Clear(); RiskRejects.Clear();
        Kept = 0; PredictionsValid = 0; ReportsApplied = 0; FillsApplied = 0;
    }
}

/// <summary>
/// Append-only JSON Lines decision log: one self-contained, auditable record per cycle, hash-chained for tamper
/// evidence. Each line ends with "prev_hash" (the previous line's hash; 64 zeros for the first) and "hash" =
/// SHA-256(prev_hash + the record JSON without these two fields). Editing, deleting, inserting or reordering any
/// record breaks the chain; <see cref="Verify"/> (CLI: trademaster verify-audit) checks it.
/// </summary>
public sealed class AuditLog : IDisposable
{
    public const string Genesis = "0000000000000000000000000000000000000000000000000000000000000000";
    private readonly FileStream _fs;
    private readonly ArrayBufferWriter<byte> _buf = new(1 << 14);
    private readonly Utf8JsonWriter _w;
    private string _prev = Genesis;
    private readonly Universe _u;
    private readonly string _model, _modelHash, _dataVersion;
    private readonly bool _synthetic;

    public string Path { get; }

    public AuditLog(string path, Universe universe, IModel model, DataManifest manifest)
    {
        Path = path;
        _fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16);
        _w = new Utf8JsonWriter(_buf, new JsonWriterOptions { Indented = false, SkipValidation = true });
        _u = universe;
        _model = $"{model.Name}@{model.Version}";
        _modelHash = model.ArtifactHash;
        _dataVersion = manifest.DataVersion;
        _synthetic = manifest.Synthetic;
    }

    private static string Snake(CycleStatus s) => JsonNamingPolicy.SnakeCaseLower.ConvertName(s.ToString());

    public void Write(CycleDecision d)
    {
        var w = _w;
        w.WriteStartObject();
        w.WriteNumber("cycle", d.CycleId);
        w.WriteString("cutoff_utc", Ns.Iso(d.CutoffTs));
        w.WriteNumber("cutoff_ts", d.CutoffTs);
        w.WriteNumber("decided_ts", d.DecidedTs);
        w.WriteString("status", Snake(d.Status));
        w.WriteString("phase", d.Phase.ToString().ToLowerInvariant());
        w.WriteBoolean("synthetic", _synthetic);
        w.WriteString("model", _model);
        w.WriteString("model_hash", _modelHash);
        w.WriteString("data_version", _dataVersion);
        w.WriteString("input_hash", d.InputHash.ToString("x16", CultureInfo.InvariantCulture));
        w.WriteNumber("valid_instruments", d.Valid);
        w.WriteStartObject("masked");
        w.WriteNumber("no_quote", d.MaskCounts[(int)MaskReason.NoQuote]);
        w.WriteNumber("halted", d.MaskCounts[(int)MaskReason.Halted]);
        w.WriteNumber("stale", d.MaskCounts[(int)MaskReason.Stale]);
        w.WriteEndObject();
        w.WriteBoolean("feed_down", d.FeedDown);
        w.WriteBoolean("kill_switch", d.KillSwitch);
        w.WriteNumber("predictions_valid", d.PredictionsValid);

        w.WriteStartObject("provenance");
        w.WriteString("bid", "observed"); w.WriteString("ask", "observed");
        w.WriteString("beta", "estimated");
        w.WriteString("mu", "inferred"); w.WriteString("sigma", "inferred"); w.WriteString("score", "inferred");
        w.WriteString("peer", "inferred");
        w.WriteEndObject();

        w.WriteStartArray("orders");
        foreach (var (id, it) in d.Submitted) WriteIntent(it, id, null);
        w.WriteEndArray();
        w.WriteNumber("kept", d.Kept);
        if (d.NotSent.Count > 0)
        {
            w.WriteStartArray("not_sent_disconnected");
            foreach (var it in d.NotSent) WriteIntent(it, null, null);
            w.WriteEndArray();
        }
        w.WriteStartArray("cancels");
        foreach (var (id, why) in d.Cancels)
        {
            w.WriteStartObject(); w.WriteNumber("id", id); w.WriteString("why", why); w.WriteEndObject();
        }
        w.WriteEndArray();
        w.WriteStartArray("risk_rejects");
        foreach (var (it, reason) in d.RiskRejects) WriteIntent(it, null, reason.ToString());
        w.WriteEndArray();
        w.WriteNumber("reports_applied", d.ReportsApplied);
        w.WriteNumber("fills_applied", d.FillsApplied);
        w.WriteNumber("open_positions", d.OpenPositions);
        w.WriteString("equity", d.Equity.ToString("0.00", CultureInfo.InvariantCulture));
        w.WriteEndObject();
        w.Flush();
        var body = _buf.WrittenSpan;                      // {...}
        string hash = Hash(_prev, body);
        _fs.Write(body[..^1]);
        _fs.Write(Encoding.ASCII.GetBytes($",\"prev_hash\":\"{_prev}\",\"hash\":\"{hash}\"}}\n"));
        _prev = hash;
        _buf.ResetWrittenCount();
        w.Reset(_buf); // next record is a new JSON root
    }

    private void WriteIntent(in OrderIntent it, long? id, string? reason)
    {
        var w = _w;
        w.WriteStartObject();
        if (id is long oid) w.WriteNumber("id", oid);
        w.WriteString("sym", _u.Instruments[it.Instrument].Symbol);
        w.WriteString("side", it.Side == Side.Buy ? "buy" : "sell");
        w.WriteNumber("qty", it.Qty);
        w.WriteString("limit", it.Limit.ToString());
        w.WriteString("bid", new Price(it.Bid).ToString());
        w.WriteString("ask", new Price(it.Ask).ToString());
        w.WriteNumber("mu_bps", Math.Round(it.Mu * 1e4, 3));
        w.WriteNumber("sigma_bps", Math.Round(it.Sigma * 1e4, 3));
        w.WriteNumber("score", Math.Round(it.Score, 4));
        w.WriteNumber("beta", Math.Round(it.Beta, 4));
        w.WriteString("rule", it.Rule);
        if (it.Peer >= 0 && it.Peer < _u.Count)
        {
            w.WriteString("peer", _u.Instruments[it.Peer].Symbol);
            w.WriteNumber("peer_weight", Math.Round(it.PeerWeight, 4));
        }
        if (reason is not null) w.WriteString("reason", reason);
        w.WriteEndObject();
    }

    private static string Hash(string prev, ReadOnlySpan<byte> body)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        h.AppendData(Encoding.ASCII.GetBytes(prev));
        h.AppendData(body);
        return Convert.ToHexStringLower(h.GetHashAndReset());
    }

    /// <summary>Re-computes the hash chain of a decision log. Returns the number of valid records and, if the chain is
    /// broken, the 1-based line where it breaks and why.</summary>
    public static (int Records, int BadLine, string? Problem) Verify(string path)
    {
        const string tag = ",\"prev_hash\":\"";
        string prev = Genesis;
        int n = 0;
        foreach (var line in File.ReadLines(path))
        {
            n++;
            int at = line.LastIndexOf(tag, StringComparison.Ordinal);
            if (at < 0 || !line.EndsWith("\"}", StringComparison.Ordinal)) return (n - 1, n, "record has no hash chain fields");
            string tail = line[(at + tag.Length)..];               // <prev>","hash":"<hash>"}
            string recPrev = tail[..64], recHash = tail[^66..^2];
            if (recPrev != prev) return (n - 1, n, "prev_hash does not match the previous record (deleted, inserted or reordered)");
            string expect = Hash(prev, Encoding.UTF8.GetBytes(line[..at] + "}"));
            if (expect != recHash) return (n - 1, n, "hash mismatch (record content was modified)");
            prev = recHash;
        }
        return (n, 0, null);
    }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _w.Flush();
        _w.Dispose();
        _fs.Dispose();
    }
}
