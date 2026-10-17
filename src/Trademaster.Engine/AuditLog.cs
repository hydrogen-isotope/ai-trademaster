using System.Globalization;
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

/// <summary>Append-only JSON Lines decision log: one self-contained, auditable record per cycle.</summary>
public sealed class AuditLog : IDisposable
{
    private readonly FileStream _fs;
    private readonly Utf8JsonWriter _w;
    private readonly Universe _u;
    private readonly string _model, _modelHash, _dataVersion;
    private readonly bool _synthetic;

    public string Path { get; }

    public AuditLog(string path, Universe universe, IModel model, DataManifest manifest)
    {
        Path = path;
        _fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16);
        _w = new Utf8JsonWriter(_fs, new JsonWriterOptions { Indented = false, SkipValidation = true });
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
        _fs.WriteByte((byte)'\n');
        w.Reset(_fs); // next record is a new JSON root
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
        if (reason is not null) w.WriteString("reason", reason);
        w.WriteEndObject();
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
