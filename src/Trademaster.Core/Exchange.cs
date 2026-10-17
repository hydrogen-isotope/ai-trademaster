using System.Globalization;

namespace Trademaster.Core;

public sealed record SessionWindow(string Open, string Close);

/// <summary>Exchange metadata. Values for real exchanges must come from official sources (<see cref="Verified"/>).</summary>
public sealed class ExchangeConfig
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    /// <summary>True only when every field has been checked against the exchange's official documents.</summary>
    public bool Verified { get; init; }
    public bool Synthetic { get; init; }
    public string Notes { get; init; } = "";
    public required string TimeZone { get; init; }
    public string Currency { get; init; } = "";
    public List<SessionWindow> Sessions { get; init; } = [];
    public List<string> Holidays { get; init; } = [];
    public List<TickBand> TickTable { get; init; } = [];
    public int LotSize { get; init; } = 1;
    /// <summary>Exchange-side price collar: orders further than this % from the reference price are rejected.</summary>
    public decimal PriceCollarPct { get; init; } = 10m;
    public decimal FeeBps { get; init; }
    public decimal MinFee { get; init; }

    public static ExchangeConfig Load(string path, bool allowUnverified = false)
    {
        var cfg = Json.Load<ExchangeConfig>(path);
        if (!cfg.Verified && !cfg.Synthetic && !allowUnverified)
            throw new InvalidOperationException(
                $"Exchange '{cfg.Id}' config is not verified against official sources ({path}). " +
                "Fill it in from the exchange's documents and set verified=true before using it.");
        if (cfg.TickTable.Count == 0) throw new InvalidDataException($"Exchange '{cfg.Id}' has no tick table.");
        if (cfg.Sessions.Count == 0) throw new InvalidDataException($"Exchange '{cfg.Id}' has no sessions.");
        return cfg;
    }

    public TickTable BuildTickTable() => new(TickTable);

    /// <summary>The tick table for one instrument: its own tick size if it has one (futures), else the exchange's.</summary>
    public TickTable BuildTickTable(Instrument instrument) =>
        instrument.TickSize is decimal t ? new([new TickBand(0, t)]) : new(TickTable);

    /// <summary>One tick table per instrument, indexed by instrument id.</summary>
    public TickTable[] BuildTickTables(Universe universe)
    {
        var shared = BuildTickTable();
        return universe.Instruments.Select(i => i.TickSize is null ? shared : BuildTickTable(i)).ToArray();
    }
}

public enum MarketPhase { Closed, Open, Break, Holiday }

/// <summary>
/// Maps a UTC timestamp to the exchange's market phase using its local sessions and holiday list.
/// A session whose close is not after its open runs overnight (e.g. CME Globex 18:00-17:00): it opens on the evening
/// before its trading date and belongs to that next trading date. Weekends and holidays are trading dates without a
/// session, so Globex opens Sunday 18:00 for Monday and closes Friday 17:00.
/// </summary>
public sealed class MarketCalendar
{
    private readonly TimeZoneInfo _tz;
    private readonly (TimeSpan Open, TimeSpan Close)[] _sessions;
    private readonly HashSet<DateOnly> _holidays;

    public MarketCalendar(ExchangeConfig cfg)
    {
        _tz = TimeZoneInfo.FindSystemTimeZoneById(cfg.TimeZone);
        _sessions = cfg.Sessions
            .Select(s => (TimeSpan.Parse(s.Open, CultureInfo.InvariantCulture), TimeSpan.Parse(s.Close, CultureInfo.InvariantCulture)))
            .OrderBy(s => s.Item1)
            .ToArray();
        _holidays = cfg.Holidays.Select(h => DateOnly.ParseExact(h, "yyyy-MM-dd", CultureInfo.InvariantCulture)).ToHashSet();
    }

    public DateTime ToLocal(long ns) => TimeZoneInfo.ConvertTimeFromUtc(Ns.ToUtc(ns), _tz);

    private bool IsTradingDate(DateOnly d) => d.DayOfWeek is not (DayOfWeek.Saturday or DayOfWeek.Sunday) && !_holidays.Contains(d);

    public bool Overnight => _sessions.Length == 1 && _sessions[0].Close <= _sessions[0].Open;

    /// <summary>The trading date a timestamp belongs to (overnight sessions roll forward at the evening open).</summary>
    public DateOnly TradingDate(long ns)
    {
        var local = ToLocal(ns);
        var day = DateOnly.FromDateTime(local);
        return Overnight && local.TimeOfDay >= _sessions[0].Open ? day.AddDays(1) : day;
    }

    public MarketPhase Phase(long ns)
    {
        var local = ToLocal(ns);
        var day = DateOnly.FromDateTime(local);
        var t = local.TimeOfDay;
        if (Overnight)
        {
            var (open, close) = _sessions[0];
            if (t >= open) return IsTradingDate(day.AddDays(1)) ? MarketPhase.Open : MarketPhase.Holiday;
            if (t < close) return IsTradingDate(day) ? MarketPhase.Open : MarketPhase.Holiday;
            return IsTradingDate(day) && IsTradingDate(day.AddDays(1)) ? MarketPhase.Break : MarketPhase.Holiday;
        }
        if (!IsTradingDate(day)) return MarketPhase.Holiday;
        foreach (var (open, close) in _sessions)
            if (t >= open && t < close) return MarketPhase.Open;
        return t >= _sessions[0].Open && t < _sessions[^1].Close ? MarketPhase.Break : MarketPhase.Closed;
    }

    /// <summary>UTC ns of the first session open on the given local date.</summary>
    public long SessionOpen(DateOnly localDate, int session = 0)
    {
        var local = localDate.ToDateTime(TimeOnly.FromTimeSpan(_sessions[session].Open), DateTimeKind.Unspecified);
        return Ns.FromUtc(TimeZoneInfo.ConvertTimeToUtc(local, _tz));
    }
}
