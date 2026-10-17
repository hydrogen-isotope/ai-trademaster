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
}

public enum MarketPhase { Closed, Open, Break, Holiday }

/// <summary>Maps a UTC timestamp to the exchange's market phase using its local sessions and holiday list.</summary>
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

    public MarketPhase Phase(long ns)
    {
        var local = ToLocal(ns);
        var day = DateOnly.FromDateTime(local);
        if (local.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday || _holidays.Contains(day)) return MarketPhase.Holiday;
        var t = local.TimeOfDay;
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
