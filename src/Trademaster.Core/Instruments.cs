using System.Text.Json;

namespace Trademaster.Core;

public enum InstrumentType { Stock, Etf, Index, Future }

public enum LiquidityClass { High, Medium, Low }

/// <summary>
/// Public instrument metadata. <see cref="Sector"/> is a public classification; for an ETF it is the
/// sector (or "MARKET") the fund publicly tracks. ETF holdings are never part of this record.
/// </summary>
public sealed record Instrument(int Id, string Symbol, InstrumentType Type, string? Sector, LiquidityClass Liquidity)
{
    /// <summary>Futures: product root, e.g. "GC". Null for stocks and ETFs.</summary>
    public string? Root { get; init; }
    /// <summary>Futures: units per contract (GC 100 troy oz, SI 5,000 troy oz). 1 for stocks.</summary>
    public decimal Multiplier { get; init; } = 1m;
    /// <summary>Per-instrument minimum price fluctuation; overrides the exchange tick table when set.</summary>
    public decimal? TickSize { get; init; }
    /// <summary>Futures: last trading day of the contract (exchange local date).</summary>
    public DateOnly? LastTradeDate { get; init; }

    /// <summary>Money value of one tick for one contract (or share).</summary>
    public decimal? TickValue => TickSize is decimal t ? t * Multiplier : null;
}

public sealed class Universe
{
    public required string Exchange { get; init; }
    public required List<Instrument> Instruments { get; init; }

    public int Count => Instruments.Count;

    public static Universe Load(string path)
    {
        var u = JsonSerializer.Deserialize<Universe>(File.ReadAllText(path), Json.Options)
                ?? throw new InvalidDataException($"Empty universe file {path}");
        for (int i = 0; i < u.Instruments.Count; i++)
            if (u.Instruments[i].Id != i)
                throw new InvalidDataException($"Universe ids must be dense 0..N-1 (index {i} has id {u.Instruments[i].Id}).");
        return u;
    }

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, Json.Options));

    /// <summary>
    /// For each instrument, the ETF that publicly tracks its sector, else the market ETF, else -1.
    /// Derived only from public sector labels — not from holdings.
    /// </summary>
    public int[] ReferenceEtfs()
    {
        var bySector = new Dictionary<string, int>();
        int market = -1;
        foreach (var ins in Instruments)
        {
            if (ins.Type != InstrumentType.Etf || ins.Sector is null) continue;
            if (ins.Sector == "MARKET") market = ins.Id;
            else bySector.TryAdd(ins.Sector, ins.Id);
        }
        var map = new int[Count];
        for (int i = 0; i < Count; i++)
        {
            var ins = Instruments[i];
            map[i] = ins.Type == InstrumentType.Stock && ins.Sector is not null && bySector.TryGetValue(ins.Sector, out var e)
                ? e
                : (ins.Type == InstrumentType.Stock ? market : -1);
        }
        return map;
    }
}
