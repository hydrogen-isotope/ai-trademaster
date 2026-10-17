using System.Text.Json;
using Trademaster.Core;
using Trademaster.Sim;

namespace Trademaster.Tests;

/// <summary>Builds isolated temp directories with a synthetic exchange config and generated data.</summary>
public static class TestData
{
    public static ExchangeConfig Exchange(int lot = 100) => new()
    {
        Id = "SYNTH", Name = "test", Synthetic = true, TimeZone = "UTC",
        Sessions = [new SessionWindow("09:00", "12:00"), new SessionWindow("13:00", "17:00")],
        TickTable = [new TickBand(0m, 0.001m), new TickBand(1m, 0.005m), new TickBand(10m, 0.01m), new TickBand(100m, 0.05m)],
        LotSize = lot, PriceCollarPct = 10m, FeeBps = 1m, MinFee = 1m,
    };

    public static string TempDir(string name)
    {
        var d = Path.Combine(Path.GetTempPath(), "trademaster-tests", name + "-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(d);
        return d;
    }

    /// <summary>Generates a small synthetic data set; returns (dataDir, exchangesDir).</summary>
    public static (string Data, string Exchanges) Generate(string name, double minutes = 3, int stocks = 30, int seed = 11)
    {
        var root = TempDir(name);
        var exDir = Path.Combine(root, "exchanges");
        Directory.CreateDirectory(exDir);
        var ex = Exchange();
        File.WriteAllText(Path.Combine(exDir, "SYNTH.json"), JsonSerializer.Serialize(ex, Json.Options));
        var data = Path.Combine(root, "data");
        new SyntheticGenerator(new GeneratorOptions { Stocks = stocks, Sectors = 3, Minutes = minutes, Seed = seed }, ex).Generate(data);
        return (data, exDir);
    }

    public static string WriteReplay(string dir, IReadOnlyList<MarketEvent> events)
    {
        var p = Path.Combine(dir, "feed.tmrp");
        ReplayFile.Write(p, events);
        return p;
    }

    public static Universe Universe(int stocks = 2) => new()
    {
        Exchange = "SYNTH",
        Instruments = Enumerable.Range(0, stocks)
            .Select(i => new Instrument(i, $"S{i}", InstrumentType.Stock, "SEC00", LiquidityClass.High))
            .Append(new Instrument(stocks, "ETF", InstrumentType.Etf, "SEC00", LiquidityClass.High))
            .ToList(),
    };

    public static long T(double seconds) => Ns.FromUtc(new DateTime(2026, 1, 5, 9, 0, 0, DateTimeKind.Utc)) + (long)(seconds * Ns.PerSec);

    public static Price P(decimal d) => Price.FromDecimal(d);
}
