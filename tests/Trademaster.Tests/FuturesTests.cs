using Trademaster.Core;

namespace Trademaster.Tests;

public class FuturesTests
{
    private static string ConfigPath(string file)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "config", "exchanges", file))) dir = dir.Parent;
        return Path.Combine(dir!.FullName, "config", "exchanges", file);
    }

    private static ExchangeConfig Comex() => ExchangeConfig.Load(ConfigPath("COMEX.json"), allowUnverified: true);

    private static long Et(int y, int mo, int d, int h, int mi)
    {
        var tz = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
        return Ns.FromUtc(TimeZoneInfo.ConvertTimeToUtc(new DateTime(y, mo, d, h, mi, 0, DateTimeKind.Unspecified), tz));
    }

    private static readonly Instrument Gold = new(0, "GCZ6", InstrumentType.Future, "METALS", LiquidityClass.High)
    { Root = "GC", Multiplier = 100m, TickSize = 0.10m, LastTradeDate = new DateOnly(2026, 12, 29) };
    private static readonly Instrument Silver = new(1, "SIZ6", InstrumentType.Future, "METALS", LiquidityClass.High)
    { Root = "SI", Multiplier = 5000m, TickSize = 0.005m, LastTradeDate = new DateOnly(2026, 12, 29) };

    [Fact]
    public void Comex_config_refuses_to_load_as_verified()
    {
        Assert.Throws<InvalidOperationException>(() => ExchangeConfig.Load(ConfigPath("COMEX.json")));
    }

    [Fact]
    public void Globex_overnight_session_follows_cme_hours()
    {
        var cal = new MarketCalendar(Comex());
        Assert.True(cal.Overnight);
        Assert.Equal(MarketPhase.Holiday, cal.Phase(Et(2026, 10, 17, 12, 0)));   // Saturday
        Assert.Equal(MarketPhase.Holiday, cal.Phase(Et(2026, 10, 18, 17, 30)));  // Sunday before the open
        Assert.Equal(MarketPhase.Open, cal.Phase(Et(2026, 10, 18, 18, 0)));      // Sunday 18:00 opens Monday's session
        Assert.Equal(new DateOnly(2026, 10, 19), cal.TradingDate(Et(2026, 10, 18, 19, 0)));
        Assert.Equal(MarketPhase.Open, cal.Phase(Et(2026, 10, 19, 3, 0)));       // overnight
        Assert.Equal(MarketPhase.Open, cal.Phase(Et(2026, 10, 19, 16, 59)));
        Assert.Equal(MarketPhase.Break, cal.Phase(Et(2026, 10, 19, 17, 30)));    // daily 60-minute break
        Assert.Equal(new DateOnly(2026, 10, 20), cal.TradingDate(Et(2026, 10, 19, 18, 0)));
        Assert.Equal(MarketPhase.Holiday, cal.Phase(Et(2026, 10, 23, 17, 30)));  // Friday after the close
        Assert.Equal(MarketPhase.Holiday, cal.Phase(Et(2026, 11, 25, 19, 0)));   // evening before Thanksgiving
    }

    [Fact]
    public void Each_contract_uses_its_own_tick_size()
    {
        var ex = Comex();
        var u = new Universe { Exchange = "COMEX", Instruments = [Gold, Silver] };
        var t = ex.BuildTickTables(u);
        Assert.True(t[0].IsValid(Price.FromDecimal(4186.7m)));
        Assert.False(t[0].IsValid(Price.FromDecimal(4186.75m)));
        Assert.True(t[1].IsValid(Price.FromDecimal(48.005m)));
        Assert.False(t[1].IsValid(Price.FromDecimal(48.002m)));
        Assert.Equal(10m, Gold.TickValue);       // $0.10 x 100 oz
        Assert.Equal(25m, Silver.TickValue);     // $0.005 x 5,000 oz
    }

    [Fact]
    public void Stocks_keep_the_exchange_tick_table()
    {
        var stock = new Instrument(0, "AAA", InstrumentType.Stock, "TECH", LiquidityClass.High);
        Assert.Null(stock.TickValue);
        Assert.Equal(1m, stock.Multiplier);
        var t = TestData.Exchange().BuildTickTables(new Universe { Exchange = "SYNTH", Instruments = [stock] });
        Assert.True(t[0].IsValid(Price.FromDecimal(10.01m)));
    }
}
