using Trademaster.Core;
using static Trademaster.Tests.TestData;

namespace Trademaster.Tests;

public class TickTableTests
{
    private readonly TickTable _t = Exchange().BuildTickTable();

    [Theory]
    [InlineData("0.9994", "0.999", "1.000")]
    [InlineData("1.0001", "1.000", "1.005")]
    [InlineData("9.9990", "9.995", "10.000")]
    [InlineData("12.345", "12.34", "12.35")]
    [InlineData("100.02", "100.00", "100.05")]
    [InlineData("12.34", "12.34", "12.34")]
    public void Rounds_conservatively_across_bands(string px, string down, string up)
    {
        var p = P(decimal.Parse(px, System.Globalization.CultureInfo.InvariantCulture));
        Assert.Equal(P(decimal.Parse(down, System.Globalization.CultureInfo.InvariantCulture)), _t.RoundDown(p));
        Assert.Equal(P(decimal.Parse(up, System.Globalization.CultureInfo.InvariantCulture)), _t.RoundUp(p));
        Assert.True(_t.IsValid(_t.RoundDown(p)));
        Assert.True(_t.IsValid(_t.RoundUp(p)));
    }

    [Fact]
    public void Validates_ticks_and_steps_across_band_boundary()
    {
        Assert.True(_t.IsValid(P(9.99m)));
        Assert.False(_t.IsValid(P(9.991m)));
        Assert.False(_t.IsValid(P(10.005m)));
        Assert.Equal(P(10.01m), _t.Step(P(9.995m), 2));
        Assert.Equal(P(9.995m), _t.Step(P(10.00m), -1));
    }

    [Fact]
    public void Rejects_misaligned_band()
    {
        Assert.Throws<ArgumentException>(() => new TickTable([new TickBand(0m, 0.01m), new TickBand(1.005m, 0.05m)]));
    }
}

public class CalendarTests
{
    [Fact]
    public void Phases_follow_sessions_weekends_and_holidays()
    {
        var cfg = Exchange();
        var cal = new MarketCalendar(cfg);
        Assert.Equal(MarketPhase.Closed, cal.Phase(T(-1)));          // 08:59:59 Monday
        Assert.Equal(MarketPhase.Open, cal.Phase(T(0)));             // 09:00
        Assert.Equal(MarketPhase.Break, cal.Phase(T(3.5 * 3600)));   // 12:30
        Assert.Equal(MarketPhase.Closed, cal.Phase(T(8 * 3600)));    // 17:00
        Assert.Equal(MarketPhase.Holiday, cal.Phase(T(-2 * 86400))); // Saturday
        var hol = new ExchangeConfig
        {
            Id = cfg.Id, Name = cfg.Name, Synthetic = true, TimeZone = cfg.TimeZone, Sessions = cfg.Sessions,
            TickTable = cfg.TickTable, Holidays = ["2026-01-05"],
        };
        Assert.Equal(MarketPhase.Holiday, new MarketCalendar(hol).Phase(T(60)));
    }

    [Fact]
    public void Unverified_real_exchange_config_is_refused()
    {
        var dir = TempDir("ex");
        var path = Path.Combine(dir, "X.json");
        File.WriteAllText(path, """{ "id": "X", "name": "x", "verified": false, "time_zone": "UTC", "sessions": [{"open":"09:00","close":"10:00"}], "tick_table": [{"from":0,"tick":0.01}] }""");
        Assert.Throws<InvalidOperationException>(() => ExchangeConfig.Load(path));
    }
}
