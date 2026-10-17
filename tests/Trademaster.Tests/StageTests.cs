using Trademaster.Core;
using Trademaster.Engine.Orders;
using Trademaster.Engine.Risk;
using Trademaster.Engine.Stages;
using static Trademaster.Tests.TestData;

namespace Trademaster.Tests;

public class IngestTests
{
    private static MarketEvent Q(long seq, double recvS, int ins = 0, decimal bid = 10.00m, decimal ask = 10.01m) =>
        MarketEvent.Quote(seq, T(recvS) - Ns.PerMs, T(recvS), ins, P(bid), P(ask), 100, 100);

    private static S1Ingest Ingest(IReadOnlyList<MarketEvent> evs, int windowMs = 50) =>
        new(new ReplayReader(WriteReplay(TempDir("ingest"), evs)), 3, windowMs);

    [Fact]
    public void Drops_duplicates_and_restores_sequence_order()
    {
        var s1 = Ingest([Q(1, 0.001), Q(3, 0.002), Q(2, 0.003), Q(3, 0.004), Q(4, 0.005)]);
        s1.ProcessUntil(T(1));
        Assert.Equal([1L, 2, 3, 4], s1.Accepted.Select(e => e.Seq));
        Assert.Equal(1, s1.Counters.Duplicates);
        Assert.Equal(1, s1.Counters.Reordered);
        Assert.Equal(0, s1.Counters.GapLost);
    }

    [Fact]
    public void Declares_gap_lost_after_window_and_drops_late_arrival()
    {
        // seq 2 is missing for longer than the 50 ms window, then arrives late.
        var s1 = Ingest([Q(1, 0.000), Q(3, 0.010), Q(4, 0.020), Q(2, 0.500)]);
        s1.ProcessUntil(T(0.050));
        Assert.Equal([1L], s1.Accepted.Select(e => e.Seq));      // 3,4 held while gap may still fill
        s1.ProcessUntil(T(1));
        Assert.Equal([3L, 4], s1.Accepted.Select(e => e.Seq));
        Assert.Equal(1, s1.Counters.GapLost);
        Assert.Equal(1, s1.Counters.LateDropped);
    }

    [Fact]
    public void Never_reads_past_the_cutoff()
    {
        var s1 = Ingest([Q(1, 0.5), Q(2, 1.0), Q(3, 1.0000001)]);
        s1.ProcessUntil(T(1.0));
        Assert.Equal([1L, 2], s1.Accepted.Select(e => e.Seq));
    }

    [Fact]
    public void Rejects_invalid_records()
    {
        var s1 = Ingest([Q(1, 0.1, ins: 99), MarketEvent.Trade(2, T(0.2), T(0.2), 0, P(10m), 0)]);
        s1.ProcessUntil(T(1));
        Assert.Empty(s1.Accepted);
        Assert.Equal(2, s1.Counters.Invalid);
    }
}

public class EvaluateTests
{
    private static S2Evaluate Eval() => new(Universe(2), new QualitySection());

    [Fact]
    public void Crossed_quote_is_flagged_and_not_used()
    {
        var s2 = Eval();
        s2.Evaluate([MarketEvent.Quote(1, T(0), T(0), 0, P(10.00m), P(10.01m), 100, 100)], T(0.5));
        s2.Evaluate([MarketEvent.Quote(2, T(0.6), T(0.6), 0, P(10.05m), P(10.00m), 100, 100)], T(1));
        Assert.Equal(1, s2.Counters.Crossed);
        Assert.Equal(P(10.00m).Raw, s2.GoodBid[0]);
        Assert.True(s2.Quality[0] < 1f);
    }

    [Fact]
    public void Masks_no_quote_stale_and_halted()
    {
        var s2 = Eval();
        s2.Evaluate([
            MarketEvent.Quote(1, T(0), T(0), 0, P(10.00m), P(10.01m), 100, 100),
            MarketEvent.Quote(2, T(0), T(0), 1, P(20.00m), P(20.01m), 100, 100),
            MarketEvent.StatusChange(3, T(0.1), T(0.1), 1, TradingStatus.Halted),
        ], T(1));
        Assert.Equal(MaskReason.Valid, s2.Mask[0]);
        Assert.Equal(MaskReason.Halted, s2.Mask[1]);
        Assert.Equal(MaskReason.NoQuote, s2.Mask[2]);
        s2.Evaluate([MarketEvent.Trade(4, T(5.5), T(5.5), 0, P(10.00m), 100)], T(6)); // quote now 6 s old > 5 s (high)
        Assert.Equal(MaskReason.Stale, s2.Mask[0]);
    }

    [Fact]
    public void Feed_down_after_timeout()
    {
        var s2 = Eval();
        s2.Evaluate([MarketEvent.Quote(1, T(0), T(0), 0, P(10.00m), P(10.01m), 100, 100)], T(1));
        Assert.False(s2.FeedDown);
        s2.Evaluate([], T(3.5));
        Assert.True(s2.FeedDown);
    }
}

public class DecisionAndRiskTests
{
    private static (DecisionPolicy Policy, S2Evaluate S2, OrderManager Om, ModelInputBatch Batch) Setup()
    {
        var u = Universe(2);
        var ex = Exchange();
        var s2 = new S2Evaluate(u, new QualitySection());
        s2.Evaluate([
            MarketEvent.Quote(1, T(0), T(0), 0, P(10.00m), P(10.02m), 100, 100),
            MarketEvent.Quote(2, T(0), T(0), 2, P(50.00m), P(50.05m), 100, 100),
        ], T(0.5));
        var batch = new ModelInputBatch { N = 3, W = 1, Fdim = F.Count, X = new float[3 * F.Count], Mask = [1, 0, 1], Quality = new float[3] };
        return (new DecisionPolicy(new DecisionSection(), new RiskSection(), ex, u), s2, new OrderManager(3, 1_000_000m), batch);
    }

    [Fact]
    public void Buy_limit_is_capped_at_ask_and_rounded_down_to_tick()
    {
        var (policy, s2, om, b) = Setup();
        var preds = new Prediction[3];
        preds[0] = new Prediction { Valid = true, Mu = 0.0040f, Sigma = 0.002f };   // strong: mid*(1+mu) > ask → capped at ask
        var intents = new List<OrderIntent>();
        policy.Decide(b, preds, s2, om, intents);
        var it = Assert.Single(intents);
        Assert.Equal(Side.Buy, it.Side);
        Assert.Equal(P(10.02m), it.Limit);
        Assert.Equal(0, it.Qty % 100);

        // Because the entry hurdle includes the half spread, an entry's limit is always ≥ ask, so v0 entries
        // are marketable. Exits can rest: holding 500, mu = -5 bps → mid·(1+mu+fee+margin) = 10.007998 → up to 10.01.
        om.Position[0] = 500;
        preds[0] = new Prediction { Valid = true, Mu = -0.0005f, Sigma = 0.001f };
        policy.Decide(b, preds, s2, om, intents);
        var exit = Assert.Single(intents);
        Assert.Equal(Side.Sell, exit.Side);
        Assert.Equal(500, exit.Qty);
        Assert.Equal(P(10.01m), exit.Limit);
    }

    [Fact]
    public void No_trade_below_cost_hurdle_or_score_or_on_masked_instrument()
    {
        var (policy, s2, om, b) = Setup();
        var preds = new Prediction[3];
        preds[0] = new Prediction { Valid = true, Mu = 0.0005f, Sigma = 0.0001f };  // high score, below spread+fee cost
        preds[1] = new Prediction { Valid = true, Mu = 0.01f, Sigma = 0.001f };     // masked instrument
        var intents = new List<OrderIntent>();
        policy.Decide(b, preds, s2, om, intents);
        Assert.Empty(intents);
    }

    [Fact]
    public void Risk_rejects_position_limit_collar_tick_and_oversell()
    {
        var (_, _, om, b) = Setup();
        var risk = new RiskManager(new RiskSection { MaxPositionNotional = 5_000, PriceCollarPct = 2 }, Exchange(), 1_000_000m);
        double gross = 0;
        OrderIntent I(Side s, long q, decimal px) => new(0, s, q, P(px), 0, 0, 0, 0, P(10.00m).Raw, P(10.02m).Raw, "t");
        Assert.Equal(RiskReason.None, risk.Check(I(Side.Buy, 400, 10.01m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.PositionLimit, risk.Check(I(Side.Buy, 600, 10.01m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.PriceCollar, risk.Check(I(Side.Buy, 100, 10.50m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.InvalidTick, risk.Check(I(Side.Buy, 100, 10.013m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.InvalidLot, risk.Check(I(Side.Buy, 150, 10.01m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.Oversell, risk.Check(I(Side.Sell, 100, 10.01m), 0, b, om, ref gross));
        Assert.Equal(RiskReason.OrderRateLimit, risk.Check(I(Side.Buy, 100, 10.01m), 20, b, om, ref gross));
    }

    [Fact]
    public void Kill_switch_latches_on_daily_loss()
    {
        var risk = new RiskManager(new RiskSection { MaxDailyLoss = 1000 }, Exchange(), 100_000m);
        risk.UpdateLoss(99_500m);
        Assert.False(risk.KillSwitch);
        risk.UpdateLoss(98_900m);
        Assert.True(risk.KillSwitch);
        risk.UpdateLoss(100_000m);
        Assert.True(risk.KillSwitch);
    }
}
