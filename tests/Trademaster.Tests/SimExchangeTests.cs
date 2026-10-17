using Trademaster.Core;
using Trademaster.Sim;
using static Trademaster.Tests.TestData;

namespace Trademaster.Tests;

public class SimExchangeTests
{
    private static SimExchange Venue(IReadOnlyList<MarketEvent> market, double touchProb = 0, double participation = 0.5,
        List<DisconnectWindow>? disconnects = null)
    {
        var path = WriteReplay(TempDir("venue"), market);
        var cfg = new ExecutionSection { OrderLatencyMs = 5, TouchFillProb = touchProb, Participation = participation, Disconnects = disconnects ?? [] };
        return new SimExchange(path, Universe(1), Exchange(), cfg, T(0));
    }

    private static MarketEvent Q(long seq, double s, decimal bid, decimal ask, long bsz = 1000, long asz = 1000) =>
        MarketEvent.Quote(seq, T(s), T(s), 0, P(bid), P(ask), bsz, asz);

    private static MarketEvent Tr(long seq, double s, decimal px, long size) => MarketEvent.Trade(seq, T(s), T(s), 0, P(px), size);

    private static List<ExecutionReport> Drain(SimExchange v, double s)
    {
        var l = new List<ExecutionReport>();
        v.AdvanceTo(T(s));
        v.DrainReports(T(s), l);
        return l;
    }

    [Fact]
    public void Marketable_buy_fills_at_ask_up_to_displayed_size()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m, asz: 300)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 500, P(10.05m), 0), T(0.5));
        var r = Drain(v, 1);
        var fill = Assert.Single(r, x => x.Kind == ReportKind.Fill);
        Assert.Equal(300, fill.FillQty);
        Assert.Equal(P(10.02m), fill.FillPrice);
        Assert.Equal(200, fill.LeavesQty);
        Assert.Equal(T(0.505), fill.Ts); // order latency
    }

    [Fact]
    public void Resting_buy_fills_completely_when_market_trades_through()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m), Tr(2, 1.0, 10.01m, 1000), Tr(3, 2.0, 10.00m, 1000), Tr(4, 3.0, 9.99m, 100)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 1000, P(10.00m), 0), T(0.5));
        var r = Drain(v, 2.5);
        Assert.DoesNotContain(r, x => x.Kind == ReportKind.Fill); // trade above, touch with prob 0
        r = Drain(v, 3.5);
        var fill = Assert.Single(r, x => x.Kind == ReportKind.Fill);
        Assert.Equal(1000, fill.FillQty);        // a trade below our bid means every bid at our price was taken
        Assert.Equal(P(10.00m), fill.FillPrice); // passive order fills at its own limit
    }

    private static SimExchange QueueVenue(IReadOnlyList<MarketEvent> market)
    {
        var path = WriteReplay(TempDir("queue"), market);
        var cfg = new ExecutionSection { OrderLatencyMs = 5, FillModel = FillModel.Queue };
        return new SimExchange(path, Universe(1), Exchange(), cfg, T(0));
    }

    [Fact]
    public void Queue_joiner_fills_only_after_displayed_size_ahead_trades()
    {
        // Join a 1,000-share bid. 600 trades at our price (we're still behind 400), then 700 more: 300 fills us.
        var v = QueueVenue([Q(1, 0.1, 10.00m, 10.02m, bsz: 1000), Tr(2, 1, 10.00m, 600), Tr(3, 2, 10.00m, 700)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 500, P(10.00m), 0), T(0.5));
        Assert.DoesNotContain(Drain(v, 1.5), x => x.Kind == ReportKind.Fill);
        var fill = Assert.Single(Drain(v, 2.5), x => x.Kind == ReportKind.Fill);
        Assert.Equal(300, fill.FillQty);
        Assert.Equal(1000, v.Orders[1].QueueAheadOnArrival);
        Assert.True(v.FillLog[0].Passive);
    }

    [Fact]
    public void Analytics_measure_spread_capture_markout_adverse_selection_and_cleanup()
    {
        // Passive buy joins 10.00 (mid 10.01). A trade through at 9.99 fills us at 10.00; one second later mid is 9.98.
        // A second passive buy never fills and is cleaned up by crossing the 10.02 ask at cancel time.
        var v = QueueVenue([Q(1, 0.1, 10.00m, 10.02m, bsz: 100), Tr(2, 1.0, 9.99m, 100), Q(3, 1.5, 9.97m, 9.99m),
                            Q(4, 3.0, 10.00m, 10.02m)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 100, P(10.00m), 0), T(0.5));
        v.Submit(new OrderRequest(2, 0, Side.Buy, 100, P(9.90m), 0), T(0.5));
        v.AdvanceTo(T(4));
        v.Cancel(2, T(4));
        v.AdvanceTo(T(70));
        var stats = Trademaster.Sim.ExecutionAnalytics.Compute(v, new Dictionary<long, string> { [1] = "filled", [2] = "unfilled" }, 1.0);
        var f = stats.Single(s => s.Group == "filled");
        Assert.Equal(1.0, f.FillRate);
        Assert.Equal((10.01 - 10.00) / 10.01 * 1e4, f.SpreadCaptureBps, 6);          // +10 bps: bought below the decision mid
        double m0 = (10.00 + 10.02) / 2;                                            // mid at the fill instant (t = 1.0); it drops at t = 1.5
        Assert.Equal((9.98 - m0) / m0 * 1e4, f.AdverseSelectionBps[0], 6);          // price fell 3 cents after we bought
        Assert.Equal((9.98 - 10.00) / 10.00 * 1e4, f.MarkoutBps[0], 6);
        var u = stats.Single(s => s.Group == "unfilled");
        Assert.Equal(0.0, u.FillRate);
        Assert.Equal(1.0, u.CleanupShare);
        Assert.Equal((10.02 - 10.01) / 10.01 * 1e4 + 1.0, u.ShortfallBps, 6);       // crossed the ask at cancel + taker fee
    }

    [Fact]
    public void Queue_cancellations_ahead_are_pro_rata_and_improving_order_is_first()
    {
        // Size at our level drops 1000 -> 500 without trades: half of the queue ahead (pro rata) is gone.
        var v = QueueVenue([Q(1, 0.1, 10.00m, 10.03m, bsz: 1000), Q(2, 1, 10.00m, 10.03m, bsz: 500), Tr(3, 2, 10.00m, 499),
                            Tr(4, 3, 10.01m, 100), Tr(5, 4, 10.00m, 101)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 100, P(10.00m), 0), T(0.5));  // joins behind 1000
        v.Submit(new OrderRequest(2, 0, Side.Buy, 100, P(10.02m), 0), T(0.5));  // improves the bid: nobody ahead
        // The 10.00 trade is below order 2's 10.02 bid, so order 2 was traded through and fills completely.
        // Order 1: 1000 ahead, halved to 500 by cancels; 499 traded leaves 1 ahead, so no fill yet.
        var r = Drain(v, 2.5).Where(x => x.Kind == ReportKind.Fill).ToList();
        Assert.Equal(100, Assert.Single(r).FillQty);
        Assert.Equal(2, r[0].OrderId);
        Assert.Equal(0, v.Orders[2].QueueAheadOnArrival);
        Assert.DoesNotContain(Drain(v, 3.5), x => x.Kind == ReportKind.Fill);   // trade at 10.01 is above order 1's price
        // A 101-share trade at our price: 1 clears the queue ahead, 100 fill us.
        Assert.Equal(100, Assert.Single(Drain(v, 4.5), x => x.Kind == ReportKind.Fill && x.OrderId == 1).FillQty);
    }

    [Fact]
    public void Touch_fills_when_probability_is_one()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m), Tr(2, 1.0, 10.00m, 400)], touchProb: 1.0);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 200, P(10.00m), 0), T(0.5));
        Assert.Equal(200, Assert.Single(Drain(v, 2), x => x.Kind == ReportKind.Fill).FillQty);
    }

    [Fact]
    public void Cancel_after_fill_is_rejected_too_late()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 100, P(10.02m), 0), T(0.5));
        v.Cancel(1, T(0.501)); // arrives 1 ms after the order filled
        var r = Drain(v, 1);
        Assert.Contains(r, x => x.Kind == ReportKind.Fill);
        Assert.Equal("TooLateToCancel", Assert.Single(r, x => x.Kind == ReportKind.CancelReject).Reason);
    }

    [Fact]
    public void Rejects_invalid_tick_lot_collar_and_closed_market()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 100, P(10.013m), 0), T(0.5));
        v.Submit(new OrderRequest(2, 0, Side.Buy, 150, P(10.00m), 0), T(0.5));
        v.Submit(new OrderRequest(3, 0, Side.Buy, 100, P(12.00m), 0), T(0.5));
        var r = Drain(v, 1).Where(x => x.Kind == ReportKind.Reject).Select(x => x.Reason).ToList();
        Assert.Equal(["InvalidTick", "InvalidLot", "PriceCollar"], r);
        v.Submit(new OrderRequest(4, 0, Side.Buy, 100, P(10.00m), 0), T(3 * 3600 + 10)); // 12:00:10 lunch break
        Assert.Equal("MarketClosed", Assert.Single(Drain(v, 3 * 3600 + 11), x => x.Kind == ReportKind.Reject).Reason);
    }

    [Fact]
    public void Disconnect_blocks_submission_cancels_on_disconnect_and_holds_reports()
    {
        var v = Venue([Q(1, 0.1, 10.00m, 10.02m)], disconnects: [new DisconnectWindow(2, 3)]);
        v.AdvanceTo(T(0.5));
        v.Submit(new OrderRequest(1, 0, Side.Buy, 100, P(9.99m), 0), T(0.5)); // rests
        Assert.Single(Drain(v, 1), x => x.Kind == ReportKind.Ack);
        Assert.Empty(Drain(v, 2.5));                                   // COD cancel generated but held
        Assert.False(v.IsConnected);
        Assert.Equal(SubmitResult.Disconnected, v.Submit(new OrderRequest(2, 0, Side.Buy, 100, P(9.99m), 0), T(2.5)));
        var r = Drain(v, 5.1);
        Assert.True(v.IsConnected);
        Assert.Equal("CancelOnDisconnect", Assert.Single(r, x => x.Kind == ReportKind.Cancelled).Reason);
        Assert.Empty(v.Snapshot(T(5.1)).OpenOrders);
    }
}
