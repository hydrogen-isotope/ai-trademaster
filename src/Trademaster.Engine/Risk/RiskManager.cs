using Trademaster.Core;
using Trademaster.Engine.Orders;
using Trademaster.Engine.Stages;

namespace Trademaster.Engine.Risk;

public enum RiskReason
{
    None, InstrumentMasked, InvalidTick, InvalidLot, PriceCollar, PositionLimit, GrossLimit, OrderRateLimit, Oversell,
}

/// <summary>
/// Hard limits applied after the policy and independent of the model. Cycle-level gates (market closed,
/// feed down, disconnected, kill switch) are enforced by the cycle runner before any order reaches here.
/// </summary>
public sealed class RiskManager
{
    private readonly RiskSection _cfg;
    private readonly TickTable[]? _ticks;
    private readonly TickTable _shared;
    private readonly int _lot;
    private readonly decimal _startEquity;

    public bool KillSwitch { get; private set; }
    public string? KillReason { get; private set; }
    public Dictionary<RiskReason, long> Rejections { get; } = [];

    public RiskManager(RiskSection cfg, ExchangeConfig exchange, decimal startEquity, Universe? universe = null)
    {
        _cfg = cfg;
        _shared = exchange.BuildTickTable();
        _ticks = universe is null ? null : exchange.BuildTickTables(universe);
        _lot = Math.Max(1, exchange.LotSize);
        _startEquity = startEquity;
    }

    /// <summary>Latches the kill switch when the daily loss limit is breached. It never resets within a run.</summary>
    public void UpdateLoss(decimal equity)
    {
        if (!KillSwitch && _startEquity - equity > (decimal)_cfg.MaxDailyLoss)
        {
            KillSwitch = true;
            KillReason = $"daily loss {_startEquity - equity:0.00} > limit {_cfg.MaxDailyLoss:0.00}";
        }
    }

    public RiskReason Check(in OrderIntent it, int approvedThisCycle, ModelInputBatch b, OrderManager om, ref double gross)
    {
        var r = Evaluate(it, approvedThisCycle, b, om, ref gross);
        if (r != RiskReason.None) Rejections[r] = Rejections.GetValueOrDefault(r) + 1;
        return r;
    }

    private RiskReason Evaluate(in OrderIntent it, int approvedThisCycle, ModelInputBatch b, OrderManager om, ref double gross)
    {
        if (approvedThisCycle >= _cfg.MaxOrdersPerCycle) return RiskReason.OrderRateLimit;
        if (b.Mask[it.Instrument] == 0) return RiskReason.InstrumentMasked;
        if (!(_ticks?[it.Instrument] ?? _shared).IsValid(it.Limit)) return RiskReason.InvalidTick;
        if (it.Qty <= 0 || it.Qty % _lot != 0) return RiskReason.InvalidLot;
        double mid = (it.Bid + it.Ask) / (2.0 * Price.Scale);
        if (Math.Abs(it.Limit.ToDouble() - mid) / mid * 100 > _cfg.PriceCollarPct) return RiskReason.PriceCollar;
        double notional = it.Qty * it.Limit.ToDouble();
        if (it.Side == Side.Buy)
        {
            if ((om.Position[it.Instrument] + it.Qty) * it.Limit.ToDouble() > _cfg.MaxPositionNotional) return RiskReason.PositionLimit;
            if (gross + notional > _cfg.MaxGrossNotional) return RiskReason.GrossLimit;
            gross += notional;
        }
        else if (it.Qty > om.Position[it.Instrument]) return RiskReason.Oversell;
        return RiskReason.None;
    }
}
