using Trademaster.Core;
using Trademaster.Engine.Orders;

namespace Trademaster.Engine.Stages;

/// <summary>
/// A desired working order for this cycle, with the inputs that produced it (for the audit record).
/// <see cref="Cutoff"/> is the worst price the decision accepts (max for buys, min for sells); <see cref="Limit"/> is
/// the price actually sent — equal to the cutoff-capped opposite quote when aggressive, our side's best when passive.
/// </summary>
public readonly record struct OrderIntent(
    int Instrument, Side Side, long Qty, Price Limit,
    float Mu, float Sigma, float Score, float Beta, long Bid, long Ask, string Rule, Price Cutoff = default,
    int Peer = -1, float PeerWeight = 0);

/// <summary>
/// Stage 6 — decision policy v0 (long-only, explicit).
/// Aggressive style: cost = half spread + fee + slippage estimate; entry when mu &gt; cost + margin and mu/sigma ≥ entry_score;
/// limit = min(ask, cutoff) where cutoff = mid·(1 + mu − fee − margin), rounded DOWN to tick.
/// Passive style: cost = fee + assumed adverse selection (no spread paid); limit = min(bid, cutoff) — joins the bid.
/// Exit (both): holding and mu/sigma &lt; −exit_score → sell the position; cutoff = mid·(1 + mu + fee + margin) rounded UP;
/// aggressive sells at max(bid, cutoff), passive rests at max(ask, cutoff).
/// Intents are desired working orders, not deltas: an existing order that still satisfies the intent is kept
/// (see <see cref="Keeps"/>), so passive orders keep their queue position.
/// </summary>
public sealed class DecisionPolicy
{
    private readonly DecisionSection _cfg;
    private readonly TickTable _ticks;
    private readonly double _feeFrac, _maxPos;
    private readonly int _lot;
    private readonly bool[] _tradeable;

    public bool Passive { get; }

    public DecisionPolicy(DecisionSection cfg, RiskSection risk, ExchangeConfig exchange, Universe universe)
    {
        _cfg = cfg;
        _ticks = exchange.BuildTickTable();
        _feeFrac = (double)exchange.FeeBps / 1e4;
        _lot = Math.Max(1, exchange.LotSize);
        _maxPos = risk.MaxPositionNotional;
        _tradeable = universe.Instruments.Select(i => i.Type == InstrumentType.Stock).ToArray();
        Passive = cfg.ExecutionStyle.Equals("passive", StringComparison.OrdinalIgnoreCase);
        if (!Passive && !cfg.ExecutionStyle.Equals("aggressive", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"decision.execution_style must be 'aggressive' or 'passive' (got '{cfg.ExecutionStyle}').");
    }

    public void Decide(ModelInputBatch b, ReadOnlySpan<Prediction> preds, S2Evaluate s2, OrderManager om, List<OrderIntent> intents)
    {
        intents.Clear();
        double margin = _cfg.MarginBps / 1e4, slip = _cfg.SlippageBps / 1e4;
        for (int i = 0; i < b.N; i++)
        {
            if (!_tradeable[i] || b.Mask[i] == 0) continue;
            var p = preds[i];
            if (!p.Valid) continue;
            long bid = s2.GoodBid[i], ask = s2.GoodAsk[i];
            double bidD = bid / (double)Price.Scale, askD = ask / (double)Price.Scale, mid = (bidD + askD) / 2;
            double cost = Passive ? _feeFrac + _cfg.PassiveCostBps / 1e4 : (askD - bidD) / 2 / mid + _feeFrac + slip;
            double mu = p.Mu, score = mu / p.Sigma;
            long pos = om.Position[i];

            if (mu > cost + margin && score >= _cfg.EntryScore)
            {
                var cutoff = _ticks.RoundDown(Price.FromDoubleFloor(mid * (1 + mu - _feeFrac - margin)));
                var limit = Price.Min(cutoff, new Price(Passive ? bid : ask));
                if (!limit.IsPositive) continue;
                double targetNotional = Math.Min(_maxPos, _cfg.BaseNotional * score / _cfg.EntryScore);
                long target = (long)(targetNotional / limit.ToDouble()) / _lot * _lot;
                long qty = target - pos;
                if (qty >= _lot)
                    intents.Add(new OrderIntent(i, Side.Buy, qty, limit, p.Mu, p.Sigma, (float)score, p.Beta, bid, ask, "entry", cutoff,
                        p.HasPeer ? p.Peer : -1, p.PeerWeight));
            }
            else if (pos > 0 && score < -_cfg.ExitScore)
            {
                var cutoff = _ticks.RoundUp(Price.FromDoubleCeiling(mid * (1 + mu + _feeFrac + margin)));
                var limit = Price.Max(cutoff, new Price(Passive ? ask : bid));
                intents.Add(new OrderIntent(i, Side.Sell, pos, limit, p.Mu, p.Sigma, (float)score, p.Beta, bid, ask, "exit", cutoff,
                    p.HasPeer ? p.Peer : -1, p.PeerWeight));
            }
        }
    }

    /// <summary>
    /// Whether an existing working order already satisfies this cycle's intent and should be kept as is.
    /// Aggressive: identical side, price and quantity. Passive: same side and quantity, still at or better than our
    /// side's best price, and not beyond the cutoff — so it keeps its place in the queue.
    /// </summary>
    public bool Keeps(WorkingOrder o, in OrderIntent it)
    {
        if (o.Req.Side != it.Side || o.Leaves != it.Qty) return false;
        if (!Passive || !it.Cutoff.IsPositive) return o.Req.Limit == it.Limit;
        long lim = o.Req.Limit.Raw;
        return it.Side == Side.Buy
            ? lim >= it.Limit.Raw && lim <= it.Cutoff.Raw
            : lim <= it.Limit.Raw && lim >= it.Cutoff.Raw;
    }
}
