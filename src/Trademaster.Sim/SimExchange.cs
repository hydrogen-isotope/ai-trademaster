using Trademaster.Core;

namespace Trademaster.Sim;

/// <summary>Venue-side record of one order, for execution analytics.</summary>
public sealed class OrderRecord
{
    public long OrderId; public int Instrument; public Side Side; public long Qty; public long Limit;
    public long SubmitTs; public double DecisionMid; public double DecisionHalfSpread;
    public long ArrivalTs; public bool RestedOnArrival; public long QueueAheadOnArrival = -1;
    public long FilledQty; public long FirstFillTs; public long EndTs; public string End = "open";
}

/// <summary>Venue-side record of one fill. Passive = we were resting (maker); otherwise we crossed (taker).</summary>
public readonly record struct FillRecord(long OrderId, int Instrument, Side Side, long Qty, long Price, long Ts, bool Passive, decimal Fee);

/// <summary>
/// Simulated execution venue driven by the true exchange stream (not the engine's defective feed).
/// Marketable limits fill at the opposite quote up to its displayed size. Resting limits fill according to
/// <see cref="ExecutionSection.FillModel"/> (see <see cref="FillModel"/>), and always when the market trades or
/// quotes through them. Messages take <see cref="ExecutionSection.OrderLatencyMs"/> each way.
/// A limit price never guarantees a fill. Every order and fill is recorded for execution analytics, and the
/// true mid-price path is kept so fills can be marked out afterwards.
/// </summary>
public sealed class SimExchange : IExecutionAdapter, IDisposable
{
    private sealed class SimOrder(OrderRequest req, OrderRecord rec)
    {
        public readonly OrderRequest Req = req;
        public readonly OrderRecord Rec = rec;
        public long Leaves = req.Qty;
        /// <summary>Displayed quantity ahead of us at our price (queue model). long.MaxValue/4 = deeper than L1, unknown.</summary>
        public long Ahead;
    }

    private const long Deep = long.MaxValue / 4;
    private enum MsgKind { New, Cancel }
    private readonly record struct Msg(MsgKind Kind, OrderRequest Req, long OrderId);

    private readonly ReplayReader _md;
    private readonly MarketCalendar _calendar;
    private readonly TickTable _ticks;
    private readonly ExchangeConfig _ex;
    private readonly ExecutionSection _cfg;
    private readonly long _latency;
    private readonly Random _rng;
    private readonly (long Start, long End)[] _disconnects;

    private readonly long[] _bid, _ask, _bidSz, _askSz;
    private readonly bool[] _halted;
    private readonly List<SimOrder>[] _resting;
    private readonly Dictionary<long, SimOrder> _open = [];
    private readonly HashSet<long> _closed = [];
    private readonly Dictionary<int, long> _positions = [];
    private readonly PriorityQueue<Msg, (long Ts, long Seq)> _inbound = new();
    private readonly Queue<(long Deliver, ExecutionReport Report)> _outbound = new();
    private long _msgSeq;
    private long _now = long.MinValue;
    private int _nextDisconnect;

    // Execution analytics
    public readonly Dictionary<long, OrderRecord> Orders = [];
    public readonly List<FillRecord> FillLog = [];
    private readonly List<long>[] _midTs;
    private readonly List<(double Bid, double Ask)>[] _mid;

    public SimExchange(string exchangeReplayPath, Universe universe, ExchangeConfig exchange, ExecutionSection cfg, long replayStartTs)
    {
        _md = new ReplayReader(exchangeReplayPath);
        _ex = exchange;
        _calendar = new MarketCalendar(exchange);
        _ticks = exchange.BuildTickTable();
        _cfg = cfg;
        _latency = cfg.OrderLatencyMs * Ns.PerMs;
        _rng = new Random(cfg.Seed);
        _disconnects = cfg.Disconnects
            .Select(d => (replayStartTs + (long)(d.AtS * Ns.PerSec), replayStartTs + (long)((d.AtS + d.DurationS) * Ns.PerSec)))
            .OrderBy(d => d.Item1).ToArray();
        int n = universe.Count;
        _bid = new long[n]; _ask = new long[n]; _bidSz = new long[n]; _askSz = new long[n];
        _halted = new bool[n];
        _resting = new List<SimOrder>[n];
        _midTs = new List<long>[n];
        _mid = new List<(double, double)>[n];
        for (int i = 0; i < n; i++) { _resting[i] = []; _midTs[i] = []; _mid[i] = []; }
    }

    public string Name => "sim";
    public bool IsSimulated => true;
    public bool IsConnected => !InDisconnect(_now);
    public long Now => _now;

    // Counters for reports
    public long Fills { get; private set; }
    public long Rejects { get; private set; }
    public long Cancels { get; private set; }
    public long CancelRejects { get; private set; }
    public long CodCancels { get; private set; }

    /// <summary>Current true best bid/ask (raw price units) and sizes for an instrument.</summary>
    public (long Bid, long Ask, long BidSize, long AskSize) Quote(int i) => (_bid[i], _ask[i], _bidSz[i], _askSz[i]);

    /// <summary>True best bid/ask (currency units) at or before <paramref name="ts"/> (NaN before the first quote).</summary>
    public (double Bid, double Ask) QuoteAt(int i, long ts)
    {
        var t = _midTs[i];
        int lo = 0, hi = t.Count - 1, ans = -1;
        while (lo <= hi) { int m = (lo + hi) >> 1; if (t[m] <= ts) { ans = m; lo = m + 1; } else hi = m - 1; }
        return ans < 0 ? (double.NaN, double.NaN) : _mid[i][ans];
    }

    /// <summary>True mid at or before <paramref name="ts"/> (NaN before the first quote).</summary>
    public double MidAt(int i, long ts)
    {
        var (b, a) = QuoteAt(i, ts);
        return (b + a) / 2;
    }

    private bool InDisconnect(long ts)
    {
        foreach (var (s, e) in _disconnects) if (ts >= s && ts < e) return true;
        return false;
    }

    private long DisconnectEnd(long ts)
    {
        foreach (var (s, e) in _disconnects) if (ts >= s && ts < e) return e;
        return ts;
    }

    public SubmitResult Submit(in OrderRequest request, long ts)
    {
        if (InDisconnect(ts)) return SubmitResult.Disconnected;
        int i = request.Instrument;
        if (i >= 0 && i < _bid.Length && !Orders.ContainsKey(request.OrderId))
        {
            double mid = _bid[i] > 0 && _ask[i] > 0 ? (_bid[i] + _ask[i]) / 2.0 / Price.Scale : double.NaN;
            Orders[request.OrderId] = new OrderRecord
            {
                OrderId = request.OrderId, Instrument = i, Side = request.Side, Qty = request.Qty, Limit = request.Limit.Raw,
                SubmitTs = ts, DecisionMid = mid, DecisionHalfSpread = (_ask[i] - _bid[i]) / 2.0 / Price.Scale,
            };
        }
        _inbound.Enqueue(new Msg(MsgKind.New, request, request.OrderId), (ts + _latency, ++_msgSeq));
        return SubmitResult.Sent;
    }

    public SubmitResult Cancel(long orderId, long ts)
    {
        if (InDisconnect(ts)) return SubmitResult.Disconnected;
        _inbound.Enqueue(new Msg(MsgKind.Cancel, default, orderId), (ts + _latency, ++_msgSeq));
        return SubmitResult.Sent;
    }

    public void AdvanceTo(long ts)
    {
        while (true)
        {
            long mdTs = _md.TryPeek(out var ev) ? ev.ExchangeTs : long.MaxValue;
            long dcTs = _nextDisconnect < _disconnects.Length ? _disconnects[_nextDisconnect].Start : long.MaxValue;
            long msgTs = _inbound.TryPeek(out _, out var pri) ? pri.Ts : long.MaxValue;
            long next = Math.Min(mdTs, Math.Min(dcTs, msgTs));
            if (next > ts) break;
            if (mdTs == next) { _md.TryRead(out ev); OnMarket(ev); }
            else if (dcTs == next) { _nextDisconnect++; OnDisconnect(next); }
            else
            {
                var msg = _inbound.Dequeue();
                _now = next;
                if (msg.Kind == MsgKind.New) OnNew(msg.Req, next); else OnCancel(msg.OrderId, next);
            }
        }
        _now = ts;
    }

    public void DrainReports(long ts, List<ExecutionReport> into)
    {
        while (_outbound.TryPeek(out var r) && r.Deliver <= ts) into.Add(_outbound.Dequeue().Report);
    }

    public AccountSnapshot Snapshot(long ts) => new(
        ts,
        _open.Values.OrderBy(o => o.Req.OrderId).Select(o => new OpenOrderSnapshot(o.Req, o.Leaves)).ToList(),
        new Dictionary<int, long>(_positions));

    // ------------------------------------------------------------------ venue internals

    private void Emit(in ExecutionReport r)
    {
        long deliver = InDisconnect(r.Ts) ? DisconnectEnd(r.Ts) + _latency : r.Ts + _latency;
        _outbound.Enqueue((deliver, r));
    }

    private void OnDisconnect(long ts)
    {
        if (!_cfg.CancelOnDisconnect) return;
        foreach (var o in _open.Values.OrderBy(o => o.Req.OrderId).ToList())
        {
            CloseOrder(o, ts, "cancel_on_disconnect");
            CodCancels++;
            Emit(new ExecutionReport(o.Req.OrderId, o.Req.Instrument, o.Req.Side, ReportKind.Cancelled, ts, 0, Price.Zero, 0, o.Leaves, "CancelOnDisconnect"));
        }
    }

    private bool Queue => _cfg.FillModel == FillModel.Queue;

    private void OnMarket(in MarketEvent ev)
    {
        _now = ev.ExchangeTs;
        int i = ev.Instrument;
        switch (ev.Kind)
        {
            case EventKind.Status:
                _halted[i] = ev.Status == TradingStatus.Halted;
                break;
            case EventKind.Quote:
            {
                long prevBid = _bid[i], prevAsk = _ask[i], prevBidSz = _bidSz[i], prevAskSz = _askSz[i];
                _bid[i] = ev.Px1; _ask[i] = ev.Px2; _bidSz[i] = ev.Sz1; _askSz[i] = ev.Sz2;
                if (ev.Px1 > 0 && ev.Px2 > ev.Px1)
                {
                    _midTs[i].Add(ev.ExchangeTs);
                    _mid[i].Add((ev.Px1 / (double)Price.Scale, ev.Px2 / (double)Price.Scale));
                }
                if (_halted[i]) break;
                var rest = _resting[i];
                for (int k = rest.Count - 1; k >= 0; k--)
                {
                    var o = rest[k];
                    long lim = o.Req.Limit.Raw;
                    // Opposite side now at or through our price: the market crossed us; we fill at our price.
                    if (o.Req.Side == Side.Buy && _ask[i] > 0 && _ask[i] <= lim)
                    {
                        Fill(o, Lots(Math.Min(o.Leaves, _askSz[i])), o.Req.Limit, ev.ExchangeTs, passive: true);
                        continue;
                    }
                    if (o.Req.Side == Side.Sell && _bid[i] > 0 && _bid[i] >= lim)
                    {
                        Fill(o, Lots(Math.Min(o.Leaves, _bidSz[i])), o.Req.Limit, ev.ExchangeTs, passive: true);
                        continue;
                    }
                    if (Queue)
                    {
                        bool buy = o.Req.Side == Side.Buy;
                        long best = buy ? _bid[i] : _ask[i], prevBest = buy ? prevBid : prevAsk;
                        long size = buy ? _bidSz[i] : _askSz[i], prevSize = buy ? prevBidSz : prevAskSz;
                        bool better = buy ? best < lim : best > lim;    // our price is now better than the displayed best
                        if (better) o.Ahead = 0;                         // the level we joined emptied: we are first
                        else if (best == lim)
                        {
                            if (prevBest == lim && size < prevSize && prevSize > 0 && o.Ahead < Deep)
                                o.Ahead -= (long)Math.Round((prevSize - size) * (double)o.Ahead / prevSize); // cancels, pro rata
                            else if (prevBest != lim)
                                o.Ahead = Math.Min(o.Ahead, size);       // our deeper level became best
                            o.Ahead = Math.Clamp(o.Ahead, 0, size);
                        }
                    }
                }
                break;
            }
            case EventKind.Trade:
                if (_halted[i]) break;
                var resting = _resting[i];
                for (int k = resting.Count - 1; k >= 0; k--)
                {
                    var o = resting[k];
                    long lim = o.Req.Limit.Raw, px = ev.Px1;
                    bool through = o.Req.Side == Side.Buy ? px < lim : px > lim;
                    if (through)
                    {
                        Fill(o, o.Leaves, o.Req.Limit, ev.ExchangeTs, passive: true);
                        continue;
                    }
                    if (px != lim) continue;
                    if (Queue)
                    {
                        if (o.Ahead >= Deep) continue;
                        long size = ev.Sz1;
                        if (o.Ahead >= size) { o.Ahead -= size; continue; }
                        long q = Lots(Math.Min(o.Leaves, size - o.Ahead));
                        o.Ahead = 0;
                        Fill(o, q, o.Req.Limit, ev.ExchangeTs, passive: true);
                    }
                    else if (_rng.NextDouble() < _cfg.TouchFillProb)
                    {
                        long cap = Lots((long)Math.Floor(ev.Sz1 * _cfg.Participation));
                        Fill(o, Math.Min(o.Leaves, cap), o.Req.Limit, ev.ExchangeTs, passive: true);
                    }
                }
                break;
        }
    }

    private long Lots(long qty)
    {
        long lot = Math.Max(1, _ex.LotSize);
        return qty / lot * lot;
    }

    private void OnNew(in OrderRequest req, long ts)
    {
        string? reason = null;
        int i = req.Instrument;
        if (_open.ContainsKey(req.OrderId) || _closed.Contains(req.OrderId)) reason = "DuplicateOrderId";
        else if (i < 0 || i >= _bid.Length) reason = "UnknownInstrument";
        else if (_calendar.Phase(ts) != MarketPhase.Open) reason = "MarketClosed";
        else if (_halted[i]) reason = "Halted";
        else if (!_ticks.IsValid(req.Limit)) reason = "InvalidTick";
        else if (req.Qty <= 0 || req.Qty % Math.Max(1, _ex.LotSize) != 0) reason = "InvalidLot";
        else if (_bid[i] > 0 && _ask[i] > 0)
        {
            double mid = (_bid[i] + _ask[i]) / 2.0;
            if (Math.Abs(req.Limit.Raw - mid) / mid * 100.0 > (double)_ex.PriceCollarPct) reason = "PriceCollar";
        }
        Orders.TryGetValue(req.OrderId, out var rec);
        if (reason is not null)
        {
            Rejects++;
            _closed.Add(req.OrderId);
            if (rec is not null) { rec.End = "rejected:" + reason; rec.EndTs = ts; }
            Emit(new ExecutionReport(req.OrderId, i, req.Side, ReportKind.Reject, ts, 0, Price.Zero, 0, 0, reason));
            return;
        }
        rec ??= Orders[req.OrderId] = new OrderRecord { OrderId = req.OrderId, Instrument = i, Side = req.Side, Qty = req.Qty, Limit = req.Limit.Raw, SubmitTs = ts, DecisionMid = double.NaN };
        rec.ArrivalTs = ts;

        var o = new SimOrder(req, rec);
        _open[req.OrderId] = o;
        Emit(new ExecutionReport(req.OrderId, i, req.Side, ReportKind.Ack, ts, 0, Price.Zero, 0, o.Leaves, null));

        // Marketable on arrival: take liquidity at the opposite quote.
        if (req.Side == Side.Buy && _ask[i] > 0 && req.Limit.Raw >= _ask[i])
        {
            long q = Lots(Math.Min(o.Leaves, _askSz[i]));
            _askSz[i] -= q;
            Fill(o, q, new Price(_ask[i]), ts, passive: false);
        }
        else if (req.Side == Side.Sell && _bid[i] > 0 && req.Limit.Raw <= _bid[i])
        {
            long q = Lots(Math.Min(o.Leaves, _bidSz[i]));
            _bidSz[i] -= q;
            Fill(o, q, new Price(_bid[i]), ts, passive: false);
        }
        if (o.Leaves > 0)
        {
            long best = req.Side == Side.Buy ? _bid[i] : _ask[i];
            long size = req.Side == Side.Buy ? _bidSz[i] : _askSz[i];
            long lim = req.Limit.Raw;
            bool improves = req.Side == Side.Buy ? lim > best : lim < best;
            o.Ahead = best <= 0 || improves ? 0 : lim == best ? size : Deep;
            rec.RestedOnArrival = true;
            rec.QueueAheadOnArrival = o.Ahead >= Deep ? -1 : o.Ahead;
            _resting[i].Add(o);
        }
    }

    private void OnCancel(long orderId, long ts)
    {
        if (_open.TryGetValue(orderId, out var o))
        {
            CloseOrder(o, ts, "cancelled");
            Cancels++;
            Emit(new ExecutionReport(orderId, o.Req.Instrument, o.Req.Side, ReportKind.Cancelled, ts, 0, Price.Zero, 0, o.Leaves, null));
        }
        else
        {
            CancelRejects++;
            Emit(new ExecutionReport(orderId, -1, Side.Buy, ReportKind.CancelReject, ts, 0, Price.Zero, 0, 0,
                _closed.Contains(orderId) ? "TooLateToCancel" : "UnknownOrder"));
        }
    }

    private void Fill(SimOrder o, long qty, Price px, long ts, bool passive)
    {
        if (qty <= 0) return;
        o.Leaves -= qty;
        Fills++;
        decimal notional = qty * (decimal)px.Raw / Price.Scale;
        decimal fee = Math.Max(_ex.MinFee, notional * _ex.FeeBps / 10_000m);
        _positions[o.Req.Instrument] = _positions.GetValueOrDefault(o.Req.Instrument) + (o.Req.Side == Side.Buy ? qty : -qty);
        if (o.Rec.FilledQty == 0) o.Rec.FirstFillTs = ts;
        o.Rec.FilledQty += qty;
        FillLog.Add(new FillRecord(o.Req.OrderId, o.Req.Instrument, o.Req.Side, qty, px.Raw, ts, passive, fee));
        Emit(new ExecutionReport(o.Req.OrderId, o.Req.Instrument, o.Req.Side, ReportKind.Fill, ts, qty, px, fee, o.Leaves, null));
        if (o.Leaves == 0) CloseOrder(o, ts, "filled");
    }

    private void CloseOrder(SimOrder o, long ts, string why)
    {
        _open.Remove(o.Req.OrderId);
        _closed.Add(o.Req.OrderId);
        _resting[o.Req.Instrument].Remove(o);
        o.Rec.End = why;
        o.Rec.EndTs = ts;
    }

    public void Dispose() => _md.Dispose();
}
