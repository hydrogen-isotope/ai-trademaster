using Trademaster.Core;

namespace Trademaster.Engine.Stages;

public sealed class IngestCounters
{
    /// <summary>Reordered = applied from the reorder buffer (arrived ahead of a missing sequence number).</summary>
    public long Received, Applied, Duplicates, LateDropped, GapLost, Reordered, Invalid;
}

/// <summary>
/// Stage 1 — market data collection. Consumes the feed in arrival order up to the cycle cutoff,
/// validates records, removes duplicates (by feed sequence number), restores sequence order within a
/// bounded reorder window, declares gaps lost after the window, and keeps the latest raw state per instrument.
/// Accepted events for the cycle are handed to stage 2 in sequence order.
/// </summary>
public sealed class S1Ingest
{
    private readonly ReplayReader _feed;
    private readonly int _n;
    private readonly long _windowNs;
    private readonly Dictionary<long, MarketEvent> _buffer = [];
    private readonly HashSet<long> _lost = [];
    private long _expected = -1;
    private long _bufferOldestRecv = long.MaxValue;

    // Latest raw state per instrument (before quality evaluation).
    public readonly long[] LastRecvTs, LastExchangeTs;

    public readonly List<MarketEvent> Accepted = new(1 << 14);
    public readonly IngestCounters Counters = new();
    public double CycleMaxFeedLatencyMs { get; private set; }

    public S1Ingest(ReplayReader feed, int instruments, int reorderWindowMs)
    {
        _feed = feed;
        _n = instruments;
        _windowNs = reorderWindowMs * Ns.PerMs;
        LastRecvTs = new long[instruments];
        LastExchangeTs = new long[instruments];
    }

    public int Buffered => _buffer.Count;

    /// <summary>Consumes every feed record with RecvTs ≤ cutoff.</summary>
    public void ProcessUntil(long cutoff)
    {
        Accepted.Clear();
        CycleMaxFeedLatencyMs = 0;
        while (_feed.TryPeek(out var ev) && ev.RecvTs <= cutoff)
        {
            _feed.TryRead(out ev);
            ExpireGaps(ev.RecvTs);
            OnEvent(ev);
        }
        ExpireGaps(cutoff);
        if (_lost.Count > 100_000) _lost.RemoveWhere(s => s < _expected - 1_000_000);
    }

    private void OnEvent(in MarketEvent ev)
    {
        Counters.Received++;
        if (_expected < 0) _expected = ev.Seq;

        if (ev.Seq == _expected)
        {
            Apply(ev);
            _expected++;
            Drain();
        }
        else if (ev.Seq < _expected)
        {
            if (_lost.Remove(ev.Seq)) Counters.LateDropped++;   // arrived after its gap was declared lost
            else Counters.Duplicates++;
        }
        else if (!_buffer.TryAdd(ev.Seq, ev)) Counters.Duplicates++;
        else if (ev.RecvTs < _bufferOldestRecv) _bufferOldestRecv = ev.RecvTs;
    }

    private void Drain()
    {
        bool drained = false;
        while (_buffer.Remove(_expected, out var next))
        {
            Counters.Reordered++;
            Apply(next);
            _expected++;
            drained = true;
        }
        if (drained) RecomputeOldest();
    }

    private void ExpireGaps(long now)
    {
        while (_buffer.Count > 0 && now - _bufferOldestRecv > _windowNs)
        {
            long minSeq = long.MaxValue;
            foreach (var s in _buffer.Keys) if (s < minSeq) minSeq = s;
            for (long s = _expected; s < minSeq; s++) _lost.Add(s);
            Counters.GapLost += minSeq - _expected;
            _expected = minSeq;
            Drain();
        }
    }

    private void RecomputeOldest()
    {
        _bufferOldestRecv = long.MaxValue;
        foreach (var e in _buffer.Values) if (e.RecvTs < _bufferOldestRecv) _bufferOldestRecv = e.RecvTs;
    }

    private void Apply(in MarketEvent ev)
    {
        if (!IsValid(ev)) { Counters.Invalid++; return; }
        Counters.Applied++;
        LastRecvTs[ev.Instrument] = ev.RecvTs;
        LastExchangeTs[ev.Instrument] = ev.ExchangeTs;
        double lat = Ns.ToMs(ev.RecvTs - ev.ExchangeTs);
        if (lat > CycleMaxFeedLatencyMs) CycleMaxFeedLatencyMs = lat;
        Accepted.Add(ev);
    }

    private bool IsValid(in MarketEvent ev)
    {
        if ((uint)ev.Instrument >= (uint)_n) return false;
        if (ev.RecvTs < ev.ExchangeTs) return false;
        return ev.Kind switch
        {
            EventKind.Quote => ev.Px1 > 0 && ev.Px2 > 0 && ev.Sz1 >= 0 && ev.Sz2 >= 0,
            EventKind.Trade => ev.Px1 > 0 && ev.Sz1 > 0,
            EventKind.Status => ev.Status is TradingStatus.Trading or TradingStatus.Halted,
            _ => false,
        };
    }
}
