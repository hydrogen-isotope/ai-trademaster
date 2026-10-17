using Trademaster.Core;

namespace Trademaster.Engine.Stages;

public enum MaskReason : byte { Valid = 0, NoQuote, Halted, Stale }

public sealed class QualityCounters
{
    public long Crossed, Locked, Jumps, OutlierTrades, StatusChanges;
}

/// <summary>
/// Stage 2 — data evaluation. Flags and weights observations; never rewrites prices.
/// Keeps the last <i>good</i> quote per instrument (crossed/locked quotes are flagged and not used),
/// flags jumps and outlier trades, tracks halts, and builds the per-instrument validity mask at the cutoff.
/// </summary>
public sealed class S2Evaluate
{
    private readonly int _n;
    private readonly long[] _maxAgeNs;
    private readonly double _jumpSigma, _outlierSpreads;
    private readonly long _feedTimeoutNs;
    private readonly double[] _absRetEwma;
    private readonly bool[] _isDirty;
    private long _lastAnyRecv;

    public readonly long[] GoodBid, GoodAsk, GoodBidSize, GoodAskSize, GoodRecvTs;
    public readonly TradingStatus[] Status;
    /// <summary>Quality weight in [0,1]: EWMA of the fraction of recent observations that passed checks.</summary>
    public readonly float[] Quality;
    public readonly MaskReason[] Mask;
    public readonly List<MarketEvent> GoodTrades = new(1 << 12);
    public readonly List<int> Dirty = new(1 << 10);
    public readonly QualityCounters Counters = new();
    public bool FeedDown { get; private set; }
    public int ValidCount { get; private set; }
    public int[] MaskCounts { get; } = new int[4];

    public S2Evaluate(Universe universe, QualitySection cfg)
    {
        _n = universe.Count;
        _maxAgeNs = universe.Instruments
            .Select(i => cfg.MaxAgeMs[i.Liquidity.ToString().ToLowerInvariant()] * Ns.PerMs).ToArray();
        _jumpSigma = cfg.JumpSigma;
        _outlierSpreads = cfg.OutlierTradeSpreads;
        _feedTimeoutNs = cfg.FeedTimeoutMs * Ns.PerMs;
        GoodBid = new long[_n]; GoodAsk = new long[_n]; GoodBidSize = new long[_n]; GoodAskSize = new long[_n]; GoodRecvTs = new long[_n];
        Status = new TradingStatus[_n];
        Quality = Enumerable.Repeat(1f, _n).ToArray();
        Mask = new MaskReason[_n];
        _absRetEwma = new double[_n];
        _isDirty = new bool[_n];
    }

    public double GoodMid(int i) => (GoodBid[i] + GoodAsk[i]) / (2.0 * Price.Scale);

    public void Evaluate(List<MarketEvent> accepted, long cutoff)
    {
        foreach (int i in Dirty) _isDirty[i] = false;
        Dirty.Clear();
        GoodTrades.Clear();

        foreach (var ev in accepted)
        {
            int i = ev.Instrument;
            if (ev.RecvTs > _lastAnyRecv) _lastAnyRecv = ev.RecvTs;
            MarkDirty(i);
            switch (ev.Kind)
            {
                case EventKind.Status:
                    Status[i] = ev.Status;
                    Counters.StatusChanges++;
                    break;
                case EventKind.Quote:
                    if (ev.Px1 >= ev.Px2)
                    {
                        if (ev.Px1 == ev.Px2) Counters.Locked++; else Counters.Crossed++;
                        Quality[i] = 0.9f * Quality[i];
                        break;
                    }
                    if (GoodBid[i] > 0)
                    {
                        double r = Math.Abs(Math.Log((ev.Px1 + ev.Px2) / (double)(GoodBid[i] + GoodAsk[i])));
                        if (_absRetEwma[i] > 0 && r > _jumpSigma * _absRetEwma[i]) Counters.Jumps++; // flagged, still accepted
                        _absRetEwma[i] = _absRetEwma[i] == 0 ? Math.Max(r, 1e-5) : 0.98 * _absRetEwma[i] + 0.02 * Math.Max(r, 1e-6);
                    }
                    GoodBid[i] = ev.Px1; GoodAsk[i] = ev.Px2; GoodBidSize[i] = ev.Sz1; GoodAskSize[i] = ev.Sz2;
                    GoodRecvTs[i] = ev.RecvTs;
                    Quality[i] = 0.9f * Quality[i] + 0.1f;
                    break;
                case EventKind.Trade:
                    if (GoodBid[i] > 0)
                    {
                        long spread = GoodAsk[i] - GoodBid[i];
                        double lo = GoodBid[i] - _outlierSpreads * spread, hi = GoodAsk[i] + _outlierSpreads * spread;
                        if (ev.Px1 < lo || ev.Px1 > hi)
                        {
                            Counters.OutlierTrades++;
                            Quality[i] = 0.9f * Quality[i];
                            break;
                        }
                    }
                    GoodTrades.Add(ev);
                    break;
            }
        }

        FeedDown = _lastAnyRecv == 0 || cutoff - _lastAnyRecv > _feedTimeoutNs;
        Array.Clear(MaskCounts);
        int valid = 0;
        for (int i = 0; i < _n; i++)
        {
            var m = GoodBid[i] == 0 ? MaskReason.NoQuote
                : Status[i] == TradingStatus.Halted ? MaskReason.Halted
                : cutoff - GoodRecvTs[i] > _maxAgeNs[i] ? MaskReason.Stale
                : MaskReason.Valid;
            Mask[i] = m;
            MaskCounts[(int)m]++;
            if (m == MaskReason.Valid) valid++;
        }
        ValidCount = valid;
    }

    private void MarkDirty(int i)
    {
        if (_isDirty[i]) return;
        _isDirty[i] = true;
        Dirty.Add(i);
    }

    /// <summary>Median and max age of the last good quote across valid instruments, in ms.</summary>
    public (double Median, double Max) DataAge(long cutoff, double[] scratch)
    {
        int k = 0;
        for (int i = 0; i < _n; i++) if (Mask[i] == MaskReason.Valid) scratch[k++] = Ns.ToMs(cutoff - GoodRecvTs[i]);
        if (k == 0) return (0, 0);
        Array.Sort(scratch, 0, k);
        return (scratch[k / 2], scratch[k - 1]);
    }
}
