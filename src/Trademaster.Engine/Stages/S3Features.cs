using Trademaster.Core;

namespace Trademaster.Engine.Stages;

/// <summary>Feature indices of the per-instrument feature vector.</summary>
public static class F
{
    public const int R1 = 0;          // log return of good mid since previous cutoff
    public const int Ret10 = 1;       // exp-decayed sum of log returns, tau 10 s
    public const int Ret60 = 2;       // exp-decayed sum of log returns, tau 60 s
    public const int Vol60 = 3;       // sqrt of exp-decayed sum of squared returns, tau 60 s
    public const int SpreadBps = 4;
    public const int MicroBps = 5;    // microprice offset from mid, bps
    public const int Imbalance = 6;   // (bidSize - askSize) / (bidSize + askSize)
    public const int Flow = 7;        // exp-decayed signed trade volume (lots), tau 10 s
    public const int Intensity = 8;   // exp-decayed trade count, tau 10 s
    public const int AgeSec = 9;      // seconds since last good quote
    public const int Count = 10;

    public static readonly string[] Names =
        ["r1", "ret10", "ret60", "vol60", "spread_bps", "micro_bps", "imbalance", "flow", "intensity", "age_s"];
}

/// <summary>
/// Stage 3 — per-instrument feature state, updated incrementally: only instruments with new events
/// are touched. Time-decayed sums are stored with their last-update timestamp and decayed lazily when
/// read, so idle instruments cost nothing. All timestamps used are RecvTs ≤ cutoff (point-in-time).
/// </summary>
public sealed class S3Features
{
    private const double Tau10 = 10.0, Tau60 = 60.0;
    private readonly int _n, _lot;
    private readonly S2Evaluate _s2;

    private readonly double[] _lastMid, _midAtPrevCutoff;
    private readonly double[] _ret10, _ret60, _var60;
    private readonly long[] _retTs;
    private readonly double[] _flow, _intensity;
    private readonly long[] _flowTs;

    public int UpdatedLastCycle { get; private set; }

    public S3Features(int instruments, int lotSize, S2Evaluate s2)
    {
        _n = instruments;
        _lot = Math.Max(1, lotSize);
        _s2 = s2;
        _lastMid = new double[_n]; _midAtPrevCutoff = new double[_n];
        _ret10 = new double[_n]; _ret60 = new double[_n]; _var60 = new double[_n]; _retTs = new long[_n];
        _flow = new double[_n]; _intensity = new double[_n]; _flowTs = new long[_n];
    }

    private static double Decay(double v, long fromTs, long toTs, double tau) =>
        v == 0 || toTs <= fromTs ? v : v * Math.Exp(-Ns.ToSec(toTs - fromTs) / tau);

    public void Update()
    {
        foreach (var t in _s2.GoodTrades)
        {
            int i = t.Instrument;
            long bid = _s2.GoodBid[i], ask = _s2.GoodAsk[i];
            int sign = bid == 0 ? 0 : t.Px1 >= ask ? 1 : t.Px1 <= bid ? -1 : 0;
            _flow[i] = Decay(_flow[i], _flowTs[i], t.RecvTs, Tau10) + sign * (t.Sz1 / (double)_lot);
            _intensity[i] = Decay(_intensity[i], _flowTs[i], t.RecvTs, Tau10) + 1.0;
            _flowTs[i] = t.RecvTs;
        }

        int updated = 0;
        foreach (int i in _s2.Dirty)
        {
            if (_s2.GoodBid[i] == 0) continue;
            double mid = _s2.GoodMid(i);
            if (mid == _lastMid[i]) continue;
            long ts = _s2.GoodRecvTs[i];
            if (_lastMid[i] > 0)
            {
                double r = Math.Log(mid / _lastMid[i]);
                _ret10[i] = Decay(_ret10[i], _retTs[i], ts, Tau10) + r;
                _ret60[i] = Decay(_ret60[i], _retTs[i], ts, Tau60) + r;
                _var60[i] = Decay(_var60[i], _retTs[i], ts, Tau60) + r * r;
            }
            else _midAtPrevCutoff[i] = mid;
            _lastMid[i] = mid;
            _retTs[i] = ts;
            updated++;
        }
        UpdatedLastCycle = updated;
    }

    /// <summary>Writes instrument <paramref name="i"/>'s raw features at <paramref name="cutoff"/> into <paramref name="dst"/>.</summary>
    public void Read(int i, long cutoff, Span<float> dst)
    {
        if (_lastMid[i] == 0) { dst.Clear(); return; }
        double mid = _lastMid[i];
        long bid = _s2.GoodBid[i], ask = _s2.GoodAsk[i], bs = _s2.GoodBidSize[i], az = _s2.GoodAskSize[i];
        double bidD = bid / (double)Price.Scale, askD = ask / (double)Price.Scale;
        double micro = bs + az > 0 ? (bidD * az + askD * bs) / (bs + az) : mid;

        dst[F.R1] = (float)(_midAtPrevCutoff[i] > 0 ? Math.Log(mid / _midAtPrevCutoff[i]) : 0);
        dst[F.Ret10] = (float)Decay(_ret10[i], _retTs[i], cutoff, Tau10);
        dst[F.Ret60] = (float)Decay(_ret60[i], _retTs[i], cutoff, Tau60);
        dst[F.Vol60] = (float)Math.Sqrt(Decay(_var60[i], _retTs[i], cutoff, Tau60));
        dst[F.SpreadBps] = (float)((askD - bidD) / mid * 1e4);
        dst[F.MicroBps] = (float)((micro - mid) / mid * 1e4);
        dst[F.Imbalance] = bs + az > 0 ? (float)((bs - az) / (double)(bs + az)) : 0f;
        dst[F.Flow] = (float)Decay(_flow[i], _flowTs[i], cutoff, Tau10);
        dst[F.Intensity] = (float)Decay(_intensity[i], _flowTs[i], cutoff, Tau10);
        dst[F.AgeSec] = (float)Ns.ToSec(cutoff - _s2.GoodRecvTs[i]);
    }

    /// <summary>Marks the current mids as the reference for next cycle's one-cycle return.</summary>
    public void EndCycle()
    {
        for (int i = 0; i < _n; i++) if (_lastMid[i] > 0) _midAtPrevCutoff[i] = _lastMid[i];
    }
}
