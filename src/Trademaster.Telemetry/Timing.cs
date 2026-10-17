using System.Diagnostics;

namespace Trademaster.Telemetry;

/// <summary>Measured pipeline sections. H2D/D2H stay zero on CPU-only runs and are reported as n/a.</summary>
public enum Stage
{
    Ingest, Evaluate, Features, Pack, HostToDevice, Inference, DeviceToHost, ExecSync, Decision, Risk, OrderInterface, Audit,
}

public enum CycleStatus { Ok, Missed, SkippedOverrun, Closed, FeedDown, KillSwitch, Disconnected }

/// <summary>One row of per-cycle telemetry (wall-clock measurements; never part of the deterministic decision log).</summary>
public struct CycleTiming
{
    public const int StageCount = (int)Stage.Audit + 1;

    public long CycleId;
    public long CutoffTs;
    public CycleStatus Status;
    public bool Warmup;
    /// <summary>Compute time exceeded the budget (in virtual mode this is a would-miss; decisions are unaffected).</summary>
    public bool OverBudget;
    public long TotalNs;
    /// <summary>Cutoff → order interface, in wall time (paced mode); equals TotalNs before audit in virtual mode.</summary>
    public long ToOrderNs;
    public long StartLagNs;
    public long GcPauseNs;
    public long AllocBytes;
    public int AcceptedEvents;
    public int DirtyInstruments;
    public int ValidInstruments;
    public double MaxDataAgeMs;
    public double MedianDataAgeMs;
    public double MaxFeedLatencyMs;
    public int OrdersSubmitted;
    public int Cancels;
    public long[] StageNs;

    public static CycleTiming Create() => new() { StageNs = new long[StageCount] };
}

/// <summary>Accumulates high-resolution stage timings for one cycle.</summary>
public sealed class StageClock
{
    private long _mark;
    private readonly long[] _acc = new long[CycleTiming.StageCount];
    private long _start;

    public static double TicksToNs => 1e9 / Stopwatch.Frequency;

    public void Begin()
    {
        Array.Clear(_acc);
        _start = _mark = Stopwatch.GetTimestamp();
    }

    /// <summary>Attributes the time since the previous mark to <paramref name="stage"/>.</summary>
    public void Lap(Stage stage)
    {
        long now = Stopwatch.GetTimestamp();
        _acc[(int)stage] += now - _mark;
        _mark = now;
    }

    public long ElapsedNs => (long)((Stopwatch.GetTimestamp() - _start) * TicksToNs);

    public void CopyTo(long[] stageNs)
    {
        for (int i = 0; i < _acc.Length; i++) stageNs[i] = (long)(_acc[i] * TicksToNs);
    }
}

public readonly record struct Percentiles(int Count, double P50, double P95, double P99, double Max, double Mean)
{
    public static Percentiles Of(IEnumerable<double> values)
    {
        var a = values.ToArray();
        if (a.Length == 0) return new(0, 0, 0, 0, 0, 0);
        Array.Sort(a);
        double Q(double q) => a[Math.Min(a.Length - 1, (int)Math.Ceiling(q * a.Length) - 1)];
        return new(a.Length, Q(0.50), Q(0.95), Q(0.99), a[^1], a.Average());
    }
}
