using System.Diagnostics;

namespace Trademaster.Engine;

/// <summary>Wall clock used for paced (real-time) runs. Abstracted so deadline handling can be tested deterministically.</summary>
public interface IWallClock
{
    long NowNs();
    void WaitUntil(long ns);
}

/// <summary>QueryPerformanceCounter-based clock. Sleeps coarsely, then spins the last 20 ms for an accurate start.</summary>
public sealed class SystemWallClock : IWallClock
{
    private static readonly double NsPerTick = 1e9 / Stopwatch.Frequency;

    public long NowNs() => (long)(Stopwatch.GetTimestamp() * NsPerTick);

    public void WaitUntil(long ns)
    {
        while (true)
        {
            long remaining = ns - NowNs();
            if (remaining <= 0) return;
            if (remaining > 20_000_000) Thread.Sleep((int)((remaining - 20_000_000) / 1_000_000));
            else Thread.SpinWait(50);
        }
    }
}

/// <summary>Manually driven clock for tests.</summary>
public sealed class FakeWallClock(long start = 0) : IWallClock
{
    public long Now { get; set; } = start;
    public long NowNs() => Now;
    public void WaitUntil(long ns) { if (Now < ns) Now = ns; }
    public void Advance(long ns) => Now += ns;
}
