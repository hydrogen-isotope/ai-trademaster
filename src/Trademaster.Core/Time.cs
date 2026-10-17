using System.Globalization;

namespace Trademaster.Core;

/// <summary>All timestamps are int64 nanoseconds since the Unix epoch, UTC.</summary>
public static class Ns
{
    public const long PerUs = 1_000;
    public const long PerMs = 1_000_000;
    public const long PerSec = 1_000_000_000;

    public static long FromUtc(DateTime utc) => (utc.ToUniversalTime() - DateTime.UnixEpoch).Ticks * 100;
    public static DateTime ToUtc(long ns) => DateTime.UnixEpoch.AddTicks(ns / 100);
    public static long FromMs(double ms) => (long)(ms * PerMs);
    public static double ToMs(long ns) => ns / (double)PerMs;
    public static double ToSec(long ns) => ns / (double)PerSec;

    public static string Iso(long ns) =>
        ToUtc(ns).ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);
}
