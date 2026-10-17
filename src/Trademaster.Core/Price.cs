using System.Globalization;

namespace Trademaster.Core;

/// <summary>
/// Fixed-point price in units of 1/10,000 of the quote currency.
/// Prices never travel as floating point between stages; only features and models use floats.
/// </summary>
public readonly record struct Price(long Raw) : IComparable<Price>
{
    public const long Scale = 10_000;
    public static readonly Price Zero = new(0);

    public bool IsPositive => Raw > 0;
    public double ToDouble() => Raw / (double)Scale;

    public static Price FromDecimal(decimal value) =>
        new((long)Math.Round(value * Scale, MidpointRounding.AwayFromZero));

    /// <summary>Largest representable price ≤ value (tolerates binary float noise).</summary>
    public static Price FromDoubleFloor(double value) => new((long)Math.Floor(value * Scale + 1e-6));

    /// <summary>Smallest representable price ≥ value (tolerates binary float noise).</summary>
    public static Price FromDoubleCeiling(double value) => new((long)Math.Ceiling(value * Scale - 1e-6));

    public int CompareTo(Price other) => Raw.CompareTo(other.Raw);
    public static bool operator <(Price a, Price b) => a.Raw < b.Raw;
    public static bool operator >(Price a, Price b) => a.Raw > b.Raw;
    public static bool operator <=(Price a, Price b) => a.Raw <= b.Raw;
    public static bool operator >=(Price a, Price b) => a.Raw >= b.Raw;
    public static Price Min(Price a, Price b) => a.Raw <= b.Raw ? a : b;
    public static Price Max(Price a, Price b) => a.Raw >= b.Raw ? a : b;

    public override string ToString() =>
        (Raw / (decimal)Scale).ToString("0.0000", CultureInfo.InvariantCulture);
}
