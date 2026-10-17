namespace Trademaster.Core;

/// <summary>A price band: from <see cref="From"/> (inclusive) upward the minimum increment is <see cref="Tick"/>.</summary>
public sealed record TickBand(decimal From, decimal Tick);

/// <summary>Exchange price-increment table with conservative rounding helpers.</summary>
public sealed class TickTable
{
    private readonly long[] _from;
    private readonly long[] _tick;

    public TickTable(IReadOnlyList<TickBand> bands)
    {
        if (bands.Count == 0) throw new ArgumentException("Tick table needs at least one band.");
        _from = new long[bands.Count];
        _tick = new long[bands.Count];
        for (int i = 0; i < bands.Count; i++)
        {
            _from[i] = Price.FromDecimal(bands[i].From).Raw;
            _tick[i] = Price.FromDecimal(bands[i].Tick).Raw;
            if (_tick[i] <= 0) throw new ArgumentException($"Band {i}: tick must be positive.");
            if (i == 0 && _from[0] != 0) throw new ArgumentException("First band must start at 0.");
            if (i > 0)
            {
                if (_from[i] <= _from[i - 1]) throw new ArgumentException("Bands must be strictly increasing.");
                if ((_from[i] - _from[i - 1]) % _tick[i - 1] != 0)
                    throw new ArgumentException($"Band {i} start is not on the previous band's tick grid.");
            }
        }
    }

    private int BandOf(long raw)
    {
        int lo = 0, hi = _from.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (_from[mid] <= raw) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

    public Price TickAt(Price p) => new(_tick[BandOf(Math.Max(0, p.Raw))]);

    public bool IsValid(Price p)
    {
        if (p.Raw <= 0) return false;
        int b = BandOf(p.Raw);
        return (p.Raw - _from[b]) % _tick[b] == 0;
    }

    /// <summary>Largest valid price ≤ p (use for buy limits).</summary>
    public Price RoundDown(Price p)
    {
        if (p.Raw <= 0) return Price.Zero;
        int b = BandOf(p.Raw);
        return new(_from[b] + (p.Raw - _from[b]) / _tick[b] * _tick[b]);
    }

    /// <summary>Smallest valid price ≥ p (use for sell limits).</summary>
    public Price RoundUp(Price p)
    {
        if (p.Raw <= 0) return new(_tick[0]);
        int b = BandOf(p.Raw);
        long off = p.Raw - _from[b];
        long v = _from[b] + (off + _tick[b] - 1) / _tick[b] * _tick[b];
        if (b + 1 < _from.Length && v > _from[b + 1]) v = _from[b + 1];
        return new(v);
    }

    /// <summary>Moves a valid price by a whole number of ticks (crossing bands correctly).</summary>
    public Price Step(Price p, int ticks)
    {
        var cur = p;
        for (int k = 0; k < Math.Abs(ticks); k++)
            cur = ticks > 0 ? RoundUp(new Price(cur.Raw + 1)) : RoundDown(new Price(cur.Raw - 1));
        return cur;
    }
}
