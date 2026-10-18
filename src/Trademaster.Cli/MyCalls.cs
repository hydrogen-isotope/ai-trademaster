using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Trademaster.Cli;

/// <summary>A user's own up/down call on a stock, scored against real closing prices. Research only; never trades.</summary>
public sealed record MyCall(string Id, DateTimeOffset Created, string Symbol, string Direction, double EntryPrice, string ByDate,
    double? Target, string? Note, double? ModelChance);

/// <summary>Stores calls in private/data/mycalls.json and scores them with daily bars from the market-data provider.</summary>
public sealed class MyCalls
{
    private static readonly JsonSerializerOptions Opts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true };
    private readonly string _path = Path.Combine("private", "data", "mycalls.json");
    private readonly object _lock = new();

    public List<MyCall> Load()
    {
        lock (_lock)
            return File.Exists(_path) ? JsonSerializer.Deserialize<List<MyCall>>(File.ReadAllText(_path), Opts) ?? [] : [];
    }

    private void Save(List<MyCall> calls)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        var tmp = _path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(calls, Opts));
        File.Move(tmp, _path, overwrite: true);
    }

    public MyCall Add(string symbol, string direction, double entry, string byDate, double? target, string? note, double? modelChance)
    {
        var c = new MyCall(Guid.NewGuid().ToString("N")[..10], DateTimeOffset.UtcNow, symbol, direction, entry, byDate, target, note, modelChance);
        lock (_lock) { var all = Load(); all.Add(c); Save(all); }
        return c;
    }

    /// <summary>Withdraw only within 15 minutes of making a call, so losing calls can't be deleted later.</summary>
    public bool Withdraw(string id)
    {
        lock (_lock)
        {
            var all = Load();
            var c = all.FirstOrDefault(x => x.Id == id);
            if (c is null || DateTimeOffset.UtcNow - c.Created > TimeSpan.FromMinutes(15)) return false;
            all.Remove(c); Save(all);
            return true;
        }
    }

    /// <summary>The model's probability that the close on <paramref name="byDate"/> is above <paramref name="price"/>,
    /// from the latest forecast file's quantiles (only for dates inside its 5-day horizon).</summary>
    public static double? ModelChanceAbove(string symbol, string byDate, double price)
    {
        var dir = Path.Combine("private", "data", "forecasts");
        if (!Directory.Exists(dir)) return null;
        foreach (var f in Directory.GetFiles(dir, "????-??-??.json").OrderByDescending(x => x))
        {
            var doc = JsonNode.Parse(File.ReadAllText(f));
            var fc = doc?["forecasts"]?.AsArray().FirstOrDefault(x => (string?)x?["symbol"] == symbol);
            var day = fc?["days"]?.AsArray().FirstOrDefault(d => (string?)d?["date"] == byDate);
            if (day is null) continue;
            (double level, string key)[] qs = [(0.025, "p2_5"), (0.05, "p5"), (0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90"), (0.95, "p95"), (0.975, "p97_5")];
            var pts = qs.Where(q => day[q.key] is not null).Select(q => ((double)day[q.key]!, q.level)).ToList();
            if (pts.Count == 0) return null;
            if (price <= pts[0].Item1) return 1 - pts[0].Item2 / 2;
            if (price >= pts[^1].Item1) return (1 - pts[^1].Item2) / 2;
            for (int i = 1; i < pts.Count; i++)
                if (price <= pts[i].Item1)
                {
                    var (x0, p0) = pts[i - 1]; var (x1, p1) = pts[i];
                    return 1 - (p0 + (p1 - p0) * (price - x0) / (x1 - x0));
                }
            return null;
        }
        return null;
    }

    /// <summary>Score every call with daily bars: right/wrong on the by-date close, and whether the target was touched.</summary>
    public async Task<object> Scored(IMarketData md, CancellationToken ct)
    {
        var calls = Load();
        var bars = new Dictionary<string, List<BarDto>>();
        foreach (var s in calls.Select(c => c.Symbol).Distinct())
        {
            try { bars[s] = await md.Bars(s, "1D", ct); } catch { bars[s] = []; }
        }
        var ny = TimeZoneInfo.FindSystemTimeZoneById(OperatingSystem.IsWindows() ? "Eastern Standard Time" : "America/New_York");
        string Day(long t) => TimeZoneInfo.ConvertTime(DateTimeOffset.FromUnixTimeSeconds(t), ny).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        var rows = new List<object>();
        int done = 0, right = 0, targetsHit = 0, withTarget = 0; double modelSum = 0; int modelN = 0;
        foreach (var c in calls.OrderByDescending(c => c.Created))
        {
            var created = TimeZoneInfo.ConvertTime(c.Created, ny).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            var after = bars[c.Symbol].Where(b => string.CompareOrdinal(Day(b.T), created) > 0 && string.CompareOrdinal(Day(b.T), c.ByDate) <= 0).ToList();
            var final = after.LastOrDefault(b => Day(b.T) == c.ByDate);
            bool closed = final is not null || string.CompareOrdinal(DateTime.UtcNow.ToString("yyyy-MM-dd"), c.ByDate) > 0 && after.Count > 0;
            final ??= closed ? after.LastOrDefault() : null;
            bool up = c.Direction == "up";
            bool? isRight = final is null ? null : up ? final.C > c.EntryPrice : final.C < c.EntryPrice;
            bool? tgt = c.Target is double t && after.Count > 0 ? (up ? after.Max(b => b.H) >= t : after.Min(b => b.L) <= t) : null;
            var last = after.LastOrDefault()?.C;
            if (isRight is bool r)
            {
                done++; if (r) right++;
                if (c.ModelChance is double m) { modelSum += up ? m : 1 - m; modelN++; }
                if (tgt is bool th) { withTarget++; if (th) targetsHit++; }
            }
            rows.Add(new
            {
                c.Id, created = c.Created, c.Symbol, c.Direction, c.EntryPrice, c.ByDate, c.Target, c.Note,
                modelChanceYourWay = c.ModelChance is double mc ? (up ? mc : 1 - mc) : (double?)null,
                status = isRight is null ? "open" : isRight.Value ? "right" : "wrong", closeOnDate = final?.C, latest = last,
                moveSoFarPct = last is double l ? (l / c.EntryPrice - 1) * 100 : (double?)null, targetTouched = tgt,
                canWithdraw = DateTimeOffset.UtcNow - c.Created <= TimeSpan.FromMinutes(15),
            });
        }
        return new
        {
            summary = new
            {
                total = calls.Count, scored = done, right, hitRate = done > 0 ? (double)right / done : (double?)null,
                modelAvgChanceYourWay = modelN > 0 ? modelSum / modelN : (double?)null, targetsHit, withTarget,
            },
            calls = rows,
        };
    }
}
