using System.Collections.Concurrent;
using System.Globalization;
using System.Net.Http.Headers;
using System.Text.Json;

namespace Trademaster.Cli;

public sealed record SnapDto(string Symbol, double? Last, double? PrevClose, double? Open, double? High, double? Low, double? Volume,
    double? Vwap, double? Trades, double? Bid, double? BidSize, double? Ask, double? AskSize, long? QuoteTime, long? TradeTime);
public sealed record BarDto(long T, double O, double H, double L, double C, double V, double? Vw);
public sealed record TickDto(long T, double P, double S);

/// <summary>Read-only market data for the dashboard. Quotes and charts only; never places orders.</summary>
public interface IMarketData
{
    string Name { get; }
    bool Live { get; }
    string Note { get; }
    Task<Dictionary<string, SnapDto>> Snapshots(IReadOnlyList<string> symbols, CancellationToken ct);
    Task<List<BarDto>> Bars(string symbol, string tf, CancellationToken ct);
    Task<List<TickDto>> Trades(string symbol, CancellationToken ct);
    IReadOnlyList<string>? Universe { get; }
}

public static class MarketDataFactory
{
    /// <summary>Alpaca if keys are configured (env vars or private/config/alpaca.json), otherwise the recorded demo.</summary>
    public static IMarketData Create(string root = ".")
    {
        var alpaca = AlpacaMarketData.TryCreate(root);
        if (alpaca is not null) return alpaca;
        return new RecordedMarketData(Path.Combine(root, "private", "data", "hist_1m_30d"));
    }
}

/// <summary>
/// Alpaca Market Data API v2 (free Basic plan: real-time IEX quotes/trades/bars, historical SIP bars).
/// Keys: env APCA_API_KEY_ID / APCA_API_SECRET_KEY, or private/config/alpaca.json {"key_id","secret_key","feed"}.
/// Keys are never logged or sent to the browser.
/// </summary>
public sealed class AlpacaMarketData : IMarketData, IDisposable
{
    private const string Base = "https://data.alpaca.markets";
    private readonly HttpClient _http;
    private readonly string _feed;

    public string Name => "alpaca";
    public bool Live => true;
    public string Note => _feed == "iex"
        ? "Alpaca · IEX real-time (one exchange: prices are real, volumes are IEX-only) · daily bars all exchanges"
        : $"Alpaca · {_feed.ToUpperInvariant()}";
    public IReadOnlyList<string>? Universe => null;

    private AlpacaMarketData(string keyId, string secret, string feed)
    {
        _feed = feed;
        _http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.Add("APCA-API-KEY-ID", keyId);
        _http.DefaultRequestHeaders.Add("APCA-API-SECRET-KEY", secret);
        _http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    public static AlpacaMarketData? TryCreate(string root)
    {
        string? id = Environment.GetEnvironmentVariable("APCA_API_KEY_ID"), secret = Environment.GetEnvironmentVariable("APCA_API_SECRET_KEY");
        string feed = "iex";
        var file = Path.Combine(root, "private", "config", "alpaca.json");
        if ((id is null || secret is null) && File.Exists(file))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(file));
            var r = doc.RootElement;
            id = r.TryGetProperty("key_id", out var a) ? a.GetString() : null;
            secret = r.TryGetProperty("secret_key", out var b) ? b.GetString() : null;
            if (r.TryGetProperty("feed", out var f) && f.GetString() is { Length: > 0 } fs) feed = fs;
        }
        if (id?.StartsWith("PASTE_") == true || secret?.StartsWith("PASTE_") == true) return null;   // template not filled in yet
        return string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(secret) ? null : new AlpacaMarketData(id!, secret!, feed);
    }

    private async Task<JsonDocument> Get(string pathAndQuery, CancellationToken ct)
    {
        using var resp = await _http.GetAsync(Base + pathAndQuery, ct);
        var body = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"Alpaca {(int)resp.StatusCode}: {(body.Length > 200 ? body[..200] : body)}");
        return JsonDocument.Parse(body);
    }

    private static double? D(JsonElement e, string name) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
    private static long? Ts(JsonElement e) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty("t", out var v) && v.ValueKind == JsonValueKind.String
        ? DateTimeOffset.Parse(v.GetString()!, CultureInfo.InvariantCulture).ToUnixTimeSeconds() : null;
    private static JsonElement P(JsonElement e, string name) => e.TryGetProperty(name, out var v) ? v : default;

    public async Task<Dictionary<string, SnapDto>> Snapshots(IReadOnlyList<string> symbols, CancellationToken ct)
    {
        var res = new Dictionary<string, SnapDto>();
        foreach (var chunk in symbols.Chunk(100))
        {
            using var doc = await Get($"/v2/stocks/snapshots?symbols={Uri.EscapeDataString(string.Join(',', chunk))}&feed={_feed}", ct);
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                var s = p.Value;
                if (s.ValueKind != JsonValueKind.Object) continue;
                JsonElement tr = P(s, "latestTrade"), q = P(s, "latestQuote"), day = P(s, "dailyBar"), prev = P(s, "prevDailyBar");
                res[p.Name] = new SnapDto(p.Name, D(tr, "p") ?? D(day, "c"), D(prev, "c"), D(day, "o"), D(day, "h"), D(day, "l"), D(day, "v"),
                    D(day, "vw"), D(day, "n"), D(q, "bp"), D(q, "bs"), D(q, "ap"), D(q, "as"), Ts(q), Ts(tr));
            }
        }
        return res;
    }

    public async Task<List<BarDto>> Bars(string symbol, string tf, CancellationToken ct)
    {
        var (frame, days, limit, feed) = tf switch
        {
            "1m" => ("1Min", 4, 1000, _feed),
            "5m" => ("5Min", 12, 1000, _feed),
            "15m" => ("15Min", 30, 1000, _feed),
            "1h" => ("1Hour", 90, 1000, _feed),
            "1W" => ("1Week", 3650, 600, "sip"),
            _ => ("1Day", 1100, 800, "sip"),
        };
        var start = DateTime.UtcNow.AddDays(-days).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        using var doc = await Get($"/v2/stocks/bars?symbols={Uri.EscapeDataString(symbol)}&timeframe={frame}&start={start}&limit={limit}&sort=desc&adjustment=split&feed={feed}", ct);
        var list = new List<BarDto>();
        if (doc.RootElement.TryGetProperty("bars", out var bars) && bars.TryGetProperty(symbol, out var arr))
            foreach (var b in arr.EnumerateArray())
                list.Add(new BarDto(Ts(b) ?? 0, D(b, "o") ?? 0, D(b, "h") ?? 0, D(b, "l") ?? 0, D(b, "c") ?? 0, D(b, "v") ?? 0, D(b, "vw")));
        list.Reverse();
        return list;
    }

    public async Task<List<TickDto>> Trades(string symbol, CancellationToken ct)
    {
        var start = DateTime.UtcNow.AddDays(-4).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        using var doc = await Get($"/v2/stocks/trades?symbols={Uri.EscapeDataString(symbol)}&start={start}&limit=60&sort=desc&feed={_feed}", ct);
        var list = new List<TickDto>();
        if (doc.RootElement.TryGetProperty("trades", out var trades) && trades.TryGetProperty(symbol, out var arr))
            foreach (var t in arr.EnumerateArray())
                list.Add(new TickDto(Ts(t) ?? 0, D(t, "p") ?? 0, D(t, "s") ?? 0));
        return list;
    }

    public void Dispose() => _http.Dispose();
}

/// <summary>
/// Demo provider: replays the recorded 1-minute bars in private/data/hist_1m_30d (last session, 30x speed, looping),
/// so the UI can be used without a data key. Clearly labelled as recorded data in the UI.
/// </summary>
public sealed class RecordedMarketData : IMarketData
{
    private static readonly TimeZoneInfo Ny = TimeZoneInfo.FindSystemTimeZoneById(OperatingSystem.IsWindows() ? "Eastern Standard Time" : "America/New_York");
    private const int Speed = 30;
    private readonly Dictionary<string, (long[] T, double[] C, double[] V)> _data = [];
    private readonly DateTime _started = DateTime.UtcNow;
    private readonly long _lastOpen;
    private readonly string? _problem;

    public string Name => "demo";
    public bool Live => false;
    public string Note => _problem ?? "DEMO · recorded 1-minute data replayed at 30x · add a free Alpaca key for live quotes";
    public IReadOnlyList<string>? Universe { get; }

    public RecordedMarketData(string dir)
    {
        var uni = Path.Combine(dir, "universe.json");
        if (!File.Exists(uni)) { _problem = "No data source: add an Alpaca key, or pull recorded data first"; Universe = []; return; }
        using var u = JsonDocument.Parse(File.ReadAllText(uni));
        var syms = u.RootElement.GetProperty("stocks").EnumerateArray().Concat(u.RootElement.GetProperty("etfs").EnumerateArray())
            .Select(x => x.GetString()!).ToList();
        foreach (var s in syms)
        {
            var f = Path.Combine(dir, s + ".json");
            if (!File.Exists(f)) continue;
            using var d = JsonDocument.Parse(File.ReadAllText(f));
            var r = d.RootElement;
            _data[s] = (r.GetProperty("ts").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
                        r.GetProperty("close").EnumerateArray().Select(x => x.GetDouble()).ToArray(),
                        r.GetProperty("volume").EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Number ? x.GetDouble() : 0).ToArray());
        }
        Universe = _data.Keys.OrderBy(x => x).ToList();
        var anyT = _data.Values.First().T;
        _lastOpen = SessionOpen(anyT[^1]);
    }

    private static long SessionOpen(long t)
    {
        var ny = TimeZoneInfo.ConvertTime(DateTimeOffset.FromUnixTimeSeconds(t), Ny);
        var open = new DateTimeOffset(ny.Year, ny.Month, ny.Day, 9, 30, 0, ny.Offset);
        return open.ToUnixTimeSeconds();
    }

    /// <summary>Virtual "now" inside the last recorded session.</summary>
    private long Cursor() => _lastOpen + 60 + (long)((DateTime.UtcNow - _started).TotalSeconds * Speed) % (390 * 60);

    private int Upto(long[] t, long cursor)
    {
        int i = Array.BinarySearch(t, cursor);
        return i >= 0 ? i : ~i - 1;
    }

    public Task<Dictionary<string, SnapDto>> Snapshots(IReadOnlyList<string> symbols, CancellationToken ct)
    {
        long cur = Cursor();
        var res = new Dictionary<string, SnapDto>();
        foreach (var s in symbols)
        {
            if (!_data.TryGetValue(s, out var d)) continue;
            int i = Upto(d.T, cur);
            if (i < 0) continue;
            int first = Upto(d.T, _lastOpen - 1) + 1;
            double? prev = first > 0 ? d.C[first - 1] : null;
            double hi = double.MinValue, lo = double.MaxValue, vol = 0, pv = 0;
            for (int k = first; k <= i; k++) { hi = Math.Max(hi, d.C[k]); lo = Math.Min(lo, d.C[k]); vol += d.V[k]; pv += d.V[k] * d.C[k]; }
            bool today = i >= first;
            res[s] = new SnapDto(s, d.C[i], prev, today ? d.C[first] : null, today ? hi : null, today ? lo : null, today ? vol : null,
                vol > 0 ? pv / vol : null, null, null, null, null, null, null, d.T[i] + 60);
        }
        return Task.FromResult(res);
    }

    public Task<List<BarDto>> Bars(string symbol, string tf, CancellationToken ct)
    {
        var list = new List<BarDto>();
        if (!_data.TryGetValue(symbol, out var d)) return Task.FromResult(list);
        int end = Upto(d.T, Cursor());
        int step = tf switch { "1m" => 60, "5m" => 300, "15m" => 900, "1h" => 3600, _ => 0 };
        BarDto? cur = null;
        long curKey = long.MinValue;
        for (int k = 0; k <= end; k++)
        {
            long key = step > 0 ? (d.T[k] - SessionOpen(d.T[k])) / step * step + SessionOpen(d.T[k]) : SessionOpen(d.T[k]) - 9 * 3600 - 1800;
            if (key != curKey)
            {
                if (cur is not null) list.Add(cur);
                curKey = key;
                cur = new BarDto(key, d.C[k], d.C[k], d.C[k], d.C[k], d.V[k], null);
            }
            else
                cur = cur! with { H = Math.Max(cur.H, d.C[k]), L = Math.Min(cur.L, d.C[k]), C = d.C[k], V = cur.V + d.V[k] };
        }
        if (cur is not null) list.Add(cur);
        return Task.FromResult(list);
    }

    public Task<List<TickDto>> Trades(string symbol, CancellationToken ct)
    {
        var list = new List<TickDto>();
        if (!_data.TryGetValue(symbol, out var d)) return Task.FromResult(list);
        int end = Upto(d.T, Cursor());
        for (int k = end; k >= Math.Max(0, end - 40); k--) list.Add(new TickDto(d.T[k] + 60, d.C[k], d.V[k]));
        return Task.FromResult(list);
    }
}

/// <summary>Small time-based cache so the browser can poll every second without exceeding API rate limits.</summary>
public sealed class MdCache
{
    private readonly ConcurrentDictionary<string, (DateTime At, object Value)> _c = new();

    public async Task<object> Get(string key, TimeSpan ttl, Func<Task<object>> load)
    {
        if (_c.TryGetValue(key, out var e) && DateTime.UtcNow - e.At < ttl) return e.Value;
        var v = await load();
        _c[key] = (DateTime.UtcNow, v);
        return v;
    }
}
