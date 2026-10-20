using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Trademaster.Cli;

/// <summary>
/// File-based link to the background Qwen stock evaluator (python -m trademaster_ml.stock_eval serve): a click drops a
/// request file into the queue, and the evaluator writes the cached result next to it. Research only.
/// </summary>
public static partial class StockEval
{
    public static readonly string Root = Path.Combine("private", "reports", "evals");
    private static readonly string QueueDir = Path.Combine(Root, "queue");
    private static readonly object StartLock = new();
    private static DateTime lastStart = DateTime.MinValue;

    [GeneratedRegex(@"^[A-Z0-9.\-]{1,15}$")]
    private static partial Regex SymbolRx();

    [GeneratedRegex("\"asof\":\\s*\"([0-9-]+)\"")]
    private static partial Regex AsofRx();

    public static bool OkSymbol(string s) => SymbolRx().IsMatch(s);

    /// <summary>The close the latest price forecasts are for (read from the head of forecast.json).</summary>
    public static string? Asof(string market)
    {
        var path = market == "sgx" ? Path.Combine("private", "reports", "sgx", "v7", "forecast.json")
                                   : Path.Combine("private", "reports", "v7", "forecast.json");
        if (!File.Exists(path)) return null;
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        var buf = new byte[256];
        int n = fs.Read(buf, 0, buf.Length);
        var m = AsofRx().Match(System.Text.Encoding.UTF8.GetString(buf, 0, n));
        return m.Success ? m.Groups[1].Value : null;
    }

    private static string[] Requests() =>
        Directory.Exists(QueueDir) ? Directory.GetFiles(QueueDir, "*.req").Select(Path.GetFileName).OfType<string>().Order(StringComparer.Ordinal).ToArray() : [];

    public static bool Queued(string market, string sym) => Requests().Any(f => f.EndsWith($"-{market}-{sym}.req", StringComparison.Ordinal));

    /// <summary>Queue a clicked stock ahead of the app-start batch (priority 0), once.</summary>
    public static void Enqueue(string market, string sym)
    {
        Directory.CreateDirectory(QueueDir);
        if (Requests().Any(f => f.StartsWith("0-", StringComparison.Ordinal) && f.EndsWith($"-{market}-{sym}.req", StringComparison.Ordinal))) return;
        File.WriteAllText(Path.Combine(QueueDir, $"0-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}-{market}-{sym}.req"), "");
    }

    /// <summary>Re-evaluate on request: the current result for the latest close is moved aside (to replaced/, never
    /// deleted) and the stock is queued first. The forward-test log keeps the first evaluation of each stock and close.</summary>
    public static bool Reevaluate(string market, string sym, string asof)
    {
        var dir = Path.Combine(Root, market);
        if (Requests().Any(f => f.StartsWith("0-", StringComparison.Ordinal) && f.EndsWith($"-{market}-{sym}.req", StringComparison.Ordinal))) return false;
        var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        foreach (var kind in new[] { "json", "error.json" })
        {
            var f = Path.Combine(dir, $"{sym}_{asof}.{kind}");
            if (!File.Exists(f)) continue;
            Directory.CreateDirectory(Path.Combine(dir, "replaced"));
            File.Move(f, Path.Combine(dir, "replaced", $"{sym}_{asof}.{stamp}.{kind}"));
        }
        Enqueue(market, sym);
        return true;
    }

    /// <summary>1-based place in the queue (0 if not queued).</summary>
    public static int Position(string market, string sym)
    {
        var r = Requests();
        var i = Array.FindIndex(r, f => f.EndsWith($"-{market}-{sym}.req", StringComparison.Ordinal));
        return i + 1;
    }

    /// <summary>The evaluator's state file as JSON ("null" if it never ran).</summary>
    public static string Daemon()
    {
        var path = Path.Combine(Root, "daemon.json");
        try { return File.Exists(path) ? File.ReadAllText(path) : "null"; }
        catch (IOException) { return "null"; }
    }

    private static bool DaemonAlive()
    {
        try
        {
            using var doc = JsonDocument.Parse(Daemon());
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return false;
            var r = doc.RootElement;
            if (r.TryGetProperty("state", out var st) && st.GetString() == "stopped") return false;
            var hb = r.GetProperty("heartbeat").GetDouble();
            return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0 - hb < 60;
        }
        catch (Exception) { return false; }
    }

    private static readonly object SymLock = new();
    private static (DateTime, DateTime, DateTime) symStamp;
    private static List<object> symList = [];

    // index stand-ins (ETFs that track them), with the words people search for
    private static readonly (string S, string N, string A)[] IndexEntries =
    [
        ("ONEQ", "Nasdaq Composite (via ONEQ ETF)", "nasdaq composite comp ixic ^ixic composite index nasdaq index"),
        ("QQQ", "Nasdaq-100 (via QQQ ETF)", "nasdaq 100 nasdaq-100 nasdaq100 ndx ^ndx nq tech 100"),
        ("SPY", "S&P 500 (via SPY ETF)", "s&p 500 sp500 s&p spx ^gspc es standard and poor's"),
    ];

    /// <summary>Everything the search boxes can find, as {s, n, m, k, a}: index stand-ins and NASDAQ sector groups
    /// (k = index / sector), then every ranked stock on NASDAQ and SGX (k = stock). Cached until the files change.</summary>
    public static List<object> Symbols()
    {
        string pn = Path.Combine("private", "reports", "v6", "rankings.json"), ps = Path.Combine("private", "reports", "sgx", "v6", "rankings.json"),
               pi = Path.Combine("private", "reports", "v6", "indexes.json");
        DateTime T(string p) => File.Exists(p) ? File.GetLastWriteTimeUtc(p) : default;
        var stamp = (T(pn), T(ps), T(pi));
        lock (SymLock)
        {
            if (stamp == symStamp && symList.Count > 0) return symList;
            var list = new List<object>();
            foreach (var (sym, name, alias) in IndexEntries) list.Add(new { s = sym, n = name, m = "nasdaq", k = "index", a = alias });
            if (File.Exists(pi))
            {
                using var fs = new FileStream(pi, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                using var doc = JsonDocument.Parse(fs);
                if (doc.RootElement.TryGetProperty("sectors", out var secs))
                    foreach (var x in secs.EnumerateArray())
                    {
                        var label = x.GetProperty("label").GetString() ?? "";
                        list.Add(new { s = x.GetProperty("key").GetString(), n = label + " (NASDAQ sector group)", m = "nasdaq", k = "sector",
                                       a = label.ToLowerInvariant().Replace("&", "and") + " sector" });
                    }
            }
            foreach (var (path, market) in new[] { (pn, "nasdaq"), (ps, "sgx") })
            {
                if (!File.Exists(path)) continue;
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                using var doc = JsonDocument.Parse(fs);
                if (!doc.RootElement.TryGetProperty("rankings", out var rows)) continue;
                foreach (var r in rows.EnumerateArray())
                    list.Add(new { s = r.GetProperty("symbol").GetString(), n = r.TryGetProperty("name", out var n) ? n.GetString() : "", m = market, k = "stock", a = "" });
            }
            symStamp = stamp;
            symList = list;
            return list;
        }
    }

    /// <summary>Start the evaluator (hidden; GPU, since the user is at the PC) unless it is running.</summary>
    public static void EnsureDaemon(int port)
    {
        lock (StartLock)
        {
            if (DaemonAlive() || DateTime.UtcNow - lastStart < TimeSpan.FromSeconds(30)) return;
            var py = Path.Combine("python", ".venv", "Scripts", "python.exe");
            if (!File.Exists(py)) return;
            Directory.CreateDirectory(Path.Combine("private", "reports", "jobs"));
            var psi = new ProcessStartInfo(Path.GetFullPath(py))
            {
                UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Directory.GetCurrentDirectory(),
            };
            foreach (var a in new[] { "-W", "ignore", "-m", "trademaster_ml.stock_eval", "serve", "--allow-gpu",
                                      "--app-url", $"http://127.0.0.1:{port}", "--log", Path.Combine("private", "reports", "jobs", "stock_eval.log") })
                psi.ArgumentList.Add(a);
            psi.Environment["PYTHONPATH"] = "python";
            Process.Start(psi);
            lastStart = DateTime.UtcNow;
        }
    }

    private static string? queuedFor;

    /// <summary>Opening a page queues every ranked NASDAQ and SGX stock not yet evaluated for the latest close
    /// (python -m trademaster_ml.stock_eval batch), once per close while the app runs, then starts the evaluator.
    /// A clicked stock is still evaluated first.</summary>
    public static bool EnsureQueue(int port)
    {
        if (NewCloseExpected(DateTimeOffset.UtcNow)) return false;     // the update queues every stock itself once the new close is in
        var key = $"{Asof("nasdaq")}|{Asof("sgx")}";
        lock (StartLock)
        {
            if (queuedFor == key) return false;
            queuedFor = key;
        }
        var py = Path.Combine("python", ".venv", "Scripts", "python.exe");
        if (!File.Exists(py)) return false;
        Task.Run(() =>
        {
            try
            {
                var psi = new ProcessStartInfo(Path.GetFullPath(py)) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Directory.GetCurrentDirectory() };
                foreach (var a in new[] { "-W", "ignore", "-m", "trademaster_ml.stock_eval", "batch" }) psi.ArgumentList.Add(a);
                psi.Environment["PYTHONPATH"] = "python";
                using var p = Process.Start(psi);
                p?.WaitForExit(120_000);
            }
            catch (Exception) { lock (StartLock) queuedFor = null; return; }
            EnsureDaemon(port);
        });
        return true;
    }

    /// <summary>US market full-day holidays by rule (same rules as python/trademaster_ml/holidays.py).</summary>
    public static bool IsUsHoliday(DateTime d)
    {
        int y = d.Year;
        static DateTime Nth(int y, int m, DayOfWeek wd, int n) { var d = new DateTime(y, m, 1); d = d.AddDays(((int)wd - (int)d.DayOfWeek + 7) % 7); return d.AddDays(7 * (n - 1)); }
        static DateTime Last(int y, int m, DayOfWeek wd) { var d = new DateTime(y, m, DateTime.DaysInMonth(y, m)); return d.AddDays(-(((int)d.DayOfWeek - (int)wd + 7) % 7)); }
        static DateTime? Obs(DateTime d, bool newYear = false) => d.DayOfWeek switch
        { DayOfWeek.Saturday => newYear ? null : d.AddDays(-1), DayOfWeek.Sunday => d.AddDays(1), _ => d };
        static DateTime Easter(int y)
        {
            int a = y % 19, b = y / 100, c = y % 100, dd = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3, h = (19 * a + b - dd - g + 15) % 30,
                i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = (a + 11 * h + 22 * l) / 451;
            return new DateTime(y, (h + l - 7 * m + 114) / 31, (h + l - 7 * m + 114) % 31 + 1);
        }
        var days = new List<DateTime?> { Obs(new DateTime(y, 1, 1), true), Nth(y, 1, DayOfWeek.Monday, 3), Nth(y, 2, DayOfWeek.Monday, 3), Easter(y).AddDays(-2),
            Last(y, 5, DayOfWeek.Monday), Obs(new DateTime(y, 7, 4)), Nth(y, 9, DayOfWeek.Monday, 1), Nth(y, 11, DayOfWeek.Thursday, 4), Obs(new DateTime(y, 12, 25)) };
        if (y >= 2022) days.Add(Obs(new DateTime(y, 6, 19)));
        return days.Contains(d.Date);
    }

    /// <summary>The last finished session's date for a market: a weekday after its close plus 20 minutes (bars settle).</summary>
    public static string LastFinishedSession(string market, DateTimeOffset now)
    {
        var (tzId, close) = market == "sgx" ? (OperatingSystem.IsWindows() ? "Singapore Standard Time" : "Asia/Singapore", new TimeSpan(17, 20, 0))
                                            : (OperatingSystem.IsWindows() ? "Eastern Standard Time" : "America/New_York", new TimeSpan(16, 20, 0));
        var local = TimeZoneInfo.ConvertTime(now, TimeZoneInfo.FindSystemTimeZoneById(tzId)).DateTime;
        var d = local.TimeOfDay >= close ? local.Date : local.Date.AddDays(-1);
        while (d.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday || (market != "sgx" && IsUsHoliday(d))) d = d.AddDays(-1);
        return d.ToString("yyyy-MM-dd", System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>True when a market has a finished session newer than the one its forecasts are built on.</summary>
    public static bool NewCloseExpected(DateTimeOffset now) =>
        new[] { "nasdaq", "sgx" }.Any(mk => Asof(mk) is string a && string.CompareOrdinal(a, LastFinishedSession(mk, now)) < 0);

    private static DateTime lastRefreshStart = DateTime.MinValue;

    /// <summary>Start the background update (new closes, news, forecasts, rankings, SGX, indexes) unless one is running or
    /// finished in the last 15 minutes. The user is at the PC (the page asked), so the news is read on the GPU.</summary>
    public static (bool Started, string Reason) StartRefresh()
    {
        lock (StartLock)
        {
            var status = Path.Combine("private", "reports", "v6", "refresh_status.json");
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(status));
                var st = doc.RootElement.GetProperty("state").GetString();
                var age = DateTime.Now - File.GetLastWriteTime(status);
                if (st == "running" && age < TimeSpan.FromHours(1)) return (false, "an update is already running");
                if (age < TimeSpan.FromMinutes(15)) return (false, "checked less than 15 minutes ago");
            }
            catch (Exception) { }
            if (DateTime.UtcNow - lastRefreshStart < TimeSpan.FromMinutes(15)) return (false, "started less than 15 minutes ago");
            var py = Path.Combine("python", ".venv", "Scripts", "python.exe");
            if (!File.Exists(py)) return (false, "python environment not found");
            var psi = new ProcessStartInfo(Path.GetFullPath(py)) { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Directory.GetCurrentDirectory() };
            foreach (var a in new[] { "-W", "ignore", "-m", "trademaster_ml.refresh", "--gpu-news" }) psi.ArgumentList.Add(a);
            psi.Environment["PYTHONPATH"] = "python";
            Process.Start(psi);
            lastRefreshStart = DateTime.UtcNow;
            return (true, "update started");
        }
    }
}
