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
    private static (DateTime, DateTime) symStamp;
    private static List<object> symList = [];

    /// <summary>Every ranked stock on NASDAQ and SGX as {s, n, m}, read from the rankings files (cached until they change).</summary>
    public static List<object> Symbols()
    {
        string pn = Path.Combine("private", "reports", "v6", "rankings.json"), ps = Path.Combine("private", "reports", "sgx", "v6", "rankings.json");
        var stamp = (File.Exists(pn) ? File.GetLastWriteTimeUtc(pn) : default, File.Exists(ps) ? File.GetLastWriteTimeUtc(ps) : default);
        lock (SymLock)
        {
            if (stamp == symStamp && symList.Count > 0) return symList;
            var list = new List<object>();
            foreach (var (path, market) in new[] { (pn, "nasdaq"), (ps, "sgx") })
            {
                if (!File.Exists(path)) continue;
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                using var doc = JsonDocument.Parse(fs);
                if (!doc.RootElement.TryGetProperty("rankings", out var rows)) continue;
                foreach (var r in rows.EnumerateArray())
                    list.Add(new { s = r.GetProperty("symbol").GetString(), n = r.TryGetProperty("name", out var n) ? n.GetString() : "", m = market });
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
}
