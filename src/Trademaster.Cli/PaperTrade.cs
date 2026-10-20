using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Trademaster.Cli;

/// <summary>
/// Auto-trade on Alpaca's PAPER account (simulated money) through python -m trademaster_ml.papertrade, which refuses any
/// key or endpoint that is not paper. Runs only while the app is open: a background loop calls `run` every few minutes
/// when the user has switched it on in the Auto-trade page.
/// </summary>
public static class PaperTrade
{
    static readonly string Dir = Path.Combine("private", "reports", "paper");
    static readonly string Control = Path.Combine(Dir, "control.json");
    static readonly SemaphoreSlim Busy = new(1, 1);
    static string lastRun = "";

    public static string Status()
    {
        var p = Path.Combine(Dir, "status.json");
        var st = File.Exists(p) ? File.ReadAllText(p) : "null";
        return $"{{\"status\":{st},\"control\":{ReadControl().ToJsonString()},\"last_run\":{JsonSerializer.Serialize(lastRun)},\"busy\":{(Busy.CurrentCount == 0 ? "true" : "false")}}}";
    }

    public static JsonObject ReadControl()
    {
        try { if (File.Exists(Control)) return JsonNode.Parse(File.ReadAllText(Control))!.AsObject(); } catch (Exception) { }
        return new JsonObject { ["enabled"] = false, ["books"] = new JsonObject { ["ai"] = true, ["rank"] = true, ["night"] = true },
            ["split"] = new JsonObject { ["ai"] = 0.45, ["rank"] = 0.45, ["night"] = 0.10 } };
    }

    /// <summary>Switches from the page: {"enabled": bool, "books": {"ai": bool, "rank": bool}}.</summary>
    public static void SetControl(JsonObject change)
    {
        var c = ReadControl();
        if (change["enabled"] is JsonValue e && e.TryGetValue(out bool en)) c["enabled"] = en;
        if (change["books"] is JsonObject b)
            foreach (var k in new[] { "ai", "rank", "night" })
                if (b[k] is JsonValue v && v.TryGetValue(out bool on)) (c["books"] ??= new JsonObject())[k] = on;
        Directory.CreateDirectory(Dir);
        File.WriteAllText(Control, c.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>Runs `python -m trademaster_ml.papertrade args` unless another call is running; returns its output.</summary>
    public static Task<string> Python(params string[] args) => Call(false, args);

    /// <summary>Like Python, but waits up to a minute for a running call (a click on Buy/Sell should not just fail).</summary>
    public static Task<string> PythonWait(params string[] args) => Call(true, args);

    static async Task<string> Call(bool wait, string[] args)
    {
        if (!await Busy.WaitAsync(wait ? 60_000 : 0)) return "busy";
        try
        {
            var py = Path.Combine("python", ".venv", "Scripts", "python.exe");
            if (!File.Exists(py)) return "python environment not found";
            var psi = new ProcessStartInfo(Path.GetFullPath(py))
            {
                UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true,
                WorkingDirectory = Directory.GetCurrentDirectory(),
            };
            foreach (var a in new[] { "-W", "ignore", "-m", "trademaster_ml.papertrade" }.Concat(args)) psi.ArgumentList.Add(a);
            psi.Environment["PYTHONPATH"] = "python";
            using var p = Process.Start(psi)!;
            var outTask = p.StandardOutput.ReadToEndAsync();
            var errTask = p.StandardError.ReadToEndAsync();
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            await p.WaitForExitAsync(cts.Token);
            var text = (await outTask) + (await errTask);
            Directory.CreateDirectory(Dir);
            File.AppendAllText(Path.Combine(Dir, "runs.log"), $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {string.Join(' ', args)}\n{text}\n");
            lastRun = $"{DateTime.Now:HH:mm} {string.Join(' ', args)}: {text.Trim().Split('\n').LastOrDefault()}";
            return text;
        }
        catch (Exception ex) { return ex.Message; }
        finally { Busy.Release(); }
    }

    /// <summary>While the app runs: every 5 minutes, if auto-trade is on, let the trader act (it checks the market clock and
    /// rebalances each book at most once per day / week).</summary>
    public static void StartLoop()
    {
        _ = Task.Run(async () =>
        {
            await Task.Delay(TimeSpan.FromSeconds(30));
            while (true)
            {
                try
                {
                    if (ReadControl()["enabled"]?.GetValue<bool>() == true) await Python("run");
                }
                catch (Exception) { }
                await Task.Delay(TimeSpan.FromMinutes(5));
            }
        });
    }
}
