using Microsoft.Extensions.Hosting;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Trademaster.Core;
using Trademaster.Engine;
using Trademaster.Engine.Stages;
using Trademaster.Sim;
using Trademaster.Telemetry;

namespace Trademaster.Cli;

/// <summary>
/// Local dashboard: runs the engine in paced (real-time) mode on a background thread and streams one snapshot per
/// cycle to the browser over Server-Sent Events. Listens on 127.0.0.1 only. Simulation only — no order entry.
/// </summary>
public static class Dashboard
{
    private const int BaseBarSec = 5;
    private const int MaxBars = 6000;

    private sealed class Bar { public long T; public double O, H, L, C; public long V; }

    private sealed class Store
    {
        public readonly object Lock = new();
        public readonly Dictionary<string, List<Bar>> Bars = [];
        public readonly Dictionary<string, List<(long T, double Mu)>> Mu = [];
        public readonly Dictionary<string, List<FillView>> Fills = [];
        public readonly Dictionary<string, double> Open = [];
        public readonly List<(long T, double Equity)> EquityCurve = [];
        public readonly Queue<double> Latency = new();
        public readonly Dictionary<string, long> StatusCounts = [];
        public long Missed, OverBudget, Cycles;
        public string Json = "{}";
        public long Version;
        public TaskCompletionSource Changed = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public string RunState = "starting";
        public int RunNumber;

        public void Reset()
        {
            Bars.Clear(); Mu.Clear(); Fills.Clear(); Open.Clear(); Latency.Clear(); StatusCounts.Clear(); EquityCurve.Clear();
            Missed = OverBudget = Cycles = 0;
        }
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public static int Run(EngineConfig cfg, int port, bool loop, string? windowPage = null)
    {
        cfg.Run.Mode = RunMode.Paced;
        var store = new Store();
        var gpu = GpuHealth.Check().Describe();
        var sys = SystemInfo.Collect();

        var engine = new Thread(() => EngineLoop(cfg, store, loop)) { IsBackground = true, Name = "engine", Priority = ThreadPriority.AboveNormal };
        engine.Start();

        var builder = WebApplication.CreateBuilder();
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.WebHost.UseUrls($"http://127.0.0.1:{port}");
        var app = builder.Build();
        string uiDir = Path.Combine(AppContext.BaseDirectory, "ui");
        // local dashboard: always revalidate, so an updated page or new daily data shows without a hard refresh
        app.Use(async (ctx, next) => { ctx.Response.Headers.CacheControl = "no-cache"; await next(); });

        app.MapGet("/", () => Results.File(Path.Combine(uiDir, "market.html"), "text/html; charset=utf-8"));
        app.MapGet("/favicon.svg", () => Results.File(Path.Combine(uiDir, "favicon.svg"), "image/svg+xml"));
        app.MapGet("/nav.js", () => Results.File(Path.Combine(uiDir, "nav.js"), "text/javascript; charset=utf-8"));
        app.MapGet("/sim", () => Results.File(Path.Combine(uiDir, "index.html"), "text/html; charset=utf-8"));

        // Real market data (read-only quotes and charts): Alpaca when a key is configured, else the recorded demo.
        var md = MarketDataFactory.Create();
        var mdCache = new MdCache();
        static bool OkSym(string? s) => s is { Length: > 0 and <= 10 } && s.All(c => char.IsAsciiLetterUpper(c) || c is '.' or '-');
        app.MapGet("/api/md/status", () => Results.Json(new { provider = md.Name, live = md.Live, note = md.Note, universe = md.Universe }, JsonOpts));
        app.MapGet("/api/md/snapshots", async (string symbols, CancellationToken ct) =>
        {
            var list = symbols.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(x => x.ToUpperInvariant()).Where(OkSym).Distinct().Take(200).ToList();
            try { return Results.Json(await mdCache.Get("s:" + string.Join(',', list), TimeSpan.FromSeconds(1), async () => await md.Snapshots(list, ct)), JsonOpts); }
            catch (Exception ex) when (ex is not OperationCanceledException) { return Results.Problem(ex.Message, statusCode: 502); }
        });
        app.MapGet("/api/md/bars", async (string symbol, string? tf, CancellationToken ct) =>
        {
            symbol = symbol.ToUpperInvariant(); tf ??= "1D";
            if (!OkSym(symbol) || tf is not ("1m" or "5m" or "15m" or "1h" or "1D" or "1W")) return Results.BadRequest();
            var ttl = TimeSpan.FromSeconds(tf is "1D" or "1W" ? 60 : 5);
            try { return Results.Json(await mdCache.Get($"b:{symbol}:{tf}", ttl, async () => await md.Bars(symbol, tf, ct)), JsonOpts); }
            catch (Exception ex) when (ex is not OperationCanceledException) { return Results.Problem(ex.Message, statusCode: 502); }
        });
        app.MapGet("/api/md/trades", async (string symbol, CancellationToken ct) =>
        {
            symbol = symbol.ToUpperInvariant();
            if (!OkSym(symbol)) return Results.BadRequest();
            try { return Results.Json(await mdCache.Get($"t:{symbol}", TimeSpan.FromSeconds(2), async () => await md.Trades(symbol, ct)), JsonOpts); }
            catch (Exception ex) when (ex is not OperationCanceledException) { return Results.Problem(ex.Message, statusCode: 502); }
        });
        // 5-trading-day forecasts (research): files written by python -m trademaster_ml.daily_forecast
        app.MapGet("/api/forecast", (string symbol) =>
        {
            symbol = symbol.ToUpperInvariant();
            if (!OkSym(symbol)) return Results.BadRequest();
            var fcDir = Path.Combine("private", "data", "forecasts");
            var repDir = Path.Combine("private", "reports", "daily_forecast");
            static System.Text.Json.Nodes.JsonNode? Read(string path)
            {
                if (!File.Exists(path)) return null;
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                return System.Text.Json.Nodes.JsonNode.Parse(fs);
            }
            static System.Text.Json.Nodes.JsonNode? FindSym(System.Text.Json.Nodes.JsonNode? doc, string sym) =>
                doc?["forecasts"]?.AsArray().FirstOrDefault(f => (string?)f?["symbol"] == sym)?.DeepClone();
            var latest = Directory.Exists(fcDir)
                ? Directory.GetFiles(fcDir, "????-??-??.json").OrderByDescending(f => f).Select(Read).FirstOrDefault(d => FindSym(d, symbol) is not null)
                : null;
            var fc = FindSym(latest, symbol);
            if (fc is null) return Results.NotFound();
            var holdout = Directory.Exists(repDir)
                ? Directory.GetFiles(repDir, "holdout_*.json").OrderByDescending(f => f).Select(Read).Select(d => FindSym(d, symbol)).FirstOrDefault(x => x is not null)
                : null;
            var validation = Read(Path.Combine(repDir, "validation_summary.json"))?["symbols"]?[symbol]?.DeepClone();
            var daily = Read(Path.Combine("private", "data", "daily", symbol + ".json"));
            var dates = daily?["dates"]?.AsArray().Select(x => (string?)x).ToList() ?? [];
            var close = daily?["close"]?.AsArray().Select(x => (double?)x).ToList() ?? [];
            int from = Math.Max(0, dates.Count - 90);
            return Results.Json(new
            {
                symbol, createdAt = (string?)latest?["created_at"], forecast = fc, holdout, validation,
                history = dates.Skip(from).Zip(close.Skip(from), (d, c) => new { date = d, close = c }),
            }, JsonOpts);
        });
        // "My calls": the user's own up/down calls, scored against real closes (research only; no orders)
        var myCalls = new MyCalls();
        app.MapGet("/api/mycalls", async (CancellationToken ct) => Results.Json(await myCalls.Scored(md, ct), JsonOpts));
        app.MapPost("/api/mycalls", async (HttpContext ctx, CancellationToken ct) =>
        {
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body, cancellationToken: ct);
            string sym = (body.TryGetProperty("symbol", out var s1) ? s1.GetString() ?? "" : "").ToUpperInvariant();
            string dir = body.TryGetProperty("direction", out var d1) ? d1.GetString() ?? "" : "";
            string by = body.TryGetProperty("byDate", out var b1) ? b1.GetString() ?? "" : "";
            double? target = body.TryGetProperty("target", out var t1) && t1.ValueKind == JsonValueKind.Number ? t1.GetDouble() : null;
            string? note = body.TryGetProperty("note", out var n1) ? n1.GetString() : null;
            if (!OkSym(sym) || dir is not ("up" or "down") || !DateOnly.TryParseExact(by, "yyyy-MM-dd", out var byD)
                || byD <= DateOnly.FromDateTime(DateTime.UtcNow) || byD > DateOnly.FromDateTime(DateTime.UtcNow.AddDays(120)))
                return Results.BadRequest(new { error = "need symbol, direction up/down and a future by-date within 120 days" });
            var snap = await md.Snapshots([sym], ct);
            if (!snap.TryGetValue(sym, out var sn) || sn.Last is not double entry) return Results.BadRequest(new { error = "no current price for " + sym });
            var chance = MyCalls.ModelChanceAbove(sym, by, entry);
            var call = myCalls.Add(sym, dir, entry, by, target, note is { Length: > 300 } ? note[..300] : note, chance);
            return Results.Json(call, JsonOpts);
        });
        app.MapDelete("/api/mycalls/{id}", (string id) => myCalls.Withdraw(id) ? Results.Ok() : Results.BadRequest(new { error = "calls can only be withdrawn within 15 minutes" }));
        app.MapGet("/api/explain", () =>
        {
            var path = Path.Combine("private", "reports", "daily_forecast", "v4_explain.json");
            if (!File.Exists(path)) return Results.NotFound();
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var sr = new StreamReader(fs);
            return Results.Text(sr.ReadToEnd(), "application/json");
        });
        app.MapGet("/api/info", () => Results.Json(new
        {
            system = sys.Lines().ToArray(), gpu, cadenceMs = cfg.Run.CadenceMs, budgetMs = cfg.Run.BudgetMs, speed = cfg.Run.Speed,
            baseBarSec = BaseBarSec, mode = "simulation", live = false,
        }, JsonOpts));
        app.MapGet("/api/state", () => Results.Text(store.Json, "application/json"));

        // v6 cross-stock NASDAQ rankings (research): written by python -m trademaster_ml.v6 rank
        app.MapGet("/markets", () => Results.File(Path.Combine(uiDir, "markets.html"), "text/html; charset=utf-8"));
        // the old single-page Rankings view now lives inside the Markets hub
        app.MapGet("/rankings", (string? market) => Results.Redirect(market == "sgx" ? "/markets#/sgx/overview" : "/markets#/nasdaq/overview"));
        app.MapGet("/api/rankings", (string? market) =>
        {
            var path = market == "sgx"
                ? Path.Combine("private", "reports", "sgx", "v6", "rankings.json")
                : Path.Combine("private", "reports", "v6", "rankings.json");
            return File.Exists(path) ? Results.Text(File.ReadAllText(path), "application/json")
                : Results.NotFound(new { error = "no rankings yet: run python -m trademaster_ml.v6 rank" });
        });
        // every ranked stock on both exchanges (symbol, name, market), for the search boxes
        app.MapGet("/api/symbols", () => Results.Json(StockEval.Symbols()));
        // NASDAQ index ETFs and sector groups (python -m trademaster_ml.indexes, run by the refresh)
        app.MapGet("/api/indexes", () =>
        {
            var path = Path.Combine("private", "reports", "v6", "indexes.json");
            return File.Exists(path) ? Results.Text(File.ReadAllText(path), "application/json")
                : Results.NotFound(new { error = "no index data yet: run python -m trademaster_ml.indexes" });
        });
        app.MapGet("/api/fairvalue", () =>
        {
            var path = Path.Combine("private", "reports", "v6", "fairvalue.json");
            return File.Exists(path) ? Results.Text(File.ReadAllText(path), "application/json")
                : Results.NotFound(new { error = "no fair values yet: run python -m trademaster_ml.fairvalue" });
        });
        app.MapGet("/api/sgx/momentum", () =>
        {
            var path = Path.Combine("private", "reports", "sgx", "momentum.json");
            return File.Exists(path) ? Results.Text(File.ReadAllText(path), "application/json") : Results.Json(new { status = "none" });
        });
        // pull new closes now (the page asks once a newer close should exist, e.g. after the US or SGX market closes)
        app.MapPost("/api/refresh", () => { var (started, reason) = StockEval.StartRefresh(); return Results.Json(new { started, reason }); });
        app.MapGet("/api/rankings/status", () =>
        {
            var path = Path.Combine("private", "reports", "v6", "refresh_status.json");
            return File.Exists(path) ? Results.Text(File.ReadAllText(path), "application/json") : Results.Json(new { state = "none" });
        });

        // v8 amendment C: per-stock Qwen analysis (python -m trademaster_ml.stock_eval). A click returns the cached
        // evaluation for the latest close, or queues one and starts the background evaluator (the user is at the PC).
        app.MapGet("/api/stock/eval", (string? market, string symbol) =>
        {
            var mk = market == "sgx" ? "sgx" : "nasdaq";
            var sym = (symbol ?? "").Trim().ToUpperInvariant();
            if (!StockEval.OkSymbol(sym)) return Results.BadRequest(new { error = "bad symbol" });
            var asof = StockEval.Asof(mk);
            if (asof is null) return Results.NotFound(new { error = "no price forecasts yet" });
            var dir = Path.Combine(StockEval.Root, mk);
            string done = Path.Combine(dir, $"{sym}_{asof}.json"), err = Path.Combine(dir, $"{sym}_{asof}.error.json");
            if (File.Exists(done)) return Results.Text($"{{\"state\":\"done\",\"result\":{File.ReadAllText(done)}}}", "application/json");
            if (File.Exists(err) && !StockEval.Queued(mk, sym)) return Results.Text($"{{\"state\":\"error\",\"result\":{File.ReadAllText(err)}}}", "application/json");
            StockEval.Enqueue(mk, sym);
            StockEval.EnsureDaemon(port);
            var partial = Path.Combine(dir, $"{sym}_{asof}.partial.json");
            var daemon = StockEval.Daemon();
            var body = $"{{\"state\":\"pending\",\"position\":{StockEval.Position(mk, sym)},\"daemon\":{daemon}," +
                       $"\"result\":{(File.Exists(partial) ? File.ReadAllText(partial) : "null")}}}";
            return Results.Text(body, "application/json");
        });
        app.MapPost("/api/stock/queue-all", () => Results.Json(new { started = StockEval.EnsureQueue(port) }));
        app.MapPost("/api/stock/reeval", (string? market, string symbol) =>
        {
            var mk = market == "sgx" ? "sgx" : "nasdaq";
            var sym = (symbol ?? "").Trim().ToUpperInvariant();
            if (!StockEval.OkSymbol(sym)) return Results.BadRequest(new { error = "bad symbol" });
            var asof = StockEval.Asof(mk);
            if (asof is null) return Results.NotFound(new { error = "no price forecasts yet" });
            var queued = StockEval.Reevaluate(mk, sym, asof);
            StockEval.EnsureDaemon(port);
            return Results.Json(new { queued });
        });
        app.MapGet("/api/stock/evals", (string? market) =>
        {
            var mk = market == "sgx" ? "sgx" : "nasdaq";
            var asof = StockEval.Asof(mk);
            var idx = Path.Combine(StockEval.Root, mk, $"index_{asof}.json");
            var score = Path.Combine(StockEval.Root, mk, "advice_score.json");
            var body = $"{{\"asof\":{JsonSerializer.Serialize(asof)},\"daemon\":{StockEval.Daemon()}," +
                       $"\"index\":{(File.Exists(idx) ? File.ReadAllText(idx) : "{}")},\"score\":{(File.Exists(score) ? File.ReadAllText(score) : "null")}}}";
            return Results.Text(body, "application/json");
        });

        // Forward test (real data, predictions only): files written by python -m trademaster_ml.forward
        string fwdRoot = Path.Combine("private", "data", "forward");
        app.MapGet("/forward", () => Results.File(Path.Combine(uiDir, "forward.html"), "text/html; charset=utf-8"));
        app.MapGet("/api/forward/runs", () =>
        {
            var runs = Directory.Exists(fwdRoot)
                ? new DirectoryInfo(fwdRoot).GetDirectories()
                    .Where(d => File.Exists(Path.Combine(d.FullName, "summary.json")))
                    .OrderByDescending(d => d.Name.StartsWith("live-")).ThenByDescending(d => d.Name)
                    .Select(d => d.Name).ToList()
                : [];
            return Results.Json(runs);
        });
        app.MapGet("/api/forward", (string? run) =>
        {
            if (!Directory.Exists(fwdRoot)) return Results.NotFound();
            var name = run ?? new DirectoryInfo(fwdRoot).GetDirectories()
                .Where(d => File.Exists(Path.Combine(d.FullName, "summary.json")))
                .OrderByDescending(d => d.Name.StartsWith("live-")).ThenByDescending(d => d.Name).Select(d => d.Name).FirstOrDefault();
            if (name is null || name.Contains("..") || name.Contains('/') || name.Contains('\\')) return Results.NotFound();
            var path = Path.Combine(fwdRoot, name, "summary.json");
            if (!File.Exists(path)) return Results.NotFound();
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var sr = new StreamReader(fs);
            return Results.Text(sr.ReadToEnd(), "application/json");
        });
        // Buy/sell indicators (research only): files written by python -m trademaster_ml.signals_live
        IEnumerable<string> SignalRuns() => Directory.Exists(fwdRoot)
            ? new DirectoryInfo(fwdRoot).GetDirectories("signals-*")
                .Where(d => File.Exists(Path.Combine(d.FullName, "signals.json")))
                .OrderByDescending(d => d.Name[^10..]).ThenByDescending(d => d.Name.StartsWith("signals-live-"))
                .Select(d => d.Name)
            : [];
        app.MapGet("/api/forward/signals/runs", () => Results.Json(SignalRuns().ToList()));
        app.MapGet("/api/forward/signals", (string? run) =>
        {
            var name = run ?? SignalRuns().FirstOrDefault();
            if (name is null || !name.StartsWith("signals-") || name.Contains("..") || name.Contains('/') || name.Contains('\\'))
                return Results.NotFound();
            var path = Path.Combine(fwdRoot, name, "signals.json");
            if (!File.Exists(path)) return Results.NotFound();
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var sr = new StreamReader(fs);
            return Results.Text(sr.ReadToEnd(), "application/json");
        });
        app.MapGet("/api/history", (string symbol) =>
        {
            lock (store.Lock)
            {
                var bars = store.Bars.TryGetValue(symbol, out var b) ? b.Select(x => new { t = x.T, o = x.O, h = x.H, l = x.L, c = x.C, v = x.V }).ToList() : [];
                var mu = store.Mu.TryGetValue(symbol, out var m) ? m.Select(x => new { t = x.T, v = Math.Round(x.Mu, 2) }).ToList() : [];
                var fills = store.Fills.TryGetValue(symbol, out var f) ? f.ToList() : [];
                return Results.Json(new { symbol, bars, mu, fills }, JsonOpts);
            }
        });
        app.MapGet("/api/equity", () =>
        {
            lock (store.Lock) return Results.Json(store.EquityCurve.Select(e => new { t = e.T, v = e.Equity }).ToList(), JsonOpts);
        });
        app.MapGet("/api/sparklines", () =>
        {
            lock (store.Lock)
                return Results.Json(store.Bars.ToDictionary(kv => kv.Key,
                    kv => kv.Value.Skip(Math.Max(0, kv.Value.Count - 120)).Select(b => Math.Round(b.C, 4)).ToList()), JsonOpts);
        });
        app.MapGet("/api/stream", async (HttpContext ctx) =>
        {
            ctx.Response.Headers.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            long seen = -1;
            var ct = ctx.RequestAborted;
            while (!ct.IsCancellationRequested)
            {
                string json; long ver; Task wait;
                lock (store.Lock) { json = store.Json; ver = store.Version; wait = store.Changed.Task; }
                try
                {
                    if (ver != seen)
                    {
                        await ctx.Response.WriteAsync("data: " + json + "\n\n", ct);
                        await ctx.Response.Body.FlushAsync(ct);
                        seen = ver;
                    }
                    else if (await Task.WhenAny(wait, Task.Delay(15_000, ct)) != wait)
                    {
                        await ctx.Response.WriteAsync(": ping\n\n", ct);
                        await ctx.Response.Body.FlushAsync(ct);
                    }
                }
                catch (OperationCanceledException) { break; }
            }
        });

        Console.WriteLine($"AI Trademaster dashboard: http://127.0.0.1:{port}/   (SIMULATION, paced x{cfg.Run.Speed}; Ctrl+C to stop)");
        if (windowPage is null)
        {
            app.Run();
            return 0;
        }
        app.Start();
        var window = AppWindow.Open($"http://127.0.0.1:{port}{windowPage}");
        if (window is null)
        {
            Console.WriteLine("Window opened (not tracked). The dashboard keeps running until Ctrl+C.");
            app.WaitForShutdown();
            return 0;
        }
        Console.WriteLine("Window open. Closing the window stops the dashboard.");
        window.WaitForExit();
        app.StopAsync().GetAwaiter().GetResult();
        return 0;
    }

    private static void EngineLoop(EngineConfig cfg, Store store, bool loop)
    {
        do
        {
            lock (store.Lock) { store.Reset(); store.RunNumber++; store.RunState = "running"; }
            try
            {
                var manifest = DataManifest.Load(cfg.Data.Dir);
                var universe = Universe.Load(Path.Combine(cfg.Data.Dir, manifest.UniverseFile));
                var exchange = ExchangeConfig.Load(Path.Combine(cfg.Data.ExchangesDir, manifest.Exchange + ".json"));
                IModel model = Trademaster.Engine.Inference.ModelFactory.Create(cfg, universe);
                using var disposeModel = model as IDisposable;
                using var exec = new SimExchange(Path.Combine(cfg.Data.Dir, manifest.ExchangeFile), universe, exchange, cfg.Execution, manifest.StartTs);
                string outDir = Path.Combine(cfg.Run.OutputDir, $"{DateTime.Now:yyyyMMdd-HHmmss}-dashboard");
                using var runner = new CycleRunner(cfg, exec, model, new SystemWallClock(), outDir);
                runner.OnCycle = v => Publish(store, v);
                runner.Run();
                lock (store.Lock) store.RunState = "finished";
            }
            catch (Exception e)
            {
                lock (store.Lock) store.RunState = "error: " + e.Message;
                Console.Error.WriteLine($"engine error: {e.Message}");
                return;
            }
            Publish(store, null);
            if (loop) Thread.Sleep(5000);
        } while (loop);
    }

    private static void Publish(Store s, CycleView? v)
    {
        lock (s.Lock)
        {
            if (v is not null)
            {
                s.Cycles++;
                s.StatusCounts[v.Status] = s.StatusCounts.GetValueOrDefault(v.Status) + 1;
                if (v.Status == "Missed") s.Missed++;
                if (v.OverBudget) s.OverBudget++;
                long sec = v.CutoffTs / Ns.PerSec;
                if (s.EquityCurve.Count == 0 || s.EquityCurve[^1].T < sec) s.EquityCurve.Add((sec, Math.Round(v.Equity, 2)));
                s.Latency.Enqueue(v.ToOrderMs);
                while (s.Latency.Count > 300) s.Latency.Dequeue();
                long bucket = v.CutoffTs / Ns.PerSec / BaseBarSec * BaseBarSec;
                foreach (var r in v.Rows)
                {
                    if (r.Bid <= 0 || r.Ask <= 0) continue;
                    double mid = (r.Bid + r.Ask) / 2;
                    s.Open.TryAdd(r.Symbol, mid);
                    if (!s.Bars.TryGetValue(r.Symbol, out var bars)) s.Bars[r.Symbol] = bars = [];
                    if (bars.Count > 0 && bars[^1].T == bucket)
                    {
                        var b = bars[^1];
                        b.H = Math.Max(b.H, mid); b.L = Math.Min(b.L, mid); b.C = mid; b.V += r.CycleVolume;
                    }
                    else
                    {
                        bars.Add(new Bar { T = bucket, O = mid, H = mid, L = mid, C = mid, V = r.CycleVolume });
                        if (bars.Count > MaxBars) bars.RemoveAt(0);
                    }
                    if (r.HasPrediction)
                    {
                        if (!s.Mu.TryGetValue(r.Symbol, out var mu)) s.Mu[r.Symbol] = mu = [];
                        if (mu.Count > 0 && mu[^1].T == bucket) mu[^1] = (bucket, r.MuBps); else mu.Add((bucket, r.MuBps));
                        if (mu.Count > MaxBars) mu.RemoveAt(0);
                    }
                }
                foreach (var f in v.Fills)
                {
                    if (!s.Fills.TryGetValue(f.Symbol, out var list)) s.Fills[f.Symbol] = list = [];
                    list.Add(f);
                }
            }
            var lat = Percentiles.Of(s.Latency);
            var payload = new
            {
                view = v,
                opens = v?.Rows.Select(r => s.Open.GetValueOrDefault(r.Symbol)).ToArray(),
                stats = new
                {
                    cycles = s.Cycles, missed = s.Missed, overBudget = s.OverBudget, statusCounts = s.StatusCounts,
                    p50 = lat.P50, p95 = lat.P95, p99 = lat.P99, max = lat.Max, window = lat.Count,
                },
                run = new { state = s.RunState, number = s.RunNumber },
            };
            s.Json = JsonSerializer.Serialize(payload, JsonOpts);
            s.Version++;
            var old = s.Changed;
            s.Changed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            old.TrySetResult();
        }
    }
}
