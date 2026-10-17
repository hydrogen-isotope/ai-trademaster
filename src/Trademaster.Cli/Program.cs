using System.Globalization;
using Trademaster.Core;
using Trademaster.Engine;
using Trademaster.Engine.Stages;
using Trademaster.Sim;
using Trademaster.Telemetry;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

var cmd = args.Length > 0 ? args[0].ToLowerInvariant() : "help";
var opts = ParseOptions(args.Skip(1).ToArray());

try
{
    return cmd switch
    {
        "doctor" => Doctor(),
        "generate" => Generate(opts),
        "run" => Run(opts),
        "bench-onnx" => BenchOnnx(opts),
        "serve" => Serve(opts),
        "exec-study" => ExecStudy(opts),
        "verify-audit" => VerifyAudit(args.Skip(1).FirstOrDefault()),
        _ => Help(),
    };
}
catch (Exception ex) when (ex is InvalidOperationException or InvalidDataException or FileNotFoundException or DirectoryNotFoundException or ArgumentException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 2;
}

static int Help()
{
    Console.WriteLine("""
        AI Trademaster — simulation only (no live trading).

          doctor                               Print hardware/runtime inventory
          generate [--out data/synth-default]  Generate a seeded synthetic market
                   [--stocks 200] [--sectors 8] [--minutes 30] [--seed 42]
          run      [--config config/default.json] [--data DIR] [--mode virtual|paced]
                   [--speed 1.0] [--max-cycles N] [--out reports/<run-id>]
                   [--model b0_factor_residual|onnx] [--model-path private/models/...] [--backend cpu|dml|...]
                   [--allow-gpu] [--dump-features]
          verify-audit <run>/decisions.jsonl                 Check the decision log's hash chain
          serve    [--speed 1] [--port 5080]                 Local dashboard
          exec-study [--data DIR] [--fill-model queue|touch]  Execution probe
          bench-onnx [--backend cpu|dml|cuda|tensorrt|migraphx[,..]] [--device N] [--allow-gpu]
                     [--n 30,300,1000] [--k 1,10,100,1000]   Inference latency (GPU needs --allow-gpu + safety gate)
        """);
    return 0;
}

static int Doctor()
{
    foreach (var l in SystemInfo.Collect().Lines()) Console.WriteLine(l);
    var providers = Trademaster.Engine.Inference.OnnxBackend.AvailableProviders();
    Console.WriteLine($"ONNX:        flavour {Trademaster.Engine.Inference.OnnxBackend.Flavor}; providers {string.Join(", ", providers)}");
    var adapters = GpuAdapters.Enumerate();
    int pick = GpuAdapters.PickDiscrete(adapters);
    foreach (var a in adapters)
        Console.WriteLine($"  DirectML device {a.Index}: {a.Name} ({a.Vendor}, {a.DedicatedVramGb} GB dedicated){(a.Software ? " [software]" : "")}{(a.Index == pick ? "  <- auto-selected for dml" : "")}");
    Console.WriteLine(GpuHealth.Check().Describe());
    var findings = GpuConfigCheck.Findings();
    Console.WriteLine(findings.Count == 0
        ? "GPU config:  no known risky settings found"
        : "GPU config:  " + findings.Count + " issue(s) — see docs/GPU-STABILITY.md");
    foreach (var f in findings) Console.WriteLine("  - " + f);
    return 0;
}

static int Generate(Dictionary<string, string> o)
{
    var g = new GeneratorOptions
    {
        Stocks = Int(o, "stocks", 200), Sectors = Int(o, "sectors", 8),
        Minutes = Dbl(o, "minutes", 30), Seed = Int(o, "seed", 42), ImpactBps = Dbl(o, "impact-bps", 0),
    };
    string outDir = o.GetValueOrDefault("out", "private/data/synth-default");
    var ex = ExchangeConfig.Load(Path.Combine(o.GetValueOrDefault("exchanges", "config/exchanges"), o.GetValueOrDefault("exchange", "SYNTH") + ".json"));
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var r = new SyntheticGenerator(g, ex).Generate(outDir);
    Console.WriteLine($"SYNTHETIC data written to {outDir} in {sw.Elapsed.TotalSeconds:0.0} s");
    Console.WriteLine($"  instruments: {g.Stocks} stocks + {g.Sectors + 1} ETFs, {g.Minutes} min, seed {g.Seed}");
    Console.WriteLine($"  exchange events: {r.ExchangeEvents:N0}, feed events: {r.FeedEvents:N0}");
    Console.WriteLine($"  data version: {r.Manifest.DataVersion}");
    return 0;
}

static int Run(Dictionary<string, string> o)
{
    var cfg = EngineConfig.Load(o.GetValueOrDefault("config", "config/default.json"));
    if (o.TryGetValue("data", out var data)) cfg.Data.Dir = data;
    if (o.TryGetValue("mode", out var mode)) cfg.Run.Mode = Enum.Parse<RunMode>(mode, ignoreCase: true);
    if (o.TryGetValue("speed", out var speed)) cfg.Run.Speed = double.Parse(speed, CultureInfo.InvariantCulture);
    if (o.TryGetValue("max-cycles", out var mc)) cfg.Run.MaxCycles = int.Parse(mc, CultureInfo.InvariantCulture);
    if (cfg.Run.Speed <= 0) throw new ArgumentException("--speed must be > 0");
    if (o.TryGetValue("exec-style", out var style)) cfg.Decision.ExecutionStyle = style;
    if (o.TryGetValue("fill-model", out var fmo)) cfg.Execution.FillModel = fmo == "touch" ? FillModel.TouchProbability : FillModel.Queue;
    if (o.TryGetValue("max-daily-loss", out var mdl)) cfg.Risk.MaxDailyLoss = double.Parse(mdl, CultureInfo.InvariantCulture);
    if (o.TryGetValue("passive-cost-bps", out var pcb)) cfg.Decision.PassiveCostBps = double.Parse(pcb, CultureInfo.InvariantCulture);

    string runId = $"{DateTime.Now:yyyyMMdd-HHmmss}-{cfg.Run.Mode.ToString().ToLowerInvariant()}";
    string outDir = o.GetValueOrDefault("out", Path.Combine(cfg.Run.OutputDir, runId));

    var manifest = DataManifest.Load(cfg.Data.Dir);
    var universe = Universe.Load(Path.Combine(cfg.Data.Dir, manifest.UniverseFile));
    var exchange = ExchangeConfig.Load(Path.Combine(cfg.Data.ExchangesDir, manifest.Exchange + ".json"));
    if (o.TryGetValue("model", out var mname)) cfg.Model.Name = mname;
    if (o.TryGetValue("model-path", out var mpath)) cfg.Model.Path = mpath;
    if (o.TryGetValue("backend", out var be)) cfg.Compute.Backend = be;
    if (o.ContainsKey("allow-gpu")) cfg.Compute.AllowGpu = true;
    if (o.ContainsKey("dump-features")) cfg.Run.DumpFeatures = true;
    IModel model = Trademaster.Engine.Inference.ModelFactory.Create(cfg, universe);
    using var disposeModel = model as IDisposable;

    using var exec = new SimExchange(Path.Combine(cfg.Data.Dir, manifest.ExchangeFile), universe, exchange, cfg.Execution, manifest.StartTs);
    var sys = SystemInfo.Collect();
    if (cfg.Run.Mode == RunMode.Paced) Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;

    using var runner = new CycleRunner(cfg, exec, model, new SystemWallClock(), outDir, Console.Out);
    var result = runner.Run();
    var summary = RunReport.Write(result, sys, outDir);
    Console.WriteLine();
    Console.WriteLine(File.ReadAllText(summary));
    Console.WriteLine($"Reports: {outDir}{Path.DirectorySeparatorChar}{{summary.md, cycles.csv, decisions.jsonl}}");
    return 0;
}

static int VerifyAudit(string? path)
{
    if (path is null || !File.Exists(path)) throw new FileNotFoundException("usage: trademaster verify-audit <run-folder>/decisions.jsonl");
    var (records, bad, problem) = AuditLog.Verify(path);
    if (problem is null) { Console.WriteLine($"OK: {records:N0} records, hash chain intact ({path})"); return 0; }
    Console.WriteLine($"TAMPERED: chain breaks at line {bad} ({problem}); {records:N0} records before it are intact");
    return 1;
}

static int ExecStudy(Dictionary<string, string> o)
{
    var cfg = EngineConfig.Load(o.GetValueOrDefault("config", "config/default.json"));
    string data = o.GetValueOrDefault("data", cfg.Data.Dir);
    var fm = o.GetValueOrDefault("fill-model", "queue") == "touch" ? FillModel.TouchProbability : FillModel.Queue;
    var exec = new ExecutionSection
    {
        OrderLatencyMs = cfg.Execution.OrderLatencyMs, Participation = cfg.Execution.Participation,
        TouchFillProb = cfg.Execution.TouchFillProb, FillModel = fm, Seed = cfg.Execution.Seed,
    };
    var po = new ProbeOptions(Dbl(o, "rate", 2), Dbl(o, "expiry", 30), Int(o, "lots", 10), Int(o, "seed", 11));
    var manifest = DataManifest.Load(data);
    var exch = ExchangeConfig.Load(Path.Combine(cfg.Data.ExchangesDir, manifest.Exchange + ".json"));
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var (venue, groups) = ExecutionProbe.Run(data, cfg.Data.ExchangesDir, exec, po);
    var stats = ExecutionAnalytics.Compute(venue, groups, (double)exch.FeeBps);
    string impact = manifest.Parameters.GetValueOrDefault("ImpactBps", "0");
    var sb = new System.Text.StringBuilder();
    sb.AppendLine($"### Execution probe — {data} (SYNTHETIC, impact {impact} bps/avg trade), fill model {fm}, expiry {po.ExpirySec}s, {po.Lots} lots, fee {exch.FeeBps} bps");
    sb.AppendLine();
    sb.AppendLine("| Placement | Orders | Fill rate | Fully filled | Median s to 1st fill | Passive share of fills | Spread capture | Markout 1s / 5s / 30s / 60s | Adverse sel. 1s / 5s / 30s / 60s | Fee | **Shortfall incl. clean-up** | Clean-up share |");
    sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|");
    string B(double v) => double.IsFinite(v) ? v.ToString("+0.00;-0.00", CultureInfo.InvariantCulture) : "—";
    foreach (var s in stats)
        sb.AppendLine(string.Create(CultureInfo.InvariantCulture,
            $"| {s.Group} | {s.Orders:N0} | {s.FillRate:P0} | {s.FullyFilledShare:P0} | {s.MedianSecondsToFill:0.0} | {s.PassiveFillShare:P0} | {B(s.SpreadCaptureBps)} | {string.Join(" / ", s.MarkoutBps.Select(B))} | {string.Join(" / ", s.AdverseSelectionBps.Select(B))} | {B(s.FeeBps)} | **{B(s.ShortfallBps)}** | {s.CleanupShare:P0} |"));
    sb.AppendLine();
    sb.AppendLine("All values in bps. Spread capture and markouts: positive = good for us. Adverse selection: negative = price moved against us after the fill. Shortfall: cost per ordered share vs decision mid incl. fees and crossing the spread for anything unfilled (positive = cost).");
    Console.WriteLine(sb.ToString());
    Console.WriteLine($"({venue.FillLog.Count:N0} fills, {sw.Elapsed.TotalSeconds:0.0}s)");
    string outDir = o.GetValueOrDefault("out", Path.Combine("private", "reports", $"{DateTime.Now:yyyyMMdd-HHmmss}-exec-study"));
    Directory.CreateDirectory(outDir);
    File.AppendAllText(Path.Combine(outDir, "summary.md"), sb.ToString() + Environment.NewLine);
    return 0;
}

static int Serve(Dictionary<string, string> o)
{
    var cfg = EngineConfig.Load(o.GetValueOrDefault("config", "config/default.json"));
    if (o.TryGetValue("data", out var data)) cfg.Data.Dir = data;
    if (o.TryGetValue("speed", out var speed)) cfg.Run.Speed = double.Parse(speed, CultureInfo.InvariantCulture);
    cfg.Run.MaxCycles = 0;
    return Trademaster.Cli.Dashboard.Run(cfg, Int(o, "port", 5080), !o.ContainsKey("no-loop"));
}

static int BenchOnnx(Dictionary<string, string> o)
{
    string model = o.GetValueOrDefault("model", "private/models/bench/b1_dynamic_n.onnx");
    int[] ns = o.GetValueOrDefault("n", "30,300,1000,3000").Split(',').Select(int.Parse).ToArray();
    int[] ks = o.GetValueOrDefault("k", "1,10,100,1000").Split(',').Select(int.Parse).ToArray();
    int threads = Int(o, "threads", 4);
    int calls = Int(o, "calls", 200);
    // --backend cpu,dml compares backends in one report. GPU backends need --allow-gpu and still pass the GPU safety
    // gate (no driver resets in 7 days) unless --override-gpu-stability is given for a supervised, on-site run.
    var backends = o.GetValueOrDefault("backend", "cpu").Split(',').Select(b => new Trademaster.Engine.Inference.OnnxBackend(
        b.Trim().ToLowerInvariant(), Int(o, "device", -1), o.ContainsKey("allow-gpu"), o.ContainsKey("override-gpu-stability"))).ToList();
    Console.WriteLine($"ONNX benchmark: {model}, threads={threads}, backends={string.Join(",", backends.Select(b => b.Name))}, ORT flavour {Trademaster.Engine.Inference.OnnxBackend.Flavor}");
    var rows = new List<Trademaster.Engine.Inference.OnnxBench.Row>();
    foreach (var be in backends) rows.AddRange(Trademaster.Engine.Inference.OnnxBench.Run(model, ns, ks, threads, calls, Console.Out, be));
    var dir = o.GetValueOrDefault("out", Path.Combine("private", "reports", $"{DateTime.Now:yyyyMMdd-HHmmss}-bench-onnx"));
    Directory.CreateDirectory(dir);
    var md = Path.Combine(dir, "summary.md");
    File.WriteAllText(md, Trademaster.Engine.Inference.OnnxBench.Markdown(rows, SystemInfo.Collect(), model));
    Console.WriteLine($"Report: {md}");
    return 0;
}

static Dictionary<string, string> ParseOptions(string[] a)
{
    var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < a.Length; i++)
    {
        if (!a[i].StartsWith("--")) throw new ArgumentException($"Unexpected argument '{a[i]}'.");
        string key = a[i][2..];
        d[key] = i + 1 < a.Length && !a[i + 1].StartsWith("--") ? a[++i] : "true";
    }
    return d;
}

static int Int(Dictionary<string, string> o, string k, int def) => o.TryGetValue(k, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : def;
static double Dbl(Dictionary<string, string> o, string k, double def) => o.TryGetValue(k, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : def;
