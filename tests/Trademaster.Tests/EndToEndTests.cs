using System.Text.Json;
using Trademaster.Core;
using Trademaster.Engine;
using Trademaster.Engine.Stages;
using Trademaster.Sim;
using Trademaster.Telemetry;

namespace Trademaster.Tests;

public class EndToEndTests
{
    private static EngineConfig Config(string data, string exchanges, RunMode mode = RunMode.Virtual, List<DisconnectWindow>? disconnects = null) => new()
    {
        Data = new DataSection { Dir = data, ExchangesDir = exchanges },
        Run = new RunSection { Mode = mode, WarmupCycles = 5, ProgressEveryCycles = 0 },
        Execution = new ExecutionSection { Disconnects = disconnects ?? [] },
        Risk = new RiskSection { MaxDailyLoss = 1e9 },
    };

    private static (RunResult Result, SimExchange Venue) RunOnce(EngineConfig cfg, string outDir, IWallClock? wall = null, Func<IModel, IModel>? wrap = null)
    {
        var m = DataManifest.Load(cfg.Data.Dir);
        var u = Universe.Load(Path.Combine(cfg.Data.Dir, m.UniverseFile));
        var ex = ExchangeConfig.Load(Path.Combine(cfg.Data.ExchangesDir, "SYNTH.json"));
        IModel model = new B0FactorResidualModel(cfg.Model, cfg.Features.Window, u);
        if (wrap is not null) model = wrap(model);
        var venue = new SimExchange(Path.Combine(cfg.Data.Dir, m.ExchangeFile), u, ex, cfg.Execution, m.StartTs);
        using var runner = new CycleRunner(cfg, venue, model, wall ?? new SystemWallClock(), outDir);
        return (runner.Run(), venue);
    }

    [Fact]
    public void Virtual_replay_is_deterministic_and_auditable()
    {
        var (data, ex) = TestData.Generate("det");
        var cfg = Config(data, ex);
        var (a, va) = RunOnce(cfg, TestData.TempDir("run-a"));
        var (b, vb) = RunOnce(cfg, TestData.TempDir("run-b"));
        va.Dispose(); vb.Dispose();

        Assert.Equal(ReplayFile.Sha256(a.DecisionsPath), ReplayFile.Sha256(b.DecisionsPath));
        Assert.True(a.OrdersSubmitted > 0, "the scenario should produce some orders");

        var lines = File.ReadAllLines(a.DecisionsPath);
        Assert.Equal(a.Cycles.Count, lines.Length);
        using var first = JsonDocument.Parse(lines.First(l => l.Contains("\"orders\":[{")));
        var r = first.RootElement;
        foreach (var key in new[] { "cycle", "cutoff_ts", "decided_ts", "status", "model", "model_hash", "data_version", "input_hash", "provenance" })
            Assert.True(r.TryGetProperty(key, out _), $"decision record missing '{key}'");
        Assert.True(r.GetProperty("synthetic").GetBoolean());
    }

    [Fact]
    public void Engine_positions_match_venue_after_disconnect_and_reconnect()
    {
        var (data, ex) = TestData.Generate("recon", minutes: 4);
        var cfg = Config(data, ex, disconnects: [new DisconnectWindow(90, 10)]);
        var (res, venue) = RunOnce(cfg, TestData.TempDir("run-recon"));
        var snap = venue.Snapshot(0);
        for (int i = 0; i < res.Universe.Count; i++)
            Assert.Equal(snap.Positions.GetValueOrDefault(i), res.Orders.Position[i]);
        Assert.Equal(1, res.Orders.Reconciliations);
        Assert.Contains(res.Cycles, c => c.Status == CycleStatus.Disconnected);
        venue.Dispose();
    }

    /// <summary>Wraps a model and burns fake wall-clock time on chosen cycles.</summary>
    private sealed class SlowModel(IModel inner, FakeWallClock clock, Dictionary<long, long> delayNsByCycle) : IModel
    {
        public string Name => inner.Name;
        public string Version => inner.Version;
        public string ArtifactHash => inner.ArtifactHash;
        public FeatureSpec Spec => inner.Spec;
        public int HorizonCycles => inner.HorizonCycles;
        public void Predict(ModelInputBatch batch, ObservedStats stats, Span<Prediction> output)
        {
            inner.Predict(batch, stats, output);
            if (delayNsByCycle.TryGetValue(batch.CycleId, out var d)) clock.Advance(d);
        }
    }

    [Fact]
    public void Paced_overrun_is_recorded_as_missed_submits_nothing_and_skips_passed_cutoffs()
    {
        var (data, ex) = TestData.Generate("miss", minutes: 2);
        var cfg = Config(data, ex, RunMode.Paced);
        var clock = new FakeWallClock(1_000 * Ns.PerSec);
        var delays = new Dictionary<long, long> { [60] = 950 * Ns.PerMs, [70] = 2_500 * Ns.PerMs };
        var (res, venue) = RunOnce(cfg, TestData.TempDir("run-miss"), clock, m => new SlowModel(m, clock, delays));
        venue.Dispose();

        var byId = res.Cycles.ToDictionary(c => c.CycleId);
        Assert.Equal(CycleStatus.Missed, byId[60].Status);
        Assert.Equal(CycleStatus.Missed, byId[70].Status);
        Assert.Equal(CycleStatus.SkippedOverrun, byId[71].Status);
        Assert.Equal(CycleStatus.SkippedOverrun, byId[72].Status);
        Assert.Equal(CycleStatus.Ok, byId[73].Status);
        Assert.Equal(CycleStatus.Ok, byId[61].Status);

        foreach (var line in File.ReadLines(res.DecisionsPath))
        {
            using var doc = JsonDocument.Parse(line);
            var st = doc.RootElement.GetProperty("status").GetString();
            if (st is "missed" or "skipped_overrun")
                Assert.Equal(0, doc.RootElement.GetProperty("orders").GetArrayLength());
        }
    }
}
