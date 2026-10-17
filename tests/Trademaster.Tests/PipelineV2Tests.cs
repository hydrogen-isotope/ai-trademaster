using Trademaster.Core;
using Trademaster.Engine;
using Trademaster.Engine.Inference;
using Trademaster.Engine.Stages;
using static Trademaster.Tests.TestData;

namespace Trademaster.Tests;

public class AuditChainTests
{
    private sealed class FakeModel : IModel
    {
        public string Name => "fake";
        public string Version => "1";
        public string ArtifactHash => "abc";
        public FeatureSpec Spec { get; } = FeatureSpec.Identity(1);
        public int HorizonCycles => 1;
        public void Predict(ModelInputBatch batch, ObservedStats stats, Span<Prediction> output) { }
    }

    private static DataManifest Manifest() => new()
    {
        Kind = "synthetic", Synthetic = true, Exchange = "SYNTH", StartTs = 0, EndTs = 1, FeedFile = "f", FeedSha256 = new string('0', 64),
        FeedEvents = 0, ExchangeFile = "e", ExchangeSha256 = new string('0', 64), ExchangeEvents = 0, UniverseFile = "u",
    };

    private static string WriteLog(int records)
    {
        var path = Path.Combine(TempDir("audit"), "decisions.jsonl");
        using var log = new AuditLog(path, Universe(2), new FakeModel(), Manifest());
        for (int k = 0; k < records; k++)
        {
            var d = new CycleDecision { CycleId = k, CutoffTs = T(k), DecidedTs = T(k) + 1, Equity = 1000m + k };
            log.Write(d);
        }
        return path;
    }

    [Fact]
    public void Intact_chain_verifies()
    {
        var (records, bad, problem) = AuditLog.Verify(WriteLog(5));
        Assert.Equal(5, records);
        Assert.Equal(0, bad);
        Assert.Null(problem);
    }

    [Fact]
    public void Edited_record_is_detected()
    {
        var path = WriteLog(5);
        var lines = File.ReadAllLines(path);
        lines[2] = lines[2].Replace("\"equity\":\"1002.00\"", "\"equity\":\"9002.00\"");
        File.WriteAllLines(path, lines);
        var (records, bad, problem) = AuditLog.Verify(path);
        Assert.Equal(3, bad);
        Assert.Equal(2, records);
        Assert.Contains("modified", problem);
    }

    [Fact]
    public void Deleted_record_is_detected()
    {
        var path = WriteLog(5);
        var lines = File.ReadAllLines(path).ToList();
        lines.RemoveAt(1);
        File.WriteAllLines(path, lines);
        var (_, bad, problem) = AuditLog.Verify(path);
        Assert.Equal(2, bad);
        Assert.Contains("prev_hash", problem);
    }
}

public class ContentDedupTests
{
    private static MarketEvent Q(double exchS, double recvS, decimal bid = 10.00m) =>
        MarketEvent.Quote(0, T(exchS), T(recvS), 0, P(bid), P(bid + 0.01m), 100, 100);

    [Fact]
    public void Drops_exact_repeats_and_stale_quotes_without_sequence_numbers()
    {
        // Seq is 0 everywhere (feed without sequence numbers). Event 2 repeats event 1; event 4 is older than event 3.
        var evs = new[] { Q(0.10, 0.11), Q(0.10, 0.12), Q(0.30, 0.31, 10.02m), Q(0.20, 0.32, 10.01m), Q(0.40, 0.41, 10.03m) };
        var s1 = new S1Ingest(new ReplayReader(WriteReplay(TempDir("content"), evs)), 3, 50, "content", 2000);
        s1.ProcessUntil(T(1));
        Assert.Equal(3, s1.Accepted.Count);
        Assert.Equal(1, s1.Counters.Duplicates);
        Assert.Equal(1, s1.Counters.StaleDropped);
        Assert.Equal(P(10.03m).Raw, s1.Accepted[^1].Px1);
    }

    [Fact]
    public void Unknown_dedup_mode_is_rejected()
    {
        var path = WriteReplay(TempDir("content2"), [Q(0.1, 0.1)]);
        Assert.Throws<ArgumentException>(() => new S1Ingest(new ReplayReader(path), 3, 50, "hash"));
    }
}

public class CrossStockModelTests
{
    [Fact]
    public void Refuses_missing_files_and_models_not_trained_on_engine_features()
    {
        var dir = TempDir("model");
        Assert.Throws<FileNotFoundException>(() => new OnnxCrossStockModel(dir, 60, Universe(2), new OnnxBackend()));
        File.WriteAllBytes(Path.Combine(dir, "model.onnx"), [0]);
        File.WriteAllText(Path.Combine(dir, "model.spec.json"), """{ "feature_source": "bars_5m", "features": ["r1"] }""");
        var ex = Assert.Throws<InvalidOperationException>(() => new OnnxCrossStockModel(dir, 60, Universe(2), new OnnxBackend()));
        Assert.Contains("engine features", ex.Message);
    }

    [Fact]
    public void Refuses_feature_list_mismatch()
    {
        var dir = TempDir("model2");
        File.WriteAllBytes(Path.Combine(dir, "model.onnx"), [0]);
        File.WriteAllText(Path.Combine(dir, "model.spec.json"),
            """{ "feature_source": "engine", "features": ["r1","something_else"], "mean": [0,0], "std": [1,1] }""");
        Assert.Throws<InvalidOperationException>(() => new OnnxCrossStockModel(dir, 60, Universe(2), new OnnxBackend()));
    }

    [Fact]
    public void Model_factory_builds_b0_by_default_and_rejects_unknown_names()
    {
        var cfg = new EngineConfig();
        Assert.IsType<B0FactorResidualModel>(ModelFactory.Create(cfg, Universe(2)));
        cfg.Model.Name = "magic";
        Assert.Throws<ArgumentException>(() => ModelFactory.Create(cfg, Universe(2)));
    }
}
