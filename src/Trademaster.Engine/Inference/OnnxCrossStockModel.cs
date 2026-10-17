using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Trademaster.Core;
using Trademaster.Engine.Stages;

namespace Trademaster.Engine.Inference;

/// <summary>
/// Contents of <c>model.spec.json</c> written by <c>python -m trademaster_ml.engine_train</c>: the contract between the
/// trained network and the engine. Extra keys (training metrics, provenance) are allowed and ignored.
/// </summary>
public sealed class ModelSpecFile
{
    public string Name { get; init; } = "b1_cross_attention";
    public string Version { get; init; } = "1";
    /// <summary>Engine feature names in order; must equal the engine's feature list exactly.</summary>
    public string[] Features { get; init; } = [];
    public float[] Mean { get; init; } = [];
    public float[] Std { get; init; } = [];
    public float Clip { get; init; } = 5f;
    /// <summary>Prediction horizon in engine cycles.</summary>
    public int HorizonCycles { get; init; } = 10;
    /// <summary>Multiplier on the model's sigma, calibrated on validation data so standardized residuals have std ≈ 1.</summary>
    public double SigmaScale { get; init; } = 1.0;
    /// <summary>Instrument order the model was trained with (its id embeddings depend on it). Null = engine universe order.</summary>
    public string[]? Symbols { get; init; }
    public string Target { get; init; } = "";
    /// <summary>"engine" for models trained on engine features; anything else is refused.</summary>
    public string FeatureSource { get; init; } = "";

    private static readonly JsonSerializerOptions Opts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Skip,
    };

    public static ModelSpecFile Load(string path) =>
        JsonSerializer.Deserialize<ModelSpecFile>(File.ReadAllText(path), Opts) ?? throw new InvalidDataException($"Empty model spec {path}");
}

/// <summary>
/// Stage 5 — cross-stock neural inference. Runs a B1-style attention network (every stock and ETF is a token, so
/// each prediction can use the state of the whole universe) through ONNX Runtime on the configured backend
/// (CPU now; DirectML/CUDA/TensorRT through <see cref="OnnxBackend"/> and the GPU safety gate).
///
/// Input: the newest time slice of the stage-4 batch (already normalized with this model's spec), mapped into
/// the model's instrument order; instruments the model doesn't know, or that are masked, are excluded.
/// Output per instrument: expected log return over <see cref="HorizonCycles"/> (μ) and a calibrated σ.
/// If the model exports an attention map, the instrument each prediction attended to most is reported as an
/// inferred relationship (never an observed one; ETF holdings are not used anywhere).
/// </summary>
public sealed class OnnxCrossStockModel : IModel, IDisposable
{
    private readonly OnnxRunner _runner;
    private readonly ModelSpecFile _spec;
    private readonly int[] _engToModel, _modelToEng;
    private readonly int _nModel;
    private readonly float _sigmaScale;

    public OnnxCrossStockModel(string modelDir, int window, Universe universe, OnnxBackend backend, int threads = 4, double? sigmaScaleOverride = null)
    {
        string onnx = Path.Combine(modelDir, "model.onnx"), specPath = Path.Combine(modelDir, "model.spec.json");
        if (!File.Exists(onnx) || !File.Exists(specPath))
            throw new FileNotFoundException($"Model directory '{modelDir}' needs model.onnx and model.spec.json " +
                                            "(train one with: python -m trademaster_ml.engine_train).");
        _spec = ModelSpecFile.Load(specPath);
        if (_spec.FeatureSource != "engine")
            throw new InvalidOperationException($"Model '{modelDir}' was not trained on engine features (feature_source='{_spec.FeatureSource}'). " +
                                                "Bar-based research models cannot drive the 1-second engine.");
        Spec = new FeatureSpec { Names = _spec.Features, Mean = _spec.Mean, Std = _spec.Std, Window = window, Clip = _spec.Clip };
        Spec.CheckCompatible();

        // Instrument mapping: engine index <-> model index.
        var engSyms = universe.Instruments.Select(i => i.Symbol).ToArray();
        var modelSyms = _spec.Symbols ?? engSyms;
        _nModel = modelSyms.Length;
        var pos = modelSyms.Select((s, k) => (s, k)).ToDictionary(t => t.s, t => t.k);
        _engToModel = engSyms.Select(s => pos.TryGetValue(s, out var k) ? k : -1).ToArray();
        _modelToEng = new int[_nModel];
        Array.Fill(_modelToEng, -1);
        for (int i = 0; i < _engToModel.Length; i++) if (_engToModel[i] >= 0) _modelToEng[_engToModel[i]] = i;
        if (_engToModel.All(k => k < 0)) throw new InvalidOperationException("No engine instrument is known to the model.");

        _runner = new OnnxRunner(onnx, _nModel, F.Count, 1, threads, backend);
        _sigmaScale = (float)(sigmaScaleOverride ?? _spec.SigmaScale);
        HorizonCycles = _spec.HorizonCycles;
        using var fs = File.OpenRead(onnx);
        ArtifactHash = Convert.ToHexStringLower(SHA256.HashData(fs))[..16];
        Name = _spec.Name;
        Version = _spec.Version;
        Coverage = _engToModel.Count(k => k >= 0);
    }

    public string Name { get; }
    public string Version { get; }
    public string ArtifactHash { get; }
    public FeatureSpec Spec { get; }
    public int HorizonCycles { get; }
    /// <summary>Number of engine instruments the model covers.</summary>
    public int Coverage { get; }
    public bool HasRelations => _runner.Attn is not null;
    public string Backend => _runner.Description;

    public void Predict(ModelInputBatch b, ObservedStats stats, Span<Prediction> output)
    {
        output.Clear();
        int slot = b.Slot(0), f = F.Count;
        var x = _runner.X;
        var m = _runner.Mask;
        Array.Clear(x);
        Array.Clear(m);
        for (int i = 0; i < b.N; i++)
        {
            int mi = _engToModel[i];
            if (mi < 0 || b.Mask[i] == 0) continue;
            Array.Copy(b.X, b.Offset(slot, i), x, mi * f, f);
            m[mi] = 1f;
        }
        _runner.Run();
        var o = _runner.Out;
        var attn = _runner.Attn;
        for (int i = 0; i < b.N; i++)
        {
            int mi = _engToModel[i];
            if (mi < 0 || m[mi] == 0) continue;
            float mu = o[mi * 2] / 1e4f;
            float sigma = MathF.Exp(0.5f * o[mi * 2 + 1]) / 1e4f * _sigmaScale;
            if (!float.IsFinite(mu) || !float.IsFinite(sigma) || sigma <= 0) continue;
            var p = new Prediction { Valid = true, Mu = mu, Sigma = sigma, Beta = (float)stats.Beta[i] };
            if (attn is not null)
            {
                int best = -1; float bw = 0;
                int row = mi * _nModel;
                for (int j = 0; j < _nModel; j++)
                {
                    if (j == mi || m[j] == 0) continue;
                    float w = attn[row + j];
                    if (w > bw) { bw = w; best = j; }
                }
                if (best >= 0 && _modelToEng[best] >= 0) { p.HasPeer = true; p.Peer = _modelToEng[best]; p.PeerWeight = bw; }
            }
            output[i] = p;
        }
    }

    public void Dispose() => _runner.Dispose();
}

/// <summary>Builds the configured stage-5 model.</summary>
public static class ModelFactory
{
    public static IModel Create(EngineConfig cfg, Universe universe) => cfg.Model.Name switch
    {
        "b0_factor_residual" => new B0FactorResidualModel(cfg.Model, cfg.Features.Window, universe),
        "onnx" => new OnnxCrossStockModel(cfg.Model.Path, cfg.Features.Window, universe, OnnxBackend.From(cfg.Compute),
                                          cfg.Model.Threads, cfg.Model.SigmaScale),
        _ => throw new ArgumentException($"Unknown model '{cfg.Model.Name}' (use b0_factor_residual or onnx)."),
    };
}
