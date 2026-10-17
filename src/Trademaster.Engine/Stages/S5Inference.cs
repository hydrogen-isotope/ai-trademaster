using System.IO.Hashing;
using System.Text;
using Trademaster.Core;

namespace Trademaster.Engine.Stages;

/// <summary>Model output for one instrument. Provenance: "inferred".</summary>
public struct Prediction
{
    public bool Valid;
    /// <summary>Expected log return over the model horizon.</summary>
    public float Mu;
    /// <summary>Uncertainty (std) of that return.</summary>
    public float Sigma;
    /// <summary>Estimated relationship used (beta to reference ETF); provenance "estimated".</summary>
    public float Beta;
    /// <summary>Model-inferred strongest relationship: the instrument this prediction attended to most (provenance "inferred").</summary>
    public bool HasPeer;
    public int Peer;
    public float PeerWeight;
}

/// <summary>Stage 5 contract. Every model — rule baseline, ONNX network — implements this.</summary>
public interface IModel
{
    string Name { get; }
    string Version { get; }
    /// <summary>Hash of the model artifact (parameters / weights) for the audit record.</summary>
    string ArtifactHash { get; }
    FeatureSpec Spec { get; }
    int HorizonCycles { get; }
    void Predict(ModelInputBatch batch, ObservedStats stats, Span<Prediction> output);
}

/// <summary>
/// B0 — non-neural cross-stock baseline. For each stock, the residual of its one-cycle returns against
/// its reference ETF (beta from <see cref="ObservedStats"/>) is summed over a lookback; the forecast is
/// partial reversion of that residual: mu = -kappa * residual, sigma = residual vol * sqrt(horizon).
/// Every neural model must beat this after costs.
/// </summary>
public sealed class B0FactorResidualModel : IModel
{
    private readonly int _lookback;
    private readonly double _kappa;
    private readonly bool[] _tradeable;

    public B0FactorResidualModel(ModelSection cfg, int window, Universe universe)
    {
        if (cfg.Lookback > window) throw new ArgumentException("model.lookback must be ≤ features.window");
        _lookback = cfg.Lookback;
        _kappa = cfg.Kappa;
        HorizonCycles = cfg.HorizonCycles;
        Spec = FeatureSpec.Identity(window);
        _tradeable = universe.Instruments.Select(i => i.Type == InstrumentType.Stock).ToArray();
        var p = $"b0|lookback={_lookback}|kappa={_kappa:R}|horizon={HorizonCycles}|window={window}";
        ArtifactHash = Convert.ToHexStringLower(XxHash64.Hash(Encoding.UTF8.GetBytes(p)));
    }

    public string Name => "b0_factor_residual";
    public string Version => "1";
    public string ArtifactHash { get; }
    public FeatureSpec Spec { get; }
    public int HorizonCycles { get; }

    public void Predict(ModelInputBatch b, ObservedStats stats, Span<Prediction> output)
    {
        output.Clear();
        if (b.Filled < _lookback) return;
        for (int i = 0; i < b.N; i++)
        {
            int j = stats.RefEtf[i];
            if (!_tradeable[i] || j < 0 || b.Mask[i] == 0 || b.Mask[j] == 0) continue;
            double beta = stats.Beta[i], sum = 0, sumSq = 0;
            for (int lag = 0; lag < _lookback; lag++)
            {
                int s = b.Slot(lag);
                double res = b.X[b.Offset(s, i) + F.R1] - beta * b.X[b.Offset(s, j) + F.R1];
                sum += res;
                sumSq += res * res;
            }
            double sigma = Math.Sqrt(sumSq / _lookback * HorizonCycles);
            if (sigma < 1e-6) continue;
            output[i] = new Prediction { Valid = true, Mu = (float)(-_kappa * sum), Sigma = (float)sigma, Beta = (float)beta };
        }
    }
}
