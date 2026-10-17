namespace Trademaster.Core;

public enum RunMode { Virtual, Paced }

public enum OnMissPolicy { CancelExpired, CancelAll }

public sealed class EngineConfig
{
    public DataSection Data { get; init; } = new();
    public RunSection Run { get; init; } = new();
    public IngestSection Ingest { get; init; } = new();
    public QualitySection Quality { get; init; } = new();
    public FeatureSection Features { get; init; } = new();
    public ModelSection Model { get; init; } = new();
    public DecisionSection Decision { get; init; } = new();
    public RiskSection Risk { get; init; } = new();
    public ExecutionSection Execution { get; init; } = new();
    public TradingSection Trading { get; init; } = new();
    public AccountSection Account { get; init; } = new();
    public ComputeSection Compute { get; init; } = new();

    public static EngineConfig Load(string path) => Json.Load<EngineConfig>(path);
}

public sealed class DataSection
{
    public string Dir { get; set; } = "private/data/synth-default";
    public string ExchangesDir { get; init; } = "config/exchanges";
}

public sealed class RunSection
{
    public RunMode Mode { get; set; } = RunMode.Virtual;
    public int CadenceMs { get; init; } = 1000;
    /// <summary>A cycle must reach the order interface within this budget after its cutoff.</summary>
    public int BudgetMs { get; init; } = 900;
    /// <summary>Paced mode only: replay speed multiple (1 = real time). Wall-clock budget scales by 1/speed.</summary>
    public double Speed { get; set; } = 1.0;
    /// <summary>Virtual mode only: fixed simulated delay between cutoff and order submission (keeps runs deterministic).</summary>
    public int VirtualDecisionLatencyMs { get; init; } = 50;
    public int WarmupCycles { get; init; } = 30;
    public int MaxCycles { get; set; }
    public OnMissPolicy OnMiss { get; init; } = OnMissPolicy.CancelExpired;
    public string OutputDir { get; set; } = "private/reports";
    public int ProgressEveryCycles { get; init; } = 60;
}

public sealed class IngestSection
{
    /// <summary>How long a sequence gap may stay open waiting for the missing events before they are declared lost.</summary>
    public int ReorderWindowMs { get; init; } = 50;
}

public sealed class QualitySection
{
    public double JumpSigma { get; init; } = 10.0;
    public double OutlierTradeSpreads { get; init; } = 3.0;
    public int FeedTimeoutMs { get; init; } = 3000;
    public Dictionary<string, int> MaxAgeMs { get; init; } = new() { ["high"] = 5000, ["medium"] = 15000, ["low"] = 60000 };
}

public sealed class FeatureSection
{
    public int Window { get; init; } = 60;
}

public sealed class ModelSection
{
    public string Name { get; init; } = "b0_factor_residual";
    public int Lookback { get; init; } = 30;
    public double Kappa { get; init; } = 0.5;
    public int HorizonCycles { get; init; } = 10;
    public int BetaHalflifeCycles { get; init; } = 300;
}

public sealed class DecisionSection
{
    public double EntryScore { get; init; } = 1.0;
    public double ExitScore { get; init; } = 0.0;
    public double MarginBps { get; init; } = 2.0;
    /// <summary>Engine's slippage estimate used in the cost hurdle (the simulator models fills separately).</summary>
    public double SlippageBps { get; init; } = 1.0;
    public double BaseNotional { get; init; } = 10_000;
    /// <summary>Orders stay valid for one cadence plus this grace period, then the engine cancels them.</summary>
    public int OrderGraceMs { get; init; } = 200;
    /// <summary>aggressive: cross the spread up to the cutoff price. passive: rest at our side's best price (join),
    /// capped by the cutoff, and keep queue position while the order is still at or better than the best price.</summary>
    public string ExecutionStyle { get; set; } = "aggressive";
    /// <summary>Passive style: assumed adverse-selection cost (bps) used in the entry hurdle instead of the half spread.</summary>
    public double PassiveCostBps { get; set; } = 1.0;
}

public sealed class RiskSection
{
    public double MaxPositionNotional { get; init; } = 50_000;
    public double MaxGrossNotional { get; init; } = 500_000;
    public double MaxDailyLoss { get; set; } = 5_000;
    public int MaxOrdersPerCycle { get; init; } = 20;
    /// <summary>Engine-side collar: limit must be within this % of mid.</summary>
    public double PriceCollarPct { get; init; } = 2.0;
}

public sealed record DisconnectWindow(double AtS, double DurationS);

/// <summary>
/// How resting limit orders get filled in the simulator.
/// TouchProbability: fill on trade-through, or on an exact touch with a fixed probability (M1 model).
/// Queue: L1 queue-position model — joining a level puts the displayed size ahead of us; trades at our price and
/// (proportionally) size decreases consume it; we fill only after it is gone, or when the price trades through.
/// </summary>
public enum FillModel { TouchProbability, Queue }

public sealed class ExecutionSection
{
    public string Adapter { get; init; } = "sim";
    public int OrderLatencyMs { get; init; } = 5;
    public double Participation { get; init; } = 0.5;
    public double TouchFillProb { get; init; } = 0.3;
    public FillModel FillModel { get; set; } = FillModel.TouchProbability;
    public bool CancelOnDisconnect { get; init; } = true;
    public List<DisconnectWindow> Disconnects { get; init; } = [];
    public int Seed { get; init; } = 7;
}

public sealed class TradingSection
{
    public bool LiveEnabled { get; init; }
    public string LiveConfirm { get; init; } = "";
}

/// <summary>
/// Device policy. GPU is off by default: the RX 7800 XT on the development machine has a history of
/// driver resets that escalate into full system hangs (docs/GPU-STABILITY.md).
/// </summary>
public sealed class ComputeSection
{
    public bool AllowGpu { get; init; }
    /// <summary>Skips the "no driver resets in 7 days" requirement. Only for supervised, on-site runs.</summary>
    public bool OverrideGpuStabilityCheck { get; init; }
}

public sealed class AccountSection
{
    public double StartingCash { get; init; } = 1_000_000;
}
