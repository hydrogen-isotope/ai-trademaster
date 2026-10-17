namespace Trademaster.Engine;

/// <summary>Per-instrument state at a cycle cutoff, for display. Prices in currency units.</summary>
public sealed record InstrumentView(
    int Id, string Symbol, string Type, string? Sector,
    double Bid, double Ask, long BidSize, long AskSize,
    double LastTrade, long CycleVolume,
    bool HasPrediction, double MuBps, double SigmaBps, double Score, double Beta,
    long Position, string Mask);

public sealed record OrderView(long Id, string Symbol, string Side, long Qty, double Limit, double MuBps, double Score, string Rule);

public sealed record FillView(long OrderId, string Symbol, string Side, long Qty, double Price, double Fee, long Ts);

public sealed record CancelView(long OrderId, string Why);

public sealed record RiskRejectView(string Symbol, string Side, long Qty, string Reason);

/// <summary>Everything the dashboard shows for one cycle. Built only when a cycle observer is attached.</summary>
public sealed record CycleView(
    long Cycle, long CutoffTs, long DecidedTs, string Status, string Phase, bool Synthetic, string DataVersion,
    string Model, double TotalMs, double ToOrderMs, double StartLagMs, bool OverBudget, double BudgetMs,
    IReadOnlyDictionary<string, double> StageMs,
    double Equity, double StartEquity, double Cash, double Fees, double GrossExposure,
    bool KillSwitch, string? KillReason, bool FeedDown, bool Connected,
    int Valid, int Instruments, double MedianDataAgeMs, double MaxDataAgeMs,
    IReadOnlyList<InstrumentView> Rows,
    IReadOnlyList<OrderView> NewOrders, IReadOnlyList<FillView> Fills,
    IReadOnlyList<CancelView> Cancels, IReadOnlyList<RiskRejectView> RiskRejects,
    int OpenOrders, long TotalFills, long TotalOrders);
