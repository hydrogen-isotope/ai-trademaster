namespace Trademaster.Core;

public enum Side : sbyte { Buy = 1, Sell = -1 }

/// <summary>A limit order. The limit (cutoff) price bounds the execution price; it does not guarantee a fill.</summary>
public readonly record struct OrderRequest(long OrderId, int Instrument, Side Side, long Qty, Price Limit, long CycleId);

public enum ReportKind { Ack, Reject, Fill, Cancelled, CancelReject }

/// <summary>Exchange → engine message. <see cref="Ts"/> is exchange time; delivery to the engine is later.</summary>
public readonly record struct ExecutionReport(
    long OrderId,
    int Instrument,
    Side Side,
    ReportKind Kind,
    long Ts,
    long FillQty,
    Price FillPrice,
    decimal Fee,
    long LeavesQty,
    string? Reason);

public enum SubmitResult { Sent, Disconnected }

public sealed record OpenOrderSnapshot(OrderRequest Request, long LeavesQty);

/// <summary>Authoritative account state from the execution venue, used to reconcile after a reconnect.</summary>
public sealed record AccountSnapshot(long Ts, IReadOnlyList<OpenOrderSnapshot> OpenOrders, IReadOnlyDictionary<int, long> Positions);

/// <summary>
/// Execution venue interface, kept separate from inference. Only a simulated implementation exists;
/// live adapters require explicit enablement (trading.live_enabled + trading.live_confirm).
/// </summary>
public interface IExecutionAdapter
{
    string Name { get; }
    bool IsSimulated { get; }
    /// <summary>Connection state as of the last <see cref="AdvanceTo"/>.</summary>
    bool IsConnected { get; }
    /// <summary>Processes venue-side activity up to and including <paramref name="ts"/>.</summary>
    void AdvanceTo(long ts);
    SubmitResult Submit(in OrderRequest request, long ts);
    SubmitResult Cancel(long orderId, long ts);
    /// <summary>Appends reports whose delivery time is ≤ <paramref name="ts"/>.</summary>
    void DrainReports(long ts, List<ExecutionReport> into);
    AccountSnapshot Snapshot(long ts);
}
