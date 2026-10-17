using Trademaster.Core;

namespace Trademaster.Engine.Orders;

public enum WorkingState { PendingNew, Working, PendingCancel }

public sealed class WorkingOrder(OrderRequest req, long submitTs, long validUntil)
{
    public OrderRequest Req { get; } = req;
    public long SubmitTs { get; } = submitTs;
    /// <summary>Engine-side expiry; extended when a later cycle re-affirms the same order.</summary>
    public long ValidUntil { get; set; } = validUntil;
    public long Leaves { get; set; } = req.Qty;
    public WorkingState State { get; set; } = WorkingState.PendingNew;
}

/// <summary>
/// Engine-side view of orders, positions and cash, driven only by execution reports
/// (an order is never assumed filled). Reconciles against the venue snapshot after reconnects.
/// </summary>
public sealed class OrderManager
{
    private readonly Dictionary<long, WorkingOrder> _open = [];
    private long _nextId = 1;

    public long[] Position { get; }
    /// <summary>Average entry price of the open position (0 when flat). Fees are tracked separately.</summary>
    public decimal[] AvgCost { get; }
    /// <summary>Realized P&amp;L per instrument from closed quantity, before fees.</summary>
    public decimal[] Realized { get; }
    public decimal Cash { get; private set; }
    public decimal Fees { get; private set; }
    public decimal StartingCash { get; }

    public long Acks, Fills, FilledQty, Rejects, Cancelled, CancelRejects, Reconciliations, ReconcileDiffs;
    public Dictionary<string, long> RejectReasons { get; } = [];

    public OrderManager(int instruments, decimal startingCash)
    {
        Position = new long[instruments];
        AvgCost = new decimal[instruments];
        Realized = new decimal[instruments];
        Cash = StartingCash = startingCash;
    }

    public IEnumerable<WorkingOrder> Open => _open.Values;
    public int OpenCount => _open.Count;

    public long NextOrderId() => _nextId++;

    public WorkingOrder? OpenFor(int instrument)
    {
        foreach (var o in _open.Values) if (o.Req.Instrument == instrument && o.State != WorkingState.PendingCancel) return o;
        return null;
    }

    public void Track(in OrderRequest req, long submitTs, long validUntil) =>
        _open[req.OrderId] = new WorkingOrder(req, submitTs, validUntil);

    public void Forget(long orderId) => _open.Remove(orderId);

    public void MarkPendingCancel(long orderId)
    {
        if (_open.TryGetValue(orderId, out var o)) o.State = WorkingState.PendingCancel;
    }

    /// <summary>Open (non-cancelling) buy quantity for an instrument — counted against position limits.</summary>
    public long PendingQty(int instrument, Side side)
    {
        long q = 0;
        foreach (var o in _open.Values)
            if (o.Req.Instrument == instrument && o.Req.Side == side && o.State != WorkingState.PendingCancel) q += o.Leaves;
        return q;
    }

    /// <summary>Average-cost accounting for a signed fill (call before updating the position).</summary>
    private void UpdateCost(int i, long signedQty, decimal px)
    {
        long pos = Position[i];
        if (pos == 0 || Math.Sign(pos) == Math.Sign(signedQty))
        {
            long newPos = pos + signedQty;
            AvgCost[i] = (AvgCost[i] * Math.Abs(pos) + px * Math.Abs(signedQty)) / Math.Abs(newPos);
            return;
        }
        long closing = Math.Min(Math.Abs(pos), Math.Abs(signedQty));
        Realized[i] += closing * (px - AvgCost[i]) * Math.Sign(pos);
        long rest = Math.Abs(signedQty) - closing;
        if (rest > 0) AvgCost[i] = px;                    // flipped through zero: the remainder opens at this price
        else if (Math.Abs(pos) == closing) AvgCost[i] = 0; // flat
    }

    public void Apply(in ExecutionReport r)
    {
        _open.TryGetValue(r.OrderId, out var o);
        switch (r.Kind)
        {
            case ReportKind.Ack:
                Acks++;
                if (o is not null && o.State == WorkingState.PendingNew) o.State = WorkingState.Working;
                break;
            case ReportKind.Reject:
                Rejects++;
                RejectReasons[r.Reason ?? "?"] = RejectReasons.GetValueOrDefault(r.Reason ?? "?") + 1;
                _open.Remove(r.OrderId);
                break;
            case ReportKind.Fill:
                Fills++;
                FilledQty += r.FillQty;
                // Applied even if the order already left the book (e.g. removed by reconciliation while
                // its fill report was still in flight): the report carries side and instrument.
                int sign = (int)r.Side;
                UpdateCost(r.Instrument, sign * r.FillQty, (decimal)r.FillPrice.Raw / Price.Scale);
                Position[r.Instrument] += sign * r.FillQty;
                Cash -= sign * r.FillQty * (decimal)r.FillPrice.Raw / Price.Scale;
                Cash -= r.Fee;
                Fees += r.Fee;
                if (o is not null)
                {
                    o.Leaves = r.LeavesQty;
                    if (r.LeavesQty == 0) _open.Remove(r.OrderId);
                }
                break;
            case ReportKind.Cancelled:
                Cancelled++;
                _open.Remove(r.OrderId);
                break;
            case ReportKind.CancelReject:
                CancelRejects++;
                if (r.Reason == "UnknownOrder") _open.Remove(r.OrderId);
                else if (o is not null) o.State = WorkingState.Working; // too late: fills will close it
                break;
        }
    }

    /// <summary>
    /// Replaces the engine's open-order book with the venue's authoritative snapshot (right after reconnect).
    /// Positions are not touched here: reports held during the disconnect are still being delivered.
    /// </summary>
    public void ReconcileOrders(AccountSnapshot snap)
    {
        Reconciliations++;
        var venue = snap.OpenOrders.ToDictionary(o => o.Request.OrderId);
        foreach (var id in _open.Keys.Order().ToList())
            if (!venue.ContainsKey(id)) { _open.Remove(id); ReconcileDiffs++; }
        foreach (var (id, vo) in venue.OrderBy(kv => kv.Key))
        {
            if (_open.TryGetValue(id, out var mine))
            {
                if (mine.Leaves != vo.LeavesQty) { mine.Leaves = vo.LeavesQty; ReconcileDiffs++; }
            }
            else
            {
                var w = new WorkingOrder(vo.Request, snap.Ts, snap.Ts) { Leaves = vo.LeavesQty, State = WorkingState.Working };
                _open[id] = w;
                ReconcileDiffs++;
            }
        }
    }

    /// <summary>Checks positions against the venue once held reports have arrived; venue wins on mismatch.</summary>
    public int VerifyPositions(AccountSnapshot snap)
    {
        int diffs = 0;
        for (int i = 0; i < Position.Length; i++)
        {
            long v = snap.Positions.GetValueOrDefault(i);
            if (Position[i] != v) { Position[i] = v; diffs++; }
        }
        ReconcileDiffs += diffs;
        return diffs;
    }

    public decimal Equity(Func<int, double> mark)
    {
        decimal eq = Cash;
        for (int i = 0; i < Position.Length; i++)
            if (Position[i] != 0) eq += Position[i] * (decimal)mark(i);
        return eq;
    }

    public double GrossNotional(Func<int, double> mark)
    {
        double g = 0;
        for (int i = 0; i < Position.Length; i++) if (Position[i] != 0) g += Math.Abs(Position[i] * mark(i));
        foreach (var o in _open.Values)
            if (o.Req.Side == Side.Buy && o.State != WorkingState.PendingCancel) g += o.Leaves * o.Req.Limit.ToDouble();
        return g;
    }
}
