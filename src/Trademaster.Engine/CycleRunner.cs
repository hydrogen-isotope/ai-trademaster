using System.Runtime;
using Trademaster.Core;
using Trademaster.Engine.Orders;
using Trademaster.Engine.Risk;
using Trademaster.Engine.Stages;
using Trademaster.Telemetry;

namespace Trademaster.Engine;

/// <summary>Everything a report needs after a run.</summary>
public sealed class RunResult
{
    public required EngineConfig Config { get; init; }
    public required DataManifest Manifest { get; init; }
    public required Universe Universe { get; init; }
    public required IModel Model { get; init; }
    public required string AdapterName { get; init; }
    public required List<CycleTiming> Cycles { get; init; }
    public required IngestCounters Ingest { get; init; }
    public required QualityCounters Quality { get; init; }
    public required OrderManager Orders { get; init; }
    public required RiskManager Risk { get; init; }
    public required string DecisionsPath { get; init; }
    public required decimal FinalEquity { get; init; }
    public required long PeakWorkingSet { get; init; }
    public required TimeSpan TotalGcPause { get; init; }
    public required double WallSeconds { get; init; }
    public required long OrdersSubmitted { get; init; }
    public required long OrdersNotSentDisconnected { get; init; }
}

/// <summary>
/// Drives the six stages once per cadence. Cycle k uses only data with RecvTs ≤ T_k.
/// Virtual mode: simulated time, deterministic decisions, compute latency still measured (would-miss flags).
/// Paced mode: replay released at wall-clock speed; a cycle that reaches the order interface after its
/// deadline is MISSED and submits nothing; cutoffs that passed during an overrun are SKIPPED, never queued.
/// </summary>
public sealed class CycleRunner : IDisposable
{
    private readonly EngineConfig _cfg;
    private readonly IExecutionAdapter _exec;
    private readonly IModel _model;
    private readonly IWallClock _wall;
    private readonly DataManifest _manifest;
    private readonly Universe _u;
    private readonly MarketCalendar _calendar;
    private readonly ReplayReader _feed;
    private readonly S1Ingest _s1;
    private readonly S2Evaluate _s2;
    private readonly S3Features _s3;
    private readonly S4Pack _s4;
    private readonly DecisionPolicy _policy;
    private readonly RiskManager _risk;
    private readonly OrderManager _om;
    private readonly AuditLog _audit;
    private readonly Prediction[] _preds;
    private readonly List<OrderIntent> _intents = new(256);
    private readonly List<OrderIntent> _desired = new(256);
    private readonly List<ExecutionReport> _reports = new(256);
    private readonly StageClock _clock = new();
    private readonly CycleDecision _d = new();
    private readonly double[] _ageScratch;
    private readonly List<CycleTiming> _timings = [];
    private readonly TextWriter _log;
    private readonly FeatureDump? _dump;
    private readonly bool _flattenOnKill;
    private readonly HashSet<long> _flattenIds = [];
    private readonly TickTable[] _ticks;

    private readonly long _cadence, _budget, _grace, _virtualLatency;
    private bool _wasConnected = true;
    private long _verifyPositionsAtCycle = -1;
    private long _ordersSubmitted, _notSent;
    private long _wallStart, _t0;

    public string OutputDir { get; }

    /// <summary>Optional display hook, called after each cycle's audit record (wall-clock data; not part of the audit).</summary>
    public Action<CycleView>? OnCycle { get; set; }

    /// <summary>Set from another thread to end the run after the current cycle.</summary>
    public volatile bool StopRequested;

    public CycleRunner(EngineConfig cfg, IExecutionAdapter exec, IModel model, IWallClock wall, string outputDir, TextWriter? log = null)
    {
        if (cfg.Trading.LiveEnabled || !exec.IsSimulated)
            throw new InvalidOperationException("Live trading is not implemented in this prototype. Only the simulated adapter may be used.");
        _cfg = cfg;
        _exec = exec;
        _model = model;
        _wall = wall;
        _log = log ?? TextWriter.Null;
        OutputDir = outputDir;
        Directory.CreateDirectory(outputDir);

        var dir = cfg.Data.Dir;
        _manifest = DataManifest.Load(dir);
        _manifest.Verify(dir);
        _u = Universe.Load(Path.Combine(dir, _manifest.UniverseFile));
        var ex = ExchangeConfig.Load(Path.Combine(cfg.Data.ExchangesDir, _manifest.Exchange + ".json"));
        _calendar = new MarketCalendar(ex);

        int n = _u.Count;
        _feed = new ReplayReader(Path.Combine(dir, _manifest.FeedFile));
        _s1 = new S1Ingest(_feed, n, cfg.Ingest.ReorderWindowMs, cfg.Ingest.Dedup, cfg.Ingest.ContentWindowMs);
        _s2 = new S2Evaluate(_u, cfg.Quality);
        _s3 = new S3Features(n, ex.LotSize, _s2);
        if (model.Spec.Window != cfg.Features.Window) throw new InvalidOperationException("Model window does not match features.window.");
        _s4 = new S4Pack(n, model.Spec, _s2, _s3, _u.ReferenceEtfs(), cfg.Model.BetaHalflifeCycles);
        _policy = new DecisionPolicy(cfg.Decision, cfg.Risk, ex, _u);
        _om = new OrderManager(n, (decimal)cfg.Account.StartingCash);
        _risk = new RiskManager(cfg.Risk, ex, _om.StartingCash, _u);
        _audit = new AuditLog(Path.Combine(outputDir, "decisions.jsonl"), _u, model, _manifest);
        _preds = new Prediction[n];
        _ageScratch = new double[n];
        if (cfg.Run.DumpFeatures) _dump = new FeatureDump(outputDir, _u, _manifest, cfg.Run.CadenceMs);
        _ticks = ex.BuildTickTables(_u);
        _flattenOnKill = cfg.Risk.KillSwitchAction switch
        {
            "hold" => false, "flatten" => true,
            _ => throw new InvalidOperationException($"risk.kill_switch_action must be 'hold' or 'flatten' (got '{cfg.Risk.KillSwitchAction}')."),
        };

        _cadence = cfg.Run.CadenceMs * Ns.PerMs;
        _budget = cfg.Run.BudgetMs * Ns.PerMs;
        _grace = cfg.Decision.OrderGraceMs * Ns.PerMs;
        _virtualLatency = cfg.Run.VirtualDecisionLatencyMs * Ns.PerMs;
        if (_budget >= _cadence) throw new InvalidOperationException("run.budget_ms must be smaller than run.cadence_ms.");
    }

    private bool Paced => _cfg.Run.Mode == RunMode.Paced;
    private long SimToWall(long t) => _wallStart + (long)((t - _t0) / _cfg.Run.Speed);
    private long WallToSim(long w) => _t0 + (long)((w - _wallStart) * _cfg.Run.Speed);

    private double Mark(int i) => _s2.GoodBid[i] > 0 ? _s2.GoodMid(i) : 0;

    public RunResult Run()
    {
        var oldMode = GCSettings.LatencyMode;
        GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
        var gcStart = GC.GetTotalPauseDuration();
        long peakWs = 0;
        long wall0 = _wall.NowNs();
        try
        {
            _t0 = (_manifest.StartTs / _cadence + 1) * _cadence;
            long total = (_manifest.EndTs + _cadence - _t0) / _cadence + 1;
            if (_cfg.Run.MaxCycles > 0) total = Math.Min(total, _cfg.Run.MaxCycles);
            _wallStart = _wall.NowNs() + 100 * Ns.PerMs;
            _log.WriteLine($"Run: {total} cycles, mode={_cfg.Run.Mode}, cadence={_cfg.Run.CadenceMs} ms, budget={_cfg.Run.BudgetMs} ms" +
                           (Paced ? $", speed={_cfg.Run.Speed}x" : "") + $", data={_manifest.DataVersion}" + (_manifest.Synthetic ? " [SYNTHETIC]" : ""));

            long k = 0;
            while (k < total && !StopRequested)
            {
                long tk = _t0 + k * _cadence;
                long startLag = 0;
                if (Paced)
                {
                    _wall.WaitUntil(SimToWall(tk));
                    startLag = _wall.NowNs() - SimToWall(tk);
                }
                RunCycle(k, tk, startLag);

                if (k % 30 == 0) peakWs = Math.Max(peakWs, SystemInfo.WorkingSetBytes());
                if (_cfg.Run.ProgressEveryCycles > 0 && k % _cfg.Run.ProgressEveryCycles == 0) Progress(k);

                long next = k + 1;
                if (Paced)
                {
                    long nowSim = WallToSim(_wall.NowNs());
                    while (next < total && _t0 + next * _cadence < nowSim) RecordSkipped(next++);
                }
                k = next;
            }
            FinalSync(_t0 + (total - 1) * _cadence + _cadence);
            peakWs = Math.Max(peakWs, SystemInfo.WorkingSetBytes());
        }
        finally
        {
            GCSettings.LatencyMode = oldMode;
            _audit.Dispose();
            _dump?.Dispose();
        }

        return new RunResult
        {
            Config = _cfg, Manifest = _manifest, Universe = _u, Model = _model, AdapterName = _exec.Name,
            Cycles = _timings, Ingest = _s1.Counters, Quality = _s2.Counters, Orders = _om, Risk = _risk,
            DecisionsPath = _audit.Path, FinalEquity = _om.Equity(Mark), PeakWorkingSet = peakWs,
            TotalGcPause = GC.GetTotalPauseDuration() - gcStart, WallSeconds = Ns.ToSec(_wall.NowNs() - wall0),
            OrdersSubmitted = _ordersSubmitted, OrdersNotSentDisconnected = _notSent,
        };
    }

    private void RunCycle(long k, long tk, long startLag)
    {
        var d = _d;
        d.Reset();
        var t = CycleTiming.Create();
        long gc0 = GC.GetTotalPauseDuration().Ticks;
        long alloc0 = GC.GetAllocatedBytesForCurrentThread();
        _clock.Begin();

        // S1–S2: upstream data preparation
        _s1.ProcessUntil(tk);
        _clock.Lap(Stage.Ingest);
        _s2.Evaluate(_s1.Accepted, tk);
        _clock.Lap(Stage.Evaluate);

        // S3–S4: intermediate processing
        _s3.Update();
        _clock.Lap(Stage.Features);
        var phase = _calendar.Phase(tk);
        bool open = phase == MarketPhase.Open;
        if (_dump is not null)
        {
            _dump.Write(tk, open, _s2, _s3); // before S4: packing resets the one-cycle return reference
            _clock.Lap(Stage.Audit);
        }
        var batch = _s4.Pack(k, tk, open);
        _clock.Lap(Stage.Pack);
        _clock.Lap(Stage.HostToDevice); // CPU-only in M1: no transfer

        // S5: cross-stock inference
        bool infer = open && !_s2.FeedDown;
        if (infer) _model.Predict(batch, _s4.Stats, _preds);
        _clock.Lap(Stage.Inference);
        _clock.Lap(Stage.DeviceToHost);

        // Execution sync: venue activity and reports up to the cutoff
        _exec.AdvanceTo(tk);
        _reports.Clear();
        _exec.DrainReports(tk, _reports);
        foreach (var r in _reports)
        {
            _om.Apply(r);
            if (r.Kind == ReportKind.Fill) d.FillsApplied++;
        }
        d.ReportsApplied = _reports.Count;
        bool connected = _exec.IsConnected;
        if (connected && !_wasConnected)
        {
            _om.ReconcileOrders(_exec.Snapshot(tk));
            _verifyPositionsAtCycle = k + 1; // after reports held during the disconnect have been delivered
        }
        else if (connected && k == _verifyPositionsAtCycle) _om.VerifyPositions(_exec.Snapshot(tk));
        _wasConnected = connected;
        _clock.Lap(Stage.ExecSync);

        var equity = _om.Equity(Mark);
        _risk.UpdateLoss(equity);
        var status = !open ? CycleStatus.Closed
            : _s2.FeedDown ? CycleStatus.FeedDown
            : _risk.KillSwitch ? CycleStatus.KillSwitch
            : !connected ? CycleStatus.Disconnected
            : CycleStatus.Ok;

        // S6: decision + risk
        var approved = _intents;
        approved.Clear();
        if (status == CycleStatus.Ok)
        {
            var desired = _desired;
            _policy.Decide(batch, _preds, _s2, _om, desired);
            _clock.Lap(Stage.Decision);
            double gross = _om.GrossNotional(Mark);
            foreach (var it in desired)
            {
                var existing = _om.OpenFor(it.Instrument);
                // An unchanged working order is re-affirmed without a new risk check (it passed when sent).
                if (existing is not null && _policy.Keeps(existing, it))
                {
                    approved.Add(it);
                    continue;
                }
                if (existing is not null && existing.Req.Side == Side.Buy) gross -= existing.Leaves * existing.Req.Limit.ToDouble();
                var reason = _risk.Check(it, approved.Count, batch, _om, ref gross);
                if (reason == RiskReason.None) approved.Add(it);
                else d.RiskRejects.Add((it, reason));
            }
            _clock.Lap(Stage.Risk);
        }
        else
        {
            _clock.Lap(Stage.Decision);
            _clock.Lap(Stage.Risk);
        }

        // Deadline: the decision must reach the order interface within the budget.
        long decidedTs;
        if (Paced)
        {
            long nowW = _wall.NowNs();
            decidedTs = WallToSim(nowW);
            if (nowW > SimToWall(tk + _budget))
            {
                if (status == CycleStatus.Ok) status = CycleStatus.Missed;
                approved.Clear(); // never submit a late (stale) decision
            }
        }
        else decidedTs = tk + _virtualLatency;

        // Order interface
        if (connected) Housekeeping(decidedTs, status);
        if (status == CycleStatus.Ok) Submit(k, tk, decidedTs, approved);
        else if (status == CycleStatus.KillSwitch && connected && _flattenOnKill) Flatten(k, tk, decidedTs);
        _clock.Lap(Stage.OrderInterface);

        // Audit record
        d.CycleId = k; d.CutoffTs = tk; d.DecidedTs = decidedTs; d.Status = status; d.Phase = phase;
        d.InputHash = batch.InputHash; d.Valid = _s2.ValidCount; Array.Copy(_s2.MaskCounts, d.MaskCounts, 4);
        d.FeedDown = _s2.FeedDown; d.KillSwitch = _risk.KillSwitch; d.Equity = equity;
        if (infer) foreach (var p in _preds) if (p.Valid) d.PredictionsValid++;
        int openPos = 0;
        foreach (var q in _om.Position) if (q != 0) openPos++;
        d.OpenPositions = openPos;
        _audit.Write(d);
        _clock.Lap(Stage.Audit);

        // Telemetry
        _clock.CopyTo(t.StageNs);
        t.CycleId = k;
        t.CutoffTs = tk;
        t.Status = status;
        t.Warmup = k < _cfg.Run.WarmupCycles;
        t.TotalNs = _clock.ElapsedNs;
        t.ToOrderNs = t.TotalNs - t.StageNs[(int)Stage.Audit];
        t.StartLagNs = startLag;
        long budgetWall = Paced ? (long)(_budget / _cfg.Run.Speed) : _budget;
        t.OverBudget = t.ToOrderNs + startLag > budgetWall;
        t.GcPauseNs = (GC.GetTotalPauseDuration().Ticks - gc0) * 100;
        t.AllocBytes = GC.GetAllocatedBytesForCurrentThread() - alloc0;
        t.AcceptedEvents = _s1.Accepted.Count;
        t.DirtyInstruments = _s2.Dirty.Count;
        t.ValidInstruments = _s2.ValidCount;
        (t.MedianDataAgeMs, t.MaxDataAgeMs) = _s2.DataAge(tk, _ageScratch);
        t.MaxFeedLatencyMs = _s1.CycleMaxFeedLatencyMs;
        t.OrdersSubmitted = d.Submitted.Count;
        t.Cancels = d.Cancels.Count;
        _timings.Add(t);

        if (OnCycle is { } observer) observer(BuildView(t, d, phase, connected, infer));
    }

    private CycleView BuildView(in CycleTiming t, CycleDecision d, MarketPhase phase, bool connected, bool infer)
    {
        int n = _u.Count;
        var lastTrade = new double[n];
        var vol = new long[n];
        foreach (var tr in _s2.GoodTrades)
        {
            lastTrade[tr.Instrument] = tr.Px1 / (double)Price.Scale;
            vol[tr.Instrument] += tr.Sz1;
        }
        var rows = new InstrumentView[n];
        for (int i = 0; i < n; i++)
        {
            var ins = _u.Instruments[i];
            var p = _preds[i];
            bool has = infer && p.Valid;
            rows[i] = new InstrumentView(i, ins.Symbol, ins.Type.ToString(), ins.Sector,
                _s2.GoodBid[i] / (double)Price.Scale, _s2.GoodAsk[i] / (double)Price.Scale, _s2.GoodBidSize[i], _s2.GoodAskSize[i],
                lastTrade[i], vol[i],
                has, has ? p.Mu * 1e4 : 0, has ? p.Sigma * 1e4 : 0, has ? p.Mu / p.Sigma : 0, has ? p.Beta : _s4.Stats.Beta[i],
                _om.Position[i], _s2.Mask[i].ToString(),
                (double)_om.AvgCost[i],
                _om.Position[i] != 0 && Mark(i) > 0 ? _om.Position[i] * (Mark(i) - (double)_om.AvgCost[i]) : 0,
                (double)_om.Realized[i]);
        }
        string Sym(int i) => i >= 0 && i < n ? _u.Instruments[i].Symbol : "?";
        var fills = _reports.Where(r => r.Kind == ReportKind.Fill)
            .Select(r => new FillView(r.OrderId, Sym(r.Instrument), r.Side == Side.Buy ? "buy" : "sell", r.FillQty, r.FillPrice.ToDouble(), (double)r.Fee, r.Ts))
            .ToList();
        var stages = new Dictionary<string, double>();
        for (int s = 0; s < CycleTiming.StageCount; s++) stages[((Stage)s).ToString()] = t.StageNs[s] / 1e6;
        return new CycleView(
            t.CycleId, t.CutoffTs, d.DecidedTs, t.Status.ToString(), phase.ToString(), _manifest.Synthetic, _manifest.DataVersion,
            $"{_model.Name}@{_model.Version}", t.TotalNs / 1e6, t.ToOrderNs / 1e6, t.StartLagNs / 1e6, t.OverBudget, _cfg.Run.BudgetMs,
            stages,
            (double)d.Equity, (double)_om.StartingCash, (double)_om.Cash, (double)_om.Fees, _om.GrossNotional(Mark),
            _risk.KillSwitch, _risk.KillReason, _s2.FeedDown, connected,
            _s2.ValidCount, n, t.MedianDataAgeMs, t.MaxDataAgeMs,
            rows,
            d.Submitted.Select(s => new OrderView(s.OrderId, Sym(s.Intent.Instrument), s.Intent.Side == Side.Buy ? "buy" : "sell",
                s.Intent.Qty, s.Intent.Limit.ToDouble(), s.Intent.Mu * 1e4, s.Intent.Score, s.Intent.Rule)).ToList(),
            fills,
            d.Cancels.Select(c => new CancelView(c.OrderId, c.Why)).ToList(),
            d.RiskRejects.Select(r => new RiskRejectView(Sym(r.Intent.Instrument), r.Intent.Side == Side.Buy ? "buy" : "sell", r.Intent.Qty, r.Reason.ToString())).ToList(),
            _om.OpenCount, _om.Fills, _ordersSubmitted);
    }

    /// <summary>Cancels expired orders always; everything on kill switch, or on a miss when on_miss = cancel_all.</summary>
    private void Housekeeping(long ts, CycleStatus status)
    {
        bool cancelAll = status == CycleStatus.KillSwitch
                         || (status == CycleStatus.Missed && _cfg.Run.OnMiss == OnMissPolicy.CancelAll);
        foreach (var o in _om.Open.Where(o => o.State != WorkingState.PendingCancel).OrderBy(o => o.Req.OrderId).ToList())
        {
            bool expired = o.ValidUntil <= ts;
            if (!cancelAll && !expired) continue;
            if (!expired && _flattenIds.Contains(o.Req.OrderId)) continue; // kill-switch flatten orders work until they expire
            if (_exec.Cancel(o.Req.OrderId, ts) == SubmitResult.Sent)
            {
                _om.MarkPendingCancel(o.Req.OrderId);
                _d.Cancels.Add((o.Req.OrderId, cancelAll ? status.ToString() : "expired"));
            }
        }
    }

    /// <summary>
    /// Kill switch with risk.kill_switch_action = flatten: close every open position with a marketable limit at the
    /// opposite good quote. One working flatten order per instrument; it is re-sent at the new price once it expires,
    /// until flat. Never sent for masked (stale/halted) instruments or invalid prices.
    /// </summary>
    private void Flatten(long k, long tk, long ts)
    {
        long validUntil = tk + _cadence + _grace;
        for (int i = 0; i < _om.Position.Length; i++)
        {
            long pos = _om.Position[i];
            if (pos == 0 || _s2.Mask[i] != MaskReason.Valid) continue;
            if (_om.Open.Any(o => o.Req.Instrument == i)) continue; // let the working / cancelling order resolve first
            var side = pos > 0 ? Side.Sell : Side.Buy;
            var limit = new Price(side == Side.Sell ? _s2.GoodBid[i] : _s2.GoodAsk[i]);
            if (!_ticks[i].IsValid(limit)) continue;
            var it = new OrderIntent(i, side, Math.Abs(pos), limit, 0, 0, 0, 0, _s2.GoodBid[i], _s2.GoodAsk[i], "kill_flatten", limit);
            var req = new OrderRequest(_om.NextOrderId(), i, side, Math.Abs(pos), limit, k);
            if (_exec.Submit(req, ts) == SubmitResult.Sent)
            {
                _om.Track(req, ts, validUntil);
                _flattenIds.Add(req.OrderId);
                _d.Submitted.Add((req.OrderId, it));
                _ordersSubmitted++;
            }
            else _notSent++;
        }
    }

    private void Submit(long k, long tk, long ts, List<OrderIntent> approved)
    {
        long validUntil = tk + _cadence + _grace;
        var byInstrument = approved.ToDictionary(a => a.Instrument);

        // Working orders not re-affirmed by this cycle are cancelled.
        foreach (var o in _om.Open.Where(o => o.State != WorkingState.PendingCancel).OrderBy(o => o.Req.OrderId).ToList())
        {
            if (byInstrument.TryGetValue(o.Req.Instrument, out var it)
                && _policy.Keeps(o, it))
            {
                o.ValidUntil = validUntil;
                byInstrument.Remove(o.Req.Instrument);
                _d.Kept++;
                continue;
            }
            if (_exec.Cancel(o.Req.OrderId, ts) == SubmitResult.Sent)
            {
                _om.MarkPendingCancel(o.Req.OrderId);
                _d.Cancels.Add((o.Req.OrderId, "replaced"));
            }
        }

        foreach (var it in approved)
        {
            if (!byInstrument.ContainsKey(it.Instrument)) continue;
            var req = new OrderRequest(_om.NextOrderId(), it.Instrument, it.Side, it.Qty, it.Limit, k);
            if (_exec.Submit(req, ts) == SubmitResult.Sent)
            {
                _om.Track(req, ts, validUntil);
                _d.Submitted.Add((req.OrderId, it));
                _ordersSubmitted++;
            }
            else
            {
                _d.NotSent.Add(it);
                _notSent++;
            }
        }
    }

    private void RecordSkipped(long k)
    {
        var d = _d;
        d.Reset();
        long tk = _t0 + k * _cadence;
        d.CycleId = k; d.CutoffTs = tk; d.DecidedTs = 0; d.Status = CycleStatus.SkippedOverrun; d.Phase = _calendar.Phase(tk);
        d.InputHash = 0; d.Valid = 0; Array.Clear(d.MaskCounts); d.FeedDown = _s2.FeedDown; d.KillSwitch = _risk.KillSwitch;
        d.Equity = _om.Equity(Mark); d.OpenPositions = 0;
        _audit.Write(d);
        var t = CycleTiming.Create();
        t.CycleId = k; t.CutoffTs = tk; t.Status = CycleStatus.SkippedOverrun; t.Warmup = k < _cfg.Run.WarmupCycles; t.OverBudget = true;
        _timings.Add(t);
    }

    private void FinalSync(long ts)
    {
        _exec.AdvanceTo(ts);
        _reports.Clear();
        _exec.DrainReports(ts, _reports);
        foreach (var r in _reports) _om.Apply(r);
    }

    private void Progress(long k)
    {
        var t = _timings[^1];
        _log.WriteLine(
            $"[{k,6}] {Ns.Iso(t.CutoffTs)} {t.Status,-14} valid={t.ValidInstruments,5} events={t.AcceptedEvents,6} " +
            $"orders={t.OrdersSubmitted,3} open={_om.OpenCount,3} fills={_om.Fills,6} equity={_om.Equity(Mark),14:N2} " +
            $"cycle={Ns.ToMs(t.TotalNs),7:0.000} ms");
    }

    public void Dispose()
    {
        _feed.Dispose();
        _audit.Dispose();
        _dump?.Dispose();
    }
}
