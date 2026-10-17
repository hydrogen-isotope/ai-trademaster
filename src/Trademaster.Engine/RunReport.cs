using System.Globalization;
using System.Text;
using Trademaster.Core;
using Trademaster.Telemetry;

namespace Trademaster.Engine;

/// <summary>Writes summary.md and cycles.csv. Only measured values are printed; anything not measured says so.</summary>
public static class RunReport
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    public static string Write(RunResult r, SystemInfo sys, string outDir)
    {
        WriteCsv(r, Path.Combine(outDir, "cycles.csv"));
        string decisionsSha = ReplayFile.Sha256(r.DecisionsPath);
        var md = BuildSummary(r, sys, decisionsSha);
        var path = Path.Combine(outDir, "summary.md");
        File.WriteAllText(path, md);
        return path;
    }

    private static string Ms(double ns) => (ns / 1e6).ToString("0.000", Inv);

    public static string BuildSummary(RunResult r, SystemInfo sys, string decisionsSha)
    {
        var cfg = r.Config;
        var all = r.Cycles;
        var measured = all.Where(c => !c.Warmup && c.Status != CycleStatus.SkippedOverrun).ToList();
        var active = measured.Where(c => c.Status is CycleStatus.Ok or CycleStatus.Missed).ToList();
        var warm = all.Where(c => c.Warmup && c.Status != CycleStatus.SkippedOverrun).ToList();
        bool paced = cfg.Run.Mode == RunMode.Paced;
        var sb = new StringBuilder();

        sb.AppendLine($"# Trademaster run report{(r.Manifest.Synthetic ? " — SYNTHETIC DATA" : "")}");
        sb.AppendLine();
        if (r.Manifest.Synthetic)
            sb.AppendLine("> All market data in this run is synthetic. The generator embeds known signals, so any P&L below is a property of the generator, **not evidence of real-market performance**.");
        sb.AppendLine();
        sb.AppendLine($"- Generated: {DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss", Inv)} UTC");
        sb.AppendLine($"- Mode: **{cfg.Run.Mode.ToString().ToLowerInvariant()}**" + (paced
            ? $" (replay released at wall-clock speed ×{cfg.Run.Speed.ToString(Inv)}; deadlines are real)"
            : " (simulated time; deterministic decisions; latency is compute-only, deadlines are evaluated as would-miss)"));
        sb.AppendLine($"- Cadence {cfg.Run.CadenceMs} ms, budget {cfg.Run.BudgetMs} ms" + (paced && cfg.Run.Speed != 1 ? $" (wall-clock budget {(cfg.Run.BudgetMs / cfg.Run.Speed).ToString("0.#", Inv)} ms)" : ""));
        sb.AppendLine($"- Model: `{r.Model.Name}@{r.Model.Version}` (hash `{r.Model.ArtifactHash}`), backend **CPU, managed C#** — no neural network and no GPU used in this run");
        sb.AppendLine($"- Execution adapter: `{r.AdapterName}` (simulated)");
        sb.AppendLine($"- Data: `{r.Manifest.DataVersion}`, {r.Universe.Count} instruments, {r.Manifest.FeedEvents:N0} feed events, {Ns.Iso(r.Manifest.StartTs)} → {Ns.Iso(r.Manifest.EndTs)}");
        sb.AppendLine($"- Wall time: {r.WallSeconds.ToString("0.0", Inv)} s");
        sb.AppendLine();
        sb.AppendLine("## Hardware");
        sb.AppendLine("```");
        foreach (var l in sys.Lines()) sb.AppendLine(l);
        sb.AppendLine("```");
        sb.AppendLine();

        sb.AppendLine("## Deadlines");
        sb.AppendLine();
        sb.AppendLine("| Status | Cycles |");
        sb.AppendLine("|---|---:|");
        foreach (var g in all.GroupBy(c => c.Status).OrderBy(g => g.Key))
            sb.AppendLine($"| {g.Key} | {g.Count():N0} |");
        sb.AppendLine($"| **Total** | {all.Count:N0} |");
        sb.AppendLine();
        int over = active.Count(c => c.OverBudget);
        sb.AppendLine(paced
            ? $"Missed deadlines: **{all.Count(c => c.Status == CycleStatus.Missed)}**, skipped (overrun): **{all.Count(c => c.Status == CycleStatus.SkippedOverrun)}**, of {all.Count(c => c.Status is CycleStatus.Ok or CycleStatus.Missed or CycleStatus.SkippedOverrun)} trading cycles."
            : $"Would-miss (compute + start lag > budget) among trading cycles after warm-up: **{over} / {active.Count}**. Virtual mode never drops decisions; run `--mode paced` for real deadline behaviour.");
        sb.AppendLine();

        sb.AppendLine($"## Latency per stage (ms) — trading cycles after {cfg.Run.WarmupCycles} warm-up cycles, n = {active.Count:N0}");
        sb.AppendLine();
        sb.AppendLine("| Stage | p50 | p95 | p99 | max | mean |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|");
        for (int s = 0; s < CycleTiming.StageCount; s++)
        {
            var st = (Stage)s;
            if (st is Stage.HostToDevice or Stage.DeviceToHost) { sb.AppendLine($"| {st} | n/a (CPU-only) | | | | |"); continue; }
            var p = Percentiles.Of(active.Select(c => (double)c.StageNs[s]));
            sb.AppendLine($"| {st} | {Ms(p.P50)} | {Ms(p.P95)} | {Ms(p.P99)} | {Ms(p.Max)} | {Ms(p.Mean)} |");
        }
        var tot = Percentiles.Of(active.Select(c => (double)c.ToOrderNs));
        sb.AppendLine($"| **Cutoff → order interface** | **{Ms(tot.P50)}** | **{Ms(tot.P95)}** | **{Ms(tot.P99)}** | **{Ms(tot.Max)}** | {Ms(tot.Mean)} |");
        var full = Percentiles.Of(active.Select(c => (double)c.TotalNs));
        sb.AppendLine($"| Full cycle incl. audit | {Ms(full.P50)} | {Ms(full.P95)} | {Ms(full.P99)} | {Ms(full.Max)} | {Ms(full.Mean)} |");
        if (paced)
        {
            var lag = Percentiles.Of(active.Select(c => (double)c.StartLagNs));
            sb.AppendLine($"| Start lag (scheduler) | {Ms(lag.P50)} | {Ms(lag.P95)} | {Ms(lag.P99)} | {Ms(lag.Max)} | {Ms(lag.Mean)} |");
        }
        var nonTrading = measured.Where(c => c.Status is not (CycleStatus.Ok or CycleStatus.Missed)).ToList();
        if (nonTrading.Count > 0)
        {
            var nt = Percentiles.Of(nonTrading.Select(c => (double)c.TotalNs));
            sb.AppendLine($"| Non-trading cycles (closed / feed down / disconnected / kill), n={nt.Count} | {Ms(nt.P50)} | {Ms(nt.P95)} | {Ms(nt.P99)} | {Ms(nt.Max)} | {Ms(nt.Mean)} |");
        }
        if (warm.Count > 0)
        {
            var w = Percentiles.Of(warm.Select(c => (double)c.TotalNs));
            sb.AppendLine($"| Warm-up cycles (JIT), n={w.Count} | {Ms(w.P50)} | {Ms(w.P95)} | {Ms(w.P99)} | {Ms(w.Max)} | {Ms(w.Mean)} |");
        }
        sb.AppendLine();
        sb.AppendLine("Ingest time includes reading the replay file from disk. In M1 ingestion is batched at the start of each cycle (a live feed would be ingested continuously).");
        sb.AppendLine();

        sb.AppendLine("## Data quality");
        sb.AppendLine();
        var ig = r.Ingest; var q = r.Quality;
        sb.AppendLine($"- Feed: received {ig.Received:N0}, applied {ig.Applied:N0}, duplicates dropped {ig.Duplicates:N0}, reordered {ig.Reordered:N0}, lost to gaps {ig.GapLost:N0}, late after gap {ig.LateDropped:N0}, invalid {ig.Invalid:N0}");
        sb.AppendLine($"- Evaluation: crossed quotes {q.Crossed:N0}, locked {q.Locked:N0}, jumps flagged {q.Jumps:N0}, outlier trades {q.OutlierTrades:N0}, status changes {q.StatusChanges:N0}");
        var age = Percentiles.Of(active.Select(c => c.MaxDataAgeMs));
        var med = Percentiles.Of(active.Select(c => c.MedianDataAgeMs));
        var lat = Percentiles.Of(active.Select(c => c.MaxFeedLatencyMs));
        sb.AppendLine($"- Data age at cutoff (valid instruments): median-per-cycle p50 {med.P50.ToString("0.0", Inv)} ms; max-per-cycle p50 {age.P50.ToString("0", Inv)} ms, p99 {age.P99.ToString("0", Inv)} ms");
        sb.AppendLine($"- Feed latency (RecvTs − ExchangeTs) worst per cycle: p50 {lat.P50.ToString("0.0", Inv)} ms, max {lat.Max.ToString("0.0", Inv)} ms");
        var valid = Percentiles.Of(active.Select(c => (double)c.ValidInstruments));
        sb.AppendLine($"- Valid instruments per trading cycle: p50 {valid.P50:0}, min {active.Select(c => c.ValidInstruments).DefaultIfEmpty().Min()} of {r.Universe.Count}");
        sb.AppendLine();

        sb.AppendLine("## Execution (simulated)");
        sb.AppendLine();
        var om = r.Orders;
        sb.AppendLine($"- Orders submitted {r.OrdersSubmitted:N0}, not sent (disconnected) {r.OrdersNotSentDisconnected:N0}, acks {om.Acks:N0}, fill reports {om.Fills:N0} ({om.FilledQty:N0} shares)");
        sb.AppendLine($"- Venue rejects {om.Rejects:N0}" + (om.RejectReasons.Count > 0 ? " — " + string.Join(", ", om.RejectReasons.OrderBy(k => k.Key).Select(k => $"{k.Key}: {k.Value}")) : ""));
        sb.AppendLine($"- Cancels confirmed {om.Cancelled:N0}, cancel rejects {om.CancelRejects:N0}, reconciliations {om.Reconciliations}, reconcile differences {om.ReconcileDiffs}");
        sb.AppendLine($"- Risk rejections: " + (r.Risk.Rejections.Count == 0 ? "none" : string.Join(", ", r.Risk.Rejections.OrderBy(k => k.Key).Select(k => $"{k.Key}: {k.Value}"))));
        sb.AppendLine($"- Kill switch: {(r.Risk.KillSwitch ? "TRIGGERED — " + r.Risk.KillReason : "not triggered")}");
        sb.AppendLine();
        sb.AppendLine($"## P&L {(r.Manifest.Synthetic ? "(SYNTHETIC — not evidence of anything)" : "")}");
        sb.AppendLine();
        sb.AppendLine($"- Start equity {om.StartingCash.ToString("N2", Inv)}, end equity {r.FinalEquity.ToString("N2", Inv)} (marked at engine's last good mid), change {(r.FinalEquity - om.StartingCash).ToString("N2", Inv)}, fees {om.Fees.ToString("N2", Inv)}");
        sb.AppendLine($"- Open positions at end: {om.Position.Count(p => p != 0)}");
        sb.AppendLine();

        sb.AppendLine("## Process");
        sb.AppendLine();
        var alloc = Percentiles.Of(active.Select(c => (double)c.AllocBytes));
        sb.AppendLine($"- Peak working set {(r.PeakWorkingSet / 1048576.0).ToString("0", Inv)} MB, total GC pause {r.TotalGcPause.TotalMilliseconds.ToString("0.0", Inv)} ms, cycles with any GC pause {active.Count(c => c.GcPauseNs > 0)}");
        sb.AppendLine($"- Managed allocation per trading cycle: p50 {(alloc.P50 / 1024).ToString("0.0", Inv)} KB, max {(alloc.Max / 1024).ToString("0.0", Inv)} KB");
        sb.AppendLine("- **Not measured in this run:** GPU utilisation, VRAM, power draw (CPU-only run; GPU counters arrive with the ONNX path in M2, power needs AMD ADLX).");
        sb.AppendLine();
        sb.AppendLine("## Reproducibility");
        sb.AppendLine();
        sb.AppendLine($"- decisions.jsonl sha256: `{decisionsSha}`" + (paced ? " (paced runs depend on timing and are not expected to be reproducible)" : " (virtual runs with identical data, config and model must reproduce this hash)"));
        sb.AppendLine($"- Feed sha256: `{r.Manifest.FeedSha256}`");
        return sb.ToString();
    }

    private static void WriteCsv(RunResult r, string path)
    {
        using var w = new StreamWriter(path);
        w.Write("cycle,cutoff_utc,status,warmup,over_budget,total_ms,to_order_ms,start_lag_ms");
        for (int s = 0; s < CycleTiming.StageCount; s++) w.Write($",{((Stage)s).ToString().ToLowerInvariant()}_ms");
        w.WriteLine(",gc_pause_ms,alloc_kb,accepted_events,dirty,valid,median_age_ms,max_age_ms,max_feed_latency_ms,orders,cancels");
        foreach (var c in r.Cycles)
        {
            w.Write(string.Create(Inv, $"{c.CycleId},{Ns.Iso(c.CutoffTs)},{c.Status},{(c.Warmup ? 1 : 0)},{(c.OverBudget ? 1 : 0)},{c.TotalNs / 1e6:0.0000},{c.ToOrderNs / 1e6:0.0000},{c.StartLagNs / 1e6:0.0000}"));
            for (int s = 0; s < CycleTiming.StageCount; s++) w.Write(string.Create(Inv, $",{c.StageNs[s] / 1e6:0.0000}"));
            w.WriteLine(string.Create(Inv, $",{c.GcPauseNs / 1e6:0.000},{c.AllocBytes / 1024.0:0.0},{c.AcceptedEvents},{c.DirtyInstruments},{c.ValidInstruments},{c.MedianDataAgeMs:0.0},{c.MaxDataAgeMs:0.0},{c.MaxFeedLatencyMs:0.00},{c.OrdersSubmitted},{c.Cancels}"));
        }
    }
}
