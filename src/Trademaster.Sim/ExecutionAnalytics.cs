using Trademaster.Core;

namespace Trademaster.Sim;

/// <summary>Execution-quality summary for one group of orders. All bps are relative to the decision mid unless noted.</summary>
public sealed record ExecStats(
    string Group, int Orders, long OrderedQty, double FillRate, double FullyFilledShare, double MedianSecondsToFill,
    int Fills, double PassiveFillShare,
    double SpreadCaptureBps,       // + = filled better than the decision mid (passive earns, aggressive pays)
    double[] MarkoutBps,           // + = good: side x (mid at fill+Δ − fill price), per filled share
    double[] AdverseSelectionBps,  // − = bad: side x (mid at fill+Δ − mid at fill), per filled share
    double FeeBps,
    double ShortfallBps,           // cost per ordered share incl. crossing the spread for any unfilled remainder at expiry
    double CleanupShare);          // share of ordered quantity that had to be completed aggressively

/// <summary>
/// Measures fills against the true market path kept by <see cref="SimExchange"/>: fill probability, time to fill,
/// spread capture, markouts and adverse selection at several horizons, and implementation shortfall with clean-up.
/// </summary>
public static class ExecutionAnalytics
{
    public static readonly double[] HorizonsSec = [1, 5, 30, 60];

    public static List<ExecStats> Compute(SimExchange ex, IReadOnlyDictionary<long, string> groups, double feeBps)
    {
        var fillsByOrder = ex.FillLog.GroupBy(f => f.OrderId).ToDictionary(g => g.Key, g => g.ToList());
        var result = new List<ExecStats>();
        foreach (var grp in groups.GroupBy(kv => kv.Value).OrderBy(g => g.Key))
        {
            var recs = grp.Select(kv => ex.Orders.GetValueOrDefault(kv.Key)).Where(r => r is not null && double.IsFinite(r.DecisionMid)
                && !r.End.StartsWith("rejected")).Cast<OrderRecord>().ToList();
            if (recs.Count == 0) continue;
            long ordered = 0, filled = 0, cleanupQty = 0;
            double capture = 0, fee = 0, shortfall = 0, passiveQty = 0;
            var mk = new double[HorizonsSec.Length];
            var adv = new double[HorizonsSec.Length];
            var mkW = new double[HorizonsSec.Length];
            var ttf = new List<double>();
            int fills = 0, full = 0;
            foreach (var r in recs)
            {
                int side = r.Side == Side.Buy ? 1 : -1;
                double dm = r.DecisionMid;
                ordered += r.Qty;
                if (r.FilledQty >= r.Qty) full++;
                if (r.FilledQty > 0) ttf.Add(Ns.ToSec(r.FirstFillTs - r.SubmitTs));
                double cost = 0;
                foreach (var f in fillsByOrder.GetValueOrDefault(r.OrderId) ?? [])
                {
                    fills++;
                    double px = f.Price / (double)Price.Scale;
                    filled += f.Qty;
                    if (f.Passive) passiveQty += f.Qty;
                    capture += f.Qty * side * (dm - px) / dm * 1e4;
                    double fbps = (double)f.Fee / (f.Qty * px) * 1e4;
                    fee += f.Qty * fbps;
                    cost += f.Qty * (side * (px - dm) / dm * 1e4 + fbps);
                    double m0 = ex.MidAt(f.Instrument, f.Ts);
                    for (int h = 0; h < HorizonsSec.Length; h++)
                    {
                        double mh = ex.MidAt(f.Instrument, f.Ts + (long)(HorizonsSec[h] * Ns.PerSec));
                        if (!double.IsFinite(mh) || !double.IsFinite(m0)) continue;
                        mk[h] += f.Qty * side * (mh - px) / px * 1e4;
                        adv[h] += f.Qty * side * (mh - m0) / m0 * 1e4;
                        mkW[h] += f.Qty;
                    }
                }
                long rest = r.Qty - r.FilledQty;
                if (rest > 0)
                {
                    // Unfilled at expiry: complete aggressively at the opposite quote (plus taker fee).
                    var (b, a) = ex.QuoteAt(r.Instrument, r.EndTs > 0 ? r.EndTs : r.SubmitTs);
                    double cross = side > 0 ? a : b;
                    if (double.IsFinite(cross)) cost += rest * (side * (cross - dm) / dm * 1e4 + feeBps);
                    cleanupQty += rest;
                }
                shortfall += cost;
            }
            ttf.Sort();
            result.Add(new ExecStats(
                grp.Key, recs.Count, ordered, (double)filled / ordered, (double)full / recs.Count,
                ttf.Count > 0 ? ttf[ttf.Count / 2] : double.NaN, fills, filled > 0 ? passiveQty / filled : 0,
                filled > 0 ? capture / filled : double.NaN,
                mk.Select((v, h) => mkW[h] > 0 ? v / mkW[h] : double.NaN).ToArray(),
                adv.Select((v, h) => mkW[h] > 0 ? v / mkW[h] : double.NaN).ToArray(),
                filled > 0 ? fee / filled : double.NaN,
                shortfall / ordered, (double)cleanupQty / ordered));
        }
        return result;
    }
}

/// <summary>
/// Signal-free execution probe: places small orders at random times on random stocks with random sides, in three
/// placements, so execution cost can be measured separately from any alpha.
///   aggressive      — limit at the opposite quote (takes liquidity)
///   passive_join    — limit at our side's best price (joins the queue)
///   passive_improve — one tick better than our side's best (only when the spread is at least 2 ticks)
/// Passive orders are cancelled after <see cref="ProbeOptions.ExpirySec"/>; any remainder is charged as crossed.
/// </summary>
public sealed record ProbeOptions(double ProbesPerSecond = 2, double ExpirySec = 30, int Lots = 10, int Seed = 11, double WarmupSec = 60);

public static class ExecutionProbe
{
    public static (SimExchange Venue, Dictionary<long, string> Groups) Run(string dataDir, string exchangesDir, ExecutionSection cfg, ProbeOptions o)
    {
        var manifest = DataManifest.Load(dataDir);
        var universe = Universe.Load(Path.Combine(dataDir, manifest.UniverseFile));
        var ex = ExchangeConfig.Load(Path.Combine(exchangesDir, manifest.Exchange + ".json"));
        var ticks = ex.BuildTickTable();
        var probeCfg = new ExecutionSection
        {
            OrderLatencyMs = cfg.OrderLatencyMs, Participation = cfg.Participation, TouchFillProb = cfg.TouchFillProb,
            FillModel = cfg.FillModel, Seed = cfg.Seed, CancelOnDisconnect = cfg.CancelOnDisconnect, Disconnects = [],
        };
        var venue = new SimExchange(Path.Combine(dataDir, manifest.ExchangeFile), universe, ex, probeCfg, manifest.StartTs);
        var stocks = universe.Instruments.Where(i => i.Type == InstrumentType.Stock).Select(i => i.Id).ToArray();
        var rng = new Random(o.Seed);
        var groups = new Dictionary<long, string>();
        var cancels = new PriorityQueue<long, long>();
        long id = 1, step = Ns.PerSec;
        long start = manifest.StartTs + (long)(o.WarmupSec * Ns.PerSec);
        long end = manifest.EndTs - (long)((o.ExpirySec + 65) * Ns.PerSec);
        string[] kinds = ["aggressive", "passive_join", "passive_improve"];
        double carry = 0;
        for (long t = start; t < end; t += step)
        {
            venue.AdvanceTo(t);
            while (cancels.TryPeek(out var oid, out var due) && due <= t) { cancels.Dequeue(); venue.Cancel(oid, t); }
            carry += o.ProbesPerSecond;
            while (carry >= 1)
            {
                carry -= 1;
                int i = stocks[rng.Next(stocks.Length)];
                var (bid, ask, _, _) = venue.Quote(i);
                if (bid <= 0 || ask <= bid) continue;
                var side = rng.NextDouble() < 0.5 ? Side.Buy : Side.Sell;
                string kind = kinds[rng.Next(kinds.Length)];
                var b = new Price(bid); var a = new Price(ask);
                Price limit;
                if (kind == "aggressive") limit = side == Side.Buy ? a : b;
                else if (kind == "passive_join") limit = side == Side.Buy ? b : a;
                else
                {
                    var better = side == Side.Buy ? ticks.Step(b, 1) : ticks.Step(a, -1);
                    if (side == Side.Buy ? better >= a : better <= b) continue; // 1-tick spread: cannot improve
                    limit = better;
                }
                var req = new OrderRequest(id++, i, side, o.Lots * Math.Max(1, ex.LotSize), limit, 0);
                venue.Submit(req, t);
                groups[req.OrderId] = kind;
                // Aggressive: whatever the displayed opposite size didn't cover is cancelled next second (counted as clean-up).
                cancels.Enqueue(req.OrderId, t + (kind == "aggressive" ? step : (long)(o.ExpirySec * Ns.PerSec)));
            }
        }
        venue.AdvanceTo(manifest.EndTs + 5 * Ns.PerSec);
        return (venue, groups);
    }
}
