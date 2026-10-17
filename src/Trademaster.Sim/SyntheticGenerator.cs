using System.Globalization;
using System.Text.Json;
using Trademaster.Core;

namespace Trademaster.Sim;

public sealed class GeneratorOptions
{
    public int Stocks { get; set; } = 200;
    public int Sectors { get; set; } = 8;
    public double Minutes { get; set; } = 30;
    public int Seed { get; set; } = 42;
    public int StepMs { get; set; } = 100;
    public string Date { get; set; } = "2026-01-05";

    // Feed defects (per event probabilities)
    public double DupProb { get; set; } = 0.002;
    public double ReorderProb { get; set; } = 0.01;
    public double LossProb { get; set; } = 0.0005;
    public double CrossedProb { get; set; } = 0.0005;

    // Scenario injections
    public bool InjectHalt { get; set; } = true;
    public bool InjectShock { get; set; } = true;
    public bool InjectOutage { get; set; } = true;
    public bool InjectBurst { get; set; } = true;

    // Embedded (known) cross-stock structure
    public double LeadLagSeconds { get; set; } = 3.0;
    public double LeadStrength { get; set; } = 0.6;

    /// <summary>
    /// Price impact per average-sized trade, in bps, applied to the stock's latent price in the trade's direction
    /// (decays with the idiosyncratic mean reversion). Creates adverse selection for passive orders. 0 = none.
    /// </summary>
    public double ImpactBps { get; set; } = 0.0;
}

/// <summary>
/// Seeded synthetic market: factor model (market + sector + mean-reverting idiosyncratic), ETFs as
/// hidden baskets, lead-lag propagation, and injected feed defects. Produces the true exchange stream
/// (for the simulated exchange) and the defective feed as received (for the engine).
/// Every output is labelled SYNTHETIC; embedded signals exist by construction and prove nothing about real markets.
/// </summary>
public sealed class SyntheticGenerator
{
    private readonly GeneratorOptions _o;
    private readonly ExchangeConfig _ex;
    private readonly TickTable _ticks;
    private readonly Random _rng;

    public SyntheticGenerator(GeneratorOptions options, ExchangeConfig exchange)
    {
        _o = options;
        _ex = exchange;
        _ticks = exchange.BuildTickTable();
        _rng = new Random(options.Seed);
    }

    private double Gauss()
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private double Uniform(double a, double b) => a + (b - a) * _rng.NextDouble();

    public sealed record Result(string Dir, DataManifest Manifest, long ExchangeEvents, long FeedEvents);

    public Result Generate(string outDir)
    {
        Directory.CreateDirectory(outDir);
        var calendar = new MarketCalendar(_ex);
        var date = DateOnly.ParseExact(_o.Date, "yyyy-MM-dd", CultureInfo.InvariantCulture);
        long start = calendar.SessionOpen(date);
        long stepNs = _o.StepMs * Ns.PerMs;
        double dt = _o.StepMs / 1000.0, sqrtDt = Math.Sqrt(dt);
        long steps = (long)(_o.Minutes * 60_000 / _o.StepMs);
        long end = start + steps * stepNs;
        double durS = steps * dt;
        int lot = Math.Max(1, _ex.LotSize);

        // ---------- universe ----------
        int nS = _o.Stocks, nSec = _o.Sectors;
        var instruments = new List<Instrument>();
        var sectorOf = new int[nS];
        for (int i = 0; i < nS; i++)
        {
            sectorOf[i] = i % nSec;
            double u = _rng.NextDouble();
            var liq = u < 0.3 ? LiquidityClass.High : u < 0.8 ? LiquidityClass.Medium : LiquidityClass.Low;
            instruments.Add(new Instrument(i, $"S{i:D4}", InstrumentType.Stock, $"SEC{sectorOf[i]:D2}", liq));
        }
        int mktEtf = instruments.Count;
        instruments.Add(new Instrument(mktEtf, "ETF_MKT", InstrumentType.Etf, "MARKET", LiquidityClass.High));
        var secEtf = new int[nSec];
        for (int s = 0; s < nSec; s++)
        {
            secEtf[s] = instruments.Count;
            instruments.Add(new Instrument(secEtf[s], $"ETF_SEC{s:D2}", InstrumentType.Etf, $"SEC{s:D2}", LiquidityClass.High));
        }
        int n = instruments.Count;
        var universe = new Universe { Exchange = _ex.Id, Instruments = instruments };

        // ---------- latent parameters (ground truth, hidden from the engine) ----------
        var p0 = new double[n];
        var betaM = new double[nS];
        var betaS = new double[nS];
        var sigI = new double[nS];
        var theta = new double[nS];
        for (int i = 0; i < nS; i++)
        {
            p0[i] = Math.Exp(Uniform(Math.Log(2), Math.Log(150)));
            betaM[i] = Uniform(0.6, 1.4);
            betaS[i] = Uniform(0.5, 1.5);
            sigI[i] = Uniform(1e-4, 3e-4);
            theta[i] = 1.0 / Uniform(30, 120);
        }
        for (int e = nS; e < n; e++) p0[e] = Math.Exp(Uniform(Math.Log(20), Math.Log(400)));
        var leader = new int[nSec];
        for (int s = 0; s < nSec; s++) leader[s] = s; // first stock of each sector (i % nSec == s)

        // ETF baskets: random positive weights over constituents (hidden).
        var etfWeights = new Dictionary<int, (int[] Members, double[] W)>();
        {
            var all = Enumerable.Range(0, nS).ToArray();
            etfWeights[mktEtf] = (all, Normalize(all.Select(_ => Uniform(0.2, 1.0)).ToArray()));
            for (int s = 0; s < nSec; s++)
            {
                var mem = all.Where(i => sectorOf[i] == s).ToArray();
                etfWeights[secEtf[s]] = (mem, Normalize(mem.Select(_ => Uniform(0.2, 1.0)).ToArray()));
            }
        }

        // ---------- scenario windows ----------
        int haltStock = Enumerable.Range(0, nS).FirstOrDefault(i => instruments[i].Liquidity == LiquidityClass.Medium && i >= nSec);
        long haltStart = start + (long)(durS * 0.30 * Ns.PerSec), haltEnd = haltStart + (long)(Math.Min(120, durS / 10) * Ns.PerSec);
        long shockStart = start + (long)(durS * 0.65 * Ns.PerSec), shockEnd = shockStart + 5 * Ns.PerSec;
        long outStart = start + (long)(durS * 0.50 * Ns.PerSec), outEnd = outStart + 8 * Ns.PerSec;
        long burstStart = start + (long)(durS * 0.40 * Ns.PerSec), burstEnd = burstStart + 3 * Ns.PerSec;

        // ---------- simulate ----------
        double mCum = 0, logV = 0;
        var sCum = new double[nSec];
        var x = new double[nS];
        var lp = new double[n];
        for (int i = 0; i < n; i++) lp[i] = Math.Log(p0[i]);
        var lp0 = (double[])lp.Clone();
        int lagSteps = Math.Max(1, (int)Math.Round(_o.LeadLagSeconds / dt));
        var leadQueue = new Queue<double>[nSec];
        for (int s = 0; s < nSec; s++) leadQueue[s] = new Queue<double>(Enumerable.Repeat(0.0, lagSteps));

        var bid = new long[n];
        var ask = new long[n];
        var halted = new bool[n];
        var truth = new List<MarketEvent>(capacity: 1 << 20);
        var stepEvents = new List<MarketEvent>(n * 3);
        double[] quoteP = [0.30, 0.12, 0.04], tradeP = [0.10, 0.04, 0.01];
        int[] minSpread = [1, 2, 3], maxSpread = [2, 3, 6];
        long seq = 0;
        bool haltOn = false;

        for (long k = 0; k < steps; k++)
        {
            long t0 = start + k * stepNs;
            bool open = calendar.Phase(t0) == MarketPhase.Open;

            // latent dynamics
            logV += -logV / 300.0 * dt + 0.05 * sqrtDt * Gauss();
            if (_o.InjectShock && t0 >= shockStart && t0 < shockEnd)
            {
                mCum += -0.03 * stepNs / (double)(shockEnd - shockStart);
                logV = Math.Max(logV, 1.0);
            }
            double v = Math.Exp(logV);
            mCum += 1e-4 * v * sqrtDt * Gauss();
            for (int s = 0; s < nSec; s++)
                sCum[s] += 8e-5 * v * sqrtDt * Gauss() + _o.LeadStrength * leadQueue[s].Dequeue();
            for (int i = 0; i < nS; i++)
            {
                double dx = -theta[i] * x[i] * dt + sigI[i] * v * sqrtDt * Gauss();
                x[i] += dx;
                if (i < nSec && leader[sectorOf[i]] == i) leadQueue[sectorOf[i]].Enqueue(dx);
                lp[i] = lp0[i] + betaM[i] * mCum + betaS[i] * sCum[sectorOf[i]] + x[i];
            }
            foreach (var (etf, (mem, w)) in etfWeights)
            {
                double acc = 0;
                for (int j = 0; j < mem.Length; j++) acc += w[j] * (lp[mem[j]] - lp0[mem[j]]);
                lp[etf] = lp0[etf] + acc;
            }
            if (!open) continue;

            stepEvents.Clear();
            long Ts() => t0 + (long)(_rng.NextDouble() * stepNs);

            if (_o.InjectHalt && !haltOn && t0 >= haltStart && t0 < haltEnd)
            {
                haltOn = true; halted[haltStock] = true;
                stepEvents.Add(MarketEvent.StatusChange(0, t0, 0, haltStock, TradingStatus.Halted));
            }
            else if (haltOn && t0 >= haltEnd)
            {
                haltOn = false; halted[haltStock] = false;
                stepEvents.Add(MarketEvent.StatusChange(0, t0, 0, haltStock, TradingStatus.Trading));
                bid[haltStock] = 0; // force a fresh quote below
            }

            for (int i = 0; i < n; i++)
            {
                if (halted[i]) continue;
                int lc = (int)instruments[i].Liquidity;
                bool forceQuote = bid[i] == 0;
                if (forceQuote || _rng.NextDouble() < quoteP[lc])
                {
                    double mid = Math.Exp(lp[i]);
                    var tick = _ticks.TickAt(Price.FromDoubleFloor(mid));
                    int spreadTicks = _rng.Next(minSpread[lc], maxSpread[lc] + 1);
                    var b = _ticks.RoundDown(Price.FromDoubleFloor(mid - spreadTicks * tick.ToDouble() / 2));
                    if (!b.IsPositive) b = new Price(tick.Raw);
                    var a = _ticks.Step(b, spreadTicks);
                    bid[i] = b.Raw; ask[i] = a.Raw;
                    stepEvents.Add(MarketEvent.Quote(0, Ts(), 0, i, b, a, lot * _rng.Next(1, 21), lot * _rng.Next(1, 21)));
                }
                if (bid[i] > 0 && _rng.NextDouble() < tradeP[lc])
                {
                    bool buy = _rng.NextDouble() < 0.5;
                    int lots = _rng.Next(1, 11);
                    stepEvents.Add(MarketEvent.Trade(0, Ts(), 0, i, new Price(buy ? ask[i] : bid[i]), lot * lots));
                    if (_o.ImpactBps > 0 && i < nS)
                        x[i] += (buy ? 1 : -1) * _o.ImpactBps * 1e-4 * lots / 5.5; // 5.5 = mean lots per trade
                }
            }
            stepEvents.Sort((p, q) => p.ExchangeTs.CompareTo(q.ExchangeTs));
            foreach (var e in stepEvents)
                truth.Add(e with { Seq = ++seq, RecvTs = e.ExchangeTs });
        }

        // ---------- feed transform (what the engine receives) ----------
        var feed = new List<MarketEvent>(truth.Count + truth.Count / 100);
        long lost = 0, dups = 0, reordered = 0, crossed = 0, outageLost = 0, burstDelayed = 0;
        foreach (var e in truth)
        {
            if (_o.InjectOutage && e.ExchangeTs >= outStart && e.ExchangeTs < outEnd) { outageLost++; continue; }
            if (_rng.NextDouble() < _o.LossProb) { lost++; continue; }
            double delayMs = 2.0 - Math.Log(1.0 - _rng.NextDouble()) * 1.0;
            if (_rng.NextDouble() < _o.ReorderProb) { delayMs += Uniform(5, 40); reordered++; }
            if (_o.InjectBurst && e.ExchangeTs >= burstStart && e.ExchangeTs < burstEnd) { delayMs += 300; burstDelayed++; }
            var f = e with { RecvTs = e.ExchangeTs + (long)(delayMs * Ns.PerMs) };
            if (f.Kind == EventKind.Quote && _rng.NextDouble() < _o.CrossedProb)
            {
                f = f with { Px1 = f.Px2, Px2 = f.Px1 }; // corrupted in transit: bid/ask swapped
                crossed++;
            }
            feed.Add(f);
            if (_rng.NextDouble() < _o.DupProb)
            {
                feed.Add(f with { RecvTs = f.RecvTs + (long)(Uniform(1, 20) * Ns.PerMs) });
                dups++;
            }
        }
        feed.Sort((p, q) => p.RecvTs != q.RecvTs ? p.RecvTs.CompareTo(q.RecvTs) : p.Seq.CompareTo(q.Seq));

        // ---------- write ----------
        const string feedFile = "feed.tmrp", exFile = "exchange.tmrp", uniFile = "universe.json", gtFile = "ground_truth.json";
        ReplayFile.Write(Path.Combine(outDir, feedFile), feed);
        ReplayFile.Write(Path.Combine(outDir, exFile), truth);
        universe.Save(Path.Combine(outDir, uniFile));

        var gt = new
        {
            synthetic = true,
            note = "Hidden generator parameters. The engine must never read this file; it is for evaluation only.",
            stocks = Enumerable.Range(0, nS).Select(i => new
            {
                symbol = instruments[i].Symbol, sector = sectorOf[i], beta_market = betaM[i], beta_sector = betaS[i],
                idio_sigma_per_sqrt_s = sigI[i], idio_mean_reversion_per_s = theta[i], is_leader = leader[sectorOf[i]] == i,
            }),
            etf_weights = etfWeights.ToDictionary(kv => instruments[kv.Key].Symbol,
                kv => kv.Value.Members.Select((m, j) => (m, j)).ToDictionary(t => instruments[t.m].Symbol, t => kv.Value.W[t.j])),
            lead_lag = new { seconds = _o.LeadLagSeconds, strength = _o.LeadStrength, leaders = leader.Select(i => instruments[i].Symbol) },
            scenarios = new
            {
                halt = _o.InjectHalt ? new { symbol = instruments[haltStock].Symbol, start = Ns.Iso(haltStart), end = Ns.Iso(haltEnd) } : null,
                shock = _o.InjectShock ? new { market_move = -0.03, start = Ns.Iso(shockStart), end = Ns.Iso(shockEnd) } : null,
                feed_outage = _o.InjectOutage ? new { start = Ns.Iso(outStart), end = Ns.Iso(outEnd), events_lost = outageLost } : null,
                delayed_burst = _o.InjectBurst ? new { start = Ns.Iso(burstStart), end = Ns.Iso(burstEnd), extra_delay_ms = 300, events = burstDelayed } : null,
            },
            feed_defects = new { random_lost = lost, duplicates = dups, reordered, crossed_quotes = crossed },
        };
        File.WriteAllText(Path.Combine(outDir, gtFile), JsonSerializer.Serialize(gt, Json.Options));

        var manifest = new DataManifest
        {
            Kind = "synthetic",
            Synthetic = true,
            Exchange = _ex.Id,
            StartTs = start,
            EndTs = end,
            FeedFile = feedFile,
            FeedSha256 = ReplayFile.Sha256(Path.Combine(outDir, feedFile)),
            FeedEvents = feed.Count,
            ExchangeFile = exFile,
            ExchangeSha256 = ReplayFile.Sha256(Path.Combine(outDir, exFile)),
            ExchangeEvents = truth.Count,
            UniverseFile = uniFile,
            GroundTruthFile = gtFile,
            Parameters = typeof(GeneratorOptions).GetProperties()
                .ToDictionary(p => p.Name, p => Convert.ToString(p.GetValue(_o), CultureInfo.InvariantCulture) ?? ""),
        };
        File.WriteAllText(Path.Combine(outDir, "manifest.json"), JsonSerializer.Serialize(manifest, Json.Options));
        return new Result(outDir, manifest, truth.Count, feed.Count);
    }

    private static double[] Normalize(double[] w)
    {
        double s = w.Sum();
        for (int i = 0; i < w.Length; i++) w[i] /= s;
        return w;
    }
}
