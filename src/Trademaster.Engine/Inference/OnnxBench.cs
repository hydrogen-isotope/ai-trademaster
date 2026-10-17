using System.Diagnostics;
using System.Globalization;
using System.Text;
using Trademaster.Telemetry;

namespace Trademaster.Engine.Inference;

/// <summary>
/// CPU latency benchmark for B1-shaped ONNX models, covering the three "inference step" modes (ARCHITECTURE §8):
///   A  one batched pass over N instruments;
///   B  K independent samples in one call (batch = K);
///   C  K sequential, dependent passes (each step feeds its output into the next step's input).
/// Inputs are random; this measures speed only.
/// </summary>
public static class OnnxBench
{
    private const int F = 8;

    public sealed record Row(string Mode, int N, int K, int Threads, Percentiles PerCallMs, double PerStepMs, double StepsPerSecond);

    public static List<Row> Run(string modelPath, int[] ns, int[] ks, int threads, int calls, TextWriter log)
    {
        var rows = new List<Row>();
        var rng = new Random(1);
        foreach (int n in ns)
        {
            using (var a = new OnnxRunner(modelPath, n, F, 1, threads))
            {
                Fill(a, rng);
                for (int i = 0; i < 20; i++) a.Run();
                var t = Time(a.Run, calls);
                rows.Add(new Row("A batched pass", n, 1, threads, t, t.P50, 1000 / t.P50));
                log.WriteLine($"A  N={n,5}            p50 {t.P50,8:0.000} ms  p99 {t.P99,8:0.000} ms");
            }
            // Attention scores are K x heads x N x N floats; cap at ~0.5 GB so the benchmark can't exhaust RAM.
            foreach (int k in ks.Where(k => k > 1 && (long)k * n * n <= 30_000_000))
            {
                using var b = new OnnxRunner(modelPath, n, F, k, threads);
                Fill(b, rng);
                for (int i = 0; i < 5; i++) b.Run();
                var t = Time(b.Run, Math.Max(5, calls / k));
                rows.Add(new Row("B parallel samples", n, k, threads, t, t.P50 / k, 1000 * k / t.P50));
                log.WriteLine($"B  N={n,5} K={k,5}    p50 {t.P50,8:0.000} ms per call ({t.P50 / k:0.0000} ms/sample)");
            }
            using (var c = new OnnxRunner(modelPath, n, F, 1, threads))
            {
                Fill(c, rng);
                for (int i = 0; i < 20; i++) c.Run();
                foreach (int k in ks)
                {
                    int reps = Math.Clamp(2000 / k, 3, 50);
                    var t = Time(() =>
                    {
                        for (int s = 0; s < k; s++)
                        {
                            c.Run();
                            for (int i = 0; i < n; i++) c.X[i * F] = c.Out[i * 2] * 1e-3f; // dependency on previous step
                        }
                    }, reps);
                    rows.Add(new Row("C sequential steps", n, k, threads, t, t.P50 / k, 1000 * k / t.P50));
                    log.WriteLine($"C  N={n,5} K={k,5}    p50 {t.P50,8:0.000} ms total ({t.P50 / k:0.0000} ms/step)  p99 {t.P99:0.000} ms");
                }
            }
        }
        return rows;
    }

    private static void Fill(OnnxRunner r, Random rng)
    {
        for (int i = 0; i < r.X.Length; i++) r.X[i] = (float)(rng.NextDouble() * 2 - 1);
        Array.Fill(r.Mask, 1f);
    }

    private static Percentiles Time(Action a, int reps)
    {
        var ms = new double[reps];
        for (int i = 0; i < reps; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            a();
            ms[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        return Percentiles.Of(ms);
    }

    public static string Markdown(List<Row> rows, SystemInfo sys, string modelPath)
    {
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.AppendLine("# ONNX inference benchmark — CPU");
        sb.AppendLine();
        sb.AppendLine($"Model `{modelPath}` (B1 architecture, random weights, 8 features, d=32, 2 attention layers). ONNX Runtime CPU execution provider. **No GPU used.**");
        sb.AppendLine();
        sb.AppendLine("```"); foreach (var l in sys.Lines()) sb.AppendLine(l); sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("| Mode | N instruments | K | Threads | p50 per call (ms) | p99 per call (ms) | Per step/sample (ms) | Steps/s |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|");
        foreach (var r in rows)
            sb.AppendLine(string.Create(inv, $"| {r.Mode} | {r.N:N0} | {r.K:N0} | {r.Threads} | {r.PerCallMs.P50:0.000} | {r.PerCallMs.P99:0.000} | {r.PerStepMs:0.0000} | {r.StepsPerSecond:N0} |"));
        sb.AppendLine();
        sb.AppendLine("Mode C (sequential, dependent steps) is the binding case for a \"1,000+ steps per 1-second cycle\" design: it fits only if K x per-step time stays under the ~890 ms left after the rest of the pipeline.");
        return sb.ToString();
    }
}
