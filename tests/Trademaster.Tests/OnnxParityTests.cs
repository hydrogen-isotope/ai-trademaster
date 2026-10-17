using Trademaster.Engine.Inference;

namespace Trademaster.Tests;

/// <summary>C# ONNX Runtime (CPU) must reproduce the PyTorch outputs saved at export time.</summary>
public class OnnxParityTests
{
    private static string? Repo()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Trademaster.slnx"))) return d.FullName;
        return null;
    }

    public static IEnumerable<object[]> ModelDirs()
    {
        var root = Path.Combine(Repo()!, "private", "models", "b1");
        var dirs = Directory.Exists(root)
            ? Directory.GetDirectories(root).Where(d => File.Exists(Path.Combine(d, "parity_out.bin"))).ToList()
            : [];
        // Nothing trained in this checkout yet: run python -m trademaster_ml.train / train_wf first.
        return dirs.Count == 0 ? [[""]] : dirs.Select(d => new object[] { Path.GetFileName(d) });
    }

    [Theory]
    [MemberData(nameof(ModelDirs))]
    public void Trained_b1_matches_pytorch_reference(string version)
    {
        if (version == "") return;
        var dir = Path.Combine(Repo()!, "private", "models", "b1", version);

        var (xs, x) = F32Tensor.Read(Path.Combine(dir, "parity_x.bin"));
        var (_, m) = F32Tensor.Read(Path.Combine(dir, "parity_mask.bin"));
        var (_, reference) = F32Tensor.Read(Path.Combine(dir, "parity_out.bin"));
        int bars = xs[0], n = xs[1], f = xs[2];

        using var r = new OnnxRunner(Path.Combine(dir, "model.onnx"), n, f, 1, 2);
        double maxDiff = 0;
        for (int b = 0; b < bars; b++)
        {
            Array.Copy(x, b * n * f, r.X, 0, n * f);
            Array.Copy(m, b * n, r.Mask, 0, n);
            r.Run();
            for (int i = 0; i < n * 2; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(r.Out[i] - reference[b * n * 2 + i]));
        }
        Assert.True(maxDiff < 1e-3, $"max |C# ORT - PyTorch| = {maxDiff}");
    }
}
