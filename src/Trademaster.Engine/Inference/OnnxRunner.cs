using Microsoft.ML.OnnxRuntime;

namespace Trademaster.Engine.Inference;

/// <summary>
/// Runs a B1-shaped ONNX model (inputs x[B,N,F], mask[B,N]; output out[B,N,2]) on the CPU execution provider.
/// Input and output buffers are allocated once and bound as OrtValues, so steady-state runs allocate nothing.
/// GPU execution providers are deliberately not wired in (compute.allow_gpu = false; docs/GPU-STABILITY.md).
/// </summary>
public sealed class OnnxRunner : IDisposable
{
    private readonly InferenceSession _session;
    private readonly RunOptions _run = new();
    private readonly OrtValue _x, _mask, _out;
    private readonly string[] _inNames = ["x", "mask"], _outNames = ["out"];
    private readonly OrtValue[] _inputs, _outputs;

    public int Batch { get; }
    public int N { get; }
    public int F { get; }
    public float[] X { get; }
    public float[] Mask { get; }
    public float[] Out { get; }
    public string ModelPath { get; }

    public OnnxRunner(string modelPath, int n, int f, int batch = 1, int intraOpThreads = 4)
    {
        ModelPath = modelPath;
        N = n; F = f; Batch = batch;
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = intraOpThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        _session = new InferenceSession(modelPath, so); // CPU execution provider only
        X = new float[batch * n * f];
        Mask = new float[batch * n];
        Out = new float[batch * n * 2];
        _x = OrtValue.CreateTensorValueFromMemory(X, [batch, n, f]);
        _mask = OrtValue.CreateTensorValueFromMemory(Mask, [batch, n]);
        _out = OrtValue.CreateTensorValueFromMemory(Out, [batch, n, 2]);
        _inputs = [_x, _mask];
        _outputs = [_out];
    }

    public IReadOnlyList<string> Providers => ["CPUExecutionProvider"];

    /// <summary>Runs one forward pass over the current contents of <see cref="X"/>/<see cref="Mask"/> into <see cref="Out"/>.</summary>
    public void Run() => _session.Run(_run, _inNames, _inputs, _outNames, _outputs);

    public void Dispose()
    {
        _x.Dispose(); _mask.Dispose(); _out.Dispose();
        _run.Dispose();
        _session.Dispose();
    }
}

/// <summary>Reads the little-endian float32 tensors written by python/trademaster_ml/train.py (write_f32).</summary>
public static class F32Tensor
{
    public static (int[] Shape, float[] Data) Read(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        int nd = r.ReadInt32();
        var shape = new int[nd];
        for (int i = 0; i < nd; i++) shape[i] = r.ReadInt32();
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var data = new float[count];
        for (long i = 0; i < count; i++) data[i] = r.ReadSingle();
        return (shape, data);
    }
}
