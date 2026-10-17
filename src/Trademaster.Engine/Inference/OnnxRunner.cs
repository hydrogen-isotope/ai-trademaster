using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Trademaster.Core;
using Trademaster.Telemetry;

namespace Trademaster.Engine.Inference;

/// <summary>
/// Where an <see cref="OnnxRunner"/> executes. GPU backends always pass <see cref="GpuGate"/> first.
/// cpu · dml (DirectX 12 / DirectML: AMD RX 7800 XT now) · cuda / tensorrt (NVIDIA, build with -p:OrtFlavor=Cuda) ·
/// migraphx (AMD ROCm on Linux, needs an ORT build with MIGraphX).
/// </summary>
public sealed record OnnxBackend(string Name = "cpu", int DeviceId = -1, bool AllowGpu = false, bool OverrideGpuStabilityCheck = false)
{
    public static readonly string[] Known = ["cpu", "dml", "cuda", "tensorrt", "migraphx"];
    public bool IsGpu => Name != "cpu";
    public static OnnxBackend From(ComputeSection c) => new(c.Backend.ToLowerInvariant(), c.DeviceId, c.AllowGpu, c.OverrideGpuStabilityCheck);

    /// <summary>ONNX Runtime build flavour compiled into this binary (DirectML, Cpu or Cuda).</summary>
    public static string Flavor =>
        typeof(OnnxBackend).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().FirstOrDefault(a => a.Key == "OrtFlavor")?.Value ?? "?";

    public static IReadOnlyList<string> AvailableProviders()
    {
        try { return OrtEnv.Instance().GetAvailableProviders(); }
        catch (Exception) { return []; }
    }
}

/// <summary>
/// Runs a B1-shaped ONNX model (inputs x[B,N,F], mask[B,N]; output out[B,N,2]) on the chosen backend.
/// Input and output buffers are allocated once and bound as OrtValues, so steady-state runs allocate nothing;
/// on a GPU backend the runtime copies them to and from device memory on each run.
/// The same model file runs on every backend: porting from AMD to NVIDIA is a build flavour + config change.
/// </summary>
public sealed class OnnxRunner : IDisposable
{
    private readonly InferenceSession _session;
    private readonly RunOptions _run = new();
    private readonly OrtValue _x, _mask, _out;
    private readonly OrtValue? _attn;
    private readonly string[] _inNames = ["x", "mask"], _outNames;
    private readonly OrtValue[] _inputs, _outputs;

    public int Batch { get; }
    public int N { get; }
    public int F { get; }
    public float[] X { get; }
    public float[] Mask { get; }
    public float[] Out { get; }
    /// <summary>Optional attention map out "attn"[B,N,N] (row i = how much instrument i attends to each j), if the model exports it.</summary>
    public float[]? Attn { get; }
    public string ModelPath { get; }

    public OnnxBackend Backend { get; }
    /// <summary>Device actually used (−1 on CPU).</summary>
    public int DeviceId { get; }

    public OnnxRunner(string modelPath, int n, int f, int batch = 1, int intraOpThreads = 4, OnnxBackend? backend = null)
    {
        Backend = backend ?? new OnnxBackend();
        if (!OnnxBackend.Known.Contains(Backend.Name))
            throw new ArgumentException($"Unknown backend '{Backend.Name}'. Use one of: {string.Join(", ", OnnxBackend.Known)}.");
        // Safety first: no GPU session is created unless GPU use is enabled and the driver has been stable.
        if (Backend.IsGpu) GpuGate.Require(Backend.AllowGpu, Backend.OverrideGpuStabilityCheck, GpuHealth.Check());

        ModelPath = modelPath;
        N = n; F = f; Batch = batch;
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = intraOpThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        DeviceId = -1;
        if (Backend.IsGpu)
        {
            var available = OnnxBackend.AvailableProviders();
            string provider = Backend.Name switch
            {
                "dml" => "DmlExecutionProvider", "cuda" => "CUDAExecutionProvider",
                "tensorrt" => "TensorrtExecutionProvider", _ => "MIGraphXExecutionProvider",
            };
            if (!available.Contains(provider))
                throw new InvalidOperationException(
                    $"Backend '{Backend.Name}' needs {provider}, which this build doesn't include (ONNX Runtime flavour '{OnnxBackend.Flavor}', " +
                    $"available: {string.Join(", ", available)}). DirectML: default Windows build. NVIDIA: build with -p:OrtFlavor=Cuda.");
            DeviceId = Backend.DeviceId >= 0 ? Backend.DeviceId
                : Backend.Name == "dml" ? Math.Max(0, GpuAdapters.PickDiscrete(GpuAdapters.Enumerate())) : 0;
            switch (Backend.Name)
            {
                case "dml":
                    so.EnableMemoryPattern = false;               // required by DirectML
                    so.AppendExecutionProvider_DML(DeviceId);
                    break;
                case "cuda": so.AppendExecutionProvider_CUDA(DeviceId); break;
                case "tensorrt": so.AppendExecutionProvider_Tensorrt(DeviceId); break;
                case "migraphx": so.AppendExecutionProvider_MIGraphX(DeviceId); break;
            }
        }
        _session = new InferenceSession(modelPath, so); // CPU EP is always present as fallback for unsupported ops
        X = new float[batch * n * f];
        Mask = new float[batch * n];
        Out = new float[batch * n * 2];
        _x = OrtValue.CreateTensorValueFromMemory(X, [batch, n, f]);
        _mask = OrtValue.CreateTensorValueFromMemory(Mask, [batch, n]);
        _out = OrtValue.CreateTensorValueFromMemory(Out, [batch, n, 2]);
        _inputs = [_x, _mask];
        if (_session.OutputMetadata.ContainsKey("attn"))
        {
            Attn = new float[batch * n * n];
            _attn = OrtValue.CreateTensorValueFromMemory(Attn, [batch, n, n]);
            _outNames = ["out", "attn"];
            _outputs = [_out, _attn];
        }
        else
        {
            _outNames = ["out"];
            _outputs = [_out];
        }
    }

    public string Description => Backend.IsGpu ? $"{Backend.Name} (device {DeviceId}, ORT flavour {OnnxBackend.Flavor})" : "cpu";

    /// <summary>Runs one forward pass over the current contents of <see cref="X"/>/<see cref="Mask"/> into <see cref="Out"/>.</summary>
    public void Run() => _session.Run(_run, _inNames, _inputs, _outNames, _outputs);

    public void Dispose()
    {
        _x.Dispose(); _mask.Dispose(); _out.Dispose(); _attn?.Dispose();
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
