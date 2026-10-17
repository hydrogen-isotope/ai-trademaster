# GPU backends for the AI (ONNX Runtime)

The AI models run through **ONNX Runtime**. There's no hand-written GPU code: the same `model.onnx` file runs on
the CPU, on AMD GPUs and, later, on NVIDIA GPUs. Moving between them is a **build flavour + one config value**.

| Backend (`compute.backend`) | Hardware | Build flavour | Status |
|---|---|---|---|
| `cpu` | any | any | default; used for all results so far |
| `dml` | **AMD RX 7800 XT now** (any DirectX 12 GPU) | `DirectML` (default on Windows; ORT 1.24.4, DirectML's last release) | built and detected; **not yet run on the GPU** (safety gate) |
| `cuda` | NVIDIA (RTX Pro 6000 later) | `Cuda` (ORT 1.31 GPU) | builds; providers present; needs NVIDIA hardware |
| `tensorrt` | NVIDIA, fastest; FP16/FP8 on Blackwell | `Cuda` | builds; needs NVIDIA hardware |
| `migraphx` | AMD via ROCm on **Linux** | an ORT build with MIGraphX | API wired; not packaged for Windows |

```bash
dotnet build -c Release
```

```bash
dotnet build -c Release -p:OrtFlavor=Cuda
```

(The first is the Windows default, CPU + DirectML. The second is for an NVIDIA machine: CPU + CUDA + TensorRT.)

## Choosing the GPU
DirectML device ids follow DirectX adapter order. `trademaster doctor` lists them and marks the one picked automatically
(the hardware adapter with the most VRAM). On this PC:

```
DirectML device 0: AMD Radeon RX 7800 XT (AMD, 15.8 GB dedicated)  <- auto-selected for dml
DirectML device 1: AMD Radeon(TM) Graphics (AMD, 2 GB dedicated)
DirectML device 2: Microsoft Basic Render Driver [software]
```

Set `compute.device_id` to override. `-1` means automatic.

## Safety gate (always on)
`OnnxRunner` checks `GpuGate` **before** creating any GPU session. A GPU backend needs both:
1. `compute.allow_gpu = true` (or `--allow-gpu` on `bench-onnx`), **and**
2. **no display-driver resets in the last 7 days** (`docs/GPU-STABILITY.md`), unless `override_gpu_stability_check` is set
   for a supervised, on-site run.

Tests verify that a GPU backend is refused before any model file is opened.

## First GPU run (after Thu 22 Oct ~21:35 SGT, with you at the PC)
1. `trademaster doctor` shows `GPU health: no display-driver resets...` and the RX 7800 XT auto-selected.
2. Small, short benchmark first:
   ```bash
   dotnet run -c Release --project src/Trademaster.Cli -- bench-onnx --backend cpu,dml --allow-gpu --n 30,300 --k 1,10 --calls 50
   ```
3. If stable, the full comparison:
   ```bash
   dotnet run -c Release --project src/Trademaster.Cli -- bench-onnx --backend cpu,dml --allow-gpu --n 30,300,1000,3000 --k 1,10,100,1000
   ```
4. Watch the screen. If it freezes, recovery should now take seconds (the TDR timeout was restored to 2 s).
   `doctor` will show any new reset, and the gate then blocks GPU use for another 7 days.

What to expect: for our small model at N ≈ 30–300, the GPU may be *slower* than the CPU. Each run copies inputs and
outputs across PCIe, and the CPU already takes about 0.1–0.4 ms. The GPU should win at N ≥ 1,000 and for larger models.
The benchmark will show where the crossover is; there are no GPU numbers yet.

## Porting to NVIDIA (RTX Pro 6000)
1. Install the NVIDIA driver, CUDA and cuDNN versions matching ONNX Runtime 1.31 (TensorRT optional).
2. `dotnet build -c Release -p:OrtFlavor=Cuda`.
3. Set `compute.backend` to `cuda` (or `tensorrt`).
4. Re-run the benchmark and the parity tests. The model files and C# code don't change.

Precision: FP16 needs an FP16 copy of the model (convert at export). FP8 is available through TensorRT on
Blackwell GPUs. The RX 7800 XT has no FP8 hardware (see the precision study plan in the spec).
