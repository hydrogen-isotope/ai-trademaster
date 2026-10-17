# Development machine — inspected 2026-10-16

Collected from Windows (CIM/WMI, registry, CLI tools). These are inventory facts, not performance measurements.

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (build 26200) |
| CPU | AMD Ryzen 7 9700X — 8 cores / 16 threads |
| RAM | 64 GB (2 × 32 GB DDR5, 5600 MT/s, Crucial CT32G56C46U5) — 61.6 GB visible |
| Discrete GPU | AMD Radeon RX 7800 XT — 16 GB VRAM — driver 32.0.31041.1004 (2026-08-17) |
| Integrated GPU | AMD Radeon Graphics (Ryzen iGPU) — 2 GB shared — driver 32.0.21045.5002 |
| Storage | Crucial P5 Plus 2 TB NVMe (CT2000P5PSSD8) — 383 GB free on C: |
| .NET | SDK 10.0.402; runtimes 8.0.32, 10.0.13 |
| Python | 3.14.4 only (system), `numpy` 2.5.2; no PyTorch / ONNX Runtime installed |
| WSL | Not installed |
| ROCm / HIP SDK | Not installed |
| Package manager | winget available |

## Implications

- The RX 7800 XT is RDNA3 (gfx1101). It has FP16/BF16 matrix instructions but **no FP8 hardware**, so FP8 can only be emulated here.
- The engine must pick the discrete GPU explicitly (adapter index/LUID), because the iGPU is also a DirectX adapter.
- Python 3.12 will be needed for the training environment in M2, since GPU-enabled PyTorch wheels usually lag the newest Python. To be confirmed at install time.
- Power draw isn't exposed by the standard Windows counters; it needs AMD ADLX (later milestone).

## Target hardware (not purchased — do not assume)

Dual NVIDIA RTX Pro 6000 (Blackwell, 96 GB each) workstation, or a DGX Spark cluster. All design choices must keep working on CPU and the RX 7800 XT.
