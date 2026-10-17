# GPU stability on the development machine (investigated 2026-10-17)

## What happened
- Windows recorded **28 AMD display-driver resets (TDR, LiveKernelEvent 0x141/0x117 in `amdkmdag.sys`) in 7 days**: 10/11 (about 15 within 6 hours, plus `dwm.exe` crashing inside `amdxx64.dll`), 10/12, 10/15 and 10/16, including one **during boot**.
- Several resets escalated into **full hangs with no display and no remote access** that needed a forced restart: 10/11 10:45, 10/11 11:07, 10/17 ~00:46 (shortly after a session unlock/display wake) and 10/9 (during sleep).
- They coincide with GPU-heavy work elsewhere on the machine (Aver Engine rendering/shader tests, Blender) and with desktop display transitions. **Trademaster has never used the GPU**: M1 is CPU-only C#, and PyTorch/ONNX Runtime are not installed.

## Project safeguards
- `compute.allow_gpu = false` by default. Every future GPU path (ONNX DirectML/ROCm, training) must call `GpuGate.Require`, which also refuses if any driver reset was recorded in the last 7 days.
- `trademaster doctor` prints the reset count.
- M2 training will run on the **CPU**. GPU runs only after the machine has been stable for a week, and never unattended or remotely.

## Settings found on the machine (2026-10-17, read-only inspection)
| Setting | Value | Default | Why it matters |
|---|---|---|---|
| `GraphicsDrivers\TdrDelay` | **120 s** | 2 s | After a GPU hang Windows waits 2 minutes before resetting the driver: black screen, frozen desktop, remote control drops. The reset often fails after that long, so you end up with a forced restart. |
| `GraphicsDrivers\TdrDdiDelay` | **120 s** | 5 s | Same, for driver calls. |
| RX 7800 XT `PP_SclkDeepSleepDisable` | **1** | not set | Non-default power-management tweak. |
| Driver versions | dGPU 32.0.31041.1004, iGPU 32.0.21045.5002 | one package | Two different AMD driver packages on one system. |
| Display | Sony TV, HDMI, 4K60, **on the RX 7800 XT** | — | Any dGPU hang takes the only screen with it. |

`trademaster doctor` lists these and shows "no known risky settings found" once they are fixed.

## Fix order
1. **Restore hang-recovery defaults** (remote OK, about 2 minutes, then reboot). From an *elevated* PowerShell in the repo:
   `powershell -NoProfile -ExecutionPolicy Bypass -File tools\gpu-restore-defaults.ps1`
   It backs up both registry keys first. Expect a short flicker and recovery instead of a freeze if the GPU hangs again.
2. **At the PC:** run the AMD Cleanup Utility, then install **one** Adrenalin package for both GPUs: 26.9.2 (latest, includes an RX 7000 driver-timeout fix) or 26.8.1 (AMD "Recommended"). Pick the custom/clean option.
3. **In Adrenalin → Performance → Tuning:** reset to factory defaults (no undervolt, overclock or custom fan/power curves).
4. **At the PC (optional, most robust for remote use):** move the TV's HDMI cable to the motherboard HDMI port. The desktop then survives a dGPU hang. The trade-off is that games and 3D apps render on the dGPU and are copied to the iGPU output, which costs a little performance.
5. **Remote-restart safety net:** BIOS "Restore on AC power loss = Power On" plus a smart plug.
6. **Verify:** `trademaster doctor` should show no config issues. GPU work unblocks automatically after 7 days with no driver resets.

## Recommended fixes (owner actions; these change hardware or system settings)
1. **Plug the monitor into the motherboard video output** (Ryzen 9700X iGPU; enable it in BIOS if needed). The desktop and remote control then no longer depend on the RX 7800 XT driver.
2. **Reset any Radeon Adrenalin tuning** (undervolt, overclock, power limit) to defaults. Unstable tuning is a common cause of TDRs under load.
3. **Clean-reinstall the AMD driver** (AMD Cleanup Utility), or try a different WHQL release than 32.0.31041.1004.
4. Check GPU temperatures and PCIe power cables and seating. Consider the PSU if resets continue at stock settings.
5. For remote use: set the BIOS to *power on after AC loss* and use a smart plug, so a hung PC can be power-cycled remotely.
