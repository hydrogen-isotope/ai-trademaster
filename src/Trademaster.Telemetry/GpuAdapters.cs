using System.Runtime.InteropServices;

namespace Trademaster.Telemetry;

/// <summary>A GPU as DirectX (DXGI) enumerates it. <see cref="Index"/> is the DirectML device id.</summary>
public sealed record GpuAdapter(int Index, string Name, uint VendorId, double DedicatedVramGb, bool Software)
{
    public string Vendor => VendorId switch { 0x1002 => "AMD", 0x10DE => "NVIDIA", 0x8086 => "Intel", 0x1414 => "Microsoft", _ => $"0x{VendorId:X4}" };
}

/// <summary>
/// Read-only DXGI adapter enumeration (no GPU work is submitted). DirectML device ids follow this order, so the
/// discrete card can be chosen explicitly instead of the Ryzen iGPU.
/// </summary>
public static unsafe class GpuAdapters
{
    private static readonly Guid IidFactory1 = new("770aae78-f26f-4dba-a829-253c83d1b387");
    private const int DxgiErrorNotFound = unchecked((int)0x887A0002);
    private const uint SoftwareFlag = 2;

    [StructLayout(LayoutKind.Sequential)]
    private struct AdapterDesc1
    {
        public fixed char Description[128];
        public uint VendorId, DeviceId, SubSysId, Revision;
        public nuint DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public uint LuidLow; public int LuidHigh;
        public uint Flags;
    }

    [DllImport("dxgi.dll")]
    private static extern int CreateDXGIFactory1(in Guid riid, out nint factory);

    public static IReadOnlyList<GpuAdapter> Enumerate()
    {
        var list = new List<GpuAdapter>();
        if (!OperatingSystem.IsWindows()) return list;
        try
        {
            if (CreateDXGIFactory1(IidFactory1, out var factory) < 0 || factory == 0) return list;
            var fv = *(void***)factory;
            var enumAdapters1 = (delegate* unmanaged<nint, uint, nint*, int>)fv[12]; // IDXGIFactory1::EnumAdapters1
            var releaseFactory = (delegate* unmanaged<nint, uint>)fv[2];
            for (uint i = 0; ; i++)
            {
                nint adapter;
                int hr = enumAdapters1(factory, i, &adapter);
                if (hr == DxgiErrorNotFound || hr < 0) break;
                var av = *(void***)adapter;
                var getDesc1 = (delegate* unmanaged<nint, AdapterDesc1*, int>)av[10]; // IDXGIAdapter1::GetDesc1
                AdapterDesc1 d;
                if (getDesc1(adapter, &d) >= 0)
                    list.Add(new GpuAdapter((int)i, new string(d.Description).TrimEnd('\0'), d.VendorId,
                        Math.Round(d.DedicatedVideoMemory / 1073741824.0, 1), (d.Flags & SoftwareFlag) != 0));
                ((delegate* unmanaged<nint, uint>)av[2])(adapter);
            }
            releaseFactory(factory);
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException) { }
        return list;
    }

    /// <summary>The hardware adapter with the most dedicated VRAM (the discrete GPU), or -1 if none.</summary>
    public static int PickDiscrete(IReadOnlyList<GpuAdapter> adapters) =>
        adapters.Where(a => !a.Software).OrderByDescending(a => a.DedicatedVramGb).Select(a => a.Index).DefaultIfEmpty(-1).First();
}
