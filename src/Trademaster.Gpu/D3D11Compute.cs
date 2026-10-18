using System.Runtime.InteropServices;
using System.Text;

namespace Trademaster.Gpu;

/// <summary>
/// Minimal Direct3D 11 compute host: device on a chosen adapter, HLSL compilation, structured float/uint buffers,
/// constant buffers, dispatch and read-back. COM is called through vtable function pointers; the slot numbers are the
/// documented d3d11.h / dxgi.h / d3dcompiler.h interface orders.
///
/// Safety: Windows resets the display driver (TDR) if one GPU command runs longer than ~2 s. Callers must split work
/// into short dispatches; <see cref="Dispatch"/> refuses grids larger than <see cref="MaxGroupsPerDispatch"/>.
/// </summary>
public sealed unsafe class D3D11Compute : IDisposable
{
    public const int MaxGroupsPerDispatch = 65535;

    // ---- vtable slots
    private const int IUnknownRelease = 2;
    private const int DevCreateBuffer = 3, DevCreateUav = 8, DevCreateComputeShader = 18;
    private const int CtxMap = 14, CtxUnmap = 15, CtxDispatch = 41, CtxCopyResource = 47, CtxUpdateSubresource = 48;
    private const int CtxCSSetUavs = 68, CtxCSSetShader = 69, CtxCSSetConstantBuffers = 71, CtxFlush = 111;
    private const int BlobGetPointer = 3, BlobGetSize = 4;
    private const int FactoryEnumAdapters1 = 12, AdapterGetDesc1 = 10;

    // ---- enums
    private const uint UsageDefault = 0, UsageStaging = 3;
    private const uint BindConstant = 0x4, BindShaderResource = 0x8, BindUav = 0x80;
    private const uint MiscStructured = 0x40, CpuRead = 0x20000, MapRead = 1;
    private const int FeatureLevel11_0 = 0xb000;

    [StructLayout(LayoutKind.Sequential)]
    private struct BufferDesc { public uint ByteWidth, Usage, BindFlags, CpuAccessFlags, MiscFlags, StructureByteStride; }

    [StructLayout(LayoutKind.Sequential)]
    private struct SubresourceData { public void* SysMem; public uint Pitch, SlicePitch; }

    [StructLayout(LayoutKind.Sequential)]
    private struct MappedSubresource { public void* Data; public uint RowPitch, DepthPitch; }

    [StructLayout(LayoutKind.Sequential)]
    private struct AdapterDesc1
    {
        public fixed char Description[128];
        public uint VendorId, DeviceId, SubSysId, Revision;
        public nuint DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public uint LuidLow; public int LuidHigh;
        public uint Flags;
    }

    [DllImport("d3d11.dll")]
    private static extern int D3D11CreateDevice(nint adapter, int driverType, nint software, uint flags, int* featureLevels, uint levels,
        uint sdkVersion, out nint device, out int featureLevel, out nint context);

    [DllImport("dxgi.dll")]
    private static extern int CreateDXGIFactory1(in Guid riid, out nint factory);

    [DllImport("d3dcompiler_47.dll", CharSet = CharSet.Ansi)]
    private static extern int D3DCompile(byte* src, nuint size, string? name, nint defines, nint include, string entry, string target,
        uint flags1, uint flags2, out nint code, out nint errors);

    private nint _device, _context;
    private readonly List<nint> _owned = [];
    public string AdapterName { get; }

    private static void** V(nint p) => *(void***)p;
    private static void Check(int hr, string what) { if (hr < 0) throw new InvalidOperationException($"{what} failed: HRESULT 0x{hr:X8}"); }
    private static void Release(nint p) { if (p != 0) ((delegate* unmanaged<nint, uint>)V(p)[IUnknownRelease])(p); }

    /// <summary>Creates a device on the DXGI adapter with the given index (see GpuAdapters), or the largest discrete GPU.</summary>
    public D3D11Compute(int adapterIndex = -1)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Direct3D 11 compute needs Windows");
        Check(CreateDXGIFactory1(new Guid("770aae78-f26f-4dba-a829-253c83d1b387"), out var factory), "CreateDXGIFactory1");
        nint chosen = 0; string name = "?"; nuint bestMem = 0;
        try
        {
            var enumAdapters1 = (delegate* unmanaged<nint, uint, nint*, int>)V(factory)[FactoryEnumAdapters1];
            for (uint i = 0; ; i++)
            {
                nint a;
                if (enumAdapters1(factory, i, &a) < 0) break;
                AdapterDesc1 d;
                ((delegate* unmanaged<nint, AdapterDesc1*, int>)V(a)[AdapterGetDesc1])(a, &d);
                bool software = (d.Flags & 2) != 0;
                bool pick = adapterIndex >= 0 ? i == adapterIndex : !software && d.DedicatedVideoMemory > bestMem;
                if (pick)
                {
                    Release(chosen);
                    chosen = a; bestMem = d.DedicatedVideoMemory; name = new string(d.Description).TrimEnd('\0');
                }
                else Release(a);
            }
        }
        finally { Release(factory); }
        if (chosen == 0) throw new InvalidOperationException("no suitable GPU adapter found");
        AdapterName = name;
        int fl = FeatureLevel11_0;
        int hr = D3D11CreateDevice(chosen, 0 /* UNKNOWN: use the given adapter */, 0, 0, &fl, 1, 7, out _device, out _, out _context);
        Release(chosen);
        Check(hr, "D3D11CreateDevice");
    }

    /// <summary>Compiles an embedded .hlsl file's entry point (cs_5_0) to a compute shader.</summary>
    public nint CompileShader(string resource, string entry, IReadOnlyDictionary<string, string>? defines = null)
    {
        using var s = typeof(D3D11Compute).Assembly.GetManifestResourceStream(resource) ?? throw new FileNotFoundException(resource);
        var src = new StreamReader(s).ReadToEnd();
        if (defines is not null)
            src = string.Concat(defines.Select(kv => $"#define {kv.Key} {kv.Value}\n")) + src;
        var bytes = Encoding.UTF8.GetBytes(src);
        nint code, errors;
        int hr;
        fixed (byte* p = bytes)
            hr = D3DCompile(p, (nuint)bytes.Length, resource, 0, 0, entry, "cs_5_0", 1u << 15 /* OPTIMIZATION_LEVEL3 */, 0, out code, out errors);
        if (hr < 0)
        {
            string msg = errors != 0 ? Marshal.PtrToStringAnsi((nint)((delegate* unmanaged<nint, void*>)V(errors)[BlobGetPointer])(errors)) ?? "" : "";
            Release(errors);
            throw new InvalidOperationException($"HLSL compile of {resource}:{entry} failed: {msg}");
        }
        Release(errors);
        void* ptr = ((delegate* unmanaged<nint, void*>)V(code)[BlobGetPointer])(code);
        nuint len = ((delegate* unmanaged<nint, nuint>)V(code)[BlobGetSize])(code);
        nint shader;
        Check(((delegate* unmanaged<nint, void*, nuint, nint, nint*, int>)V(_device)[DevCreateComputeShader])(_device, ptr, len, 0, &shader), "CreateComputeShader");
        Release(code);
        _owned.Add(shader);
        return shader;
    }

    /// <summary>A GPU structured buffer of 4-byte elements with an unordered-access view.</summary>
    public sealed class GpuBuffer
    {
        internal nint Buffer, Uav;
        public int Length { get; internal init; }
    }

    public GpuBuffer CreateBuffer<T>(ReadOnlySpan<T> init, int length = -1) where T : unmanaged
    {
        if (sizeof(T) != 4) throw new ArgumentException("buffers hold 4-byte elements (float, int, uint)");
        int n = length >= 0 ? length : init.Length;
        var desc = new BufferDesc { ByteWidth = (uint)(n * 4), Usage = UsageDefault, BindFlags = BindUav | BindShaderResource, MiscFlags = MiscStructured, StructureByteStride = 4 };
        nint buf, uav;
        fixed (T* p = init)
        {
            var data = new SubresourceData { SysMem = p };
            Check(((delegate* unmanaged<nint, BufferDesc*, SubresourceData*, nint*, int>)V(_device)[DevCreateBuffer])(
                _device, &desc, init.Length >= n && n > 0 ? &data : null, &buf), "CreateBuffer");
        }
        Check(((delegate* unmanaged<nint, nint, void*, nint*, int>)V(_device)[DevCreateUav])(_device, buf, null, &uav), "CreateUnorderedAccessView");
        _owned.Add(uav); _owned.Add(buf);
        return new GpuBuffer { Buffer = buf, Uav = uav, Length = n };
    }

    public GpuBuffer CreateBuffer(int length) => CreateBuffer<float>(ReadOnlySpan<float>.Empty, length);

    /// <summary>A constant buffer for kernel parameters (any blittable struct, padded to 16 bytes).</summary>
    public nint CreateConstants<T>(in T value) where T : unmanaged
    {
        uint size = (uint)((sizeof(T) + 15) / 16 * 16);
        var desc = new BufferDesc { ByteWidth = size, Usage = UsageDefault, BindFlags = BindConstant };
        var tmp = stackalloc byte[(int)size];
        new Span<byte>(tmp, (int)size).Clear();
        fixed (T* v = &value) Buffer.MemoryCopy(v, tmp, size, sizeof(T));
        var data = new SubresourceData { SysMem = tmp };
        nint cb;
        Check(((delegate* unmanaged<nint, BufferDesc*, SubresourceData*, nint*, int>)V(_device)[DevCreateBuffer])(_device, &desc, &data, &cb), "CreateBuffer(constants)");
        _owned.Add(cb);
        return cb;
    }

    public void UpdateConstants<T>(nint cb, in T value) where T : unmanaged
    {
        uint size = (uint)((sizeof(T) + 15) / 16 * 16);
        var tmp = stackalloc byte[(int)size];
        new Span<byte>(tmp, (int)size).Clear();
        fixed (T* v = &value) Buffer.MemoryCopy(v, tmp, size, sizeof(T));
        ((delegate* unmanaged<nint, nint, uint, void*, void*, uint, uint, void>)V(_context)[CtxUpdateSubresource])(_context, cb, 0, null, tmp, 0, 0);
    }

    /// <summary>Binds a shader, its UAV slots (u0..) and constant buffer (b0), and dispatches the given thread-group grid.</summary>
    public void Dispatch(nint shader, nint constants, uint groupsX, uint groupsY, params GpuBuffer[] uavs)
    {
        if (groupsX == 0 || groupsY == 0) return;
        if (groupsX > MaxGroupsPerDispatch || groupsY > MaxGroupsPerDispatch)
            throw new ArgumentException("dispatch grid too large: split the work into chunks");
        var views = stackalloc nint[uavs.Length];
        for (int i = 0; i < uavs.Length; i++) views[i] = uavs[i].Uav;
        ((delegate* unmanaged<nint, nint, void*, uint, void>)V(_context)[CtxCSSetShader])(_context, shader, null, 0);
        ((delegate* unmanaged<nint, uint, uint, nint*, uint*, void>)V(_context)[CtxCSSetUavs])(_context, 0, (uint)uavs.Length, views, null);
        if (constants != 0)
            ((delegate* unmanaged<nint, uint, uint, nint*, void>)V(_context)[CtxCSSetConstantBuffers])(_context, 0, 1, &constants);
        ((delegate* unmanaged<nint, uint, uint, uint, void>)V(_context)[CtxDispatch])(_context, groupsX, groupsY, 1);
        var nulls = stackalloc nint[uavs.Length];
        for (int i = 0; i < uavs.Length; i++) nulls[i] = 0;
        ((delegate* unmanaged<nint, uint, uint, nint*, uint*, void>)V(_context)[CtxCSSetUavs])(_context, 0, (uint)uavs.Length, nulls, null);
    }

    /// <summary>Overwrites a GPU buffer from CPU memory (same length).</summary>
    public void Write<T>(GpuBuffer dst, ReadOnlySpan<T> src) where T : unmanaged
    {
        if (sizeof(T) != 4 || src.Length != dst.Length) throw new ArgumentException("source must match the buffer length");
        fixed (T* p = src)
            ((delegate* unmanaged<nint, nint, uint, void*, void*, uint, uint, void>)V(_context)[CtxUpdateSubresource])(_context, dst.Buffer, 0, null, p, 0, 0);
    }

    public void Flush() => ((delegate* unmanaged<nint, void>)V(_context)[CtxFlush])(_context);

    /// <summary>Copies a GPU buffer back to the CPU (waits for the GPU to finish the work that writes it).</summary>
    public void Read<T>(GpuBuffer src, Span<T> dst) where T : unmanaged
    {
        if (dst.Length > src.Length) throw new ArgumentException("destination longer than buffer");
        var desc = new BufferDesc { ByteWidth = (uint)(src.Length * 4), Usage = UsageStaging, CpuAccessFlags = CpuRead };
        nint staging;
        Check(((delegate* unmanaged<nint, BufferDesc*, SubresourceData*, nint*, int>)V(_device)[DevCreateBuffer])(_device, &desc, null, &staging), "CreateBuffer(staging)");
        try
        {
            ((delegate* unmanaged<nint, nint, nint, void>)V(_context)[CtxCopyResource])(_context, staging, src.Buffer);
            MappedSubresource m;
            Check(((delegate* unmanaged<nint, nint, uint, uint, uint, MappedSubresource*, int>)V(_context)[CtxMap])(_context, staging, 0, MapRead, 0, &m), "Map");
            new ReadOnlySpan<T>(m.Data, dst.Length).CopyTo(dst);
            ((delegate* unmanaged<nint, nint, uint, void>)V(_context)[CtxUnmap])(_context, staging, 0);
        }
        finally { Release(staging); }
    }

    public void Dispose()
    {
        for (int i = _owned.Count - 1; i >= 0; i--) Release(_owned[i]);
        _owned.Clear();
        Release(_context); _context = 0;
        Release(_device); _device = 0;
    }
}
