using System.Diagnostics;
using System.Net.Sockets;

namespace Trademaster.Cli;

/// <summary>
/// Opens the dashboard in its own app-style window (Microsoft Edge "--app" mode: no tabs, no address bar),
/// with a dedicated profile so the window's lifetime can be tracked. Edge ships with Windows.
/// The window runs with GPU acceleration off: this PC's AMD driver has hung the machine before.
/// </summary>
public static class AppWindow
{
    public static string ProfileDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AITrademaster", "window");

    public static string? FindEdge()
    {
        if (!OperatingSystem.IsWindows()) return null;
        string[] roots =
        [
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        ];
        return roots.Select(r => Path.Combine(r, "Microsoft", "Edge", "Application", "msedge.exe")).FirstOrDefault(File.Exists);
    }

    public static bool PortInUse(int port)
    {
        try
        {
            using var c = new TcpClient();
            return c.ConnectAsync("127.0.0.1", port).Wait(300) && c.Connected;
        }
        catch { return false; }
    }

    /// <summary>Opens the window. Returns the window's process when its lifetime can be tracked, else null.</summary>
    public static Process? Open(string url)
    {
        var edge = FindEdge();
        if (edge is null)
        {
            Console.WriteLine($"Microsoft Edge not found; opening {url} in the default browser instead.");
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            return null;
        }
        Directory.CreateDirectory(ProfileDir);
        var psi = new ProcessStartInfo(edge) { UseShellExecute = false };
        foreach (var a in new[]
        {
            $"--app={url}", $"--user-data-dir={ProfileDir}", "--window-size=1500,950",
            "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--disable-features=Translate",
        })
            psi.ArgumentList.Add(a);
        var p = Process.Start(psi);
        // If a window with this profile is already open, Edge hands the request over and exits at once.
        if (p is null || p.WaitForExit(3000)) return null;
        return p;
    }
}
