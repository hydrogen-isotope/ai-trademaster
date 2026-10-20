using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace Trademaster.Cli;

/// <summary>
/// Phone access over Tailscale (the user's private network; nothing is exposed to the internet). The dashboard also
/// listens on this PC's Tailscale address, and every request that does not come from this PC itself must carry the
/// sign-in code (private/config/remote.json, created on first use and shown only to the PC's own screen).
/// </summary>
public static class RemoteAccess
{
    const string Cookie = "tm_code";
    static readonly string ConfigPath = Path.Combine("private", "config", "remote.json");

    /// <summary>This PC's IPv4 address on the Tailscale interface (100.64.0.0/10), or null.</summary>
    public static IPAddress? TailscaleAddress()
    {
        foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
        {
            if (nic.OperationalStatus != OperationalStatus.Up || !nic.Name.Contains("Tailscale", StringComparison.OrdinalIgnoreCase)) continue;
            foreach (var a in nic.GetIPProperties().UnicastAddresses)
            {
                var b = a.Address.GetAddressBytes();
                if (a.Address.AddressFamily == AddressFamily.InterNetwork && b[0] == 100 && b[1] >= 64 && b[1] <= 127) return a.Address;
            }
        }
        return null;
    }

    public static string Code()
    {
        try
        {
            if (File.Exists(ConfigPath) && JsonDocument.Parse(File.ReadAllText(ConfigPath)).RootElement.TryGetProperty("code", out var c)
                && c.GetString() is { Length: >= 8 } s) return s;
        }
        catch (Exception) { }
        const string alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";             // no 0/O or 1/I to mistype
        var code = string.Concat(Enumerable.Range(0, 10).Select(_ => alphabet[RandomNumberGenerator.GetInt32(alphabet.Length)]));
        Directory.CreateDirectory(Path.GetDirectoryName(ConfigPath)!);
        File.WriteAllText(ConfigPath, JsonSerializer.Serialize(new { code }));
        return code;
    }

    static bool Local(HttpContext ctx) => ctx.Connection.RemoteIpAddress is { } ip && IPAddress.IsLoopback(ip);

    static bool Signed(HttpContext ctx) =>
        ctx.Request.Cookies.TryGetValue(Cookie, out var v) && v is not null &&
        CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(v), Encoding.UTF8.GetBytes(Code()));

    /// <summary>Sign-in gate for requests from other devices, plus the phone-access info endpoint for this PC.</summary>
    public static void Use(WebApplication app, int port)
    {
        app.Use(async (ctx, next) =>
        {
            if (Local(ctx) || Signed(ctx) || ctx.Request.Path.StartsWithSegments("/login") || ctx.Request.Path == "/favicon.svg")
            {
                await next();
                return;
            }
            if (ctx.Request.Path.StartsWithSegments("/api"))
            {
                ctx.Response.StatusCode = 401;
                return;
            }
            ctx.Response.Redirect("/login");
        });
        app.MapGet("/login", () => Results.Content(LoginPage(null), "text/html; charset=utf-8"));
        app.MapPost("/login", async (HttpContext ctx) =>
        {
            var form = await ctx.Request.ReadFormAsync();
            var given = (form["code"].ToString() ?? "").Trim().ToUpperInvariant();
            await Task.Delay(500);                                                 // slows guessing
            if (!CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(given), Encoding.UTF8.GetBytes(Code())))
                return Results.Content(LoginPage("That code is not right."), "text/html; charset=utf-8");
            ctx.Response.Cookies.Append(Cookie, Code(), new CookieOptions { HttpOnly = true, SameSite = SameSiteMode.Strict, MaxAge = TimeSpan.FromDays(60) });
            return Results.Redirect("/autotrade");
        });
        // shown only on this PC (the Auto-trade page): where to point the phone, and the code
        app.MapGet("/api/remote/info", (HttpContext ctx) =>
        {
            if (!Local(ctx)) return Results.StatusCode(403);
            var ip = TailscaleAddress();
            return Results.Json(new { tailscale = ip?.ToString(), host = Dns.GetHostName().ToLowerInvariant(), port, code = Code(),
                url = ip is null ? null : $"http://{ip}:{port}/autotrade" });
        });
    }

    static string LoginPage(string? error) => $$"""
        <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI Trademaster sign-in</title>
        <style>:root{color-scheme:dark}body{margin:0;background:#0b0e11;color:#e6e9ee;font:16px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
        display:flex;min-height:100vh;align-items:center;justify-content:center}form{background:#14181d;border:1px solid #262c34;border-radius:14px;
        padding:24px;width:min(340px,90vw)}h1{font-size:18px;font-weight:600;margin:0 0 6px}p{color:#8a93a0;font-size:13px;margin:0 0 16px}
        input{width:100%;box-sizing:border-box;background:#1b2027;border:1px solid #262c34;color:#e6e9ee;border-radius:8px;padding:12px;font-size:20px;
        letter-spacing:3px;text-transform:uppercase}button{margin-top:12px;width:100%;background:#3b82f6;color:#fff;border:0;border-radius:8px;
        padding:12px;font-size:16px;font-weight:600}.err{color:#f6465d;font-size:13px;margin-top:10px}</style></head>
        <body><form method="post" action="/login"><h1>AI Trademaster</h1><p>Enter the phone access code shown on the PC (Auto-trade page).</p>
        <input name="code" autocomplete="one-time-code" autocapitalize="characters" autofocus><button>Sign in</button>
        {{(error is null ? "" : $"<div class=\"err\">{error}</div>")}}</form></body></html>
        """;
}
