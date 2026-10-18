// Cross-sectional stock features on a [T, N] panel (row t = trading day, column n = stock), all float32.
// Missing values are NaN. Kept bit-for-bit simple so the CPU reference in CpuFeatures.cs matches it.

cbuffer Params : register(b0)
{
    uint N;        // stocks
    uint T;        // days
    uint T0;       // first day of this chunk (dispatches are split by day to stay far below the 2 s driver limit)
    uint Feature;  // which feature to compute (Feature kernel) / unused (Rank kernel)
};

RWStructuredBuffer<float> A : register(u0);   // Feature: log close.   Rank: values
RWStructuredBuffer<float> B : register(u1);   // Feature: volume.      Rank: mask (1 = in universe)
RWStructuredBuffer<float> O : register(u2);   // output [T, N]
RWStructuredBuffer<float> C : register(u3);   // Feature: market log close [T].   Industry: industry code [N] (< 0 = none)
RWStructuredBuffer<float> D : register(u4);   // Industry: leader flag [T, N] (1 = large stock in its industry that day)

static const float NAN = asfloat(0x7fc00000);
bool bad(float x) { return (asuint(x) & 0x7fffffff) >= 0x7f800000; }   // NaN or inf, immune to fast-math

float lc(int t, uint n) { return t < 0 ? NAN : A[(uint)t * N + n]; }
float mk(int t) { return t < 0 ? NAN : C[(uint)t]; }

// Market beta from daily log returns over the 252 days ending at t (needs >= 200 days with both returns).
float beta252(int t, uint n)
{
    float sr = 0, sm = 0, srm = 0, smm = 0; int cnt = 0;
    for (int j = 0; j < 252; j++)
    {
        float r = lc(t - j, n) - lc(t - j - 1, n), m = mk(t - j) - mk(t - j - 1);
        if (bad(r) || bad(m)) continue;
        sr += r; sm += m; srm += r * m; smm += m * m; cnt++;
    }
    if (cnt < 200) return NAN;
    float vm = smm / cnt - (sm / cnt) * (sm / cnt);
    return vm > 0 ? (srm / cnt - (sr / cnt) * (sm / cnt)) / vm : NAN;
}

// Mean / std of residual returns r - beta * m over days t-j for j in [j0, j1).
void resid(int t, uint n, float b, int j0, int j1, out float mean, out float sd, out int cnt)
{
    float s = 0, s2 = 0; cnt = 0;
    for (int j = j0; j < j1; j++)
    {
        float r = lc(t - j, n) - lc(t - j - 1, n), m = mk(t - j) - mk(t - j - 1);
        if (bad(r) || bad(m)) continue;
        float e = r - b * m;
        s += e; s2 += e * e; cnt++;
    }
    mean = cnt > 0 ? s / cnt : NAN;
    sd = cnt > 1 ? sqrt(max(s2 / cnt - mean * mean, 0.0)) : NAN;
}

// Feature ids (keep in sync with GpuFeatures.Names):
//  0 mom5  1 mom21  2 mom63  3 mom126  4 mom12_1 (t-252..t-21)  5 vol20  6 vol60
//  7 dist from 252-day high  8 volume surprise (log v - mean log v over prior 60)  9 log avg dollar volume 20d
// 10 beta 252d vs market  11 residual momentum (mean/std of residual returns t-251..t-21)
// 12 max daily return over 21 days  13 idiosyncratic vol 60d
float feature(uint f, int t, uint n)
{
    float c = lc(t, n);
    if (bad(c)) return NAN;
    if (f <= 3)
    {
        int k = f == 0 ? 5 : f == 1 ? 21 : f == 2 ? 63 : 126;
        float p = lc(t - k, n);
        return bad(p) ? NAN : c - p;
    }
    if (f == 4)
    {
        float a = lc(t - 21, n), b = lc(t - 252, n);
        return bad(a) || bad(b) ? NAN : a - b;
    }
    if (f == 5 || f == 6)
    {
        int k = f == 5 ? 20 : 60;
        float s = 0, s2 = 0;
        for (int j = 0; j < k; j++)
        {
            float r = lc(t - j, n) - lc(t - j - 1, n);
            if (bad(r)) return NAN;
            s += r; s2 += r * r;
        }
        float m = s / k;
        return sqrt(max(s2 / k - m * m, 0.0));
    }
    if (f == 7)
    {
        if (t < 251) return NAN;
        float hi = c;
        for (int j = 1; j < 252; j++) { float p = lc(t - j, n); if (!bad(p)) hi = max(hi, p); }
        return c - hi;
    }
    if (f == 8)
    {
        float v = B[(uint)t * N + n];
        if (bad(v) || v <= 0 || t < 60) return NAN;
        float s = 0; int cnt = 0;
        for (int j = 1; j <= 60; j++) { float p = B[(uint)(t - j) * N + n]; if (!bad(p) && p > 0) { s += log(p); cnt++; } }
        return cnt < 40 ? NAN : log(v) - s / cnt;
    }
    if (f == 9)
    {
        if (t < 19) return NAN;
        float s = 0;
        for (int j = 0; j < 20; j++)
        {
            float p = lc(t - j, n), v = B[(uint)(t - j) * N + n];
            if (bad(p) || bad(v)) return NAN;
            s += exp(p) * v;
        }
        return s > 0 ? log(s / 20) : NAN;
    }
    if (f == 10) return beta252(t, n);
    if (f == 11 || f == 13)
    {
        float b = beta252(t, n);
        if (bad(b)) return NAN;
        float m, sd; int cnt;
        if (f == 11) { resid(t, n, b, 21, 252, m, sd, cnt); return cnt < 150 || !(sd > 0) ? NAN : m / sd; }
        resid(t, n, b, 0, 60, m, sd, cnt);
        return cnt < 40 ? NAN : sd;
    }
    if (f == 12)
    {
        float hi = -1e30;
        for (int j = 0; j < 21; j++)
        {
            float r = lc(t - j, n) - lc(t - j - 1, n);
            if (bad(r)) return NAN;
            hi = max(hi, r);
        }
        return hi;
    }
    return NAN;
}

[numthreads(256, 1, 1)]
void FeatureKernel(uint3 g : SV_DispatchThreadID)
{
    uint n = g.x, t = T0 + g.y;
    if (n >= N || t >= T) return;
    O[t * N + n] = feature(Feature, (int)t, n);
}

// Cross-sectional percentile rank per day among stocks with mask = 1 and a finite value:
// rank = (#less + 0.5 * #equal-others) / (count - 1), in [0, 1]; NaN outside the universe or with < 2 stocks.
// One thread per (stock, day); each group streams the day's row through shared memory in tiles of 256.
groupshared float tile[256];

[numthreads(256, 1, 1)]
void RankKernel(uint3 g : SV_DispatchThreadID, uint3 l : SV_GroupThreadID)
{
    uint n = g.x, t = T0 + g.y;
    bool live = n < N && t < T;
    float x = NAN;
    if (live) { float v = A[t * N + n]; x = B[t * N + n] > 0.5 && !bad(v) ? v : NAN; }
    float less = 0, eq = 0, cnt = 0;
    for (uint base = 0; base < N; base += 256)
    {
        uint j = base + l.x;
        float y = NAN;
        if (j < N && t < T) { float v = A[t * N + j]; y = B[t * N + j] > 0.5 && !bad(v) ? v : NAN; }
        tile[l.x] = y;
        GroupMemoryBarrierWithGroupSync();
        uint m = min(256u, N - base);
        for (uint k = 0; k < m; k++)
        {
            float z = tile[k];
            if (!bad(z)) { cnt += 1; less += z < x ? 1 : 0; eq += z == x ? 1 : 0; }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (live) O[t * N + n] = bad(x) || cnt < 2 ? NAN : (less + 0.5 * (eq - 1)) / (cnt - 1);
}

// Industry aggregates per day over universe stocks (B = mask) sharing stock n's industry code (C):
//  Feature 0: mean of A        1: fraction of A > 0        2: mean of A over industry leaders (D = 1), excluding n
// NaN when stock n has no industry or too few contributing stocks (3; 1 for leaders).
groupshared float tv[256];
groupshared float tg[256];
groupshared float tl[256];

[numthreads(256, 1, 1)]
void IndustryKernel(uint3 g : SV_DispatchThreadID, uint3 l : SV_GroupThreadID)
{
    uint n = g.x, t = T0 + g.y;
    bool live = n < N && t < T;
    float ind = live ? C[n] : -1;
    float s = 0, cnt = 0;
    for (uint base = 0; base < N; base += 256)
    {
        uint j = base + l.x;
        float v = NAN, gi = -1, ld = 0;
        if (j < N && t < T && B[t * N + j] > 0.5) { v = A[t * N + j]; gi = C[j]; ld = D[t * N + j]; }
        tv[l.x] = v; tg[l.x] = gi; tl[l.x] = ld;
        GroupMemoryBarrierWithGroupSync();
        uint m = min(256u, N - base);
        if (ind >= 0)
            for (uint k = 0; k < m; k++)
            {
                float z = tv[k];
                if (tg[k] != ind || bad(z)) continue;
                if (Feature == 2 && tl[k] < 0.5) continue;
                if (Feature == 2 && base + k == n) continue;
                s += Feature == 1 ? (z > 0 ? 1 : 0) : z;
                cnt += 1;
            }
        GroupMemoryBarrierWithGroupSync();
    }
    // breadth is snapped to a 1/4096 grid so equal fractions (7/12, 14/24) tie exactly despite GPU division rounding
    if (live) O[t * N + n] = ind < 0 || cnt < (Feature == 2 ? 1 : 3) ? NAN : Feature == 1 ? round(s / cnt * 4096) / 4096 : s / cnt;
}
