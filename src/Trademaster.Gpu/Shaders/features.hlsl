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

static const float NAN = asfloat(0x7fc00000);
bool bad(float x) { return (asuint(x) & 0x7fffffff) >= 0x7f800000; }   // NaN or inf, immune to fast-math

float lc(int t, uint n) { return t < 0 ? NAN : A[(uint)t * N + n]; }

// Feature ids (keep in sync with GpuFeatures.Names):
//  0 mom5  1 mom21  2 mom63  3 mom126  4 mom12_1 (t-252..t-21)  5 vol20  6 vol60
//  7 dist from 252-day high  8 volume surprise (log v - mean log v over prior 60)  9 log avg dollar volume 20d
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
