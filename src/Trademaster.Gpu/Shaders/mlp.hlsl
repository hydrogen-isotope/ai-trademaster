// Small MLP (F -> H ReLU -> H ReLU -> 1), trained with mean squared error and AdamW, for S independent seeds at once.
// Parameters of seed s live at P[s * NP ...] in this order:
//   W1 [F][H] | b1 [H] | W2 [H][H] | b2 [H] | W3 [H] | b3 [1]          NP = F*H + H*H + 3H + 1
// One training step = Forward1, Forward2, Output, Back2, Back1, GradAdam (6 short dispatches).
// CPU reference: GpuMlp.CpuStep (same arithmetic order per value).

cbuffer Params : register(b0)
{
    uint S;      // seeds
    uint B;      // batch size
    uint F;      // inputs
    uint H;      // hidden width
    uint NP;     // parameters per seed
    uint Off;    // offset of this batch in each seed's index list
    uint NIdx;   // length of each seed's index list
    uint R0;     // Predict: first row of this chunk
    uint RN;     // Predict: rows in this chunk
    float Lr;
    float Wd;    // decoupled weight decay (weights only)
    float C1;    // 1 - beta1^t
    float C2;    // 1 - beta2^t
};

RWStructuredBuffer<float> X   : register(u0);   // [rows][F]
RWStructuredBuffer<float> Y   : register(u1);   // [rows]          Predict: output [S][RN]
RWStructuredBuffer<uint>  Idx : register(u2);   // [S][NIdx] shuffled training rows per seed
RWStructuredBuffer<float> P   : register(u3);   // [S][NP]
RWStructuredBuffer<float> Act : register(u4);   // [2][S][B][H]    post-ReLU activations of layers 1 and 2
RWStructuredBuffer<float> Dlt : register(u5);   // [S][B][2H+1]    dL/d(pre-activation) layer 1, layer 2, output
RWStructuredBuffer<float> MV  : register(u6);   // [S][NP][2]      Adam first / second moments

uint oW1() { return 0; }
uint oB1() { return F * H; }
uint oW2() { return F * H + H; }
uint oB2() { return F * H + H + H * H; }
uint oW3() { return F * H + 2 * H + H * H; }
uint oB3() { return F * H + 3 * H + H * H; }

uint row(uint s, uint b) { return Idx[s * NIdx + Off + b]; }
uint a1(uint s, uint b, uint j) { return (s * B + b) * H + j; }
uint a2(uint s, uint b, uint k) { return S * B * H + (s * B + b) * H + k; }
uint d1(uint s, uint b, uint j) { return (s * B + b) * (2 * H + 1) + j; }
uint d2(uint s, uint b, uint k) { return (s * B + b) * (2 * H + 1) + H + k; }
uint dO(uint s, uint b) { return (s * B + b) * (2 * H + 1) + 2 * H; }

[numthreads(256, 1, 1)]
void Forward1(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, b = g.x / H, j = g.x % H;
    if (s >= S || b >= B) return;
    uint p = s * NP, r = row(s, b);
    float z = P[p + oB1() + j];
    for (uint f = 0; f < F; f++) z += X[r * F + f] * P[p + oW1() + f * H + j];
    Act[a1(s, b, j)] = max(z, 0.0);
}

[numthreads(256, 1, 1)]
void Forward2(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, b = g.x / H, k = g.x % H;
    if (s >= S || b >= B) return;
    uint p = s * NP;
    float z = P[p + oB2() + k];
    for (uint j = 0; j < H; j++) z += Act[a1(s, b, j)] * P[p + oW2() + j * H + k];
    Act[a2(s, b, k)] = max(z, 0.0);
}

[numthreads(256, 1, 1)]
void Output(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, b = g.x;
    if (s >= S || b >= B) return;
    uint p = s * NP;
    float z = P[p + oB3()];
    for (uint k = 0; k < H; k++) z += Act[a2(s, b, k)] * P[p + oW3() + k];
    Dlt[dO(s, b)] = 2.0 * (z - Y[row(s, b)]) / B;
}

[numthreads(256, 1, 1)]
void Back2(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, b = g.x / H, k = g.x % H;
    if (s >= S || b >= B) return;
    float d = Dlt[dO(s, b)] * P[s * NP + oW3() + k];
    Dlt[d2(s, b, k)] = Act[a2(s, b, k)] > 0 ? d : 0.0;
}

[numthreads(256, 1, 1)]
void Back1(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, b = g.x / H, j = g.x % H;
    if (s >= S || b >= B) return;
    uint p = s * NP;
    float d = 0;
    for (uint k = 0; k < H; k++) d += Dlt[d2(s, b, k)] * P[p + oW2() + j * H + k];
    Dlt[d1(s, b, j)] = Act[a1(s, b, j)] > 0 ? d : 0.0;
}

// One thread per (parameter, seed): gradient summed over the batch, then the AdamW update.
[numthreads(64, 1, 1)]
void GradAdam(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, q = g.x;
    if (s >= S || q >= NP) return;
    float gr = 0;
    bool weight = true;
    if (q < oB1())
    {
        uint f = q / H, j = q % H;
        for (uint b = 0; b < B; b++) gr += X[row(s, b) * F + f] * Dlt[d1(s, b, j)];
    }
    else if (q < oW2())
    {
        uint j = q - oB1(); weight = false;
        for (uint b = 0; b < B; b++) gr += Dlt[d1(s, b, j)];
    }
    else if (q < oB2())
    {
        uint j = (q - oW2()) / H, k = (q - oW2()) % H;
        for (uint b = 0; b < B; b++) gr += Act[a1(s, b, j)] * Dlt[d2(s, b, k)];
    }
    else if (q < oW3())
    {
        uint k = q - oB2(); weight = false;
        for (uint b = 0; b < B; b++) gr += Dlt[d2(s, b, k)];
    }
    else if (q < oB3())
    {
        uint k = q - oW3();
        for (uint b = 0; b < B; b++) gr += Act[a2(s, b, k)] * Dlt[dO(s, b)];
    }
    else
    {
        weight = false;
        for (uint b = 0; b < B; b++) gr += Dlt[dO(s, b)];
    }
    uint i = s * NP + q;
    float m = 0.9 * MV[2 * i] + 0.1 * gr;
    float v = 0.999 * MV[2 * i + 1] + 0.001 * gr * gr;
    MV[2 * i] = m;
    MV[2 * i + 1] = v;
    float w = P[i];
    if (weight) w -= Lr * Wd * w;
    P[i] = w - Lr * (m / C1) / (sqrt(v / C2) + 1e-8);
}

// Inference: one thread per (row in chunk, seed). Writes Y[s * RN + i] (Y is the output buffer here).
[numthreads(256, 1, 1)]
void Predict(uint3 g : SV_DispatchThreadID)
{
    uint s = g.y, i = g.x;
    if (s >= S || i >= RN) return;
    uint p = s * NP, r = R0 + i;
    float h1[64];
    for (uint j = 0; j < H; j++)
    {
        float z = P[p + oB1() + j];
        for (uint f = 0; f < F; f++) z += X[r * F + f] * P[p + oW1() + f * H + j];
        h1[j] = max(z, 0.0);
    }
    float yhat = P[p + oB3()];
    for (uint k = 0; k < H; k++)
    {
        float z = P[p + oB2() + k];
        for (uint j2 = 0; j2 < H; j2++) z += h1[j2] * P[p + oW2() + j2 * H + k];
        yhat += max(z, 0.0) * P[p + oW3() + k];
    }
    Y[s * RN + i] = yhat;
}
