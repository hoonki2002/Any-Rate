// AR (UHHHHHH) decode GEMV, v8 = v6 with cheaper lookups (x always fp32 in shared memory):
//   * table entries are 256 B apart (entry e of lane l at byte e*256 + 4l, or e*256 + 8l for float2 pair tables), so the shared address
//     of a lookup is one byte_perm: the index byte is dropped into byte 1 of the lane's base offset (no shift/IMAD per lookup).
//   * modes k=3 and k=4 (2 and 2.5 bit): after the 4x8 bit transpose each byte holds the indices of columns j and j+4 (two nibbles);
//     that byte indexes a float2 pair table directly (one lookup per two weights, no nibble masking). k>=5: one lookup per weight.
// v6 header follows.
// AR (UHHHHHH) decode GEMV, v6: one LANE per row.
//   * A warp owns a tile of 32 rows; lane l accumulates row l. The per-row mode table lives in shared memory interleaved
//     (entry j of row l at word j*32 + l), so every lookup of lane l hits bank l: no bank conflicts for any table size.
//   * Planes are repacked tile-interleaved (same bits, different order): for chunk c (64 columns) the U words of the 32 rows
//     are 32 consecutive uint2 (256 B) and each H plane 32 consecutive uint32 (128 B) -> fully coalesced warp loads.
//   * x is read from shared memory with all lanes on the same address (broadcast).
//   * W warps of a block split the columns of the block's slice; the grid splits the columns into S slices
//     (split-K) so that small matrices still fill the GPU. Partial sums go through a fp32 workspace; the last block of a
//     tile (atomic counter) adds the S partials in a fixed order (deterministic) and writes y.
// Decode of a 64-column chunk (Hamming nibble -> codeword, bit transposes) is identical to v4 (ar_gemv.cu).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "ham8.h"

struct A6 {
    const uint2* U;            // [tiles][nchunk][32] uint2
    const uint32_t* H[6];      // [tiles][nchunk][32] uint32, plane 1..6
    const __half* tab;         // K<=6: [tiles][2^K][32] fp16 mode table
    const uint32_t* codes;     // K==7: [tiles][32 (j/4)][32] uint32 (4 codes of one row)
    const __half* lo; const __half* st;   // K==7: [tiles*32]
    const __half* x; __half* y;
    float* ws; int* cnt;       // split-K workspace [S][NB][rows_pad], counters [tiles]
    int rows, rows_pad, nchunk, S, bf16;
};

__device__ __forceinline__ uint32_t dec4(uint32_t h16) {
    uint32_t sel = h16 & 0x7777u;
    uint32_t lo4 = __byte_perm(HAM_W0, HAM_W1, sel);
    uint32_t hi4 = __byte_perm(HAM_W2, HAM_W3, sel);
    uint32_t sel2 = 0x3210u | (((h16 >> 3) & 0x1111u) << 2);
    return __byte_perm(lo4, hi4, sel2);
}
__device__ __forceinline__ void btrans4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t* o) {
    uint32_t t0 = __byte_perm(w0, w1, 0x5140), t1 = __byte_perm(w2, w3, 0x5140);
    uint32_t t2 = __byte_perm(w0, w1, 0x7362), t3 = __byte_perm(w2, w3, 0x7362);
    o[0] = __byte_perm(t0, t1, 0x5410); o[1] = __byte_perm(t0, t1, 0x7632);
    o[2] = __byte_perm(t2, t3, 0x5410); o[3] = __byte_perm(t2, t3, 0x7632);
}
__device__ __forceinline__ void bt8(uint32_t& A, uint32_t& B) {
    uint32_t t = (A ^ (B << 4)) & 0xF0F0F0F0u; A ^= t; B ^= t >> 4;
    t = (A ^ (A >> 14)) & 0x0000CCCCu; A ^= t ^ (t << 14);
    t = (B ^ (B >> 14)) & 0x0000CCCCu; B ^= t ^ (t << 14);
    t = (A ^ (A >> 7)) & 0x00AA00AAu; A ^= t ^ (t << 7);
    t = (B ^ (B >> 7)) & 0x00AA00AAu; B ^= t ^ (t << 7);
}

// table loads: shared address = byte_perm result + 1024 (dynamic shared memory starts after the 1 KB the driver reserves per block on
// sm_90+/sm_120; the launcher checks cudaDevAttrReservedSharedMemoryPerBlock == 1024), so no address arithmetic per lookup.
__device__ __forceinline__ float lds1(uint32_t off) { float v; asm volatile("ld.shared.f32 %0, [%1+1024];" : "=f"(v) : "r"(off)); return v; }
__device__ __forceinline__ float2 lds2(uint32_t off) { float2 v; asm volatile("ld.shared.v2.f32 {%0, %1}, [%2+1024];" : "=f"(v.x), "=f"(v.y) : "r"(off)); return v; }

template <int K>
__device__ __forceinline__ void load_chunk(const A6& a, size_t base, uint2& u, uint32_t* h) {
    u = __ldcs(a.U + base);
#pragma unroll
    for (int p = 1; p < K; p++) h[p] = __ldcs(a.H[p - 1] + base);
}

template <int NB, int K, int W, bool XF>
__global__ void __launch_bounds__(W * 32) ar8_kernel(A6 a) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* Tf = reinterpret_cast<float*>(smem);                       // [2^K][32]
    constexpr bool PAIR = (K <= 4);
    constexpr int TBYTES = PAIR ? (K == 3 ? 120 : 256) * 256 : (1 << K) * 256;
    float* red = Tf + TBYTES / 4;                                     // [W][NB][32]
    __half* xs = reinterpret_cast<__half*>(red + W * NB * 32);        // [NB][slice*64] fp16 (XF=false) or fp32 (XF=true)
    float* xsf = red + W * NB * 32;
    const int t = blockIdx.x, s = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cps = (a.nchunk + a.S - 1) / a.S, c0 = s * cps, c1 = min(a.nchunk, c0 + cps), ncol = (c1 - c0) * 64;
    const size_t tb = (size_t)t * a.nchunk * 32 + lane;
    // first chunk requested before staging
    int c = c0 + warp;
    uint2 un = make_uint2(0, 0); uint32_t hn[K];
#pragma unroll
    for (int p = 0; p < K; p++) hn[p] = 0u;
    if (c < c1) load_chunk<K>(a, tb + (size_t)c * 32, un, hn);
    // x slice -> shared
    const int ncols_total = a.nchunk * 64;
    for (int b = 0; b < NB; b++)
        for (int i = threadIdx.x * 8; i < ncol; i += W * 32 * 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(a.x + (size_t)b * ncols_total + c0 * 64 + i);
            if (!XF) *reinterpret_cast<uint4*>(xs + b * ncol + i) = v;
            else {
                const __half2* vh = reinterpret_cast<const __half2*>(&v); const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v);
                float f[8];
#pragma unroll
                for (int q = 0; q < 4; q++) { const float2 t2 = a.bf16 ? __bfloat1622float2(vb[q]) : __half22float2(vh[q]); f[2 * q] = t2.x; f[2 * q + 1] = t2.y; }
                *reinterpret_cast<float4*>(xsf + b * ncol + i) = make_float4(f[0], f[1], f[2], f[3]);
                *reinterpret_cast<float4*>(xsf + b * ncol + i + 4) = make_float4(f[4], f[5], f[6], f[7]);
            }
        }
    // mode table -> shared (fp32, interleaved)
    if (K == 7) {
        for (int i = threadIdx.x; i < 32 * 32; i += W * 32) {
            const int j4 = i >> 5, l = i & 31, r = t * 32 + l;
            const uint32_t c4 = a.codes[(size_t)t * 1024 + i];
            const float lo = __half2float(a.lo[r]), st = __half2float(a.st[r]);
#pragma unroll
            for (int q = 0; q < 4; q++) Tf[(j4 * 4 + q) * 64 + l] = __half2float(__float2half(lo + (float)((c4 >> (8 * q)) & 0xffu) * st));
        }
    } else if (PAIR) {
        const __half* tg = a.tab + ((size_t)t << K) * 32;
        float2* T2 = reinterpret_cast<float2*>(Tf);
        for (int i = threadIdx.x; i < (TBYTES / 256) * 32; i += W * 32) {
            const int e = i >> 5, l = i & 31;
            const int lo4 = e & 15, hi4 = e >> 4;
            T2[i] = (lo4 < (1 << K) && hi4 < (1 << K)) ? make_float2(__half2float(tg[lo4 * 32 + l]), __half2float(tg[hi4 * 32 + l])) : make_float2(0.f, 0.f);
        }
    } else {
        const __half* tg = a.tab + ((size_t)t << K) * 32;
        for (int i = threadIdx.x; i < (32 << K); i += W * 32) Tf[(i >> 5) * 64 + (i & 31)] = __half2float(tg[i]);
    }
    __syncthreads();
    const uint32_t base = PAIR ? lane * 8u : lane * 4u;   // byte 1 (bits 8-15) is free for the index
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    for (; c < c1; c += W) {
        const uint2 u = un; uint32_t hw[K];
#pragma unroll
        for (int p = 1; p < K; p++) hw[p] = hn[p];
        if (c + W < c1) load_chunk<K>(a, tb + (size_t)(c + W) * 32, un, hn);
        uint32_t Wa[8], Wb[8];
#pragma unroll
        for (int q = 0; q < 8; q++) { Wa[q] = 0u; Wb[q] = 0u; }
        Wa[K - 1] = u.x; Wb[K - 1] = u.y;
#pragma unroll
        for (int p = 1; p < K; p++) { Wa[K - 1 - p] = dec4(hw[p] & 0xffffu); Wb[K - 1 - p] = dec4(hw[p] >> 16); }
        uint32_t LA[4], HA[4], LB[4], HB[4];
        btrans4(Wa[0], Wa[1], Wa[2], Wa[3], LA); btrans4(Wb[0], Wb[1], Wb[2], Wb[3], LB);
        if (K > 4) { btrans4(Wa[4], Wa[5], Wa[6], Wa[7], HA); btrans4(Wb[4], Wb[5], Wb[6], Wb[7], HB); }
        const float* xcf = xsf + (c - c0) * 64;
#pragma unroll
        for (int g = 0; g < 8; g++) {
            uint32_t A = g < 4 ? LA[g & 3] : LB[g & 3];
            float xv[8];
            {
                const float4 x0 = *reinterpret_cast<const float4*>(xcf + g * 8), x1 = *reinterpret_cast<const float4*>(xcf + g * 8 + 4);
                xv[0] = x0.x; xv[1] = x0.y; xv[2] = x0.z; xv[3] = x0.w; xv[4] = x1.x; xv[5] = x1.y; xv[6] = x1.z; xv[7] = x1.w;
            }
            if constexpr (PAIR) {
                uint32_t tt = (A ^ (A >> 14)) & 0x0000CCCCu; A ^= tt ^ (tt << 14);
                tt = (A ^ (A >> 7)) & 0x00AA00AAu; A ^= tt ^ (tt << 7);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t off = __byte_perm(A, base, 0x7604u | (j << 4));
                    const float2 w = lds2(off);
                    acc[0] = fmaf(w.x, xv[j], acc[0]); acc[0] = fmaf(w.y, xv[4 + j], acc[0]);
                }
            } else {
                uint32_t B = g < 4 ? HA[g & 3] : HB[g & 3];
                bt8(A, B);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float wa = lds1(__byte_perm(A, base, 0x7604u | (j << 4)));
                    const float wb = lds1(__byte_perm(B, base, 0x7604u | (j << 4)));
                    acc[0] = fmaf(wa, xv[j], acc[0]); acc[0] = fmaf(wb, xv[4 + j], acc[0]);
                }
            }
        }
    }
    // reduce the W warps
#pragma unroll
    for (int b = 0; b < NB; b++) red[(warp * NB + b) * 32 + lane] = acc[b];
    __syncthreads();
    if (warp != 0) return;
    const int row = t * 32 + lane;
    float v[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) { v[b] = 0.f; for (int w = 0; w < W; w++) v[b] += red[(w * NB + b) * 32 + lane]; }
    if (a.S == 1) {
        if (row < a.rows)
#pragma unroll
            for (int b = 0; b < NB; b++) reinterpret_cast<unsigned short*>(a.y)[(size_t)b * a.rows + row] = a.bf16 ? __bfloat16_as_ushort(__float2bfloat16(v[b])) : __half_as_ushort(__float2half(v[b]));
        return;
    }
#pragma unroll
    for (int b = 0; b < NB; b++) __stcg(a.ws + ((size_t)s * NB + b) * a.rows_pad + row, v[b]);
    __threadfence();
    int prev = 0;
    if (lane == 0) prev = atomicAdd(a.cnt + t, 1);
    prev = __shfl_sync(0xffffffffu, prev, 0);
    if (prev != a.S - 1) return;
    __threadfence();
#pragma unroll
    for (int b = 0; b < NB; b++) {
        float sum = 0.f;
        for (int ss = 0; ss < a.S; ss++) sum += __ldcg(a.ws + ((size_t)ss * NB + b) * a.rows_pad + row);
        if (row < a.rows) reinterpret_cast<unsigned short*>(a.y)[(size_t)b * a.rows + row] = a.bf16 ? __bfloat16_as_ushort(__float2bfloat16(sum)) : __half_as_ushort(__float2half(sum));
    }
    if (lane == 0) a.cnt[t] = 0;
}

template <int NB, int K, int W, bool XF>
static int launch(const A6& a, cudaStream_t st) {
    const int cps = (a.nchunk + a.S - 1) / a.S;
    size_t sm = (size_t)(K <= 4 ? (K == 3 ? 120 : 256) * 256 : (1 << K) * 256) + (size_t)W * NB * 32 * 4 + (size_t)NB * cps * 64 * (XF ? 4 : 2);
    static size_t cur = 0;
    if (sm > cur) { if (cudaFuncSetAttribute(ar8_kernel<NB, K, W, XF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) return -5; cur = sm; }
    dim3 grid(a.rows_pad / 32, a.S);
    ar8_kernel<NB, K, W, XF><<<grid, W * 32, sm, st>>>(a);
    return (int)cudaGetLastError();
}
template <int NB, int W, bool XF>
static int launch_k(const A6& a, int k, cudaStream_t st) {
    switch (k) {
        case 3: return launch<NB, 3, W, XF>(a, st); case 4: return launch<NB, 4, W, XF>(a, st); case 5: return launch<NB, 5, W, XF>(a, st);
        case 6: return launch<NB, 6, W, XF>(a, st); case 7: return launch<NB, 7, W, XF>(a, st);
    }
    return -1;
}
template <int NB, bool XF>
static int launch_w(const A6& a, int k, int W, cudaStream_t st) {
    switch (W) { case 2: return launch_k<NB, 2, XF>(a, k, st); case 4: return launch_k<NB, 4, XF>(a, k, st); case 8: return launch_k<NB, 8, XF>(a, k, st); case 16: return launch_k<NB, 16, XF>(a, k, st); }
    return -4;
}

extern "C" int ar8_gemv(const void* U, const void* const* H, const void* tab, const void* codes, const void* lo, const void* st,
                        const void* x, void* y, void* ws, void* cnt, int rows, int cols, int k, int nb, int S, int W, int bf16, int xf, void* stream) {
    A6 a; a.U = (const uint2*)U;
    for (int p = 0; p < 6; p++) a.H[p] = (const uint32_t*)H[p];
    a.tab = (const __half*)tab; a.codes = (const uint32_t*)codes; a.lo = (const __half*)lo; a.st = (const __half*)st;
    a.x = (const __half*)x; a.y = (__half*)y; a.ws = (float*)ws; a.cnt = (int*)cnt;
    a.rows = rows; a.rows_pad = (rows + 31) / 32 * 32; a.nchunk = cols / 64; a.S = S; a.bf16 = bf16;
    if (cols % 64) return -2;
    if (nb != 1) return -3;
    static int reserved = -1;
    if (reserved < 0) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev); }
    if (reserved != 1024) return -6;
    return launch_w<1, true>(a, k, W, (cudaStream_t)stream);
}
