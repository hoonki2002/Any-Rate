// AR (UHHHHHH) decode GEMV, v11: v8 (one lane per row, conflict-free interleaved tables, byte_perm addressing, fp32 x in shared memory)
// with the Hamming decode moved to the MESSAGE domain.
//   The H code is RM(1,3) ([8,4,4] extended Hamming): codeword bit of column j (j = 4 j2 + 2 j1 + j0) is m3 ^ j2 m0 ^ j1 m1 ^ j0 m2.
//   Hence, with r_i = (bit i of the message of every H plane, one plane per index bit), the H part of the index of column j is
//     j=0: r3, 1: r3^r2, 2: r3^r1, 3: r3^r1^r2, 4..7: the same ^ r0.
//   So we transpose the 4-bit MESSAGES of the K-1 H planes (32 bits per plane and chunk instead of 64 codeword bits), then form the
//   indices of four columns at once with byte permutes and one 3-input XOR, for all planes simultaneously. No codeword expansion.
//   U bits are stored in a phase order (same bits, different order inside the word) so that one shift and one LOP3 merge them into the
//   index bytes: word h (h = 0, 1) of a lane and chunk, bit 8j + 4p + 2(g&1) + x = U of block 2g+p (g = 2h + ((bit>>1)&1)), column 4x + j.
// Index bit q: U -> q = K-1, H plane p -> q = K-1-p (as ar_pack / v6).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

struct A11 {
    const uint2* U;            // phase-ordered U words [tiles32][nchunk][32]
    const uint32_t* H[6];      // v6 layout [tiles32][nchunk][32]
    const __half* tab;         // k<7: [tiles32][2^k][32] fp16
    const uint32_t* codes;     // k=7: [tiles32][32 (j/4)][32]
    const __half* lo; const __half* st;
    const __half* x; __half* y; float* ws; int* cnt;
    int rows, rows_pad, nchunk, S;
};

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
__device__ __forceinline__ void bt4(uint32_t& A) {
    uint32_t t = (A ^ (A >> 14)) & 0x0000CCCCu; A ^= t ^ (t << 14);
    t = (A ^ (A >> 7)) & 0x00AA00AAu; A ^= t ^ (t << 7);
}
template <int SH> __device__ __forceinline__ uint32_t shl(uint32_t v) { if constexpr (SH >= 0) return v << SH; else return v >> (-SH); }
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) { return a ^ b ^ c; }
// H index bytes of columns 0-3 (X1) and 4-7 (X2) from R = bytes (r0, r1, r2, r3) (bytes may hold two nibble-packed blocks)
__device__ __forceinline__ void combos(uint32_t R, uint32_t& X1, uint32_t& X2) {
    X1 = xor3(__byte_perm(R, 0u, 0x3333), __byte_perm(R, 0u, 0x1124), __byte_perm(R, 0u, 0x2444));
    X2 = X1 ^ __byte_perm(R, 0u, 0x0000);
}
__device__ __forceinline__ float lds1(uint32_t off) { float v; asm("ld.shared.f32 %0, [%1+1024];" : "=f"(v) : "r"(off)); return v; }
__device__ __forceinline__ float2 lds2(uint32_t off) { float2 v; asm("ld.shared.v2.f32 {%0, %1}, [%2+1024];" : "=f"(v.x), "=f"(v.y) : "r"(off)); return v; }
#define SEL(j) (0x7604u | ((j) << 4))
// k=3 pair index: byte lo | hi << 4 (lo, hi < 8) -> dense lo | hi << 3 (64 entries)
__device__ __forceinline__ uint32_t compact3(uint32_t X) { return X; }


template <int K> struct Cfg11 { static constexpr int TBYTES = (K == 3 ? 120 : (1 << K)) * 256; };

// single-lookup group: 4 index bytes in I, x of those 4 columns at xp[0..3]
__device__ __forceinline__ void look4(uint32_t I, uint32_t base, const float* xp, float& acc) {
    const float4 xv = *reinterpret_cast<const float4*>(xp);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(0))), xv.x, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(1))), xv.y, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(2))), xv.z, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(3))), xv.w, acc);
}
// pair-lookup group (k=3): byte j of I = idx(col a_j) | idx(col b_j) << 4 ; x of a at xa[0..3], of b at xb[0..3]
__device__ __forceinline__ void look4p(uint32_t I, uint32_t base, const float* xa, const float* xb, float& acc, float& acc2) {
    const float4 va = *reinterpret_cast<const float4*>(xa), vb = *reinterpret_cast<const float4*>(xb);
    float2 w;
    w = lds2(__byte_perm(I, base, SEL(0))); acc = fmaf(w.x, va.x, acc); acc2 = fmaf(w.y, vb.x, acc2);
    w = lds2(__byte_perm(I, base, SEL(1))); acc = fmaf(w.x, va.y, acc); acc2 = fmaf(w.y, vb.y, acc2);
    w = lds2(__byte_perm(I, base, SEL(2))); acc = fmaf(w.x, va.z, acc); acc2 = fmaf(w.y, vb.z, acc2);
    w = lds2(__byte_perm(I, base, SEL(3))); acc = fmaf(w.x, va.w, acc); acc2 = fmaf(w.y, vb.w, acc2);
}

template <int K>
__device__ __forceinline__ void do_chunk(uint2 u, const uint32_t* hw, const float* xc, uint32_t base, float& acc, float& acc2) {
    uint32_t P[8];
#pragma unroll
    for (int q = 0; q < 8; q++) P[q] = (q <= K - 2) ? hw[K - 1 - q] : 0u;
    uint32_t L[4];
    btrans4(P[0], P[1], P[2], P[3], L);
    if constexpr (K <= 5) {
#pragma unroll
        for (int g = 0; g < 4; g++) {
            uint32_t R = L[g]; bt4(R);                       // byte i = r_i(block 2g) | r_i(block 2g+1) << 4
            uint32_t X1, X2; combos(R, X1, X2);
            const uint32_t uw = g < 2 ? u.x : u.y;
            constexpr int o0 = 0;                             // phase of (g&1, x): 2(g&1) + x
            const float* xg = xc + 16 * g;
            if constexpr (K <= 4) {
                if (g & 1) { X1 |= shl<K - 1 - 2>(uw) & (0x11111111u << (K - 1)); X2 |= shl<K - 1 - 3>(uw) & (0x11111111u << (K - 1)); }
                else       { X1 |= shl<K - 1 - 0>(uw) & (0x11111111u << (K - 1)); X2 |= shl<K - 1 - 1>(uw) & (0x11111111u << (K - 1)); }
                if constexpr (K == 3) {
                    look4p(compact3(X1), base, xg, xg + 8, acc, acc2);
                    look4p(compact3(X2), base, xg + 4, xg + 12, acc, acc2);
                } else {
                    look4(X1 & 0x0F0F0F0Fu, base, xg, acc); look4((X1 >> 4) & 0x0F0F0F0Fu, base, xg + 8, acc2);
                    look4(X2 & 0x0F0F0F0Fu, base, xg + 4, acc); look4((X2 >> 4) & 0x0F0F0F0Fu, base, xg + 12, acc2);
                }
            } else {   // K == 5: 5-bit indices, split nibbles to bytes, U to bit 4
                const int ph = 2 * (g & 1);
                uint32_t A1 = X1 & 0x0F0F0F0Fu, B1 = (X1 >> 4) & 0x0F0F0F0Fu, A2 = X2 & 0x0F0F0F0Fu, B2 = (X2 >> 4) & 0x0F0F0F0Fu;
                if (ph == 0) {
                    A1 |= shl<4 - 0>(uw) & 0x10101010u; B1 |= shl<4 - 0 - 4>(uw) & 0x10101010u;
                    A2 |= shl<4 - 1>(uw) & 0x10101010u; B2 |= shl<4 - 1 - 4>(uw) & 0x10101010u;
                } else {
                    A1 |= shl<4 - 2>(uw) & 0x10101010u; B1 |= shl<4 - 2 - 4>(uw) & 0x10101010u;
                    A2 |= shl<4 - 3>(uw) & 0x10101010u; B2 |= shl<4 - 3 - 4>(uw) & 0x10101010u;
                }
                look4(A1, base, xg, acc); look4(B1, base, xg + 8, acc2); look4(A2, base, xg + 4, acc); look4(B2, base, xg + 12, acc2);
                (void)o0;
            }
        }
    } else {
        uint32_t Hh[4];
        btrans4(P[4], P[5], P[6], P[7], Hh);
#pragma unroll
        for (int g = 0; g < 4; g++) {
            uint32_t RA = L[g], RB = Hh[g]; bt8(RA, RB);       // RA: bytes r_i of block 2g, RB: of block 2g+1
            const uint32_t uw = g < 2 ? u.x : u.y;
            const float* xg = xc + 16 * g;
            uint32_t A1, A2, B1, B2; combos(RA, A1, A2); combos(RB, B1, B2);
            constexpr uint32_t M = 0x01010101u << (K - 1);
            if (g & 1) {
                A1 |= shl<K - 1 - 2>(uw) & M; A2 |= shl<K - 1 - 3>(uw) & M; B1 |= shl<K - 1 - 6>(uw) & M; B2 |= shl<K - 1 - 7>(uw) & M;
            } else {
                A1 |= shl<K - 1 - 0>(uw) & M; A2 |= shl<K - 1 - 1>(uw) & M; B1 |= shl<K - 1 - 4>(uw) & M; B2 |= shl<K - 1 - 5>(uw) & M;
            }
            look4(A1, base, xg, acc); look4(A2, base, xg + 4, acc2); look4(B1, base, xg + 8, acc); look4(B2, base, xg + 12, acc2);
        }
    }
}

template <int K, int W>
__global__ void __launch_bounds__(W * 32) ar11_kernel(A11 a) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int TB = Cfg11<K>::TBYTES;
    float* Tf = reinterpret_cast<float*>(smem);
    float* red = reinterpret_cast<float*>(smem + TB);         // [W][32]
    float* xs = red + W * 32;                                  // fp32 x slice
    const int t = blockIdx.x, s = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cps = (a.nchunk + a.S - 1) / a.S, c0 = s * cps, c1 = min(a.nchunk, c0 + cps), ncol = max(0, c1 - c0) * 64;
    const size_t tb = (size_t)t * a.nchunk * 32 + lane;
    int c = c0 + warp;
    uint2 un = make_uint2(0, 0); uint32_t hn[K];
#pragma unroll
    for (int p = 0; p < K; p++) hn[p] = 0u;
    auto load = [&](int cc) {
        un = __ldcs(a.U + tb + (size_t)cc * 32);
#pragma unroll
        for (int p = 1; p < K; p++) hn[p] = __ldcs(a.H[p - 1] + tb + (size_t)cc * 32);
    };
    if (c < c1) load(c);
    for (int i = threadIdx.x * 8; i < ncol; i += W * 32 * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(a.x + (size_t)c0 * 64 + i);
        const __half2* vh = reinterpret_cast<const __half2*>(&v);
        const float2 f0 = __half22float2(vh[0]), f1 = __half22float2(vh[1]), f2 = __half22float2(vh[2]), f3 = __half22float2(vh[3]);
        *reinterpret_cast<float4*>(xs + i) = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(xs + i + 4) = make_float4(f2.x, f2.y, f3.x, f3.y);
    }
    if constexpr (K == 7) {
        for (int i = threadIdx.x; i < 32 * 32; i += W * 32) {
            const int j4 = i >> 5, l = i & 31, r = t * 32 + l;
            const uint32_t c4 = a.codes[(size_t)t * 1024 + i];
            const float lo = __half2float(a.lo[r]), st = __half2float(a.st[r]);
#pragma unroll
            for (int q = 0; q < 4; q++) Tf[(j4 * 4 + q) * 64 + l] = __half2float(__float2half(lo + (float)((c4 >> (8 * q)) & 0xffu) * st));
        }
    } else if constexpr (K == 3) {
        const __half* tg = a.tab + ((size_t)t << K) * 32;
        float2* T2 = reinterpret_cast<float2*>(Tf);
        for (int i = threadIdx.x; i < 64 * 32; i += W * 32) {
            const int e = i >> 5, l = i & 31, lo4 = e & 7, hi4 = e >> 3;
            T2[(lo4 | (hi4 << 4)) * 32 + l] = make_float2(__half2float(tg[lo4 * 32 + l]), __half2float(tg[hi4 * 32 + l]));
        }
    } else {
        const __half* tg = a.tab + ((size_t)t << K) * 32;
        for (int i = threadIdx.x; i < (32 << K); i += W * 32) Tf[(i >> 5) * 64 + (i & 31)] = __half2float(tg[i]);
    }
    __syncthreads();
    uint32_t base; asm volatile("mov.u32 %0, %1;" : "=r"(base) : "r"(K == 3 ? lane * 8u : lane * 4u) : "memory");
    float acc = 0.f, acc2 = 0.f;
    for (; c < c1; c += W) {
        const uint2 u = un; uint32_t hw[K];
#pragma unroll
        for (int p = 1; p < K; p++) hw[p] = hn[p];
        if (c + W < c1) load(c + W);
        do_chunk<K>(u, hw, xs + (c - c0) * 64, base, acc, acc2);
    }
    red[warp * 32 + lane] = acc + acc2;
    __syncthreads();
    if (warp != 0) return;
    float v = 0.f;
    for (int w = 0; w < W; w++) v += red[w * 32 + lane];
    const int row = t * 32 + lane;
    if (a.S == 1) { if (row < a.rows) a.y[row] = __float2half(v); return; }
    __stcg(a.ws + (size_t)s * a.rows_pad + row, v);
    __threadfence();
    int prev = 0;
    if (lane == 0) prev = atomicAdd(a.cnt + t, 1);
    prev = __shfl_sync(0xffffffffu, prev, 0);
    if (prev != a.S - 1) return;
    __threadfence();
    float sum = 0.f;
    for (int ss = 0; ss < a.S; ss++) sum += __ldcg(a.ws + (size_t)ss * a.rows_pad + row);
    if (row < a.rows) a.y[row] = __float2half(sum);
    if (lane == 0) a.cnt[t] = 0;
}

template <int K, int W>
static int launch(const A11& a, cudaStream_t st) {
    const int cps = (a.nchunk + a.S - 1) / a.S;
    size_t sm = (size_t)Cfg11<K>::TBYTES + W * 32 * 4 + (size_t)cps * 64 * 4;
    static size_t cur = 0;
    if (sm > cur) { if (cudaFuncSetAttribute(ar11_kernel<K, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) return -5; cur = sm; }
    ar11_kernel<K, W><<<dim3(a.rows_pad / 32, a.S), W * 32, sm, st>>>(a);
    return (int)cudaGetLastError();
}
template <int W>
static int launch_k(const A11& a, int k, cudaStream_t st) {
    switch (k) { case 3: return launch<3, W>(a, st); case 4: return launch<4, W>(a, st); case 5: return launch<5, W>(a, st);
                 case 6: return launch<6, W>(a, st); case 7: return launch<7, W>(a, st); }
    return -1;
}
extern "C" int ar11_gemv(const void* U, const void* const* H, const void* tab, const void* codes, const void* lo, const void* st,
                         const void* x, void* y, void* ws, void* cnt, int rows, int cols, int k, int S, int W, void* stream) {
    static int reserved = -1;
    if (reserved < 0) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev); }
    if (reserved != 1024) return -6;
    if (cols % 64) return -2;
    A11 a; a.U = (const uint2*)U;
    for (int p = 0; p < 6; p++) a.H[p] = (const uint32_t*)H[p];
    a.tab = (const __half*)tab; a.codes = (const uint32_t*)codes; a.lo = (const __half*)lo; a.st = (const __half*)st;
    a.x = (const __half*)x; a.y = (__half*)y; a.ws = (float*)ws; a.cnt = (int*)cnt;
    a.rows = rows; a.rows_pad = (rows + 31) / 32 * 32; a.nchunk = cols / 64; a.S = S;
    cudaStream_t s = (cudaStream_t)stream;
    switch (W) { case 4: return launch_k<4>(a, k, s); case 8: return launch_k<8>(a, k, s); case 16: return launch_k<16>(a, k, s); case 32: return launch_k<32>(a, k, s); }
    return -4;
}
