// Grouped Any-Rate decode GEMV over up to 24 matrices of one mode K (the selected experts of one MoE layer), used by moe/engine.py.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "ham8.h"

__device__ __forceinline__ uint64_t transpose8(uint64_t x) {
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAull;  x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCull; x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ull; x ^= t ^ (t << 28);
    return x;
}

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

// 8x8 bit transpose of rows (A: rows 0-3, B: rows 4-7), byte i = row i; afterwards byte j of (A,B) holds bit j of every row
__device__ __forceinline__ void bt8(uint32_t& A, uint32_t& B) {
    uint32_t t = (A ^ (B << 4)) & 0xF0F0F0F0u; A ^= t; B ^= t >> 4;
    t = (A ^ (A >> 14)) & 0x0000CCCCu; A ^= t ^ (t << 14);
    t = (B ^ (B >> 14)) & 0x0000CCCCu; B ^= t ^ (t << 14);
    t = (A ^ (A >> 7)) & 0x00AA00AAu; A ^= t ^ (t << 7);
    t = (B ^ (B >> 7)) & 0x00AA00AAu; B ^= t ^ (t << 7);
}


// Grouped AR decode GEMV for the MoE experts: blockIdx.y selects one of up to 24 matrices (8 experts x gate/up, or 8 x down), one warp per row,
// one token (bf16). The mode table is read precomputed (fp16, 2^K entries per row) for K <= 6; for K = 7 it is built from the stored
// 8-bit codes (+ per-row lo, step), as in ar_gemv.cu. Inner loop identical to ar_gemv.cu (ham8 decode of the H planes).
struct Mat { const uint8_t* plane[7]; const uint8_t* tab; const __nv_bfloat16* x; __nv_bfloat16* y; int rows, cols; };
struct Group { Mat m[24]; int n; };

template <int K>
__global__ void __launch_bounds__(256) ar_moe_kernel(Group g) {
    const Mat& a = g.m[blockIdx.y];
    extern __shared__ __align__(16) unsigned char smem[];
    const int nchunk = a.cols >> 6;
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* tb = reinterpret_cast<float*>(smem + (size_t)(2048 / 64) * 72 * 2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* T = tb + warp * 128;
    const int row = blockIdx.x * 8 + warp;
    if (blockIdx.x * 8 >= a.rows) return;
    const size_t ustride = a.cols >> 3, hstride = a.cols >> 4;
    for (int i = threadIdx.x * 8; i < a.cols; i += 256 * 8)
        *reinterpret_cast<uint4*>(xs + (i >> 6) * 72 + (i & 63)) = *reinterpret_cast<const uint4*>(a.x + i);
    if (row < a.rows) {
        if (K <= 6) {
            const __half* tab = reinterpret_cast<const __half*>(a.tab) + (size_t)row * (1 << K);
            for (int j = lane; j < (1 << K); j += 32) T[j] = __half2float(tab[j]);
        } else {
            const __half* lo = reinterpret_cast<const __half*>(a.tab + (size_t)a.rows * 128); const __half* st = lo + a.rows;
            const float l = __half2float(lo[row]), s = __half2float(st[row]);
            const uint32_t c4 = *reinterpret_cast<const uint32_t*>(a.tab + (size_t)row * 128 + lane * 4);
#pragma unroll
            for (int q = 0; q < 4; q++) T[lane * 4 + q] = __half2float(__float2half(l + (float)((c4 >> (8 * q)) & 0xffu) * s));
        }
    }
    __syncthreads();
    if (row >= a.rows) return;
    float acc = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        const uint2 u = *reinterpret_cast<const uint2*>(a.plane[0] + (size_t)row * ustride + c * 8);
        uint32_t Wa[8], Wb[8];
#pragma unroll
        for (int s = 0; s < 8; s++) { Wa[s] = 0u; Wb[s] = 0u; }
        Wa[K - 1] = u.x; Wb[K - 1] = u.y;
#pragma unroll
        for (int p = 1; p < K; p++) {
            const uint32_t h = *reinterpret_cast<const uint32_t*>(a.plane[p] + (size_t)row * hstride + c * 4);
            Wa[K - 1 - p] = dec4(h & 0xffffu); Wb[K - 1 - p] = dec4(h >> 16);
        }
        uint32_t LA[4], HA[4], LB[4], HB[4];
        btrans4(Wa[0], Wa[1], Wa[2], Wa[3], LA); btrans4(Wb[0], Wb[1], Wb[2], Wb[3], LB);
        if (K > 4) { btrans4(Wa[4], Wa[5], Wa[6], Wa[7], HA); btrans4(Wb[4], Wb[5], Wb[6], Wb[7], HB); }
        else { HA[0] = HA[1] = HA[2] = HA[3] = 0u; HB[0] = HB[1] = HB[2] = HB[3] = 0u; }
#pragma unroll
        for (int gg = 0; gg < 8; gg++) {
            uint32_t A = gg < 4 ? LA[gg & 3] : LB[gg & 3], B = gg < 4 ? HA[gg & 3] : HB[gg & 3];
            if (K > 4) bt8(A, B);
            else {
                uint32_t t = (A ^ (A >> 14)) & 0x0000CCCCu; A ^= t ^ (t << 14);
                t = (A ^ (A >> 7)) & 0x00AA00AAu; A ^= t ^ (t << 7);
                B = (A >> 4) & 0x0F0F0F0Fu; A &= 0x0F0F0F0Fu;
            }
            float wv[8];
#pragma unroll
            for (int j = 0; j < 4; j++) { wv[j] = T[__byte_perm(A, 0, 0x4440 + j)]; wv[4 + j] = T[__byte_perm(B, 0, 0x4440 + j)]; }
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + c * 72 + gg * 8);
            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
            for (int q = 0; q < 4; q++) { const float2 xf = __bfloat1622float2(xb[q]); acc += wv[2 * q] * xf.x + wv[2 * q + 1] * xf.y; }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) a.y[row] = __float2bfloat16(acc);
}

template <int K>
static int launch(const Group& g, int max_rows, cudaStream_t s) {
    const size_t sm = (size_t)(2048 / 64) * 72 * 2 + 8 * 128 * sizeof(float);
    dim3 grid((max_rows + 7) / 8, g.n);
    ar_moe_kernel<K><<<grid, 256, sm, s>>>(g);
    return (int)cudaGetLastError();
}

// planes: n x 7 pointers, tabs: n pointers, xs/ys: n pointers; all matrices of one call share K. cols <= 2048, cols % 64 == 0.
extern "C" int ar_moe(const void* const* planes, const void* const* tabs, const void* const* xs, void* const* ys, const int* rows, const int* cols, int n, int k, void* stream) {
    if (n > 24) return -4;
    Group g; g.n = n; int mr = 0;
    for (int i = 0; i < n; i++) {
        for (int p = 0; p < 7; p++) g.m[i].plane[p] = (const uint8_t*)planes[i * 7 + p];
        g.m[i].tab = (const uint8_t*)tabs[i]; g.m[i].x = (const __nv_bfloat16*)xs[i]; g.m[i].y = (__nv_bfloat16*)ys[i];
        g.m[i].rows = rows[i]; g.m[i].cols = cols[i]; if (cols[i] % 64 || cols[i] > 2048) return -2; if (rows[i] > mr) mr = rows[i];
    }
    cudaStream_t s = (cudaStream_t)stream;
    switch (k) { case 3: return launch<3>(g, mr, s); case 4: return launch<4>(g, mr, s); case 5: return launch<5>(g, mr, s); case 6: return launch<6>(g, mr, s); case 7: return launch<7>(g, mr, s); }
    return -1;
}

// Grouped residual GEMV of the 5-8-bit modes (4 + K bits, K = 1..4), bf16 activations: y[r] += sum_c lev[r][idx_K(r, c)] * x[c] for up to 24
// matrices (blockIdx.y), added to the 4-bit output y. idx_K: the first K residual planes, plane p at P + p * plane_words (uint32 words),
// column 8b + j at bit 7 - j of byte b (as ar_residual.cu); lev: fp16 [rows, 2^K], the offsets of the K-bit prefixes (demo/moe_pack.py).
// One warp per 4 rows (the 32 inputs of a word read once for the 4 rows), fp32 accumulation.
struct RMat { const uint32_t* P; size_t plane_words; const __half* lev; const __nv_bfloat16* x; __nv_bfloat16* y; int rows, cols; };
struct RGroup { RMat m[24]; int n; };

template <int K, int R>
__global__ void __launch_bounds__(256) ar_res_moe_kernel(RGroup g) {
    const RMat& a = g.m[blockIdx.y];
    __shared__ float tab[8][R][16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= a.rows) return;
#pragma unroll
    for (int i = 0; i < R; i++)
        if (lane < (1 << K) && row0 + i < a.rows) tab[warp][i][lane] = __half2float(a.lev[(size_t)(row0 + i) * (1 << K) + lane]);
    __syncwarp();
    const int nw = a.cols >> 5; float acc[R];
#pragma unroll
    for (int i = 0; i < R; i++) acc[i] = 0.f;
    for (int w = lane; w < nw; w += 32) {
        float xf[32]; const uint4* xv = reinterpret_cast<const uint4*>(a.x + 32 * w);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            uint4 v = __ldg(xv + q); const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(h[j]); xf[8 * q + 2 * j] = f.x; xf[8 * q + 2 * j + 1] = f.y; }
        }
#pragma unroll
        for (int i = 0; i < R; i++) {
            if (row0 + i >= a.rows) break;
            const uint32_t* pr = a.P + (size_t)(row0 + i) * nw + w; uint32_t b[K];
#pragma unroll
            for (int p = 0; p < K; p++) b[p] = __ldg(pr + p * a.plane_words);
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint64_t m = 0;
#pragma unroll
                for (int p = 0; p < K; p++) m |= (uint64_t)((b[p] >> (8 * q)) & 0xFF) << (8 * (K - 1 - p));
                m = transpose8(m);
#pragma unroll
                for (int j = 0; j < 8; j++) s += tab[warp][i][(m >> (8 * (7 - j))) & 0xFF] * xf[8 * q + j];
            }
            acc[i] += s;
        }
    }
#pragma unroll
    for (int i = 0; i < R; i++) {
        float s = acc[i];
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0 && row0 + i < a.rows) a.y[row0 + i] = __float2bfloat16(__bfloat162float(a.y[row0 + i]) + s);
    }
}

// planes: n pointers (plane 0 of each matrix), plane_bytes: the distance between its planes; levs/xs/ys: n pointers; cols % 32 == 0.
extern "C" int ar_res_moe(const void* const* planes, const long long* plane_bytes, const void* const* levs, const void* const* xs, void* const* ys,
                          const int* rows, const int* cols, int n, int k, void* stream) {
    if (n > 24) return -4;
    constexpr int R = 4; RGroup g; g.n = n; int mr = 0;
    for (int i = 0; i < n; i++) {
        if (cols[i] % 32 || plane_bytes[i] % 4) return -2;
        g.m[i] = {(const uint32_t*)planes[i], (size_t)plane_bytes[i] / 4, (const __half*)levs[i], (const __nv_bfloat16*)xs[i], (__nv_bfloat16*)ys[i], rows[i], cols[i]};
        if (rows[i] > mr) mr = rows[i];
    }
    dim3 grid((mr + 8 * R - 1) / (8 * R), n); cudaStream_t s = (cudaStream_t)stream;
    switch (k) {
        case 1: ar_res_moe_kernel<1, R><<<grid, 256, 0, s>>>(g); break;
        case 2: ar_res_moe_kernel<2, R><<<grid, 256, 0, s>>>(g); break;
        case 3: ar_res_moe_kernel<3, R><<<grid, 256, 0, s>>>(g); break;
        case 4: ar_res_moe_kernel<4, R><<<grid, 256, 0, s>>>(g); break;
        default: return -1;
    }
    return (int)cudaGetLastError();
}
