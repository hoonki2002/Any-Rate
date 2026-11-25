// Any-Rate residual GEMV of the 5-8-bit modes, batch 1: y[r] += sum_c lev[r][off_k + idx_k(r, c)] * x[c], added to the 4-bit output y.
// idx_k(r, c): the first k residual planes (most significant first), plane p stored as (rows, cols/8) bytes with column 8b + j at bit 7 - j
// of byte b (any_rate/core/pack.py: pack_residual); lev: (rows, 30) fp16, the offsets of the prefixes of 1, 2, 3, 4 bits at 0, 2, 6, 14.
// One warp per 4 rows (the 32 inputs of a word are read once for the 4 rows), 32 columns per lane step; the rows' 2^k offsets in shared
// memory, fp32 accumulation; the k-bit indices of 8 columns come from one 8x8 bit transpose of the k plane bytes.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint64_t transpose8(uint64_t x) {             // 8x8 bit matrix, one row per byte
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAull;  x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCull; x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ull; x ^= t ^ (t << 28);
    return x;
}

template <int K, int R>
__global__ void __launch_bounds__(256) ar_res_kernel(const uint32_t* __restrict__ P, const __half* __restrict__ lev, const __half* __restrict__ x,
                                                     __half* __restrict__ y, int rows, int cols, size_t plane_words, int off) {
    __shared__ float tab[8][R][16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = (blockIdx.x * 8 + warp) * R;
    if (row0 >= rows) return;
#pragma unroll
    for (int i = 0; i < R; i++)
        if (lane < (1 << K) && row0 + i < rows) tab[warp][i][lane] = __half2float(lev[(size_t)(row0 + i) * 30 + off + lane]);
    __syncwarp();
    const int nw = cols >> 5; float acc[R];
#pragma unroll
    for (int i = 0; i < R; i++) acc[i] = 0.f;
    for (int w = lane; w < nw; w += 32) {
        float xf[32]; const uint4* xv = reinterpret_cast<const uint4*>(x + 32 * w);     // the 32 inputs of this word, read once for R rows
#pragma unroll
        for (int q = 0; q < 4; q++) {
            uint4 v = __ldg(xv + q); const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int j = 0; j < 4; j++) { float2 f = __half22float2(h[j]); xf[8 * q + 2 * j] = f.x; xf[8 * q + 2 * j + 1] = f.y; }
        }
#pragma unroll
        for (int i = 0; i < R; i++) {
            if (row0 + i >= rows) break;
            const uint32_t* pr = P + (size_t)(row0 + i) * nw + w; uint32_t b[K];
#pragma unroll
            for (int p = 0; p < K; p++) b[p] = __ldg(pr + p * plane_words);
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < 4; q++) {                                               // byte q: columns 8q .. 8q+7 at bits 7 .. 0
                uint64_t m = 0;
#pragma unroll
                for (int p = 0; p < K; p++) m |= (uint64_t)((b[p] >> (8 * q)) & 0xFF) << (8 * (K - 1 - p));
                m = transpose8(m);                                                      // byte 7 - j: the index of column 8q + j
#pragma unroll
                for (int j = 0; j < 8; j++) a += tab[warp][i][(m >> (8 * (7 - j))) & 0xFF] * xf[8 * q + j];
            }
            acc[i] += a;
        }
    }
#pragma unroll
    for (int i = 0; i < R; i++) {
        float a = acc[i];
#pragma unroll
        for (int s = 16; s; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0 && row0 + i < rows) y[row0 + i] = __float2half(__half2float(y[row0 + i]) + a);
    }
}

extern "C" int ar_res_gemv(const void* planes, const void* lev, const void* x, void* y, int rows, int cols, int k, void* stream) {
    static const int OFF[5] = {0, 0, 2, 6, 14};
    if (k < 1 || k > 4 || cols % 32) return 1;
    constexpr int R = 4; dim3 grid((rows + 8 * R - 1) / (8 * R)), block(256); size_t pw = (size_t)rows * cols / 32; cudaStream_t s = (cudaStream_t)stream;
    const uint32_t* P = (const uint32_t*)planes; const __half* L = (const __half*)lev; const __half* X = (const __half*)x; __half* Y = (__half*)y;
    switch (k) {
        case 1: ar_res_kernel<1, R><<<grid, block, 0, s>>>(P, L, X, Y, rows, cols, pw, OFF[1]); break;
        case 2: ar_res_kernel<2, R><<<grid, block, 0, s>>>(P, L, X, Y, rows, cols, pw, OFF[2]); break;
        case 3: ar_res_kernel<3, R><<<grid, block, 0, s>>>(P, L, X, Y, rows, cols, pw, OFF[3]); break;
        case 4: ar_res_kernel<4, R><<<grid, block, 0, s>>>(P, L, X, Y, rows, cols, pw, OFF[4]); break;
    }
    return (int)cudaGetLastError();
}
