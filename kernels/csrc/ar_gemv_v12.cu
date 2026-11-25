// AR (UHHHHHH) decode GEMV, v12 = v11 compute (message-domain Hamming decode, one lane per row, conflict-free tables, byte_perm
// addressing) in a PERSISTENT STREAM-K kernel:
//   * grid = number of blocks that fit on the GPU; block b takes the equal share [b*N/G, (b+1)*N/G) of the N = tiles x chunks iterations
//     (tile-major), so there is no tail wave. A tile cut between blocks is summed with fp32 atomics into a zeroed workspace; the block that
//     completes the tile's chunk count (atomic counter) writes y and zeroes the workspace and the counter again (graph-replay safe).
//   * x is staged once per block as fp16 (converted when used); the next tile's table source (fp16 mode table or 8-bit codes) is
//     prefetched into registers while the current tile streams, so a tile switch costs no global round trip.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

struct A12 {
    const uint2* U; const uint32_t* H[6];
    const __half* tab; const uint32_t* codes; const __half* lo; const __half* st;
    const __half* x; __half* y; float* ws; int* cnt;
    int rows, rows_pad, nchunk, ntile;
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
__device__ __forceinline__ void combos(uint32_t R, uint32_t& X1, uint32_t& X2) {
    X1 = __byte_perm(R, 0u, 0x3333) ^ __byte_perm(R, 0u, 0x1124) ^ __byte_perm(R, 0u, 0x2444);
    X2 = X1 ^ __byte_perm(R, 0u, 0x0000);
}
__device__ __forceinline__ float lds1(uint32_t off) { float v; asm("ld.shared.f32 %0, [%1+1024];" : "=f"(v) : "r"(off)); return v; }
__device__ __forceinline__ float2 lds2(uint32_t off) { float2 v; asm("ld.shared.v2.f32 {%0, %1}, [%2+1024];" : "=f"(v.x), "=f"(v.y) : "r"(off)); return v; }
#define SEL(j) (0x7604u | ((j) << 4))
__device__ __forceinline__ float4 x4(const __half* p) {
    const uint2 v = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void look4(uint32_t I, uint32_t base, const __half* xp, float& acc) {
    const float4 xv = x4(xp);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(0))), xv.x, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(1))), xv.y, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(2))), xv.z, acc);
    acc = fmaf(lds1(__byte_perm(I, base, SEL(3))), xv.w, acc);
}
__device__ __forceinline__ void look4p(uint32_t I, uint32_t base, const __half* xa, const __half* xb, float& acc, float& acc2) {
    const float4 va = x4(xa), vb = x4(xb);
    float2 w;
    w = lds2(__byte_perm(I, base, SEL(0))); acc = fmaf(w.x, va.x, acc); acc2 = fmaf(w.y, vb.x, acc2);
    w = lds2(__byte_perm(I, base, SEL(1))); acc = fmaf(w.x, va.y, acc); acc2 = fmaf(w.y, vb.y, acc2);
    w = lds2(__byte_perm(I, base, SEL(2))); acc = fmaf(w.x, va.z, acc); acc2 = fmaf(w.y, vb.z, acc2);
    w = lds2(__byte_perm(I, base, SEL(3))); acc = fmaf(w.x, va.w, acc); acc2 = fmaf(w.y, vb.w, acc2);
}

template <int K>
__device__ __forceinline__ void do_chunk(uint2 u, const uint32_t* hw, const __half* xc, uint32_t base, float& acc, float& acc2) {
    uint32_t P[8];
#pragma unroll
    for (int q = 0; q < 8; q++) P[q] = (q <= K - 2) ? hw[K - 1 - q] : 0u;
    uint32_t L[4];
    btrans4(P[0], P[1], P[2], P[3], L);
    if constexpr (K <= 5) {
#pragma unroll
        for (int g = 0; g < 4; g++) {
            uint32_t R = L[g]; bt4(R);
            uint32_t X1, X2; combos(R, X1, X2);
            const uint32_t uw = g < 2 ? u.x : u.y;
            const __half* xg = xc + 16 * g;
            if constexpr (K <= 4) {
                if (g & 1) { X1 |= shl<K - 3>(uw) & (0x11111111u << (K - 1)); X2 |= shl<K - 4>(uw) & (0x11111111u << (K - 1)); }
                else       { X1 |= shl<K - 1>(uw) & (0x11111111u << (K - 1)); X2 |= shl<K - 2>(uw) & (0x11111111u << (K - 1)); }
                if constexpr (K == 3) { look4p(X1, base, xg, xg + 8, acc, acc2); look4p(X2, base, xg + 4, xg + 12, acc, acc2); }
                else {
                    look4(X1 & 0x0F0F0F0Fu, base, xg, acc); look4((X1 >> 4) & 0x0F0F0F0Fu, base, xg + 8, acc2);
                    look4(X2 & 0x0F0F0F0Fu, base, xg + 4, acc); look4((X2 >> 4) & 0x0F0F0F0Fu, base, xg + 12, acc2);
                }
            } else {
                uint32_t A1 = X1 & 0x0F0F0F0Fu, B1 = (X1 >> 4) & 0x0F0F0F0Fu, A2 = X2 & 0x0F0F0F0Fu, B2 = (X2 >> 4) & 0x0F0F0F0Fu;
                if (g & 1) { A1 |= shl<2>(uw) & 0x10101010u; B1 |= shl<-2>(uw) & 0x10101010u; A2 |= shl<1>(uw) & 0x10101010u; B2 |= shl<-3>(uw) & 0x10101010u; }
                else       { A1 |= shl<4>(uw) & 0x10101010u; B1 |= uw & 0x10101010u;          A2 |= shl<3>(uw) & 0x10101010u; B2 |= shl<-1>(uw) & 0x10101010u; }
                look4(A1, base, xg, acc); look4(B1, base, xg + 8, acc2); look4(A2, base, xg + 4, acc); look4(B2, base, xg + 12, acc2);
            }
        }
    } else {
        uint32_t Hh[4];
        btrans4(P[4], P[5], P[6], P[7], Hh);
#pragma unroll
        for (int g = 0; g < 4; g++) {
            uint32_t RA = L[g], RB = Hh[g]; bt8(RA, RB);
            const uint32_t uw = g < 2 ? u.x : u.y;
            const __half* xg = xc + 16 * g;
            uint32_t A1, A2, B1, B2; combos(RA, A1, A2); combos(RB, B1, B2);
            constexpr uint32_t M = 0x01010101u << (K - 1);
            if (g & 1) { A1 |= shl<K - 3>(uw) & M; A2 |= shl<K - 4>(uw) & M; B1 |= shl<K - 7>(uw) & M; B2 |= shl<K - 8>(uw) & M; }
            else       { A1 |= shl<K - 1>(uw) & M; A2 |= shl<K - 2>(uw) & M; B1 |= shl<K - 5>(uw) & M; B2 |= shl<K - 6>(uw) & M; }
            look4(A1, base, xg, acc); look4(A2, base, xg + 4, acc2); look4(B1, base, xg + 8, acc); look4(B2, base, xg + 12, acc2);
        }
    }
}

template <int K> struct Cfg12 {
    static constexpr int TBYTES = (K == 3 ? 120 : (1 << K)) * 256;
    static constexpr int SRC16 = (K == 7) ? (4096 + 128) / 16 : ((32 << K) * 2) / 16;   // uint4 count of a tile's table source
};

// table source of tile t -> registers (uint4 each); K=7: codes (256 uint4) then lo/st (8 uint4: lo 32 halves, st 32 halves)
template <int K, int NT, int PER>
__device__ __forceinline__ void src_load(const A12& a, int t, uint4* r) {
#pragma unroll
    for (int q = 0; q < PER; q++) {
        const int i = threadIdx.x + q * NT;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (i < Cfg12<K>::SRC16) {
            if constexpr (K == 7) {
                if (i < 256) v = __ldcg(reinterpret_cast<const uint4*>(a.codes + (size_t)t * 1024) + i);
                else if (i < 260) v = __ldcg(reinterpret_cast<const uint4*>(a.lo + (size_t)t * 32) + (i - 256));
                else v = __ldcg(reinterpret_cast<const uint4*>(a.st + (size_t)t * 32) + (i - 260));
            } else v = __ldcg(reinterpret_cast<const uint4*>(a.tab + ((size_t)t << K) * 32) + i);
        }
        r[q] = v;
    }
}

template <int K, int W>
__global__ void __launch_bounds__(W * 32) ar12_kernel(A12 a) {
    constexpr int NT = W * 32, PER = (Cfg12<K>::SRC16 + NT - 1) / NT;
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int TB = Cfg12<K>::TBYTES;
    float* Tf = reinterpret_cast<float*>(smem);
    float* red = reinterpret_cast<float*>(smem + TB);                       // [W][32]
    uint4* stg = reinterpret_cast<uint4*>(red + W * 32);                    // table source staging
    __half* xs = reinterpret_cast<__half*>(stg + Cfg12<K>::SRC16);          // full x, fp16
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long N = (long long)a.ntile * a.nchunk;
    long long it = (long long)blockIdx.x * N / gridDim.x;
    const long long it_end = (long long)(blockIdx.x + 1) * N / gridDim.x;
    const int ncols = a.nchunk * 64;
    for (int i = threadIdx.x * 8; i < ncols; i += NT * 8) *reinterpret_cast<uint4*>(xs + i) = *reinterpret_cast<const uint4*>(a.x + i);
    uint4 srcr[PER];
    if (it < it_end) src_load<K, NT, PER>(a, (int)(it / a.nchunk), srcr);
    uint32_t base0; asm volatile("mov.u32 %0, %1;" : "=r"(base0) : "r"(K == 3 ? lane * 8u : lane * 4u) : "memory");
    while (it < it_end) {
        const int t = (int)(it / a.nchunk), cs = (int)(it - (long long)t * a.nchunk);
        const int ce = (int)min((long long)a.nchunk, cs + (it_end - it));
        const size_t tb = (size_t)t * a.nchunk * 32 + lane;
        int c = cs + warp;
        uint2 un = make_uint2(0, 0); uint32_t hn[K];
#pragma unroll
        for (int p = 0; p < K; p++) hn[p] = 0u;
        auto load = [&](int cc) {
            un = __ldcs(a.U + tb + (size_t)cc * 32);
#pragma unroll
            for (int p = 1; p < K; p++) hn[p] = __ldcs(a.H[p - 1] + tb + (size_t)cc * 32);
        };
        if (c < ce) load(c);
        // table of tile t from the prefetched source
        __syncthreads();                                                    // previous tile: table and red no longer in use
#pragma unroll
        for (int q = 0; q < PER; q++) { const int i = threadIdx.x + q * NT; if (i < Cfg12<K>::SRC16) stg[i] = srcr[q]; }
        __syncthreads();
        if constexpr (K == 7) {
            const uint32_t* cw = reinterpret_cast<const uint32_t*>(stg);    // codes [32 (j/4)][32 lanes]
            const __half* lo = reinterpret_cast<const __half*>(stg + 256); const __half* st = lo + 32;
            for (int i = threadIdx.x; i < 1024; i += NT) {
                const int j4 = i >> 5, l = i & 31; const uint32_t c4 = cw[i];
                const float flo = __half2float(lo[l]), fst = __half2float(st[l]);
#pragma unroll
                for (int q = 0; q < 4; q++) Tf[(j4 * 4 + q) * 64 + l] = __half2float(__float2half(flo + (float)((c4 >> (8 * q)) & 0xffu) * fst));
            }
        } else if constexpr (K == 3) {
            const __half* tg = reinterpret_cast<const __half*>(stg);
            float2* T2 = reinterpret_cast<float2*>(Tf);
            for (int i = threadIdx.x; i < 120 * 32; i += NT) {
                const int e = i >> 5, l = i & 31, lo4 = e & 15, hi4 = e >> 4;
                T2[i] = (lo4 < 8 && hi4 < 8) ? make_float2(__half2float(tg[lo4 * 32 + l]), __half2float(tg[hi4 * 32 + l])) : make_float2(0.f, 0.f);
            }
        } else {
            const __half* tg = reinterpret_cast<const __half*>(stg);
            for (int i = threadIdx.x; i < (32 << K); i += NT) Tf[(i >> 5) * 64 + (i & 31)] = __half2float(tg[i]);
        }
        // prefetch the next tile's table source (arrives while this tile streams)
        if (ce == a.nchunk && it + (ce - cs) < it_end) src_load<K, NT, PER>(a, t + 1, srcr);
        __syncthreads();
        uint32_t base; asm volatile("mov.u32 %0, %1;" : "=r"(base) : "r"(base0) : "memory");
        float acc = 0.f, acc2 = 0.f;
        for (; c < ce; c += W) {
            const uint2 u = un; uint32_t hw[K];
#pragma unroll
            for (int p = 1; p < K; p++) hw[p] = hn[p];
            if (c + W < ce) load(c + W);
            do_chunk<K>(u, hw, xs + c * 64, base, acc, acc2);
        }
        red[warp * 32 + lane] = acc + acc2;
        __syncthreads();
        if (warp == 0) {
            float v = 0.f;
            for (int w = 0; w < W; w++) v += red[w * 32 + lane];
            const int row = t * 32 + lane;
            if (cs == 0 && ce == a.nchunk) { if (row < a.rows) a.y[row] = __float2half(v); }
            else {
                atomicAdd(a.ws + row, v);
                __threadfence();
                int prev = 0;
                if (lane == 0) prev = atomicAdd(a.cnt + t, ce - cs);
                prev = __shfl_sync(0xffffffffu, prev, 0);
                if (prev + (ce - cs) == a.nchunk) {
                    __threadfence();
                    const float sum = __ldcg(a.ws + row);
                    if (row < a.rows) a.y[row] = __float2half(sum);
                    a.ws[row] = 0.f;
                    if (lane == 0) a.cnt[t] = 0;
                }
            }
        }
        it += ce - cs;
    }
}

template <int K, int W>
static int launch(const A12& a, int grid, cudaStream_t st) {
    size_t sm = (size_t)Cfg12<K>::TBYTES + W * 32 * 4 + Cfg12<K>::SRC16 * 16 + (size_t)a.nchunk * 64 * 2;
    static size_t cur = 0; static int occ = 0; static int cur_cols = -1;
    if (sm > cur) { if (cudaFuncSetAttribute(ar12_kernel<K, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) return -5; cur = sm; }
    if (cur_cols != a.nchunk) { cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ar12_kernel<K, W>, W * 32, sm); cur_cols = a.nchunk; }
    if (occ < 1) return -7;
    int dev, nsm; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    int G = grid > 0 ? grid : nsm * occ;
    ar12_kernel<K, W><<<G, W * 32, sm, st>>>(a);
    return (int)cudaGetLastError();
}
template <int W>
static int launch_k(const A12& a, int k, int grid, cudaStream_t st) {
    switch (k) { case 3: return launch<3, W>(a, grid, st); case 4: return launch<4, W>(a, grid, st); case 5: return launch<5, W>(a, grid, st);
                 case 6: return launch<6, W>(a, grid, st); case 7: return launch<7, W>(a, grid, st); }
    return -1;
}
extern "C" int ar12_gemv(const void* U, const void* const* H, const void* tab, const void* codes, const void* lo, const void* st,
                         const void* x, void* y, void* ws, void* cnt, int rows, int cols, int k, int grid, int W, void* stream) {
    static int reserved = -1;
    if (reserved < 0) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev); }
    if (reserved != 1024) return -6;
    if (cols % 64) return -2;
    A12 a; a.U = (const uint2*)U;
    for (int p = 0; p < 6; p++) a.H[p] = (const uint32_t*)H[p];
    a.tab = (const __half*)tab; a.codes = (const uint32_t*)codes; a.lo = (const __half*)lo; a.st = (const __half*)st;
    a.x = (const __half*)x; a.y = (__half*)y; a.ws = (float*)ws; a.cnt = (int*)cnt;
    a.rows = rows; a.rows_pad = (rows + 31) / 32 * 32; a.ntile = a.rows_pad / 32; a.nchunk = cols / 64;
    cudaStream_t s = (cudaStream_t)stream;
    switch (W) { case 4: return launch_k<4>(a, k, grid, s); case 8: return launch_k<8>(a, k, grid, s); case 16: return launch_k<16>(a, k, grid, s); case 32: return launch_k<32>(a, k, grid, s); }
    return -4;
}
