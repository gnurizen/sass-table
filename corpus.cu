// corpus.cu — CUDA kernel corpus designed to exercise the widest possible
// range of SASS instructions across SM75+.
//
// Each kernel targets a different instruction family. We use volatile loads/
// stores and __device__ noinline to prevent the compiler from optimizing
// away the instructions we care about.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// Helpers to force the compiler to keep values alive.
template <typename T>
__device__ __forceinline__ void use(T v) {
    asm volatile("" ::"r"((int)v));
}
__device__ __forceinline__ void use64(double v) {
    asm volatile("" ::"d"(v));
}
__device__ __forceinline__ void use64l(long long v) {
    asm volatile("" ::"l"(v));
}

// ---------------------------------------------------------------------------
// 1. Integer arithmetic: IADD3, IMAD, IABS, IMNMX, LEA, LOP3, SHF, SHR, SHL,
//    POPC, FLO, BREV
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_int_arith(int *out, const int *a, const int *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx], y = b[idx];

    int sum = x + y;
    int diff = x - y;
    int prod = x * y;
    int mad = x * y + sum;
    int mn = min(x, y);
    int mx = max(x, y);
    int absx = abs(x);
    int lop = (x & y) | (~x & y) ^ (x | y);
    int sh_l = x << (y & 31);
    int sh_r = x >> (y & 31);
    unsigned ux = (unsigned)x;
    int popc_val = __popc(ux);
    int flo_val = __clz(ux);
    int brev_val = __brev(ux);

    out[idx] = sum + diff + prod + mad + mn + mx + absx + lop + sh_l + sh_r +
               popc_val + flo_val + brev_val;
}

// ---------------------------------------------------------------------------
// 2. 64-bit integer: IADD3 wide, IMAD wide
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_int64(long long *out, const long long *a,
                                       const long long *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long x = a[idx], y = b[idx];
    long long sum = x + y;
    long long prod = x * y;
    long long mad = x * y + sum;
    long long sh = x << (y & 63);
    out[idx] = sum + prod + mad + sh;
}

// ---------------------------------------------------------------------------
// 3. FP32: FADD, FMUL, FFMA, FMNMX, FSETP, FABS, FNEG, FRND
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fp32(float *out, const float *a,
                                      const float *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[idx], y = b[idx];

    float sum = x + y;
    float diff = x - y;
    float prod = x * y;
    float fma_val = fmaf(x, y, sum);
    float mn = fminf(x, y);
    float mx = fmaxf(x, y);
    float absx = fabsf(x);
    float neg = -x;
    float rnd = rintf(x);
    float flr = floorf(x);
    float cel = ceilf(x);
    float trn = truncf(x);

    // Comparison / select
    float sel = (x > y) ? sum : diff;

    out[idx] = sum + diff + prod + fma_val + mn + mx + absx + neg + rnd + flr +
               cel + trn + sel;
}

// ---------------------------------------------------------------------------
// 4. FP64: DADD, DMUL, DFMA, DMNMX, DSETP
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fp64(double *out, const double *a,
                                      const double *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double x = a[idx], y = b[idx];

    double sum = x + y;
    double diff = x - y;
    double prod = x * y;
    double fma_val = fma(x, y, sum);
    double mn = fmin(x, y);
    double mx = fmax(x, y);
    double absx = fabs(x);
    double neg = -x;
    double rnd = rint(x);
    double flr = floor(x);
    double cel = ceil(x);

    out[idx] = sum + diff + prod + fma_val + mn + mx + absx + neg + rnd + flr + cel;
}

// ---------------------------------------------------------------------------
// 5. Half precision: HADD2, HMUL2, HFMA2
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fp16(__half2 *out, const __half2 *a,
                                      const __half2 *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 x = a[idx], y = b[idx];

    __half2 sum = __hadd2(x, y);
    __half2 prod = __hmul2(x, y);
    __half2 fma_val = __hfma2(x, y, sum);

    out[idx] = __hadd2(__hadd2(sum, prod), fma_val);
}

// ---------------------------------------------------------------------------
// 6. Special functions (MUFU): sin, cos, exp2, lg2, rcp, rsqrt, sqrt
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_mufu(float *out, const float *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[idx];

    float s = __sinf(x);
    float c = __cosf(x);
    float e = exp2f(x);
    float l = __log2f(x);
    float r = __frcp_rn(x);
    float rs = rsqrtf(x);
    float sq = sqrtf(x);
    float ex = __expf(x);

    out[idx] = s + c + e + l + r + rs + sq + ex;
}

// ---------------------------------------------------------------------------
// 7. Conversions: I2F, F2I, I2I, F2F
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_convert(float *fout, int *iout,
                                         const float *fin, const int *iin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float fv = fin[idx];
    int iv = iin[idx];

    // F2I
    int f2i_rn = __float2int_rn(fv);
    int f2i_rz = __float2int_rz(fv);
    int f2i_ru = __float2int_ru(fv);
    int f2i_rd = __float2int_rd(fv);
    unsigned f2u = __float2uint_rn(fv);

    // I2F
    float i2f_rn = __int2float_rn(iv);
    float i2f_rz = __int2float_rz(iv);
    float u2f = __uint2float_rn((unsigned)iv);

    // F2F (double<->float)
    double d = (double)fv;
    float f = (float)d;

    iout[idx] = f2i_rn + f2i_rz + f2i_ru + f2i_rd + (int)f2u;
    fout[idx] = i2f_rn + i2f_rz + u2f + f + (float)d;
}

// ---------------------------------------------------------------------------
// 8. MOV, SEL, PRMT, SHFL
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_mov_sel(int *out, const int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx];
    int lane = threadIdx.x & 31;

    // MOV
    int r = x;
    // SEL
    int s = (lane < 16) ? x : -x;
    // PRMT — byte permute
    int p = __byte_perm(x, ~x, 0x3210);
    // SHFL
    int shfl_up = __shfl_up_sync(0xffffffff, x, 1);
    int shfl_dn = __shfl_down_sync(0xffffffff, x, 1);
    int shfl_xor = __shfl_xor_sync(0xffffffff, x, 1);
    int shfl_idx = __shfl_sync(0xffffffff, x, 0);

    out[idx] = r + s + p + shfl_up + shfl_dn + shfl_xor + shfl_idx;
}

// ---------------------------------------------------------------------------
// 9. Warp vote / match / redux
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_warp_vote(int *out, const int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx];

    int any_v = __any_sync(0xffffffff, x > 0);
    int all_v = __all_sync(0xffffffff, x > 0);
    unsigned ballot = __ballot_sync(0xffffffff, x > 0);
    unsigned match = __match_any_sync(0xffffffff, x);

#if __CUDA_ARCH__ >= 800
    // REDUX (SM80+)
    int redux_add = __reduce_add_sync(0xffffffff, x);
    int redux_min = __reduce_min_sync(0xffffffff, x);
    int redux_max = __reduce_max_sync(0xffffffff, x);
    int redux_and = __reduce_and_sync(0xffffffff, x);
    int redux_or  = __reduce_or_sync(0xffffffff, x);
    out[idx] = any_v + all_v + (int)ballot + (int)match +
               redux_add + redux_min + redux_max + redux_and + redux_or;
#else
    out[idx] = any_v + all_v + (int)ballot + (int)match;
#endif
}

// ---------------------------------------------------------------------------
// 10. Shared memory: LDS, STS, LDSM (if possible via inline asm)
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_shared(int *out, const int *a) {
    __shared__ int smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // STS
    smem[tid] = a[idx];
    __syncthreads();

    // LDS
    int v0 = smem[tid];
    int v1 = smem[(tid + 1) & 255];
    int v2 = smem[(tid + 2) & 255];
    int v3 = smem[(tid + 3) & 255];
    __syncthreads();

    // Shared atomics
    atomicAdd(&smem[tid & 15], v0);
    __syncthreads();

    out[idx] = v0 + v1 + v2 + v3 + smem[tid & 15];
}

// ---------------------------------------------------------------------------
// 11. Global memory: LDG, STG, LDL, STL, various widths
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_global_mem(int4 *out4, const int4 *a4,
                                            const int *a1, const int2 *a2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 32-bit load
    int v1 = a1[idx];
    // 64-bit load
    int2 v2 = a2[idx];
    // 128-bit load
    int4 v4 = a4[idx];

    // 128-bit store
    int4 result;
    result.x = v4.x + v2.x + v1;
    result.y = v4.y + v2.y;
    result.z = v4.z;
    result.w = v4.w;
    out4[idx] = result;
}

// ---------------------------------------------------------------------------
// 12. Atomics: ATOM, RED (global)
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_atomics(int *out, unsigned long long *out64,
                                         const int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx];

    atomicAdd(out, x);
    atomicSub(out + 1, x);
    atomicMin(out + 2, x);
    atomicMax(out + 3, x);
    atomicAnd((unsigned *)out + 4, (unsigned)x);
    atomicOr((unsigned *)out + 5, (unsigned)x);
    atomicXor((unsigned *)out + 6, (unsigned)x);
    atomicExch(out + 7, x);
    atomicCAS(out + 8, 0, x);

    // 64-bit atomics
    atomicAdd(out64, (unsigned long long)x);
    atomicExch(out64 + 1, (unsigned long long)x);
    atomicCAS(out64 + 2, 0ULL, (unsigned long long)x);
}

// ---------------------------------------------------------------------------
// 13. Control flow: BRA, BSSY, BSYNC, EXIT, RET, CALL (via device func)
// ---------------------------------------------------------------------------
__device__ __noinline__ int device_helper(int x) {
    return x * x + 1;
}

extern "C" __global__ void kern_control_flow(int *out, const int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx];

    // Branch
    int r;
    if (x > 0) {
        r = device_helper(x);
    } else if (x < -10) {
        r = device_helper(-x);
    } else {
        r = 0;
    }

    // Loop
    for (int i = 0; i < (x & 7); i++) {
        r += i;
    }

    // Switch
    switch (x & 3) {
    case 0: r += 10; break;
    case 1: r += 20; break;
    case 2: r += 30; break;
    case 3: r += 40; break;
    }

    out[idx] = r;
}

// ---------------------------------------------------------------------------
// 14. System registers: S2R (SR_TID, SR_CTAID, SR_LANEID, SR_CLOCK, etc.)
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_sysreg(int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bid_z = blockIdx.z;
    int ntid_x = blockDim.x;
    int ntid_y = blockDim.y;

    unsigned clk;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(clk));

    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    unsigned laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));

    unsigned warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));

    out[idx] = tid_x + tid_y + tid_z + bid_x + bid_y + bid_z + ntid_x +
               ntid_y + (int)clk + (int)smid + (int)laneid + (int)warpid;
}

// ---------------------------------------------------------------------------
// 15. Predicate ops: ISETP, FSETP, PLOP3, P2R, R2P
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_predicates(int *out, const int *a,
                                            const float *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = a[idx];
    float fx = b[idx];

    // Integer comparisons
    int eq = (ix == 42);
    int ne = (ix != 0);
    int lt = (ix < 100);
    int gt = (ix > -100);
    int le = (ix <= 50);
    int ge = (ix >= -50);

    // Float comparisons
    int feq = (fx == 0.0f);
    int flt = (fx < 1.0f);
    int fgt = (fx > -1.0f);
    int fnan = isnan(fx);

    out[idx] = eq + ne + lt + gt + le + ge + feq + flt + fgt + fnan;
}

// ---------------------------------------------------------------------------
// 16. LEA (load effective address) — array indexing patterns
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_lea(int *out, const int *a, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / stride;
    int col = idx % stride;

    // LEA instructions typically generated for address calculations
    int v0 = a[row * stride + col];
    int v1 = a[(row + 1) * stride + col];
    int v2 = a[row * stride + col + 1];

    out[idx] = v0 + v1 + v2;
}

// ---------------------------------------------------------------------------
// 17. Uniform datapath: UMOV, UIADD3, ULOP3, ULEPC, etc. (SM75+)
//     These are triggered by uniform (non-divergent) operations.
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_uniform(int *out, const int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Uniform operations (same value across warp)
    int block_base = blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    int niters = (n + grid_size - 1) / grid_size;

    int sum = 0;
    for (int i = 0; i < niters; i++) {
        int global_idx = block_base + i * grid_size + threadIdx.x;
        if (global_idx < n) {
            sum += a[global_idx];
        }
    }
    out[idx] = sum;
}

// ---------------------------------------------------------------------------
// 18. Texture / surface ops (TEX, TLD, SUST — limited without actual texture
//     objects, but we can exercise the instructions via inline asm stubs)
// ---------------------------------------------------------------------------
// We use a placeholder kernel; real texture instructions need texture objects
// which require a running GPU. The compiler still emits the instruction
// sequences if we reference texture intrinsics.
// (Skipping actual texture objects — these will be present in real workloads.)

// ---------------------------------------------------------------------------
// 19. CS2R, S2UR — SM80+ special register reads
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_special_regs(unsigned long long *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long clk64;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk64));

    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(globaltimer));

    out[idx] = clk64 + globaltimer;
}

// ---------------------------------------------------------------------------
// 20. Barrier / sync: BAR, MEMBAR, DEPBAR
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_barriers(int *out, const int *a) {
    __shared__ int smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    smem[tid] = a[idx];
    __syncthreads();  // BAR.SYNC

    __threadfence_block();  // MEMBAR.CTA
    __threadfence();        // MEMBAR.GL

    int v = smem[(tid + 1) & 255];
    __syncthreads();

    out[idx] = v;
}

// ---------------------------------------------------------------------------
// 21. DP4A, DP2A — integer dot product (SM75+)
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_dp4a(int *out, const int *a, const int *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = a[idx], y = b[idx];

    int dp4a_val = __dp4a(x, y, 0);
    int dp2a_lo = __dp2a_lo(x, y, 0);
    int dp2a_hi = __dp2a_hi(x, y, 0);

    out[idx] = dp4a_val + dp2a_lo + dp2a_hi;
}

// ---------------------------------------------------------------------------
// 22. FP32 fused operations and more MUFU variants
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fp_extra(float *out, const float *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[idx];

    float tanh_v = tanhf(x);
    float atan_v = atanf(x);
    float asin_v = asinf(x);
    float acos_v = acosf(x);
    float pow_v = powf(x, 2.5f);
    float log_v = logf(x + 1.0f);
    float log10_v = log10f(x + 1.0f);
    float exp_v = expf(x);
    float cbrt_v = cbrtf(x);
    float hypot_v = hypotf(x, x + 1.0f);

    out[idx] = tanh_v + atan_v + asin_v + acos_v + pow_v + log_v + log10_v +
               exp_v + cbrt_v + hypot_v;
}

// ---------------------------------------------------------------------------
// 23. Constant memory loads (LDC)
// ---------------------------------------------------------------------------
__constant__ int const_data[256];

extern "C" __global__ void kern_const_mem(int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v = const_data[idx & 255];
    int v2 = const_data[(idx + 1) & 255];
    int v3 = const_data[(idx + 2) & 255];
    out[idx] = v + v2 + v3;
}

// ---------------------------------------------------------------------------
// 24. BMMA / IMMA / HMMA — tensor core via wmma API (SM75+)
// ---------------------------------------------------------------------------
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void kern_wmma_fp16(half *d_a, half *d_b, float *d_c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::load_matrix_sync(a_frag, d_a, 16);
    wmma::load_matrix_sync(b_frag, d_b, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(d_c, c_frag, 16, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// 25. IMMA — int8 tensor core (SM75+)
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_wmma_int8(signed char *d_a, signed char *d_b,
                                           int *d_c) {
    wmma::fragment<wmma::matrix_a, 8, 32, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 32, 16, int> c_frag;

    wmma::load_matrix_sync(a_frag, d_a, 16);
    wmma::load_matrix_sync(b_frag, d_b, 32);
    wmma::fill_fragment(c_frag, 0);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(d_c, c_frag, 32, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// 26. BFI, BFE — bit field insert / extract
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_bitfield(unsigned *out, const unsigned *a,
                                          const unsigned *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = a[idx], y = b[idx];

    unsigned bfe_val;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bfe_val) : "r"(x), "r"(8), "r"(8));

    unsigned bfi_val;
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(bfi_val)
        : "r"(x), "r"(y), "r"(8), "r"(8));

    out[idx] = bfe_val + bfi_val;
}

// ---------------------------------------------------------------------------
// 27. SULD / SUST / SURED — surface operations via inline asm
//     (compiler emits these with surface reference)
// ---------------------------------------------------------------------------
// Skipping — requires surface objects and a running GPU context.

// ---------------------------------------------------------------------------
// 28. FP32 with rounding modes
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fp_rounding(float *out, const float *a,
                                             const float *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[idx], y = b[idx];

    float add_rn, add_rz, mul_rn, mul_rz;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_rn) : "f"(x), "f"(y));
    asm("add.rz.f32 %0, %1, %2;" : "=f"(add_rz) : "f"(x), "f"(y));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(mul_rn) : "f"(x), "f"(y));
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(mul_rz) : "f"(x), "f"(y));

    out[idx] = add_rn + add_rz + mul_rn + mul_rz;
}
