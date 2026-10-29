// ============================================================================
// batch_spmv.cu — 批量 SpMV 常驻内核基准（策略①：连续批处理）
//
// 问题（mp_gpu_bench.json 实测）：n≤5k 的 SpMV fp32/fp64 收益 ≈1.0×
//   —— µs 级内核，启动/调度延迟主导（与 LLM decode 侧"小请求启动贵"同构）。
// 方案：把 k 个"结构相同、数值不同"的小矩阵放进**一次 kernel 启动**
//   （grid = k 个 block，每 block 处理一个矩阵）→ 摊薄启动开销。
//
// 编译（race7 根）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -o batch_spmv_bench \
//     src/solver/mp_bench/batch_spmv.cu
// 用法：./batch_spmv_bench <s0.mtx> [k_max] [iters]
// 输出：k=1/8/16/32 的 单启动批量 vs 循环单启动 总耗时；fp64 vs fp32 批内收益
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

typedef struct { int n, nnz; int *rowptr, *col; double *val; } s0mat;

static int read_s0(const char *path, s0mat *M)
{
    FILE *f = fopen(path, "r"); char line[512];
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    long cap = 1024;
    int *cnt = (int*)calloc(M->n + 1, sizeof(int));
    M->col = (int*)malloc(cap * sizeof(int));
    M->val = (double*)malloc(cap * sizeof(double));
    M->nnz = 0;
    rewind(f); fgets(line, sizeof line, f);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (M->nnz >= cap) { cap *= 2; M->col = (int*)realloc(M->col, cap*sizeof(int));
                             M->val = (double*)realloc(M->val, cap*sizeof(double)); }
        M->col[M->nnz] = j - 1; M->val[M->nnz] = v; M->nnz++;
        cnt[i - 1]++;
    }
    fclose(f);
    M->rowptr = (int*)malloc((size_t)(M->n + 1) * sizeof(int));
    M->rowptr[0] = 0;
    for (int i = 0; i < M->n; i++) M->rowptr[i + 1] = M->rowptr[i] + cnt[i];
    free(cnt);
    return 0;
}

// ---- 单矩阵 SpMV（grid=1，供"循环单启动"基线）----
template <typename VT>
__global__ void spmv_one(const int n, const int *rowptr, const int *col,
                         const VT *val, const double *x, double *y)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = 0;
        for (int k = rowptr[i]; k < rowptr[i + 1]; k++) acc += (double)val[k] * x[col[k]];
        y[i] = acc;
    }
}

// ---- 批量 SpMV：一次启动处理 k 个同构矩阵（grid=k，每 block 一个矩阵）----
// val 布局: [k][nnz] 连续; x/y: [k][n] 连续
template <typename VT>
__global__ void spmv_batch(const int n, const int nnz, const int k,
                           const int *rowptr, const int *col,
                           const VT *val, const double *x, double *y)
{
    int b = blockIdx.x;                     // 矩阵 b
    const VT *v = val + (size_t)b * nnz;
    const double *xv = x + (size_t)b * n;
    double *yv = y + (size_t)b * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = 0;
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++) acc += (double)v[j] * xv[col[j]];
        yv[i] = acc;
    }
}

static double bench_loop(int n, int nnz, int k, int iters, const int *rp, const int *col,
                         const double *d_val64, const float *d_val32,
                         const double *d_x, double *d_y, int fp32)
{
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++)
        for (int b = 0; b < k; b++) {
            if (fp32) spmv_one<float><<<1, 256>>>(n, rp, col, d_val32 + (size_t)b * nnz, d_x + (size_t)b * n, d_y + (size_t)b * n);
            else      spmv_one<double><<<1, 256>>>(n, rp, col, d_val64 + (size_t)b * nnz, d_x + (size_t)b * n, d_y + (size_t)b * n);
        }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
    return ms / iters;
}

static double bench_batch(int n, int nnz, int k, int iters, const int *rp, const int *col,
                          const double *d_val64, const float *d_val32,
                          const double *d_x, double *d_y, int fp32)
{
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++) {
        if (fp32) spmv_batch<float><<<k, 256>>>(n, nnz, k, rp, col, d_val32, d_x, d_y);
        else      spmv_batch<double><<<k, 256>>>(n, nnz, k, rp, col, d_val64, d_x, d_y);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
    return ms / iters;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <s0.mtx> [k_max] [iters]\n", argv[0]); return 1; }
    s0mat M; if (read_s0(argv[1], &M)) { fprintf(stderr, "read fail %s\n", argv[1]); return 1; }
    int k_max = argc > 2 ? atoi(argv[2]) : 32;
    int iters = argc > 3 ? atoi(argv[3]) : 50;
    int n = M.n, nnz = M.nnz;

    /* 生成 k_max 个同构数值变体：值扰动 ±10%（结构不变） */
    double *h_val64 = (double*)malloc((size_t)k_max * nnz * 8);
    float  *h_val32 = (float*)malloc((size_t)k_max * nnz * 4);
    double *h_x = (double*)malloc((size_t)k_max * n * 8);
    double *h_y = (double*)malloc((size_t)k_max * n * 8);
    unsigned seed = 42;
    for (int b = 0; b < k_max; b++) {
        double *v64 = h_val64 + (size_t)b * nnz;
        float  *v32 = h_val32 + (size_t)b * nnz;
        double *xb = h_x + (size_t)b * n;
        for (int j = 0; j < nnz; j++) {
            double p = 1.0 + 0.2 * (((double)rand_r(&seed) / RAND_MAX) - 0.5);
            v64[j] = M.val[j] * p; v32[j] = (float)v64[j];
        }
        for (int i = 0; i < n; i++) xb[i] = 1.0 + 0.1 * sin(i * 0.01 + b);
    }

    int *d_rp, *d_col; double *d_val64, *d_x, *d_y; float *d_val32;
    CK(cudaMalloc(&d_rp, (size_t)(n+1)*4)); CK(cudaMalloc(&d_col, (size_t)nnz*4));
    CK(cudaMalloc(&d_val64, (size_t)k_max*nnz*8)); CK(cudaMalloc(&d_val32, (size_t)k_max*nnz*4));
    CK(cudaMalloc(&d_x, (size_t)k_max*n*8)); CK(cudaMalloc(&d_y, (size_t)k_max*n*8));
    CK(cudaMemcpy(d_rp, M.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_col, M.col, (size_t)nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val64, h_val64, (size_t)k_max*nnz*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val32, h_val32, (size_t)k_max*nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, h_x, (size_t)k_max*n*8, cudaMemcpyHostToDevice));

    /* 正确性（k=2, fp64 批 vs scipy 参考——用主机参考） */
    {
        spmv_batch<double><<<2, 256>>>(n, nnz, 2, d_rp, d_col, d_val64, d_x, d_y);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_y, d_y, (size_t)2*n*8, cudaMemcpyDeviceToHost));
        double err = 0, nrm = 0;
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < n; i++) {
                double ref = 0;
                const double *v64 = h_val64 + (size_t)b * nnz;
                const double *xb = h_x + (size_t)b * n;
                for (int j = M.rowptr[i]; j < M.rowptr[i+1]; j++) ref += v64[j] * xb[M.col[j]];
                double d = h_y[b*n+i] - ref; err += d*d; nrm += ref*ref;
            }
        printf("== batch_spmv %s n=%d nnz=%d ==\n正确性(批k=2 vs 主机参考): rel_err=%.2e %s\n",
               argv[1], n, nnz, sqrt(err/nrm), sqrt(err/nrm) < 1e-10 ? "✅" : "❌");
    }

    printf("%-5s | %-16s | %-16s | %-10s | %-10s\n", "k", "循环单启动(ms)", "批量单启动(ms)", "批量/循环", "批内fp32/fp64");
    for (int k : {1, 8, 16, 32}) {
        if (k > k_max) continue;
        double tl64 = bench_loop(n, nnz, k, iters, d_rp, d_col, d_val64, d_val32, d_x, d_y, 0);
        double tb64 = bench_batch(n, nnz, k, iters, d_rp, d_col, d_val64, d_val32, d_x, d_y, 0);
        double tl32 = bench_loop(n, nnz, k, iters, d_rp, d_col, d_val64, d_val32, d_x, d_y, 1);
        double tb32 = bench_batch(n, nnz, k, iters, d_rp, d_col, d_val64, d_val32, d_x, d_y, 1);
        printf("%-5d | %-16.4f | %-16.4f | %-10.2f | %-10.2f\n",
               k, tl64, tb64, tb64/tl64, tb64/tb32);
    }
    return 0;
}
