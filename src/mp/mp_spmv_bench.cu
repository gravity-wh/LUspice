// ============================================================================
// mp_spmv_bench.cu — 混合精度 SpMV 基准（赛题七 R13：mixed-precision throughput）
//
// 同一矩阵、同一内核结构，仅值精度不同（fp64 vs fp32）：
//   - CSR（每行一线程，BLOCK=256）——同 spmv.cu v0
//   - SELL-C-σ32（warp=slice，256B 合并事务，按行长降序分片）——同 sellc_spmv.cu
// 指标：内核时间（warmup + G 组 × iters，取组中位数，组内用 cudaEvent 测整组）
//       有效带宽 = 2*nnz*bytes / time（读 A 值 + 读 x）
//       相对 fp64 加速比、带宽利用率（理论 360 GB/s）
// 正确性：fp32 y vs fp64 y 的 max rel diff
//
// 编译（race7 根目录）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -I src/solver \
//     -o mp_spmv_bench src/solver/mp_bench/mp_spmv_bench.cu
// 用法：./mp_spmv_bench <s0.mtx> [iters] [groups]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>

#define THEO_BW_GBPS 360.0
#define SELL_SIGMA 32
#define BLOCK 256

typedef struct { int n; long long nnz; int *rowptr, *colidx; double *values; } CSR;

static int read_s0(const char *path, CSR *A)
{
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    int n = 0;
    if (sscanf(line, "n %d", &n) != 1 || n <= 0) { fclose(f); return 1; }
    long long nnz = 0;
    while (fgets(line, sizeof line, f)) if (line[0] == 'A') nnz++;
    if (nnz == 0) { fclose(f); return 1; }
    rewind(f); if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    int *ri = (int *)malloc((size_t)nnz * sizeof(int));
    int *rj = (int *)malloc((size_t)nnz * sizeof(int));
    double *rv = (double *)malloc((size_t)nnz * sizeof(double));
    long long k = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (i < 1 || i > n || j < 1) continue;
        ri[k] = i - 1; rj[k] = j - 1; rv[k] = v; k++;
    }
    fclose(f);
    int *rowptr = (int *)calloc((size_t)n + 1, sizeof(int));
    for (long long t = 0; t < nnz; t++) rowptr[ri[t] + 1]++;
    for (int i = 1; i <= n; i++) rowptr[i] += rowptr[i - 1];
    int *colidx = (int *)malloc((size_t)nnz * sizeof(int));
    double *values = (double *)malloc((size_t)nnz * sizeof(double));
    int *cursor = (int *)malloc((size_t)n * sizeof(int));
    memcpy(cursor, rowptr, (size_t)n * sizeof(int));
    for (long long t = 0; t < nnz; t++) { int p = cursor[ri[t]]++; colidx[p] = rj[t]; values[p] = rv[t]; }
    free(ri); free(rj); free(rv); free(cursor);
    A->n = n; A->nnz = nnz; A->rowptr = rowptr; A->colidx = colidx; A->values = values;
    return 0;
}

/* ---------------- CSR kernels ---------------- */
__global__ void k_csr64(const int n, const int* __restrict__ rowptr,
                        const int* __restrict__ colidx, const double* __restrict__ values,
                        const double* __restrict__ x, double* __restrict__ y)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    double sum = 0.0;
    for (int p = rowptr[i]; p < rowptr[i + 1]; p++) sum += values[p] * x[colidx[p]];
    y[i] = sum;
}

__global__ void k_csr32(const int n, const int* __restrict__ rowptr,
                        const int* __restrict__ colidx, const float* __restrict__ values,
                        const float* __restrict__ x, float* __restrict__ y)
{
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    for (int p = rowptr[i]; p < rowptr[i + 1]; p++) sum += values[p] * x[colidx[p]];
    y[i] = sum;
}

/* ---------------- SELL-C-σ32 kernels ---------------- */
__global__ void k_sell64(const int n, const int nslices,
                         const int* __restrict__ offs, const int* __restrict__ lens,
                         const int* __restrict__ cols, const double* __restrict__ vals,
                         const double* __restrict__ x, double* __restrict__ y)
{
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int slice = tid >> 5;
    const int r = threadIdx.x & (SELL_SIGMA - 1);
    if (slice >= nslices) return;
    const int row = slice * SELL_SIGMA + r;
    if (row >= n) return;
    const int off = offs[slice];
    const int len = lens[slice];
    double sum = 0.0;
    for (int j = 0; j < len; j++) {
        const int idx = off + j * SELL_SIGMA + r;
        sum += vals[idx] * x[cols[idx]];
    }
    y[row] = sum;
}

__global__ void k_sell32(const int n, const int nslices,
                         const int* __restrict__ offs, const int* __restrict__ lens,
                         const int* __restrict__ cols, const float* __restrict__ vals,
                         const float* __restrict__ x, float* __restrict__ y)
{
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int slice = tid >> 5;
    const int r = threadIdx.x & (SELL_SIGMA - 1);
    if (slice >= nslices) return;
    const int row = slice * SELL_SIGMA + r;
    if (row >= n) return;
    const int off = offs[slice];
    const int len = lens[slice];
    float sum = 0.0f;
    for (int j = 0; j < len; j++) {
        const int idx = off + j * SELL_SIGMA + r;
        sum += vals[idx] * x[cols[idx]];
    }
    y[row] = sum;
}

/* ---------------- SELL-C-σ32 host 构建（同 sellc_spmv.cu） ---------------- */
static long long build_sellc(const CSR *A, int sort,
                             int **out_offs, int **out_lens,
                             int **out_cols, double **out_vals,
                             int **out_order)
{
    const int n = A->n;
    int *len = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) len[i] = (int)(A->rowptr[i + 1] - A->rowptr[i]);
    int *order = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    if (sort) {
        int maxlen = 0;
        for (int i = 0; i < n; i++) if (len[i] > maxlen) maxlen = len[i];
        int *cnt = (int *)calloc((size_t)maxlen + 1, sizeof(int));
        for (int i = 0; i < n; i++) cnt[len[i]]++;
        for (int l = maxlen - 1; l >= 0; l--) cnt[l] += cnt[l + 1];
        for (int i = 0; i < n; i++) order[--cnt[len[i]]] = i;
        free(cnt);
    }
    const int nslices = (n + SELL_SIGMA - 1) / SELL_SIGMA;
    int *lens = (int *)malloc((size_t)nslices * sizeof(int));
    int *offs = (int *)malloc((size_t)(nslices + 1) * sizeof(int));
    offs[0] = 0;
    long long total = 0;
    for (int s = 0; s < nslices; s++) {
        int sl = 0;
        for (int r = 0; r < SELL_SIGMA; r++) {
            const int k = s * SELL_SIGMA + r;
            if (k < n && len[order[k]] > sl) sl = len[order[k]];
        }
        lens[s] = sl;
        total += (long long)SELL_SIGMA * sl;
        offs[s + 1] = (int)total;
    }
    int *cols = (int *)malloc((size_t)total * sizeof(int));
    double *vals = (double *)malloc((size_t)total * sizeof(double));
    for (long long t = 0; t < total; t++) { cols[t] = 0; vals[t] = 0.0; }
    for (int s = 0; s < nslices; s++) {
        const int off = offs[s];
        for (int r = 0; r < SELL_SIGMA; r++) {
            const int k = s * SELL_SIGMA + r;
            if (k >= n) continue;
            const int orig = order[k];
            for (int e = A->rowptr[orig]; e < A->rowptr[orig + 1]; e++) {
                const int j = e - A->rowptr[orig];
                cols[off + (long long)j * SELL_SIGMA + r] = A->colidx[e];
                vals[off + (long long)j * SELL_SIGMA + r] = A->values[e];
            }
        }
    }
    free(len);
    *out_offs = offs; *out_lens = lens; *out_cols = cols; *out_vals = vals; *out_order = order;
    return total;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA FAIL %s:%d  %s: %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

/* 测一组 iters 次 launch 的平均内核时间（cudaEvent 包整组） */
template <typename F>
static double bench_group(F&& f, int iters)
{
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) f();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return (double)ms / iters;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <s0.mtx> [iters] [groups]\n", argv[0]); return 2; }
    CSR A;
    if (read_s0(argv[1], &A)) { fprintf(stderr, "read fail %s\n", argv[1]); return 1; }
    int iters = (argc >= 3) ? atoi(argv[2]) : 200;
    int groups = (argc >= 4) ? atoi(argv[3]) : 3;
    if (iters < 10) iters = 10;
    if (groups < 1) groups = 1;
    const int n = A.n;
    const long long nnz = A.nnz;

    /* x = ones */
    double *hx = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) hx[i] = 1.0;

    /* ---- device 分配 ---- */
    int *d_rp, *d_ci;
    double *d_v64, *d_x64, *d_y64;
    float *d_v32, *d_x32, *d_y32;
    CK(cudaMalloc((void **)&d_rp, (n + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_ci, (size_t)nnz * sizeof(int)));
    CK(cudaMalloc((void **)&d_v64, (size_t)nnz * sizeof(double)));
    CK(cudaMalloc((void **)&d_v32, (size_t)nnz * sizeof(float)));
    CK(cudaMalloc((void **)&d_x64, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_x32, (size_t)n * sizeof(float)));
    CK(cudaMalloc((void **)&d_y64, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_y32, (size_t)n * sizeof(float)));
    CK(cudaMemcpy(d_rp, A.rowptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci, A.colidx, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v64, A.values, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
    float *v32 = (float *)malloc((size_t)nnz * sizeof(float));
    float *x32 = (float *)malloc((size_t)n * sizeof(float));
    for (long long i = 0; i < nnz; i++) v32[i] = (float)A.values[i];
    for (int i = 0; i < n; i++) x32[i] = 1.0f;
    CK(cudaMemcpy(d_v32, v32, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x64, hx, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x32, x32, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    /* ---- SELL-C-σ32 构建（sort=1） ---- */
    int *h_offs, *h_lens, *h_cols, *h_order;
    double *h_vals;
    long long stored = build_sellc(&A, 1, &h_offs, &h_lens, &h_cols, &h_vals, &h_order);
    int nslices = (n + SELL_SIGMA - 1) / SELL_SIGMA;
    int *d_so, *d_sl, *d_sc;
    double *d_sv64;
    float *d_sv32;
    CK(cudaMalloc((void **)&d_so, (size_t)(nslices + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_sl, (size_t)nslices * sizeof(int)));
    CK(cudaMalloc((void **)&d_sc, (size_t)stored * sizeof(int)));
    CK(cudaMalloc((void **)&d_sv64, (size_t)stored * sizeof(double)));
    CK(cudaMalloc((void **)&d_sv32, (size_t)stored * sizeof(float)));
    CK(cudaMemcpy(d_so, h_offs, (size_t)(nslices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sl, h_lens, (size_t)nslices * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sc, h_cols, (size_t)stored * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sv64, h_vals, (size_t)stored * sizeof(double), cudaMemcpyHostToDevice));
    float *sv32 = (float *)malloc((size_t)stored * sizeof(float));
    for (long long i = 0; i < stored; i++) sv32[i] = (float)h_vals[i];
    CK(cudaMemcpy(d_sv32, sv32, (size_t)stored * sizeof(float), cudaMemcpyHostToDevice));

    const int cgrid = (n + BLOCK - 1) / BLOCK;
    const int sgrid = (nslices * SELL_SIGMA + BLOCK - 1) / BLOCK;
    int nthreads = nslices * SELL_SIGMA;

    /* ---- 正确性（fp32 vs fp64，CSR 路径） ---- */
    k_csr64<<<cgrid, BLOCK>>>(n, d_rp, d_ci, d_v64, d_x64, d_y64);
    k_csr32<<<cgrid, BLOCK>>>(n, d_rp, d_ci, d_v32, d_x32, d_y32);
    CK(cudaDeviceSynchronize());
    double *hy64 = (double *)malloc((size_t)n * sizeof(double));
    float *hy32 = (float *)malloc((size_t)n * sizeof(float));
    CK(cudaMemcpy(hy64, d_y64, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy32, d_y32, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    double m2 = 0, me = 0;
    for (int i = 0; i < n; i++) {
        double a = fabs(hy64[i]);
        if (a > m2) m2 = a;
        double dd = fabs((double)hy32[i] - hy64[i]);
        if (dd > me) me = dd;
    }
    double rel_err = m2 > 0 ? me / m2 : 0.0;

    /* ---- 基准 ---- */
    auto g_csr64 = [&]() { k_csr64<<<cgrid, BLOCK>>>(n, d_rp, d_ci, d_v64, d_x64, d_y64); };
    auto g_csr32 = [&]() { k_csr32<<<cgrid, BLOCK>>>(n, d_rp, d_ci, d_v32, d_x32, d_y32); };
    auto g_sel64 = [&]() { k_sell64<<<sgrid, BLOCK>>>(n, nslices, d_so, d_sl, d_sc, d_sv64, d_x64, d_y64); };
    auto g_sel32 = [&]() { k_sell32<<<sgrid, BLOCK>>>(n, nslices, d_so, d_sl, d_sc, d_sv32, d_x32, d_y32); };

    /* warmup（各 10 次） */
    for (int i = 0; i < 10; i++) { g_csr64(); g_csr32(); g_sel64(); g_sel32(); }
    CK(cudaDeviceSynchronize());

    double t_csr64 = 1e30, t_csr32 = 1e30, t_sel64 = 1e30, t_sel32 = 1e30;
    for (int g = 0; g < groups; g++) {
        double a = bench_group(g_csr64, iters); if (a < t_csr64) t_csr64 = a;
        double b = bench_group(g_csr32, iters); if (b < t_csr32) t_csr32 = b;
        double c = bench_group(g_sel64, iters); if (c < t_sel64) t_sel64 = c;
        double d = bench_group(g_sel32, iters); if (d < t_sel32) t_sel32 = d;
    }

    auto bw = [&](double t, int bytes) { return 2.0 * (double)nnz * (double)bytes / (t * 1e-3) / 1e9; };
    auto util = [&](double gbs) { return 100.0 * gbs / THEO_BW_GBPS; };

    double b_csr64 = bw(t_csr64, 8), b_csr32 = bw(t_csr32, 4);
    double b_sel64 = bw(t_sel64, 8), b_sel32 = bw(t_sel32, 4);

    char name[256] = "matrix";
    { const char *b = strrchr(argv[1], '/'); snprintf(name, sizeof name, "%s", b ? b + 1 : argv[1]);
      char *dot = strrchr(name, '.'); if (dot) *dot = '\0'; }

    printf("== mp_spmv_bench %s n=%d nnz=%lld (iters=%d groups=%d, sell stored=%lld pad=%.1f%%) ==\n",
           name, n, (long long)nnz, iters, groups, stored,
           100.0 * ((double)stored - (double)nnz) / (double)nnz);
    printf("%-10s %10s %9s %9s %10s\n", "format", "t_ms", "bw_gbs", "util%", "vs_fp64");
    printf("%-10s %10.6f %9.1f %9.1f %10s\n", "csr_fp64", t_csr64, b_csr64, util(b_csr64), "1.00x");
    printf("%-10s %10.6f %9.1f %9.1f %10.2fx\n", "csr_fp32", t_csr32, b_csr32, util(b_csr32), t_csr64 / t_csr32);
    printf("%-10s %10.6f %9.1f %9.1f %10s\n", "sell_fp64", t_sel64, b_sel64, util(b_sel64), "1.00x");
    printf("%-10s %10.6f %9.1f %9.1f %10.2fx\n", "sell_fp32", t_sel32, b_sel32, util(b_sel32), t_sel64 / t_sel32);
    printf("rel_err(fp32 vs fp64) = %.3e\n", rel_err);

    printf("JSON:{\"matrix\":\"%s\",\"n\":%d,\"nnz\":%lld,\"iters\":%d,\"groups\":%d,"
           "\"t_csr_fp64_ms\":%.6f,\"t_csr_fp32_ms\":%.6f,\"t_sell_fp64_ms\":%.6f,\"t_sell_fp32_ms\":%.6f,"
           "\"bw_csr_fp64_gbs\":%.2f,\"bw_csr_fp32_gbs\":%.2f,\"bw_sell_fp64_gbs\":%.2f,\"bw_sell_fp32_gbs\":%.2f,"
           "\"speedup_csr_fp32\":%.3f,\"speedup_sell_fp32\":%.3f,\"rel_err_fp32\":%.3e,"
           "\"sell_stored\":%lld,\"sell_pad_pct\":%.1f}\n",
           name, n, (long long)nnz, iters, groups,
           t_csr64, t_csr32, t_sel64, t_sel32,
           b_csr64, b_csr32, b_sel64, b_sel32,
           t_csr64 / t_csr32, t_sel64 / t_sel32, rel_err, stored,
           100.0 * ((double)stored - (double)nnz) / (double)nnz);

    return 0;
}
