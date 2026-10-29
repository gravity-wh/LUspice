// ============================================================================
// sell_c_spmv.cu — SELL-C-σ32 SpMV GPU 实测（赛题七 M1 收窄版：CSR 对照）
//
// 目标：对 3 个代表矩阵（pg100k / pg5k / rajat12）实测 SELL-C-σ（σ=32）SpMV
//       带宽，与 CSR 每行一线程 v0 内核对照，产出格式决策数据。
//
// 纪律（同 spmv.cu / cuda_compat.h）：只用标准 CUDA 子集 + DSH_* 抽象接口，
//   不使用 cuSparse / cuBLAS / cudaGraph / 纹理内存；迁移 MXMACA 时零改动。
//
// SELL-C-σ32 格式：
//   - 分片（slice）宽 σ=32 行（= 一个 warp），nslices = ceil(n/σ)；
//   - 块内按该片最大行长度 len[s] 填充（ELL 子结构），列主序存储：
//     cols/vals 下标 = off[s] + j*σ + r（j=0..len[s]-1 列, r=0..σ-1 局部行）
//     → 同一列 σ 个元素连续 = 一个 256B 合并事务；
//   - 可选按行长降序重排分片（--sort，标准做法；--nosort 用自然行序），
//     y 输出为分片序，host 侧用 perm 逆置换回原行序；
//   - 每行一个线程：warp 内 threadIdx.r 为局部行，slice = tid>>5。
//   - 填充槽位 cols=0、vals=0.0（x[0] 恒合法，无需分支哨兵）。
//
// 计时（protocol §2）：warmup 10 次；G 组每组 N 次 launch，组均 = 组时间/N；
//   t_ms = G 个组均的中位数。带宽 = 2*nnz*8B / t（任务定义，不扣除填充）。
//
// 构建:
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -I /home/gravity/DSH/race7/src/solver \
//       -o sell_c_spmv sell_c_spmv.cu
// 用法:
//   ./sell_c_spmv <s0.mtx> <s0.rhs|ones> [--sort|--nosort] [--iters N] [--groups G] \
//                 [--json PATH] [--out PREFIX]
//   默认 --sort；--out PREFIX 把 y_csr 与 y_sell 写入 <PREFIX>_csr.txt / <PREFIX>_sell.txt
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "spmv.h"
#include "cuda_compat.h"

#define THEO_BW_GBPS 360.0   // RTX 3060 GDDR6 理论带宽参考
#define SELL_SIGMA 32        // SELL-C 列块宽 σ（= warp 大小）
#define BLOCK 256            // 线程块大小（8 个 warp/块 → 8 个 slice/块）

// ----------------------------------------------------------------------------
// s0 格式读取（与 spmv.cu 完全一致：两遍扫描 → 计数 → 前缀和 → 游标填充 CSR）
// ----------------------------------------------------------------------------

int read_s0_to_csr(const char* path, CSRMatrix* A) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[sellc] error: cannot open %s\n", path); return 1; }

    char line[512];
    int n = 0;
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    if (sscanf(line, "n %d", &n) != 1 || n <= 0) {
        fprintf(stderr, "[sellc] error: bad header line: %s", line);
        fclose(f);
        return 1;
    }

    long long nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'A') nnz++;
    }
    if (nnz == 0) { fclose(f); return 1; }

    int* ri = (int*)malloc((size_t)nnz * sizeof(int));
    int* rj = (int*)malloc((size_t)nnz * sizeof(int));
    double* rv = (double*)malloc((size_t)nnz * sizeof(double));

    rewind(f);
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    long long k = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j;
        double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (i < 1 || i > n || j < 1) continue;
        ri[k] = i - 1;
        rj[k] = j - 1;
        rv[k] = v;
        k++;
    }
    fclose(f);
    if (k != nnz) {
        fprintf(stderr, "[sellc] warning: pass1=%lld vs pass2=%lld entries\n", nnz, k);
        nnz = k;
    }

    int* rowptr = (int*)calloc((size_t)n + 1, sizeof(int));
    for (long long t = 0; t < nnz; t++) rowptr[ri[t] + 1]++;
    for (int i = 1; i <= n; i++) rowptr[i] += rowptr[i - 1];

    int* colidx = (int*)malloc((size_t)nnz * sizeof(int));
    double* values = (double*)malloc((size_t)nnz * sizeof(double));
    int* cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, rowptr, (size_t)n * sizeof(int));
    for (long long t = 0; t < nnz; t++) {
        int p = cursor[ri[t]]++;
        colidx[p] = rj[t];
        values[p] = rv[t];
    }

    free(ri); free(rj); free(rv); free(cursor);

    A->n = n;
    A->nnz = nnz;
    A->rowptr = rowptr;
    A->colidx = colidx;
    A->values = values;
    return 0;
}

int read_rhs(const char* path, int n, double* x) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[sellc] error: cannot open %s\n", path); return 1; }
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    int cnt = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'R') continue;
        int i;
        double v;
        if (sscanf(line, "RHS %d %lf", &i, &v) != 2) continue;
        if (i >= 1 && i <= n) { x[i - 1] = v; cnt++; }
    }
    fclose(f);
    return cnt > 0 ? 0 : 1;
}

void csr_free(CSRMatrix* A) {
    if (!A) return;
    free(A->rowptr);
    free(A->colidx);
    free(A->values);
    A->rowptr = A->colidx = NULL;
    A->values = NULL;
    A->n = 0;
    A->nnz = 0;
}

// ----------------------------------------------------------------------------
// CSR 内核（同 spmv.cu v0：每行一线程）
// ----------------------------------------------------------------------------

__global__ void k_csr(const int n,
                      const int* __restrict__ rowptr,
                      const int* __restrict__ colidx,
                      const double* __restrict__ values,
                      const double* __restrict__ x,
                      double* __restrict__ y) {
    const int row = blockIdx.x * BLOCK + threadIdx.x;
    if (row < n) {
        double sum = 0.0;
        for (int kk = rowptr[row]; kk < rowptr[row + 1]; kk++) {
            sum += values[kk] * x[colidx[kk]];
        }
        y[row] = sum;
    }
}

// ----------------------------------------------------------------------------
// SELL-C-σ32 内核：warp = 一个 slice，threadIdx&31 为片内行 r
//   slice = tid>>5；grid 覆盖 nslices*σ 个线程（含末片填充行，row>=n 直接返回）
//   y 输出为分片序（host 用 perm 逆置换）
// ----------------------------------------------------------------------------

__global__ void k_sellc(const int n, const int nslices,
                        const int* __restrict__ offs,
                        const int* __restrict__ lens,
                        const int* __restrict__ cols,
                        const double* __restrict__ vals,
                        const double* __restrict__ x,
                        double* __restrict__ y) {
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int slice = tid >> 5;          // /σ
    const int r = threadIdx.x & (SELL_SIGMA - 1);
    if (slice >= nslices) return;
    const int row = slice * SELL_SIGMA + r;
    if (row >= n) return;
    const int off = offs[slice];
    const int len = lens[slice];
    double sum = 0.0;
    // 列主序：固定 j，σ 个 r 连续 → 合并 256B 事务
    for (int j = 0; j < len; j++) {
        const int idx = off + j * SELL_SIGMA + r;
        sum += vals[idx] * x[cols[idx]];
    }
    y[row] = sum;
}

// ----------------------------------------------------------------------------
// SELL-C-σ32 host 构建：可选按行长降序分片；输出分片元数据 + 列主序数组
// 返回 padded 存储槽位总数 stored（= Σ len[s]*σ）
// ----------------------------------------------------------------------------

static long long build_sellc(const CSRMatrix* A, int sort,
                             int** out_offs, int** out_lens,
                             int** out_cols, double** out_vals,
                             int** out_order) {
    const int n = A->n;
    int* len = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) len[i] = (int)(A->rowptr[i + 1] - A->rowptr[i]);

    // order[k] = 分片序第 k 个位置对应的原行号（sort=0 时即恒等）
    int* order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    if (sort) {
        // 按长度降序计数排序 O(n + maxlen)
        int maxlen = 0;
        for (int i = 0; i < n; i++) if (len[i] > maxlen) maxlen = len[i];
        int* cnt = (int*)calloc((size_t)maxlen + 1, sizeof(int));
        for (int i = 0; i < n; i++) cnt[len[i]]++;
        for (int l = maxlen - 1; l >= 0; l--) cnt[l] += cnt[l + 1];  // 后缀和→降序位
        for (int i = 0; i < n; i++) order[--cnt[len[i]]] = i;
        free(cnt);
    }

    const int nslices = (n + SELL_SIGMA - 1) / SELL_SIGMA;
    int* lens = (int*)malloc((size_t)nslices * sizeof(int));
    int* offs = (int*)malloc((size_t)(nslices + 1) * sizeof(int));
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

    int* cols = (int*)malloc((size_t)total * sizeof(int));
    double* vals = (double*)malloc((size_t)total * sizeof(double));
    for (long long t = 0; t < total; t++) { cols[t] = 0; vals[t] = 0.0; }  // 填充槽：col=0, val=0
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

// ----------------------------------------------------------------------------
// 设备分配 + launch + 计时（DSH_* 接口）
// ----------------------------------------------------------------------------

static double bench_loop_csr(const CSRMatrix* A,
                             const int* d_rowptr, const int* d_colidx, const double* d_values,
                             const double* d_x, double* d_y,
                             int warmup, int groups, int iters) {
    for (int i = 0; i < warmup; i++)
        k_csr<<<(A->n + BLOCK - 1) / BLOCK, BLOCK>>>(A->n, d_rowptr, d_colidx, d_values, d_x, d_y);
    DSH_CHECK(DSH_DeviceSynchronize());

    double* gms = (double*)calloc((size_t)groups, sizeof(double));
    for (int g = 0; g < groups; g++) {
        DSH_Event_t start, stop;
        DSH_CHECK(DSH_EventCreate(&start));
        DSH_CHECK(DSH_EventCreate(&stop));
        DSH_CHECK(DSH_EventRecord(start, DSH_DEFAULT_STREAM));
        for (int i = 0; i < iters; i++)
            k_csr<<<(A->n + BLOCK - 1) / BLOCK, BLOCK>>>(A->n, d_rowptr, d_colidx, d_values, d_x, d_y);
        DSH_CHECK(DSH_EventRecord(stop, DSH_DEFAULT_STREAM));
        DSH_CHECK(DSH_EventSync(stop));
        float ms = 0.f;
        DSH_CHECK(DSH_EventElapsed(&ms, start, stop));
        gms[g] = (double)ms / (double)iters;
        DSH_CHECK(DSH_EventDestroy(start));
        DSH_CHECK(DSH_EventDestroy(stop));
    }
    std::sort(gms, gms + groups);
    double med = gms[groups / 2];
    free(gms);
    return med;
}

static double bench_loop_sellc(const int n, const int nslices,
                               const int* d_offs, const int* d_lens,
                               const int* d_cols, const double* d_vals,
                               const double* d_x, double* d_y,
                               int warmup, int groups, int iters) {
    const int nthreads = nslices * SELL_SIGMA;
    const int grid = (nthreads + BLOCK - 1) / BLOCK;
    for (int i = 0; i < warmup; i++)
        k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_x, d_y);
    DSH_CHECK(DSH_DeviceSynchronize());

    double* gms = (double*)calloc((size_t)groups, sizeof(double));
    for (int g = 0; g < groups; g++) {
        DSH_Event_t start, stop;
        DSH_CHECK(DSH_EventCreate(&start));
        DSH_CHECK(DSH_EventCreate(&stop));
        DSH_CHECK(DSH_EventRecord(start, DSH_DEFAULT_STREAM));
        for (int i = 0; i < iters; i++)
            k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_x, d_y);
        DSH_CHECK(DSH_EventRecord(stop, DSH_DEFAULT_STREAM));
        DSH_CHECK(DSH_EventSync(stop));
        float ms = 0.f;
        DSH_CHECK(DSH_EventElapsed(&ms, start, stop));
        gms[g] = (double)ms / (double)iters;
        DSH_CHECK(DSH_EventDestroy(start));
        DSH_CHECK(DSH_EventDestroy(stop));
    }
    std::sort(gms, gms + groups);
    double med = gms[groups / 2];
    free(gms);
    return med;
}

// max_rel_err = max|y1-y2| / max|y2|
static double max_rel_err(const double* y1, const double* y2, int n) {
    double m2 = 0.0, me = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(y2[i]);
        if (a > m2) m2 = a;
        double d = fabs(y1[i] - y2[i]);
        if (d > me) me = d;
    }
    return (m2 > 0.0) ? me / m2 : 0.0;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <s0.mtx> <s0.rhs|ones> [--sort|--nosort] [--iters N] [--groups G] [--json PATH] [--out PREFIX]\n",
                argv[0]);
        return 1;
    }
    const char* mtx_path = argv[1];
    const char* rhs_arg = argv[2];
    int sort = 1, iters = 2000, groups = 5;
    const char* json_path = NULL;
    const char* out_prefix = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--sort")) sort = 1;
        else if (!strcmp(argv[i], "--nosort")) sort = 0;
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--groups") && i + 1 < argc) groups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_prefix = argv[++i];
        else { fprintf(stderr, "[sellc] unknown arg: %s\n", argv[i]); return 1; }
    }
    if (iters < 1) iters = 1;
    if (groups < 1) groups = 1;

    CSRMatrix A;
    memset(&A, 0, sizeof A);
    if (read_s0_to_csr(mtx_path, &A)) return 1;
    const int n = A.n;
    const long long nnz = A.nnz;

    double* x = (double*)malloc(sizeof(double) * (size_t)n);
    if (!strcmp(rhs_arg, "ones")) {
        for (int i = 0; i < n; i++) x[i] = 1.0;
    } else {
        if (read_rhs(rhs_arg, n, x)) {
            fprintf(stderr, "[sellc] error: cannot read rhs %s\n", rhs_arg);
            return 1;
        }
    }
    double* y_csr = (double*)malloc(sizeof(double) * (size_t)n);
    double* y_sell = (double*)malloc(sizeof(double) * (size_t)n);
    double* y_sell_orig = (double*)malloc(sizeof(double) * (size_t)n);

    // ---- 设备缓冲 ----
    int *d_rp = NULL, *d_ci = NULL;
    double* d_cv = NULL;
    int *d_offs = NULL, *d_lens = NULL, *d_cols = NULL;
    double *d_vals = NULL, *d_x = NULL, *d_y = NULL;
    const size_t nb_rp = sizeof(int) * ((size_t)n + 1);
    const size_t nb_ci = sizeof(int) * (size_t)nnz;
    const size_t nb_vl = sizeof(double) * (size_t)nnz;

    DSH_CHECK(DSH_Malloc((void**)&d_rp, nb_rp));
    DSH_CHECK(DSH_Malloc((void**)&d_ci, nb_ci));
    DSH_CHECK(DSH_Malloc((void**)&d_cv, nb_vl));
    DSH_CHECK(DSH_Memcpy(d_rp, A.rowptr, nb_rp, DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_ci, A.colidx, nb_ci, DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_cv, A.values, nb_vl, DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Malloc((void**)&d_x, sizeof(double) * (size_t)n));
    DSH_CHECK(DSH_Malloc((void**)&d_y, sizeof(double) * (size_t)n));
    DSH_CHECK(DSH_Memcpy(d_x, x, sizeof(double) * (size_t)n, DSH_MEMCPY_HOST_TO_DEVICE));

    // ---- SELL-C-σ32 host 构建（同一 CSR 源）----
    int *h_offs = NULL, *h_lens = NULL, *h_cols = NULL, *h_order = NULL;
    double* h_vals = NULL;
    long long stored = build_sellc(&A, sort, &h_offs, &h_lens, &h_cols, &h_vals, &h_order);
    const int nslices = (n + SELL_SIGMA - 1) / SELL_SIGMA;

    DSH_CHECK(DSH_Malloc((void**)&d_offs, sizeof(int) * ((size_t)nslices + 1)));
    DSH_CHECK(DSH_Malloc((void**)&d_lens, sizeof(int) * (size_t)nslices));
    DSH_CHECK(DSH_Malloc((void**)&d_cols, sizeof(int) * (size_t)stored));
    DSH_CHECK(DSH_Malloc((void**)&d_vals, sizeof(double) * (size_t)stored));
    DSH_CHECK(DSH_Memcpy(d_offs, h_offs, sizeof(int) * ((size_t)nslices + 1), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_lens, h_lens, sizeof(int) * (size_t)nslices, DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_cols, h_cols, sizeof(int) * (size_t)stored, DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_vals, h_vals, sizeof(double) * (size_t)stored, DSH_MEMCPY_HOST_TO_DEVICE));

    const int grid_csr = (n + BLOCK - 1) / BLOCK;
    const int grid_sell = (nslices * SELL_SIGMA + BLOCK - 1) / BLOCK;

    // ---- 正确性 ----
    k_csr<<<grid_csr, BLOCK>>>(n, d_rp, d_ci, d_cv, d_x, d_y);
    DSH_CHECK(DSH_GetLastError());
    DSH_CHECK(DSH_Memcpy(y_csr, d_y, sizeof(double) * (size_t)n, DSH_MEMCPY_DEVICE_TO_HOST));

    k_sellc<<<grid_sell, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_x, d_y);
    DSH_CHECK(DSH_GetLastError());
    DSH_CHECK(DSH_Memcpy(y_sell, d_y, sizeof(double) * (size_t)n, DSH_MEMCPY_DEVICE_TO_HOST));
    for (int k = 0; k < n; k++) y_sell_orig[h_order[k]] = y_sell[k];  // perm 逆置换
    const double rel_err = max_rel_err(y_sell_orig, y_csr, n);

    // ---- 计时 ----
    const double t_csr = bench_loop_csr(&A, d_rp, d_ci, d_cv, d_x, d_y, 10, groups, iters);
    const double t_sell = bench_loop_sellc(n, nslices, d_offs, d_lens, d_cols, d_vals, d_x, d_y,
                                           10, groups, iters);

    // ---- 汇报 ----
    const double bytes = 2.0 * (double)nnz * 8.0;
    const double bw_csr = bytes / (t_csr * 1e-3) / 1e9;
    const double bw_sell = bytes / (t_sell * 1e-3) / 1e9;
    const double pad_pct = 100.0 * ((double)stored - (double)nnz) / (double)nnz;

    char mtx_name[256] = "matrix";
    {
        const char* b = strrchr(mtx_path, '/');
        snprintf(mtx_name, sizeof mtx_name, "%s", b ? b + 1 : mtx_path);
        char* dot = strrchr(mtx_name, '.');
        if (dot) *dot = '\0';
    }

    printf("== SELL-C-sigma32 vs CSR (%s, n=%d nnz=%lld, sort=%d) ==\n", mtx_name, n, (long long)nnz, sort);
    printf("format        sigma  t_ms        bw_gbs    pct360  max_rel_err\n");
    printf("csr           %5d  %10.6f  %8.1f  %6.1f  %12.3e\n", 0, t_csr, bw_csr, 100.0 * bw_csr / THEO_BW_GBPS, 0.0);
    printf("sellc32       %5d  %10.6f  %8.1f  %6.1f  %12.3e\n", SELL_SIGMA, t_sell, bw_sell,
           100.0 * bw_sell / THEO_BW_GBPS, rel_err);
    printf("padded_stored=%lld pad_waste_pct=%.2f nslices=%d\n", (long long)stored, pad_pct, nslices);

    // ---- 输出 y（供 scipy 独立对照）----
    if (out_prefix) {
        char p[1024];
        snprintf(p, sizeof p, "%s_csr.txt", out_prefix);
        FILE* f = fopen(p, "w");
        if (f) { for (int i = 0; i < n; i++) fprintf(f, "%.17g\n", y_csr[i]); fclose(f); }
        snprintf(p, sizeof p, "%s_sell.txt", out_prefix);
        f = fopen(p, "w");
        if (f) { for (int i = 0; i < n; i++) fprintf(f, "%.17g\n", y_sell_orig[i]); fclose(f); }
    }

    // ---- JSON ----
    if (json_path) {
        FILE* f = fopen(json_path, "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"matrix\": \"%s\",\n", mtx_name);
            fprintf(f, "  \"n\": %d,\n", n);
            fprintf(f, "  \"nnz\": %lld,\n", (long long)nnz);
            fprintf(f, "  \"sort\": %d,\n", sort);
            fprintf(f, "  \"iters\": %d,\n", iters);
            fprintf(f, "  \"groups\": %d,\n", groups);
            fprintf(f, "  \"padded_stored\": %lld,\n", (long long)stored);
            fprintf(f, "  \"pad_waste_pct\": %.2f,\n", pad_pct);
            fprintf(f, "  \"rows\": [\n");
            fprintf(f, "    {\"format\": \"csr\", \"sigma\": 0, \"t_ms\": %.6f, \"bw_gbs\": %.1f, \"bw_pct_vs_360\": %.1f, \"max_rel_err\": 0.0},\n",
                    t_csr, bw_csr, 100.0 * bw_csr / THEO_BW_GBPS);
            fprintf(f, "    {\"format\": \"sellc32\", \"sigma\": %d, \"t_ms\": %.6f, \"bw_gbs\": %.1f, \"bw_pct_vs_360\": %.1f, \"max_rel_err\": %.3e}\n",
                    SELL_SIGMA, t_sell, bw_sell, 100.0 * bw_sell / THEO_BW_GBPS, rel_err);
            fprintf(f, "  ]\n}\n");
            fclose(f);
        }
    }

    // ---- 清理 ----
    DSH_Free(d_rp); DSH_Free(d_ci); DSH_Free(d_cv);
    DSH_Free(d_offs); DSH_Free(d_lens); DSH_Free(d_cols); DSH_Free(d_vals);
    DSH_Free(d_x); DSH_Free(d_y);
    free(h_offs); free(h_lens); free(h_cols); free(h_vals); free(h_order);
    csr_free(&A);
    free(x); free(y_csr); free(y_sell); free(y_sell_orig);
    return 0;
}
