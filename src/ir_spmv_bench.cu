// ============================================================================
// ir_spmv_bench.cu — R7: IR 循环 SpMV 真实每迭代开销 A/B（host C 循环 vs GPU SELL-C-σ32）
//
// 与 test_lu_gpu_f32 的 GMRES-IR 完全同口径：
//   - host 路径 = IR 原代码的 C 循环（w[rr] += M.val[i]*z[cc]，自然序）
//   - GPU 路径 = 每次调用含 H2D(z) + SELL kernel + D2H(w) + sync + 排序分片逆置换
//     （即 sellc_spmv_ir 的每迭代真实开销；IR 内层另有 trisolve 输出直喂 kernel
//       省一次 D2H 的变体，一并测：devx 路径）
// 另报纯 kernel 时间（无传输，R6 口径）作对照。
//
// 用法:
//   nvcc -O3 -arch=sm_86 -I src/solver -o ir_spmv_bench ir_spmv_bench.cu
//   ./ir_spmv_bench <s0.mtx> [iters N] [groups G]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <time.h>

#include "cuda_compat.h"

#define THEO_BW_GBPS 360.0
#define SELL_SIGMA 32
#define BLOCK 256

typedef struct { int n; long long nnz; int *rowptr, *colidx; double *values; } CSR;

static int read_s0(const char* path, CSR* A) {
    FILE* f = fopen(path, "r");
    if (!f) return 1;
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    int n = 0;
    if (sscanf(line, "n %d", &n) != 1 || n <= 0) { fclose(f); return 1; }
    long long nnz = 0;
    while (fgets(line, sizeof line, f)) if (line[0] == 'A') nnz++;
    if (nnz == 0) { fclose(f); return 1; }
    rewind(f); if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    int* ri = (int*)malloc((size_t)nnz * sizeof(int));
    int* rj = (int*)malloc((size_t)nnz * sizeof(int));
    double* rv = (double*)malloc((size_t)nnz * sizeof(double));
    long long k = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (i < 1 || i > n || j < 1) continue;
        ri[k] = i - 1; rj[k] = j - 1; rv[k] = v; k++;
    }
    fclose(f);
    int* rowptr = (int*)calloc((size_t)n + 1, sizeof(int));
    for (long long t = 0; t < nnz; t++) rowptr[ri[t] + 1]++;
    for (int i = 1; i <= n; i++) rowptr[i] += rowptr[i - 1];
    int* colidx = (int*)malloc((size_t)nnz * sizeof(int));
    double* values = (double*)malloc((size_t)nnz * sizeof(double));
    int* cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, rowptr, (size_t)n * sizeof(int));
    for (long long t = 0; t < nnz; t++) { int p = cursor[ri[t]]++; colidx[p] = rj[t]; values[p] = rv[t]; }
    free(ri); free(rj); free(rv); free(cursor);
    A->n = n; A->nnz = nnz; A->rowptr = rowptr; A->colidx = colidx; A->values = values;
    return 0;
}

__global__ void k_sellc(const int n, const int nslices,
                        const int* __restrict__ offs, const int* __restrict__ lens,
                        const int* __restrict__ cols, const double* __restrict__ vals,
                        const double* __restrict__ x, double* __restrict__ y) {
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

static long long build_sellc(const CSR* A, int sort, int** out_offs, int** out_lens,
                             int** out_cols, double** out_vals, int** out_order) {
    const int n = A->n;
    int* len = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) len[i] = (int)(A->rowptr[i + 1] - A->rowptr[i]);
    int* order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    if (sort) {
        int maxlen = 0;
        for (int i = 0; i < n; i++) if (len[i] > maxlen) maxlen = len[i];
        int* cnt = (int*)calloc((size_t)maxlen + 1, sizeof(int));
        for (int i = 0; i < n; i++) cnt[len[i]]++;
        for (int l = maxlen - 1; l >= 0; l--) cnt[l] += cnt[l + 1];
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

static void host_spmv(const CSR* A, const double* z, double* w) {
    /* 与 IR 原循环同口径：逐行累加（s0 文件行主序，CSR 行内保持文件序） */
    for (int i = 0; i < A->n; i++) {
        double acc = 0.0;
        for (int p = A->rowptr[i]; p < A->rowptr[i + 1]; p++) acc += A->values[p] * z[A->colidx[p]];
        w[i] = acc;
    }
}

static double ms_since(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

static double median(double* v, int g) {
    std::sort(v, v + g);
    return g % 2 ? v[g / 2] : 0.5 * (v[g / 2 - 1] + v[g / 2]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <s0.mtx> [iters] [groups]\n", argv[0]); return 1; }
    int iters = argc >= 3 ? atoi(argv[2]) : 200;
    int groups = argc >= 4 ? atoi(argv[3]) : 7;
    if (iters < 10) iters = 10;

    CSR A;
    memset(&A, 0, sizeof A);
    if (read_s0(argv[1], &A)) { fprintf(stderr, "read fail\n"); return 1; }
    const int n = A.n;
    const long long nnz = A.nnz;

    double* z = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) z[i] = 1.0 + 0.5 * (double)(i % 7);
    double* w_host = (double*)malloc((size_t)n * sizeof(double));
    double* w_gpu = (double*)malloc((size_t)n * sizeof(double));

    // SELL build + device buffers
    int *h_offs = NULL, *h_lens = NULL, *h_cols = NULL, *h_order = NULL;
    double* h_vals = NULL;
    long long stored = build_sellc(&A, 1, &h_offs, &h_lens, &h_cols, &h_vals, &h_order);
    const int nslices = (n + SELL_SIGMA - 1) / SELL_SIGMA;
    int *d_offs = NULL, *d_lens = NULL, *d_cols = NULL;
    double *d_vals = NULL, *d_z = NULL, *d_w = NULL;
    DSH_CHECK(DSH_Malloc((void**)&d_offs, (size_t)(nslices + 1) * sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_lens, (size_t)nslices * sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_cols, (size_t)stored * sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_vals, (size_t)stored * sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_z, (size_t)n * sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_w, (size_t)n * sizeof(double)));
    DSH_CHECK(DSH_Memcpy(d_offs, h_offs, (size_t)(nslices + 1) * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_lens, h_lens, (size_t)nslices * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_cols, h_cols, (size_t)stored * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_vals, h_vals, (size_t)stored * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    const int grid = (nslices * SELL_SIGMA + BLOCK - 1) / BLOCK;

    // 正确性
    host_spmv(&A, z, w_host);
    DSH_CHECK(DSH_Memcpy(d_z, z, (size_t)n * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
    k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
    DSH_CHECK(DSH_DeviceSynchronize());
    DSH_CHECK(DSH_Memcpy(w_gpu, d_w, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST));
    memcpy(tmp, w_gpu, (size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) w_gpu[h_order[k]] = tmp[k];
    double m2 = 0, me = 0;
    for (int i = 0; i < n; i++) {
        double a = fabs(w_host[i]);
        if (a > m2) m2 = a;
        double d = fabs(w_gpu[i] - w_host[i]);
        if (d > me) me = d;
    }
    double rel_err = m2 > 0 ? me / m2 : 0.0;

    // ---- 计时 ----
    double* g_host = (double*)malloc((size_t)groups * sizeof(double));
    double* g_gpu = (double*)malloc((size_t)groups * sizeof(double));   // 含传输（IR 真实口径）
    double* g_gpudx = (double*)malloc((size_t)groups * sizeof(double)); // trisolve 输出直喂（省 D2H）
    double* g_kern = (double*)malloc((size_t)groups * sizeof(double));  // 纯 kernel（R6 口径）

    struct timespec ta, tb;
    for (int gi = 0; gi < groups; gi++) {
        // host
        for (int w_ = 0; w_ < 5; w_++) host_spmv(&A, z, w_host);
        clock_gettime(CLOCK_MONOTONIC, &ta);
        for (int i = 0; i < iters; i++) host_spmv(&A, z, w_host);
        clock_gettime(CLOCK_MONOTONIC, &tb);
        g_host[gi] = ms_since(ta, tb) / iters;
        // gpu 含传输
        for (int w_ = 0; w_ < 5; w_++) {
            DSH_Memcpy(d_z, z, (size_t)n * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
            k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
            DSH_DeviceSynchronize();
            DSH_Memcpy(w_gpu, d_w, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
        }
        clock_gettime(CLOCK_MONOTONIC, &ta);
        for (int i = 0; i < iters; i++) {
            DSH_Memcpy(d_z, z, (size_t)n * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
            k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
            DSH_DeviceSynchronize();
            DSH_Memcpy(w_gpu, d_w, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
        }
        clock_gettime(CLOCK_MONOTONIC, &tb);
        g_gpu[gi] = ms_since(ta, tb) / iters;
        // gpu devx（z 已在设备端：只有 kernel + D2H）
        for (int w_ = 0; w_ < 5; w_++) {
            k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
            DSH_DeviceSynchronize();
            DSH_Memcpy(w_gpu, d_w, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
        }
        clock_gettime(CLOCK_MONOTONIC, &ta);
        for (int i = 0; i < iters; i++) {
            k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
            DSH_DeviceSynchronize();
            DSH_Memcpy(w_gpu, d_w, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
        }
        clock_gettime(CLOCK_MONOTONIC, &tb);
        g_gpudx[gi] = ms_since(ta, tb) / iters;
        // 纯 kernel
        DSH_Event_t es, ee;
        DSH_CHECK(DSH_EventCreate(&es));
        DSH_CHECK(DSH_EventCreate(&ee));
        DSH_CHECK(DSH_EventRecord(es, DSH_DEFAULT_STREAM));
        for (int i = 0; i < iters; i++) k_sellc<<<grid, BLOCK>>>(n, nslices, d_offs, d_lens, d_cols, d_vals, d_z, d_w);
        DSH_CHECK(DSH_EventRecord(ee, DSH_DEFAULT_STREAM));
        DSH_CHECK(DSH_EventSync(ee));
        float msf = 0.f;
        DSH_CHECK(DSH_EventElapsed(&msf, es, ee));
        g_kern[gi] = (double)msf / iters;
        DSH_CHECK(DSH_EventDestroy(es));
        DSH_CHECK(DSH_EventDestroy(ee));
    }

    const double t_host = median(g_host, groups);
    const double t_gpu = median(g_gpu, groups);
    const double t_gpudx = median(g_gpudx, groups);
    const double t_kern = median(g_kern, groups);
    const double bytes = 2.0 * (double)nnz * 8.0;

    char name[256] = "matrix";
    {
        const char* b = strrchr(argv[1], '/');
        snprintf(name, sizeof name, "%s", b ? b + 1 : argv[1]);
        char* dot = strrchr(name, '.');
        if (dot) *dot = '\0';
    }

    printf("== IR SpMV per-call A/B (%s n=%d nnz=%lld, iters=%d groups=%d) ==\n",
           name, n, (long long)nnz, iters, groups);
    printf("path                t_ms/call    bw_gbs    pct360   vs_host\n");
    printf("host C-loop        %10.6f  %8.1f  %6.1f     1.00x\n", t_host, bytes/(t_host*1e-3)/1e9, 100.0*bytes/(t_host*1e-3)/1e9/THEO_BW_GBPS);
    printf("gpu sell (H2D+k+D2H)%10.6f  %8.1f  %6.1f   %6.2fx\n", t_gpu, bytes/(t_gpu*1e-3)/1e9, 100.0*bytes/(t_gpu*1e-3)/1e9/THEO_BW_GBPS, t_host/t_gpu);
    printf("gpu sell (devx+k+D2H)%10.6f  %8.1f  %6.1f   %6.2fx\n", t_gpudx, bytes/(t_gpudx*1e-3)/1e9, 100.0*bytes/(t_gpudx*1e-3)/1e9/THEO_BW_GBPS, t_host/t_gpudx);
    printf("gpu sell (kernel only)%9.6f  %8.1f  %6.1f   %6.2fx\n", t_kern, bytes/(t_kern*1e-3)/1e9, 100.0*bytes/(t_kern*1e-3)/1e9/THEO_BW_GBPS, t_host/t_kern);
    printf("max_rel_err(host vs gpu) = %.3e  pad_waste=%.1f%%\n", rel_err, 100.0*((double)stored-(double)nnz)/(double)nnz);

    printf("JSON:{\"matrix\":\"%s\",\"n\":%d,\"nnz\":%lld,\"t_host_ms\":%.6f,\"t_gpu_full_ms\":%.6f,\"t_gpu_devx_ms\":%.6f,\"t_kernel_ms\":%.6f,\"speedup_full\":%.2f,\"speedup_devx\":%.2f,\"max_rel_err\":%.3e}\n",
           name, n, (long long)nnz, t_host, t_gpu, t_gpudx, t_kern, t_host/t_gpu, t_host/t_gpudx, rel_err);

    DSH_Free(d_offs); DSH_Free(d_lens); DSH_Free(d_cols); DSH_Free(d_vals); DSH_Free(d_z); DSH_Free(d_w);
    free(h_offs); free(h_lens); free(h_cols); free(h_vals); free(h_order); free(tmp);
    free(A.rowptr); free(A.colidx); free(A.values);
    free(z); free(w_host); free(w_gpu);
    free(g_host); free(g_gpu); free(g_gpudx); free(g_kern);
    return 0;
}
