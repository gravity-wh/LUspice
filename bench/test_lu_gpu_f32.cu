// ============================================================================
// test_lu_gpu.cu — M3 GPU LU 正确性验证（host 驱动）
// 复用 test_lu.c 的读取/COLAMD/.flt pin 逻辑 + GPU 上传/launch/下载/残差验证
// 编译：见 lu_gpu.cu 头部
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>

#include "cuda_compat.h"
#include "ngspice/colamd.h"
#include "lu_gpu_f32.cu"   // fp32 因子 kernel
// lu_factors 最小结构（splu.h 是 C 代码，nvcc C++ 下不 include）
typedef struct { int n; int *pinv;
    struct { int *Lp, *Li; double *Lx; } L;
    struct { int *Up, *Ui; double *Ux; } U; } dsh_lu_factors_min;

typedef struct { int n, nnz; int *row, *col; double *val; double *b; } s0mat;

static int read_s0(const char *mtx_path, const char *rhs_path, s0mat *M)
{
    FILE *f = fopen(mtx_path, "r");
    char line[512]; long cap = 1024;
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    M->row = (int*)malloc(cap * sizeof(int)); M->col = (int*)malloc(cap * sizeof(int));
    M->val = (double*)malloc(cap * sizeof(double)); M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        if (M->nnz >= cap) { cap *= 2; M->row = (int*)realloc(M->row, cap*sizeof(int));
            M->col = (int*)realloc(M->col, cap*sizeof(int)); M->val = (double*)realloc(M->val, cap*sizeof(double)); }
        sscanf(line, "A %d %d %lf", &M->row[M->nnz], &M->col[M->nnz], &M->val[M->nnz]);
        M->nnz++;
    }
    fclose(f);
    M->b = (double*)calloc((size_t)M->n + 1, sizeof(double));
    if (rhs_path) {
        f = fopen(rhs_path, "r");
        if (f) { while (fgets(line, sizeof line, f)) {
            int idx; double v;
            if (sscanf(line, "RHS %d %lf", &idx, &v) == 2 && idx >= 1 && idx <= M->n) M->b[idx] = v; }
            fclose(f); }
    }
    for (int i = 1; i <= M->n; i++) if (M->b[i] == 0) M->b[i] = 1.0;
    return 0;
}

// COO -> CSC（0-based），应用列序 q（q[j]=旧列；新列 j 含旧列 q[j] 的元素）
static int build_csc_q(const s0mat *M, const int *q,
                       int **Ap_out, int **Ai_out, double **Ax_out)
{
    int n = M->n;
    int *invq = NULL;
    if (q) {
        invq = (int*)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; j++) invq[q[j]] = j;   /* invq[旧列] = 新列 */
    }
    int *Ap = (int*)calloc((size_t)n + 1, sizeof(int));
    for (int i = 0; i < M->nnz; i++) {
        int old_c = M->col[i] - 1;
        int c = q ? invq[old_c] : old_c;
        if (c >= 0 && c < n) Ap[c + 1]++;
    }
    for (int i = 0; i < n; i++) Ap[i + 1] += Ap[i];
    int *Ai = (int*)malloc((size_t)M->nnz * sizeof(int));
    double *Ax = (double*)malloc((size_t)M->nnz * sizeof(double));
    int *cur = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cur, Ap, (size_t)n * sizeof(int));
    for (int i = 0; i < M->nnz; i++) {
        int old_c = M->col[i] - 1;
        int c = q ? invq[old_c] : old_c;
        int p = cur[c]++;
        Ai[p] = M->row[i] - 1;
        Ax[p] = M->val[i];
    }
    free(cur);
    if (invq) free(invq);
    /* 注意：lu_tool 的 CSC 行内未排序（tie-break 依赖行序）——此处对齐 lu_tool
       不排序，保证因子结构与 lu_tool 一致（M4 病态收敛关键） */
    *Ap_out = Ap; *Ai_out = Ai; *Ax_out = Ax;
    return 0;
}

// ============================================================================
// R7: IR SpMV 加速——host C 循环 vs GPU SELL-C-σ32（原矩阵自然序）
// 布局/kernel 复用 sellc_spmv.cu（σ=32 分片、列主序 256B 合并事务、排序分片）
// GPU 路径含每次调用的真实开销（z H2D / kernel / w D2H / sync）——IR 循环口径
// ============================================================================
#define IR_SELL_SIGMA 32
#define IR_SELL_BLOCK 256

__global__ void k_sellc_ir(const int n, const int nslices,
                           const int* __restrict__ offs,
                           const int* __restrict__ lens,
                           const int* __restrict__ cols,
                           const double* __restrict__ vals,
                           const double* __restrict__ x,
                           double* __restrict__ y) {
    const int tid = blockIdx.x * IR_SELL_BLOCK + threadIdx.x;
    const int slice = tid >> 5;              /* /σ */
    const int r = threadIdx.x & (IR_SELL_SIGMA - 1);
    if (slice >= nslices) return;
    const int row = slice * IR_SELL_SIGMA + r;
    if (row >= n) return;
    const int off = offs[slice];
    const int len = lens[slice];
    double sum = 0.0;
    for (int j = 0; j < len; j++) {
        const int idx = off + j * IR_SELL_SIGMA + r;
        sum += vals[idx] * x[cols[idx]];
    }
    y[row] = sum;                            /* 分片序；host 用 order 逆置换 */
}

/* 与 sellc_spmv.cu build_sellc 相同：按行长降序分片 + 列主序存储 */
static long long build_sellc_ir(int n, const int* rowptr, const int* colidx, const double* values,
                                int sort, int** out_offs, int** out_lens,
                                int** out_cols, double** out_vals, int** out_order) {
    int* len = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) len[i] = (int)(rowptr[i + 1] - rowptr[i]);
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
    const int nslices = (n + IR_SELL_SIGMA - 1) / IR_SELL_SIGMA;
    int* lens = (int*)malloc((size_t)nslices * sizeof(int));
    int* offs = (int*)malloc((size_t)(nslices + 1) * sizeof(int));
    offs[0] = 0;
    long long total = 0;
    for (int s = 0; s < nslices; s++) {
        int sl = 0;
        for (int r = 0; r < IR_SELL_SIGMA; r++) {
            const int k = s * IR_SELL_SIGMA + r;
            if (k < n && len[order[k]] > sl) sl = len[order[k]];
        }
        lens[s] = sl;
        total += (long long)IR_SELL_SIGMA * sl;
        offs[s + 1] = (int)total;
    }
    int* cols = (int*)malloc((size_t)total * sizeof(int));
    double* vals = (double*)malloc((size_t)total * sizeof(double));
    for (long long t = 0; t < total; t++) { cols[t] = 0; vals[t] = 0.0; }
    for (int s = 0; s < nslices; s++) {
        const int off = offs[s];
        for (int r = 0; r < IR_SELL_SIGMA; r++) {
            const int k = s * IR_SELL_SIGMA + r;
            if (k >= n) continue;
            const int orig = order[k];
            for (int e = rowptr[orig]; e < rowptr[orig + 1]; e++) {
                const int j = e - rowptr[orig];
                cols[off + (long long)j * IR_SELL_SIGMA + r] = colidx[e];
                vals[off + (long long)j * IR_SELL_SIGMA + r] = values[e];
            }
        }
    }
    free(len);
    *out_offs = offs; *out_lens = lens; *out_cols = cols; *out_vals = vals; *out_order = order;
    return total;
}

/* host C 循环 SpMV（IR 原路径，逐项同 test 原代码） */
static void host_spmv(const s0mat* M, const double* z, double* w) {
    for (int i = 0; i < M->n; i++) w[i] = 0.0;
    for (int i = 0; i < M->nnz; i++) {
        int rr = M->row[i] - 1, cc = M->col[i] - 1;
        w[rr] += M->val[i] * z[cc];
    }
}

typedef struct {
    int n, nslices;
    int *d_offs, *d_lens, *d_cols;
    double *d_vals, *d_z, *d_w;
    int* h_order;              /* 排序分片 → 逆置换；NULL=自然序 */
    double* h_tmp;
} SellC_IR;

/* GPU SELL SpMV：host x → host y（含 H2D + kernel + D2H + perm） */
static void sellc_spmv_ir(SellC_IR* S, const double* h_x, double* h_y) {
    DSH_CHECK(DSH_Memcpy(S->d_z, h_x, (size_t)S->n * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
    const int nthreads = S->nslices * IR_SELL_SIGMA;
    k_sellc_ir<<<(nthreads + IR_SELL_BLOCK - 1) / IR_SELL_BLOCK, IR_SELL_BLOCK>>>(
        S->n, S->nslices, S->d_offs, S->d_lens, S->d_cols, S->d_vals, S->d_z, S->d_w);
    DSH_CHECK(DSH_DeviceSynchronize());
    DSH_CHECK(DSH_Memcpy(h_y, S->d_w, (size_t)S->n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST));
    if (S->h_order) {
        memcpy(S->h_tmp, h_y, (size_t)S->n * sizeof(double));
        for (int k = 0; k < S->n; k++) h_y[S->h_order[k]] = S->h_tmp[k];
    }
}

/* GPU SELL SpMV：设备 x（trisolve 输出 d_xv 直接喂，省一次 D2H）→ host y */
static void sellc_spmv_ir_devx(SellC_IR* S, const double* d_x, double* h_y) {
    const int nthreads = S->nslices * IR_SELL_SIGMA;
    k_sellc_ir<<<(nthreads + IR_SELL_BLOCK - 1) / IR_SELL_BLOCK, IR_SELL_BLOCK>>>(
        S->n, S->nslices, S->d_offs, S->d_lens, S->d_cols, S->d_vals, d_x, S->d_w);
    DSH_CHECK(DSH_DeviceSynchronize());
    DSH_CHECK(DSH_Memcpy(h_y, S->d_w, (size_t)S->n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST));
    if (S->h_order) {
        memcpy(S->h_tmp, h_y, (size_t)S->n * sizeof(double));
        for (int k = 0; k < S->n; k++) h_y[S->h_order[k]] = S->h_tmp[k];
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [pin] [flt] [shift]\n", argv[0]); return 2; }
    s0mat M;
    if (read_s0(argv[1], argv[2], &M)) return 1;
    double pinv = (argc >= 4) ? atof(argv[3]) : 0.0;
    double shift = (argc >= 6) ? atof(argv[5]) : 0.0;
    printf("n=%d nnz=%d pin=%.1e shift=%.1e\n", M.n, M.nnz, pinv, shift);

    // .flt 浮动节点 pin 注入（同 test_lu.c）
    if (argc >= 5 && pinv > 0) {
        FILE *ff = fopen(argv[4], "r");
        if (ff) {
            char rb[256];
            while (fgets(rb, sizeof rb, ff)) {
                int row;
                if (sscanf(rb, "F %d", &row) == 1) {
                    int i = row - 1, found = 0;
                    for (int k = 0; k < M.nnz; k++)
                        if (M.row[k] - 1 == i && M.col[k] - 1 == i) { M.val[k] += pinv; found = 1; break; }
                    if (!found) {
                        M.row = (int*)realloc(M.row, (M.nnz + 1) * sizeof(int));
                        M.col = (int*)realloc(M.col, (M.nnz + 1) * sizeof(int));
                        M.val = (double*)realloc(M.val, (M.nnz + 1) * sizeof(double));
                        M.row[M.nnz] = i + 1; M.col[M.nnz] = i + 1; M.val[M.nnz] = pinv; M.nnz++;
                    }
                }
            }
            fclose(ff);
        }
    }

    /* Ruiz 行/列均衡（DSH_RUIZ=1 启用；参考 lu_tool.c ruiz，30 迭代） */
    double *R_sc = NULL, *C_sc = NULL;
    if (getenv("DSH_RUIZ")) {
        R_sc = (double*)malloc((size_t)M.n * sizeof(double));
        C_sc = (double*)malloc((size_t)M.n * sizeof(double));
        for (int i = 0; i < M.n; i++) { R_sc[i] = 1.0; C_sc[i] = 1.0; }
        double *rowmax = (double*)malloc((size_t)M.n * sizeof(double));
        double *colmax = (double*)malloc((size_t)M.n * sizeof(double));
        for (int it = 0; it < 30; it++) {
            for (int i = 0; i < M.n; i++) rowmax[i] = 0;
            for (int k = 0; k < M.nnz; k++) { double av = fabs(M.val[k]); if (av > rowmax[M.row[k]-1]) rowmax[M.row[k]-1] = av; }
            for (int i = 0; i < M.n; i++) { double s = rowmax[i] > 0 ? 1.0/sqrt(rowmax[i]) : 1.0; R_sc[i] *= s; }
            for (int k = 0; k < M.nnz; k++) M.val[k] *= R_sc[M.row[k]-1];
            for (int j = 0; j < M.n; j++) colmax[j] = 0;
            for (int k = 0; k < M.nnz; k++) { double av = fabs(M.val[k]); if (av > colmax[M.col[k]-1]) colmax[M.col[k]-1] = av; }
            for (int j = 0; j < M.n; j++) { double t = colmax[j] > 0 ? 1.0/sqrt(colmax[j]) : 1.0; C_sc[j] *= t; }
            for (int k = 0; k < M.nnz; k++) M.val[k] *= C_sc[M.col[k]-1];
        }
        /* b 行缩放：M.b[i] *= R_sc[i]（b_s = R b） */
        for (int i = 1; i <= M.n; i++) M.b[i] *= R_sc[i-1];
        free(rowmax); free(colmax);
        printf("ruiz applied\n");
    }

    // COLAMD（同 test_lu.c 逻辑；DSH_NATURAL=1 跳过用自然序）
    int *q = NULL;
    if (getenv("DSH_NATURAL")) { printf("natural order (skip colamd)\n"); }
    else {
        int *cp = (int*)calloc((size_t)M.n + 1, sizeof(int));
        long Alen = colamd_recommended(M.nnz, M.n, M.n);
        if (Alen < 2L * M.nnz + 16L * M.n + 16) Alen = 2L * M.nnz + 16L * M.n + 16;
        int *Ai_copy = (int*)malloc((size_t)Alen * sizeof(int));
        int *Ap_t = (int*)calloc((size_t)M.n + 1, sizeof(int));
        for (int i = 0; i < M.nnz; i++) Ap_t[M.col[i]]++;
        for (int i = 0; i < M.n; i++) { Ap_t[i + 1] += Ap_t[i]; cp[i] = Ap_t[i]; }
        cp[M.n] = M.nnz;
        int *cur = (int*)malloc((size_t)M.n * sizeof(int));
        memcpy(cur, Ap_t, (size_t)M.n * sizeof(int));
        for (int i = 0; i < M.nnz; i++) { int p = cur[M.col[i] - 1]++; Ai_copy[p] = M.row[i] - 1; }
        free(cur);
        for (int c_ = 0; c_ < M.n; c_++) {
            int lo = Ap_t[c_], hi = Ap_t[c_ + 1];
            for (int a = lo + 1; a < hi; a++) {
                int key = Ai_copy[a], b = a - 1;
                while (b >= lo && Ai_copy[b] > key) { Ai_copy[b + 1] = Ai_copy[b]; b--; }
                Ai_copy[b + 1] = key;
            }
        }
        double knobs[COLAMD_KNOBS] = {0}; int stats[COLAMD_STATS] = {0};
        int ok = colamd(M.n, M.n, (int)Alen, Ai_copy, cp, knobs, stats);
        if (!ok) { fprintf(stderr, "colamd failed stats=%d\n", stats[0]); return 1; }
        q = (int*)malloc((size_t)M.n * sizeof(int));
        for (int j = 0; j < M.n; j++) q[j] = cp[j];
        free(cp); free(Ai_copy); free(Ap_t);
        printf("colamd ok\n");
    }

    // CSC（COLAMD 序应用）
    int *Ap, *Ai; double *Ax;
    build_csc_q(&M, q, &Ap, &Ai, &Ax);

    // GPU 分配
    int n = M.n;

    /* ---- R7: SELL-C-σ32 SpMV 准备（IR 用；原矩阵自然序 CSR→SELL）---- */
    SellC_IR sellc; memset(&sellc, 0, sizeof sellc);
    const int gpu_spmv = getenv("DSH_GPU_SPMV") != NULL;
    const int skip_lutool = getenv("DSH_SKIP_LUTOOL") != NULL;
    {
        int* crp = (int*)calloc((size_t)n + 1, sizeof(int));
        for (int i = 0; i < M.nnz; i++) crp[M.row[i] - 1 + 1]++;
        for (int i = 0; i < n; i++) crp[i + 1] += crp[i];
        int* cci = (int*)malloc((size_t)M.nnz * sizeof(int));
        double* cvv = (double*)malloc((size_t)M.nnz * sizeof(double));
        int* cur = (int*)malloc((size_t)n * sizeof(int));
        memcpy(cur, crp, (size_t)n * sizeof(int));
        for (int i = 0; i < M.nnz; i++) { int p = cur[M.row[i] - 1]++; cci[p] = M.col[i] - 1; cvv[p] = M.val[i]; }
        free(cur);
        int* h_offs = NULL; int* h_lens = NULL; int* h_cols = NULL; int* h_order = NULL; double* h_vals = NULL;
        long long stored = build_sellc_ir(n, crp, cci, cvv, 1, &h_offs, &h_lens, &h_cols, &h_vals, &h_order);
        sellc.n = n; sellc.nslices = (n + IR_SELL_SIGMA - 1) / IR_SELL_SIGMA;
        sellc.h_order = h_order;
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_offs, (size_t)(sellc.nslices + 1) * sizeof(int)));
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_lens, (size_t)sellc.nslices * sizeof(int)));
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_cols, (size_t)stored * sizeof(int)));
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_vals, (size_t)stored * sizeof(double)));
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_z, (size_t)n * sizeof(double)));
        DSH_CHECK(DSH_Malloc((void**)&sellc.d_w, (size_t)n * sizeof(double)));
        DSH_CHECK(DSH_Memcpy(sellc.d_offs, h_offs, (size_t)(sellc.nslices + 1) * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
        DSH_CHECK(DSH_Memcpy(sellc.d_lens, h_lens, (size_t)sellc.nslices * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
        DSH_CHECK(DSH_Memcpy(sellc.d_cols, h_cols, (size_t)stored * sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
        DSH_CHECK(DSH_Memcpy(sellc.d_vals, h_vals, (size_t)stored * sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
        sellc.h_tmp = (double*)malloc((size_t)n * sizeof(double));
        free(h_offs); free(h_lens); free(h_cols); free(h_vals);
        free(crp); free(cci); free(cvv);
        printf("R7 SELL-C-s32 IR SpMV: n=%d nnz=%d stored=%lld pad_waste=%.1f%% nslices=%d gpu_spmv=%d\n",
               n, M.nnz, stored, 100.0 * ((double)stored - (double)M.nnz) / (double)M.nnz, sellc.nslices, gpu_spmv);
    }
    long cap = (long)M.nnz * 24 + 1024;
    int *d_Ap, *d_Ai; float *d_Ax;
    int *d_Lp, *d_Li; float *d_Lx;
    int *d_Up, *d_Ui; float *d_Ux;
    int *d_pinv, *d_xi; float *d_x; double *d_xv;
    int *d_lnnz, *d_unnz; int *d_sing;
    /* host fp32 副本（Ax/Lx/Ux 存储 fp32） */
    float *Ax32 = (float*)malloc((size_t)M.nnz * sizeof(float));
    for (int i = 0; i < M.nnz; i++) Ax32[i] = (float)Ax[i];
    if (shift > 0) {
        /* shift 因子：对角 + shift*|diag|（仅因子化；参考 lu_tool32c_m1 --shift） */
        for (int i = 0; i < M.n; i++) {
            for (int k = 0; k < M.nnz; k++) {
                if (M.row[k]-1 == i && M.col[k]-1 == i) {
                    Ax32[k] = (float)(Ax[k] + shift * fabs(Ax[k]));
                    break;
                }
            }
        }
    }
    DSH_CHECK(DSH_Malloc((void**)&d_Ap, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ai, (size_t)M.nnz*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ax, (size_t)M.nnz*sizeof(float)));
    DSH_CHECK(DSH_Malloc((void**)&d_Lp, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Li, cap*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Lx, cap*sizeof(float)));
    DSH_CHECK(DSH_Malloc((void**)&d_Up, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ui, cap*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ux, cap*sizeof(float)));
    DSH_CHECK(DSH_Malloc((void**)&d_pinv, n*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_xi, (2*n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_x, n*sizeof(float)));
    DSH_CHECK(DSH_Malloc((void**)&d_xv, n*sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_lnnz, sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_unnz, sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_sing, sizeof(int)));

    DSH_CHECK(DSH_Memcpy(d_Ap, Ap, (n+1)*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_Ai, Ai, (size_t)M.nnz*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_Ax, Ax32, (size_t)M.nnz*sizeof(float), DSH_MEMCPY_HOST_TO_DEVICE));

    // launch LU + trisolve
    /* diagrow[k] = 重排前对角行（q 序：q[k]；自然序：k） */
    int *diagrow = (int*)malloc((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) diagrow[k] = q ? q[k] : k;
    int *d_diag;
    DSH_CHECK(DSH_Malloc((void**)&d_diag, (size_t)n*sizeof(int)));
    DSH_CHECK(DSH_Memcpy(d_diag, diagrow, (size_t)n*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    lu_factor_f32_kernel<<<1, 1>>>(n, d_Ap, d_Ai, d_Ax, d_diag,
                               d_Lp, d_Li, d_Lx, cap,
                               d_Up, d_Ui, d_Ux, cap,
                               d_pinv, d_xi, d_x,
                               d_lnnz, d_unnz, d_sing);
    {
        /* 上传 RHS b、列序 q、工作数组 */
        double *h_b = (double*)malloc((size_t)n * sizeof(double));
        for (int i = 0; i < n; i++) h_b[i] = M.b[i + 1];
        double *d_b; int *d_q, *d_work_i; double *d_work_d;
        DSH_CHECK(DSH_Malloc((void**)&d_b, (size_t)n*sizeof(double)));
        d_q = NULL;
        if (q) {
            DSH_CHECK(DSH_Malloc((void**)&d_q, (size_t)n*sizeof(int)));
            DSH_CHECK(DSH_Memcpy(d_q, q, (size_t)n*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
        }
        DSH_CHECK(DSH_Malloc((void**)&d_work_i, (size_t)n*sizeof(int)));
        DSH_CHECK(DSH_Malloc((void**)&d_work_d, (size_t)(2*n)*sizeof(double)));
        DSH_CHECK(DSH_Memcpy(d_b, h_b, (size_t)n*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
        /* ---- 对照：lu_tool.c 版 GMRES（复制逻辑，GPU 因子下载后 host 预条件）----
           （R7: DSH_SKIP_LUTOOL=1 跳过——fp32 因子下该对照路径发散，非本工具目标） */
        if (skip_lutool) { fprintf(stderr, "[skip] lutool-gmres comparison block\n"); }
        else {
            /* 下载 GPU fp32 因子 → fp64 lu_factors */
            int *h_Lp=(int*)malloc((size_t)(n+1)*sizeof(int)), *h_Li=(int*)malloc((size_t)cap*sizeof(int));
            float *h_Lx=(float*)malloc((size_t)cap*sizeof(float));
            int *h_Up=(int*)malloc((size_t)(n+1)*sizeof(int)), *h_Ui=(int*)malloc((size_t)cap*sizeof(int));
            float *h_Ux=(float*)malloc((size_t)cap*sizeof(float));
            int *h_pinv=(int*)malloc((size_t)n*sizeof(int));
            DSH_Memcpy(h_Lp, d_Lp, (size_t)(n+1)*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_Li, d_Li, (size_t)cap*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_Lx, d_Lx, (size_t)cap*sizeof(float), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_Up, d_Up, (size_t)(n+1)*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_Ui, d_Ui, (size_t)cap*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_Ux, d_Ux, (size_t)cap*sizeof(float), DSH_MEMCPY_DEVICE_TO_HOST);
            DSH_Memcpy(h_pinv, d_pinv, (size_t)n*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
            /* 构造 lu_factors（fp64 数值：fp32 读出转 double） */
            dsh_lu_factors_min lf; memset(&lf, 0, sizeof(lf));
            lf.n = n; lf.pinv = h_pinv;
            lf.n = n; lf.L.Lp = h_Lp; lf.L.Li = h_Li;
            lf.L.Lx = (double*)malloc((size_t)cap*sizeof(double));
            for (long k2=0;k2<cap;k2++) lf.L.Lx[k2] = (double)h_Lx[k2];
            lf.U.Up = h_Up; lf.U.Ui = h_Ui;
            lf.U.Ux = (double*)malloc((size_t)cap*sizeof(double));
            for (long k2=0;k2<cap;k2++) lf.U.Ux[k2] = (double)h_Ux[k2];
            /* lu_tool 版 GMRES 主体（与 lu_tool.c gmres 相同逻辑） */
            double *x0 = (double*)calloc((size_t)n, sizeof(double));
            double *r = (double*)malloc((size_t)n*sizeof(double));
            double *av = (double*)malloc((size_t)n*sizeof(double));
            double *v = (double*)malloc((size_t)51*n*sizeof(double));
            double *z = (double*)malloc((size_t)n*sizeof(double));
            double *w = (double*)malloc((size_t)n*sizeof(double));
            double *Hm = (double*)calloc((size_t)51*51, sizeof(double));
            double *cs = (double*)calloc(51, sizeof(double)), *sn = (double*)calloc(51, sizeof(double));
            double *g = (double*)calloc(51, sizeof(double)), *y = (double*)calloc(51, sizeof(double));
            double *temp = (double*)malloc((size_t)n*sizeof(double));
            double *bvec = (double*)malloc((size_t)n*sizeof(double));
            for (int i=0;i<n;i++) bvec[i] = M.b[i+1];
            double normb = 0; for (int i=0;i<n;i++) normb += bvec[i]*bvec[i]; normb = sqrt(normb);
            int its = 0, outer = 0;
            double fres = 0;
            /* matvec av = A x0 */
            for (int i=0;i<n;i++) av[i]=0;
            for (int i=0;i<M.nnz;i++){ int rr=M.row[i]-1, cc=M.col[i]-1; av[rr]+=M.val[i]*x0[cc]; }
            for (int i=0;i<n;i++) r[i]=bvec[i]-av[i];
            double beta = 0; for (int i=0;i<n;i++) beta += r[i]*r[i]; beta = sqrt(beta);
            fres = beta/normb;
            while (beta/normb > 1e-10 && its < 300) {
                outer++;
                for (int i=0;i<n;i++) v[i]=r[i]/beta;
                g[0]=beta;
                int jj = 0;
                for (int j=0;j<50;j++) {
                    /* lu_apply: z = F^-1 v_j */
                    { double *pb=(double*)malloc(n*sizeof(double)); double *yy=(double*)malloc(n*sizeof(double));
                      int *rowat=(int*)malloc(n*sizeof(int));
                      for (int k2=0;k2<n;k2++) rowat[k2]=-1;
                      for (int i2=0;i2<n;i2++) if (lf.pinv[i2]>=0) rowat[lf.pinv[i2]]=i2;
                      for (int k2=0;k2<n;k2++) pb[k2]=rowat[k2]>=0?v[rowat[k2]]:0.0;
                      for (int k2=0;k2<n;k2++){ double acc=pb[k2]/lf.L.Lx[lf.L.Lp[k2]];
                        for (int p2=lf.L.Lp[k2]+1;p2<lf.L.Lp[k2+1];p2++) pb[lf.L.Li[p2]]-=lf.L.Lx[p2]*acc;
                        yy[k2]=acc; }
                      for (int k2=n-1;k2>=0;k2--){ yy[k2]/=lf.U.Ux[lf.U.Up[k2+1]-1];
                        for (int p2=lf.U.Up[k2];p2<lf.U.Up[k2+1]-1;p2++) yy[lf.U.Ui[p2]]-=lf.U.Ux[p2]*yy[k2]; }
                      for (int k2=0;k2<n;k2++) z[q?q[k2]:k2]=yy[k2];
                      free(pb);free(yy);free(rowat); }
                    /* w = A z */
                    for (int i2=0;i2<n;i2++) w[i2]=0;
                    for (int i2=0;i2<M.nnz;i2++){ int rr=M.row[i2]-1, cc=M.col[i2]-1; w[rr]+=M.val[i2]*z[cc]; }
                    for (int i2=0;i2<=j;i2++){
                        double h=0; for (int k2=0;k2<n;k2++) h+=w[k2]*v[i2*n+k2];
                        Hm[j*51+i2]=h;
                        for (int k2=0;k2<n;k2++) w[k2]-=h*v[i2*n+k2];
                    }
                    double hh=0; for (int k2=0;k2<n;k2++) hh+=w[k2]*w[k2]; hh=sqrt(hh);
                    Hm[j*51+j+1]=hh;
                    for (int i2=0;i2<j;i2++){
                        double tmp=cs[i2]*Hm[j*51+i2]+sn[i2]*Hm[j*51+i2+1];
                        Hm[j*51+i2+1]=-sn[i2]*Hm[j*51+i2]+cs[i2]*Hm[j*51+i2+1];
                        Hm[j*51+i2]=tmp;
                    }
                    double hjj=Hm[j*51+j], hjn=Hm[j*51+j+1];
                    double t=sqrt(hjj*hjj+hjn*hjn);
                    if (t==0){ jj=j; break; }
                    cs[j]=hjj/t; sn[j]=hjn/t;
                    Hm[j*51+j]=t; Hm[j*51+j+1]=0;
                    g[j+1]=-sn[j]*g[j]; g[j]=cs[j]*g[j];
                    its++;
                    if (fabs(g[j+1])/normb <= 1e-10){ jj=j+1; break; }
                    if (hh==0){ jj=j+1; break; }
                    for (int i2=0;i2<n;i2++) v[(j+1)*n+i2]=w[i2]/hh;
                }
                int mk = jj>0?jj:50;
                for (int j=mk-1;j>=0;j--){ double acc=g[j];
                  for (int i=j+1;i<mk;i++) acc-=Hm[j*51+i]*y[i];
                  y[j]=Hm[j*51+j]!=0?acc/Hm[j*51+j]:0; }
                for (int i=0;i<n;i++) temp[i]=0;
                for (int j=0;j<mk;j++) if (y[j]!=0) for (int i=0;i<n;i++) temp[i]+=y[j]*v[j*n+i];
                /* lu_apply(temp) → z */
                { double *pb=(double*)malloc(n*sizeof(double)); double *yy=(double*)malloc(n*sizeof(double));
                  int *rowat=(int*)malloc(n*sizeof(int));
                  for (int k2=0;k2<n;k2++) rowat[k2]=-1;
                  for (int i2=0;i2<n;i2++) if (lf.pinv[i2]>=0) rowat[lf.pinv[i2]]=i2;
                  for (int k2=0;k2<n;k2++) pb[k2]=rowat[k2]>=0?temp[rowat[k2]]:0.0;
                  for (int k2=0;k2<n;k2++){ double acc=pb[k2]/lf.L.Lx[lf.L.Lp[k2]];
                    for (int p2=lf.L.Lp[k2]+1;p2<lf.L.Lp[k2+1];p2++) pb[lf.L.Li[p2]]-=lf.L.Lx[p2]*acc;
                    yy[k2]=acc; }
                  for (int k2=n-1;k2>=0;k2--){ yy[k2]/=lf.U.Ux[lf.U.Up[k2+1]-1];
                    for (int p2=lf.U.Up[k2];p2<lf.U.Up[k2+1]-1;p2++) yy[lf.U.Ui[p2]]-=lf.U.Ux[p2]*yy[k2]; }
                  for (int k2=0;k2<n;k2++) z[q?q[k2]:k2]=yy[k2];
                  free(pb);free(yy);free(rowat); }
                for (int i=0;i<n;i++) x0[i]+=z[i];
                for (int i=0;i<n;i++) av[i]=0;
                for (int i=0;i<M.nnz;i++){ int rr=M.row[i]-1, cc=M.col[i]-1; av[rr]+=M.val[i]*x0[cc]; }
                for (int i=0;i<n;i++) r[i]=bvec[i]-av[i];
                beta=0; for (int i=0;i<n;i++) beta+=r[i]*r[i]; beta=sqrt(beta);
                fres=beta/normb;
                fprintf(stderr, "[lutgmres] outer=%d res=%g\n", outer, fres);
                if (fres <= 1e-10) break;
            }
            printf("LUTOOL-GMRES: outer=%d its=%d final=%.3e %s\n", outer, its, fres,
                   fres<=1e-6?"PASS":"FAIL");
            free(x0);free(r);free(av);free(v);free(z);free(w);free(Hm);free(cs);free(sn);free(g);free(y);free(temp);free(bvec);
            free(h_Lp);free(h_Li);free(h_Lx);free(h_Up);free(h_Ui);free(h_Ux);free(h_pinv);
            free(lf.L.Lx);free(lf.U.Ux);
        }

        /* ---- GMRES-IR（host GMRES(50) + GPU trisolve 预条件 + SpMV）----
           （R7: DSH_GPU_SPMV=1 时 SpMV 用 GPU SELL-C-σ32，含每迭代真实传输开销） */
        double *xcur = (double*)calloc((size_t)n, sizeof(double));
        double *xg2 = (double*)malloc((size_t)n * sizeof(double));
        /* 右预条件 GMRES：解 A F^{-1} u = b, x = F^{-1} u
           Arnoldi 正交化 + Givens 最小二乘；M1 处方参数 outer<=7/inner=50 */
        {
            int m = 50;
            double **V = (double**)malloc((size_t)(m+1) * sizeof(double*));
            for (int i = 0; i <= m; i++) V[i] = (double*)malloc((size_t)n * sizeof(double));
            double *H = (double*)calloc((size_t)(m+1)*m, sizeof(double));
            double *cs = (double*)calloc((size_t)m, sizeof(double));
            double *sn = (double*)calloc((size_t)m, sizeof(double));
            double *g = (double*)calloc((size_t)(m+1), sizeof(double));
            double *w = (double*)malloc((size_t)n * sizeof(double));
            double *z = (double*)malloc((size_t)n * sizeof(double));
            double *r0 = (double*)malloc((size_t)n * sizeof(double));
            double *rhs = (double*)malloc((size_t)n * sizeof(double));
            for (int i = 0; i < n; i++) rhs[i] = M.b[i+1];
            double norm_b = 0.0;
            for (int i = 0; i < n; i++) norm_b += rhs[i]*rhs[i];
            norm_b = sqrt(norm_b);
            int total_iters = 0;
            double final_rel = 0.0;
            /* R7: IR 计时 + SpMV 计数；R8: 并行 trisolve（默认开，DSH_TRISOLVE_SERIAL=1 回退单 thread） */
            struct timespec ta, tb, t_ir0;
            clock_gettime(CLOCK_MONOTONIC, &t_ir0);
            long long spmv_calls = 0;
            double spmv_ms = 0.0;
            double trisolve_ms = 0.0;
            const int par_trisolve = getenv("DSH_TRISOLVE_SERIAL") == NULL;

            for (int outer = 0; outer < 7; outer++) {
                /* r0 = b - A xcur（R7: host SpMV 或 GPU SELL-C-σ32） */
                clock_gettime(CLOCK_MONOTONIC, &ta);
                if (gpu_spmv) {
                    sellc_spmv_ir(&sellc, xcur, w);
                    for (int i = 0; i < n; i++) r0[i] = rhs[i] - w[i];
                } else {
                    host_spmv(&M, xcur, w);
                    for (int i = 0; i < n; i++) r0[i] = rhs[i] - w[i];
                }
                clock_gettime(CLOCK_MONOTONIC, &tb);
                spmv_ms += (tb.tv_sec-ta.tv_sec)*1e3 + (tb.tv_nsec-ta.tv_nsec)/1e6;
                spmv_calls++;
                double beta = 0.0;
                for (int i = 0; i < n; i++) beta += r0[i]*r0[i];
                beta = sqrt(beta);
                fprintf(stderr, "[gmres] outer=%d res=%g\n", outer, beta/norm_b);
                if (beta/norm_b <= 1e-10) { final_rel = beta/norm_b; break; }
                for (int i = 0; i < n; i++) V[0][i] = r0[i]/beta;
                g[0] = beta;
                int j = 0;
                for (j = 0; j < m; j++) {
                    /* z = F^{-1} v_j（GPU trisolve 预条件；R8: 默认并行版） */
                    DSH_Memcpy(d_b, V[j], (size_t)n*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
                    clock_gettime(CLOCK_MONOTONIC, &ta);
                    if (par_trisolve) {
                        trisolve_par_f32_kernel<<<1, 256>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                                            d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
                    } else {
                        trisolve_f32_kernel<<<1, 1>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                                      d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
                    }
                    DSH_DeviceSynchronize();
                    clock_gettime(CLOCK_MONOTONIC, &tb);
                    trisolve_ms += (tb.tv_sec-ta.tv_sec)*1e3 + (tb.tv_nsec-ta.tv_nsec)/1e6;
                    /* w = A z（R7: host SpMV 或 GPU SELL-C-σ32；GPU 路径 z 留设备端
                       直接喂 kernel——省一次 D2H，trisolve 输出 d_xv 即自然序） */
                    clock_gettime(CLOCK_MONOTONIC, &ta);
                    if (gpu_spmv) {
                        sellc_spmv_ir_devx(&sellc, d_xv, w);
                    } else {
                        DSH_Memcpy(z, d_xv, (size_t)n*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
                        host_spmv(&M, z, w);
                    }
                    clock_gettime(CLOCK_MONOTONIC, &tb);
                    spmv_ms += (tb.tv_sec-ta.tv_sec)*1e3 + (tb.tv_nsec-ta.tv_nsec)/1e6;
                    spmv_calls++;
                    /* Arnoldi 正交化（两遍 GS，病态矩阵必需） */
                    for (int pass = 0; pass < 2; pass++) {
                        for (int i = 0; i <= j; i++) {
                            double h = 0.0;
                            for (int k2 = 0; k2 < n; k2++) h += V[i][k2] * w[k2];
                            if (pass == 0) H[i*m + j] += h;
                            else H[i*m + j] += h;
                            for (int k2 = 0; k2 < n; k2++) w[k2] -= h * V[i][k2];
                        }
                    }
                    double hn = 0.0;
                    for (int k2 = 0; k2 < n; k2++) hn += w[k2]*w[k2];
                    hn = sqrt(hn);
                    if (hn < 1e-14) { j++; break; }
                    H[(j+1)*m + j] = hn;
                    for (int k2 = 0; k2 < n; k2++) V[j+1][k2] = w[k2]/hn;
                    /* Givens 旋转消 H 下对角 */
                    for (int i = 0; i < j; i++) {
                        double t = H[i*m+j];
                        H[i*m+j] = cs[i]*t + sn[i]*H[(i+1)*m+j];
                        H[(i+1)*m+j] = -sn[i]*t + cs[i]*H[(i+1)*m+j];
                    }
                    double t1 = H[j*m+j], t2 = H[(j+1)*m+j];
                    double r = sqrt(t1*t1 + t2*t2);
                    cs[j] = t1/r; sn[j] = t2/r;
                    H[j*m+j] = r; H[(j+1)*m+j] = 0.0;
                    g[j+1] = -sn[j]*g[j];
                    g[j] = cs[j]*g[j];
                    if (fabs(g[j+1])/norm_b <= 1e-10) { j++; break; }
                }
                total_iters += j;
                /* 解上三角 H y = g（回代，R = H[0..j-1, 0..j-1]） */
                double *y = (double*)calloc((size_t)j, sizeof(double));
                for (int i = j-1; i >= 0; i--) {
                    double s = g[i];
                    for (int k2 = i+1; k2 < j; k2++) s -= H[i*m+k2]*y[k2];
                    y[i] = s / H[i*m+i];
                }
                /* xcur += F^{-1} (V y) */
                for (int i = 0; i < n; i++) z[i] = 0.0;
                for (int k2 = 0; k2 < j; k2++)
                    for (int i = 0; i < n; i++) z[i] += y[k2]*V[k2][i];
                DSH_Memcpy(d_b, z, (size_t)n*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
                clock_gettime(CLOCK_MONOTONIC, &ta);
                if (par_trisolve) {
                    trisolve_par_f32_kernel<<<1, 256>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                                        d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
                } else {
                    trisolve_f32_kernel<<<1, 1>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                                  d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
                }
                DSH_DeviceSynchronize();
                clock_gettime(CLOCK_MONOTONIC, &tb);
                trisolve_ms += (tb.tv_sec-ta.tv_sec)*1e3 + (tb.tv_nsec-ta.tv_nsec)/1e6;
                DSH_Memcpy(xg2, d_xv, (size_t)n*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
                for (int i = 0; i < n; i++) xcur[i] += xg2[i];
                free(y);
                /* 检查收敛（R7: host SpMV 或 GPU SELL-C-σ32） */
                clock_gettime(CLOCK_MONOTONIC, &ta);
                if (gpu_spmv) {
                    sellc_spmv_ir(&sellc, xcur, w);
                    for (int i = 0; i < n; i++) r0[i] = rhs[i] - w[i];
                } else {
                    host_spmv(&M, xcur, w);
                    for (int i = 0; i < n; i++) r0[i] = rhs[i] - w[i];
                }
                clock_gettime(CLOCK_MONOTONIC, &tb);
                spmv_ms += (tb.tv_sec-ta.tv_sec)*1e3 + (tb.tv_nsec-ta.tv_nsec)/1e6;
                spmv_calls++;
                double nr = 0.0;
                for (int i = 0; i < n; i++) nr += r0[i]*r0[i];
                final_rel = sqrt(nr)/norm_b;
                if (final_rel <= 1e-10) break;
                /* 重启：V 复用于下轮 */
            }
            for (int j2 = 0; j2 <= m; j2++) free(V[j2]);
            free(V); free(H); free(cs); free(sn); free(g); free(w); free(z); free(r0); free(rhs);
            /* 最终残差 */
            {
                double num = 0, den = 0;
                double *bh = (double*)calloc((size_t)n, sizeof(double));
                for (int i = 0; i < M.nnz; i++) { int rr=M.row[i]-1, cc=M.col[i]-1; bh[rr] += M.val[i]*xcur[cc]; }
                for (int i = 0; i < n; i++) { double d = bh[i]-M.b[i+1]; num += d*d; den += M.b[i+1]*M.b[i+1]; }
                free(bh);
                double rel = sqrt(num/den);
                printf("GPU GMRES-IR: inner_total=%d final_res=%.3e  %s\n", total_iters, rel,
                       rel <= 1e-6 ? "PASS (S1: L2<=1e-6)" : "FAIL");
            }
            /* R7: IR 计时汇总 */
            clock_gettime(CLOCK_MONOTONIC, &tb);
            double ir_ms = (tb.tv_sec-t_ir0.tv_sec)*1e3 + (tb.tv_nsec-t_ir0.tv_nsec)/1e6;
            printf("IR-TIMING: spmv_mode=%s trisolve=%s spmv_calls=%lld spmv_ms=%.3f trisolve_ms=%.3f ir_ms=%.3f (spmv_share=%.1f%% trisolve_share=%.1f%%)\n",
                   gpu_spmv ? "gpu-sellc" : "host", par_trisolve ? "par(256)" : "serial(1)",
                   spmv_calls, spmv_ms, trisolve_ms, ir_ms,
                   ir_ms > 0 ? 100.0*spmv_ms/ir_ms : 0.0,
                   ir_ms > 0 ? 100.0*trisolve_ms/ir_ms : 0.0);
        }
        /* 最终残差（用 xcur） */

        /* 最终残差（用 xcur） */
        {
            double num = 0, den = 0;
            double *bh = (double*)calloc((size_t)n, sizeof(double));
            (void)0;
        }
        free(xcur); free(xg2);
        DSH_CHECK(DSH_DeviceSynchronize());
        free(h_b);
        DSH_Free(d_b); DSH_Free(d_q); DSH_Free(d_work_i); DSH_Free(d_work_d);
    }

    DSH_CHECK(DSH_DeviceSynchronize());

    int lnnz, unnz, sing;
    DSH_CHECK(DSH_Memcpy(&lnnz, d_lnnz, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    DSH_CHECK(DSH_Memcpy(&unnz, d_unnz, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    DSH_CHECK(DSH_Memcpy(&sing, d_sing, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    printf("GPU LU(f32): lnnz=%d unnz=%d singular=%d\n", lnnz, unnz, sing);
    printf("MEM: Ax fp32 %.2f MB + L/U fp32 %.2f MB (vs fp64 %.2f MB, 2.00x 减半)\n",
           (double)M.nnz*4/1048576, (double)(lnnz+unnz)*4/1048576,
           (double)(M.nnz+(lnnz+unnz))*8/1048576);
    {
        int *h_Up2=(int*)malloc((n+1)*sizeof(int));
        DSH_Memcpy(h_Up2, d_Up, (n+1)*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
        int cnt[8]={0};
        for (int k=0;k<n;k++){ int len=h_Up2[k+1]-h_Up2[k]; cnt[len<=1?0:len<=2?1:len<=4?2:len<=8?3:len<=16?4:len<=32?5:len<=64?6:7]++; }
        { FILE *lf=fopen("/tmp/l_gpu.txt","w");
          int *h_Lp2=(int*)malloc((n+1)*sizeof(int));
          int *h_Li2=(int*)malloc(cap*sizeof(int)); double *h_Lx2=(double*)malloc(cap*sizeof(double));
          DSH_Memcpy(h_Lp2, d_Lp, (n+1)*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
          DSH_Memcpy(h_Li2, d_Li, cap*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
          DSH_Memcpy(h_Lx2, d_Lx, cap*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
          for (int k=0;k<n;k++){ fprintf(lf,"%d %d |", k, h_Lp2[k+1]-h_Lp2[k]);
            for (int p=h_Lp2[k];p<h_Lp2[k+1];p++) fprintf(lf," (%d,%.6g)", h_Li2[p], h_Lx2[p]); fprintf(lf,"\n"); }
          fclose(lf); free(h_Lp2); free(h_Li2); free(h_Lx2); }
        { FILE *uf=fopen("/tmp/u_gpu.txt","w");
          int *h_Ui2=(int*)malloc(cap*sizeof(int)); double *h_Ux2=(double*)malloc(cap*sizeof(double));
          DSH_Memcpy(h_Ui2, d_Ui, cap*sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST);
          DSH_Memcpy(h_Ux2, d_Ux, cap*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
          for (int k=0;k<n;k++){ fprintf(uf,"%d %d |", k, h_Up2[k+1]-h_Up2[k]);
            for (int p=h_Up2[k];p<h_Up2[k+1];p++) fprintf(uf," (%d,%.6g)", h_Ui2[p], h_Ux2[p]); fprintf(uf,"\n"); }
          fclose(uf); free(h_Ui2); free(h_Ux2); }
        printf("GPU U 列长分布 [1,2,4,8,16,32,64,>64]: %d %d %d %d %d %d %d %d\n", cnt[0],cnt[1],cnt[2],cnt[3],cnt[4],cnt[5],cnt[6],cnt[7]);
        free(h_Up2);
    }
    if (sing >= 0) return 1;

    // 残差验证 ||b - A x||/||b||
    double *xg = (double*)malloc((size_t)n * sizeof(double));
    DSH_CHECK(DSH_Memcpy(xg, d_xv, (size_t)n * sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST));
    {
        double num = 0, den = 0;
        double *bh = (double*)calloc((size_t)n, sizeof(double));
        for (int i = 0; i < M.nnz; i++) {
            int r = M.row[i] - 1, c = M.col[i] - 1;
            bh[r] += M.val[i] * xg[c];
        }
        for (int i = 0; i < n; i++) {
            double d = bh[i] - M.b[i + 1];
            num += d * d; den += M.b[i + 1] * M.b[i + 1];
        }
        free(bh);
        double rel = sqrt(num / den);
        printf("GPU RESIDUAL ||b-Ax||/||b|| = %.3e  %s\n", rel,
               rel <= 1e-6 ? "PASS (S1: L2<=1e-6)" : "FAIL");
    }

    DSH_Free(d_Ap); DSH_Free(d_Ai); DSH_Free(d_Ax);
    DSH_Free(d_Lp); DSH_Free(d_Li); DSH_Free(d_Lx);
    DSH_Free(d_Up); DSH_Free(d_Ui); DSH_Free(d_Ux);
    DSH_Free(d_pinv); DSH_Free(d_xi); DSH_Free(d_x); DSH_Free(d_xv);
    DSH_Free(d_lnnz); DSH_Free(d_unnz); DSH_Free(d_sing); DSH_Free(d_diag); free(diagrow);
    free(Ap); free(Ai); free(Ax); free(Ax32); free(q); free(xg);
    DSH_Free(sellc.d_offs); DSH_Free(sellc.d_lens); DSH_Free(sellc.d_cols);
    DSH_Free(sellc.d_vals); DSH_Free(sellc.d_z); DSH_Free(sellc.d_w);
    free(sellc.h_order); free(sellc.h_tmp);
    free(M.row); free(M.col); free(M.val); free(M.b);
    return 0;
}
