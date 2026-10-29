// ============================================================================
// test_lu_gpu_par.cu — M3 GPU 稀疏左视 LU 并行版验证（host 驱动）
//
// 同一数据通路（s0 → .flt pin → COLAMD q → q 序 CSC）跑三路：
//   1. CPU 参考（splu.h 封装 cpu_ref_lu.c）→ x_cpu
//   2. 单 thread GPU（lu_gpu.cu 已验证版）→ x_single（时间基线）
//   3. 并行 GPU（lu_gpu_par.cu）→ x_par（本里程碑）
// 验收：x_par vs x_cpu（或 x_single）相对 L2 <= 1e-6 + 残差 + 时间 + Occupancy。
//
// 编译：
//   nvcc -O3 -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src \
//        -I /home/gravity/n46-mix/src/include -o test_lu_gpu_par \
//        src/solver/test_lu_gpu_par.cu src/solver/lu_gpu_par.cu \
//        src/solver/lu_gpu.cu src/solver/cpu_ref_lu.c \
//        /home/gravity/n46-mix/src/maths/KLU/colamd.c \
//        /home/gravity/n46-mix/src/maths/KLU/colamd_global.c -lm
// 用法：./test_lu_gpu_par <s0.mtx> <s0.rhs> [pin] [flt]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include "ngspice/colamd.h"
#include "lu_gpu.h"
#include "lu_gpu_par.h"
#include "cpu_ref_lu.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA FAIL %s:%d  %s: %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

typedef struct { int n, nnz; int *row, *col; double *val; double *b; } s0mat;

/* ---- s0 格式读取（同 test_lu.c）---- */
static int read_s0(const char *mtx_path, const char *rhs_path, s0mat *M)
{
    FILE *f = fopen(mtx_path, "r");
    char line[512]; long cap = 1024;
    if (!f) { fprintf(stderr, "open fail %s\n", mtx_path); return 1; }
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    M->row = (int *)malloc(cap * sizeof(int));
    M->col = (int *)malloc(cap * sizeof(int));
    M->val = (double *)malloc(cap * sizeof(double));
    M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        if (M->nnz >= cap) {
            cap *= 2;
            M->row = (int *)realloc(M->row, cap * sizeof(int));
            M->col = (int *)realloc(M->col, cap * sizeof(int));
            M->val = (double *)realloc(M->val, cap * sizeof(double));
        }
        sscanf(line, "A %d %d %lf", &M->row[M->nnz], &M->col[M->nnz],
               &M->val[M->nnz]);
        M->nnz++;
    }
    fclose(f);
    M->b = (double *)calloc((size_t)M->n + 1, sizeof(double));
    if (rhs_path) {
        f = fopen(rhs_path, "r");
        if (f) {
            while (fgets(line, sizeof line, f)) {
                int idx; double v;
                if (sscanf(line, "RHS %d %lf", &idx, &v) == 2 &&
                    idx >= 1 && idx <= M->n)
                    M->b[idx] = v;
            }
            fclose(f);
        }
    }
    for (int i = 1; i <= M->n; i++) if (M->b[i] == 0) M->b[i] = 1.0;
    return 0;
}

/* ---- .flt 浮动节点 → COO 对角 pin（同 test_lu.c）---- */
static void apply_flt_pin(s0mat *M, const char *flt_path, double pinv)
{
    if (!flt_path || pinv <= 0) return;
    FILE *ff = fopen(flt_path, "r");
    if (!ff) return;
    char rb[256];
    while (fgets(rb, sizeof rb, ff)) {
        int row;
        if (sscanf(rb, "F %d", &row) == 1) {
            int i = row - 1, found = 0;
            for (int k = 0; k < M->nnz; k++)
                if (M->row[k] - 1 == i && M->col[k] - 1 == i) {
                    M->val[k] += pinv; found = 1; break;
                }
            if (!found) {
                M->row = (int *)realloc(M->row, (M->nnz + 1) * sizeof(int));
                M->col = (int *)realloc(M->col, (M->nnz + 1) * sizeof(int));
                M->val = (double *)realloc(M->val, (M->nnz + 1) * sizeof(double));
                M->row[M->nnz] = i + 1; M->col[M->nnz] = i + 1;
                M->val[M->nnz] = pinv; M->nnz++;
            }
        }
    }
    fclose(ff);
}

/* ---- COLAMD 列序（同 test_lu.c：cp 初始化 + 每列行号升序）---- */
static int run_colamd(const s0mat *M, int **q_out)
{
    int *cp = (int *)calloc((size_t)M->n + 1, sizeof(int));
    long Alen = colamd_recommended(M->nnz, M->n, M->n);
    if (Alen < 2L * M->nnz + 16L * M->n + 16)
        Alen = 2L * M->nnz + 16L * M->n + 16;
    int *Ai_copy = (int *)malloc((size_t)Alen * sizeof(int));
    int *Ap_t = (int *)calloc((size_t)M->n + 1, sizeof(int));
    for (int i = 0; i < M->nnz; i++) Ap_t[M->col[i]]++;
    for (int i = 0; i < M->n; i++) { Ap_t[i + 1] += Ap_t[i]; cp[i] = Ap_t[i]; }
    cp[M->n] = M->nnz;
    int *cur = (int *)malloc((size_t)M->n * sizeof(int));
    memcpy(cur, Ap_t, (size_t)M->n * sizeof(int));
    for (int i = 0; i < M->nnz; i++) {
        int p = cur[M->col[i] - 1]++;
        Ai_copy[p] = M->row[i] - 1;
    }
    free(cur);
    for (int c = 0; c < M->n; c++) {       /* colamd 要求每列行号升序 */
        int lo = Ap_t[c], hi = Ap_t[c + 1];
        for (int a = lo + 1; a < hi; a++) {
            int key = Ai_copy[a], bb = a - 1;
            while (bb >= lo && Ai_copy[bb] > key) {
                Ai_copy[bb + 1] = Ai_copy[bb]; bb--;
            }
            Ai_copy[bb + 1] = key;
        }
    }
    double knobs[COLAMD_KNOBS] = {0};
    int stats[COLAMD_STATS] = {0};
    int ok = colamd(M->n, M->n, (int)Alen, Ai_copy, cp, knobs, stats);
    if (!ok) {
        fprintf(stderr, "colamd failed stats[0..3]=%d %d %d %d\n",
                stats[0], stats[1], stats[2], stats[3]);
        free(cp); free(Ai_copy); free(Ap_t);
        return 1;
    }
    *q_out = (int *)malloc((size_t)M->n * sizeof(int));
    for (int j = 0; j < M->n; j++) (*q_out)[j] = cp[j];
    free(cp); free(Ai_copy); free(Ap_t);
    return 0;
}

/* ---- COO -> CSC（0-based），应用列序 q ---- */
static void build_csc_q(const s0mat *M, const int *q,
                        int **Ap_out, int **Ai_out, double **Ax_out,
                        int **diagrow_out)
{
    int n = M->n;
    int *invq = NULL;
    if (q) {
        invq = (int *)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; j++) invq[q[j]] = j;
    }
    int *Ap = (int *)calloc((size_t)n + 1, sizeof(int));
    for (int i = 0; i < M->nnz; i++) {
        int old_c = M->col[i] - 1;
        int c = q ? invq[old_c] : old_c;
        if (c >= 0 && c < n) Ap[c + 1]++;
    }
    for (int i = 0; i < n; i++) Ap[i + 1] += Ap[i];
    int *Ai = (int *)malloc((size_t)M->nnz * sizeof(int));
    double *Ax = (double *)malloc((size_t)M->nnz * sizeof(double));
    int *cur = (int *)malloc((size_t)n * sizeof(int));
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
    *Ap_out = Ap; *Ai_out = Ai; *Ax_out = Ax;
    if (diagrow_out) {
        int *dg = (int *)malloc((size_t)n * sizeof(int));
        for (int k = 0; k < n; k++) dg[k] = q ? q[k] : k;
        *diagrow_out = dg;
    }
}

static double residual(const s0mat *M, const double *x)
{
    double num = 0, den = 0;
    double *bh = (double *)calloc((size_t)M->n, sizeof(double));
    for (int i = 0; i < M->nnz; i++) {
        int r = M->row[i] - 1, c = M->col[i] - 1;
        bh[r] += M->val[i] * x[c];
    }
    for (int i = 0; i < M->n; i++) {
        double d = bh[i] - M->b[i + 1];
        num += d * d;
        den += M->b[i + 1] * M->b[i + 1];
    }
    free(bh);
    return sqrt(num / den);
}

static double rel_l2(int n, const double *a, const double *b)
{
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return sqrt(num / den);
}

static double kernel_ms(cudaEvent_t t0, cudaEvent_t t1)
{
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    return (double)ms;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [pin] [flt]\n", argv[0]);
        return 2;
    }
    s0mat M;
    if (read_s0(argv[1], argv[2], &M)) return 1;
    double pin_val = (argc >= 4) ? atof(argv[3]) : 0.0;
    printf("== %s ==\nn=%d nnz=%d pin=%.1e\n",
           argv[1], M.n, M.nnz, pin_val);

    apply_flt_pin(&M, (argc >= 5) ? argv[4] : NULL, pin_val);

    /* Ruiz 行/列均衡（DSH_RUIZ=1；同 test_lu_gpu.cu） */
    double *R_sc = NULL, *C_sc = NULL;
    s0mat M_orig;
    if (getenv("DSH_RUIZ")) {
        M_orig = M;
        M_orig.row = (int*)malloc((size_t)M.nnz*sizeof(int));
        M_orig.col = (int*)malloc((size_t)M.nnz*sizeof(int));
        M_orig.val = (double*)malloc((size_t)M.nnz*sizeof(double));
        M_orig.b = (double*)malloc((size_t)(M.n+1)*sizeof(double));
        memcpy(M_orig.row, M.row, (size_t)M.nnz*sizeof(int));
        memcpy(M_orig.col, M.col, (size_t)M.nnz*sizeof(int));
        memcpy(M_orig.val, M.val, (size_t)M.nnz*sizeof(double));
        memcpy(M_orig.b, M.b, (size_t)(M.n+1)*sizeof(double));
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
            double dev = 0;
            for (int i = 0; i < M.n; i++) { double d = fabs(rowmax[i]*R_sc[i]*R_sc[i] - 1.0); if (d > dev) dev = d; }
            for (int j = 0; j < M.n; j++) { double d = fabs(colmax[j]*C_sc[j]*C_sc[j] - 1.0); if (d > dev) dev = d; }
            if (dev < 1e-6) break;
        }
        for (int i = 1; i <= M.n; i++) M.b[i] *= R_sc[i-1];
        { double rmin=1e300,rmax=0;
          for (int i=0;i<M.n;i++){ if(R_sc[i]<rmin)rmin=R_sc[i]; if(R_sc[i]>rmax)rmax=R_sc[i]; }
          printf("ruiz applied R[%.3g..%.3g]\n", rmin,rmax); }
        free(rowmax); free(colmax);
    }

    int *q = NULL;
    if (run_colamd(&M, &q)) return 1;
    printf("colamd ok\n");

    int *Ap, *Ai, *diagrow;
    double *Ax;
    build_csc_q(&M, NULL, &Ap, &Ai, &Ax, NULL);
    int n = M.n;

    /* ---- CPU 参考 ---- */
    double *x_cpu = (double *)malloc((size_t)n * sizeof(double));
    long lnnz_cpu = 0, unnz_cpu = 0;
    int rc = cpu_lu_ref(n, Ap, Ai, Ax, q, M.b + 1, 1.0, x_cpu,
                        &lnnz_cpu, &unnz_cpu);
    if (rc >= 0) { fprintf(stderr, "CPU reference singular at column %d\n", rc); return 1; }
    printf("CPU ref: lnnz=%ld unnz=%ld\n", lnnz_cpu, unnz_cpu);

    /* ---- q 序 CSC（GPU 用） ---- */
    int *Apq, *Aiq;
    double *Axq;
    build_csc_q(&M, q, &Apq, &Aiq, &Axq, &diagrow);
    free(Ap); free(Ai); free(Ax);
    Ap = Apq; Ai = Aiq; Ax = Axq;

    long Lcap = (long)n * (n + 1) / 2 + 16;
    long Ucap = Lcap;
    int *d_Ap, *d_Ai, *d_diagrow, *d_q;
    double *d_Ax;
    int *d_Lp, *d_Li, *d_Up, *d_Ui;
    double *d_Lx, *d_Ux;
    int *d_pinv, *d_xi_s, *d_xi_p, *d_lnnz, *d_unnz, *d_sing;
    double *d_x, *d_b, *d_xs, *d_xp, *d_wd, *d_wd_p;
    int *d_wi;
    CK(cudaMalloc((void **)&d_Ap, (n + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_Ai, (size_t)M.nnz * sizeof(int)));
    CK(cudaMalloc((void **)&d_Ax, (size_t)M.nnz * sizeof(double)));
    CK(cudaMalloc((void **)&d_diagrow, (size_t)n * sizeof(int)));
    CK(cudaMalloc((void **)&d_q, (size_t)n * sizeof(int)));
    CK(cudaMalloc((void **)&d_Lp, (n + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_Li, (size_t)Lcap * sizeof(int)));
    CK(cudaMalloc((void **)&d_Lx, (size_t)Lcap * sizeof(double)));
    CK(cudaMalloc((void **)&d_Up, (n + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_Ui, (size_t)Ucap * sizeof(int)));
    CK(cudaMalloc((void **)&d_Ux, (size_t)Ucap * sizeof(double)));
    CK(cudaMalloc((void **)&d_pinv, (size_t)n * sizeof(int)));
    CK(cudaMalloc((void **)&d_xi_s, (size_t)(2 * n + 1) * sizeof(int)));
    CK(cudaMalloc((void **)&d_xi_p,
                  (size_t)(n + 4 * LU_PAR_CAP) * sizeof(int)));
    CK(cudaMalloc((void **)&d_x, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_b, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_xs, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_xp, (size_t)n * sizeof(double)));
    CK(cudaMalloc((void **)&d_wi, (size_t)n * sizeof(int)));
    CK(cudaMalloc((void **)&d_wd, (size_t)(2 * n) * sizeof(double)));
    CK(cudaMalloc((void **)&d_wd_p, (size_t)(2 * n) * sizeof(double)));
    CK(cudaMalloc((void **)&d_lnnz, sizeof(int)));
    CK(cudaMalloc((void **)&d_unnz, sizeof(int)));
    CK(cudaMalloc((void **)&d_sing, sizeof(int)));

    CK(cudaMemcpy(d_Ap, Ap, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Ai, Ai, (size_t)M.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Ax, Ax, (size_t)M.nnz * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_diagrow, diagrow, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, q, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, M.b + 1, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    double t_f_s = 0, t_t_s = 0, t_f_p = 0, t_t_p = 0;
    int lnnz_s = 0, unnz_s = 0, lnnz_p = 0, unnz_p = 0;

    /* ============ 路 2：单 thread GPU（已验证基线） ============ */
    CK(cudaEventRecord(e0));
    lu_factor_kernel<<<1, 1>>>(n, d_Ap, d_Ai, d_Ax, d_diagrow,
                               d_Lp, d_Li, d_Lx, Lcap,
                               d_Up, d_Ui, d_Ux, Ucap,
                               d_pinv, d_xi_s, d_x,
                               d_lnnz, d_unnz, d_sing);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    t_f_s = kernel_ms(e0, e1);
    CK(cudaMemcpy(&lnnz_s, d_lnnz, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&unnz_s, d_unnz, sizeof(int), cudaMemcpyDeviceToHost));
    int sing_s = 0;
    CK(cudaMemcpy(&sing_s, d_sing, sizeof(int), cudaMemcpyDeviceToHost));
    if (sing_s == -2) { fprintf(stderr, "single-thread capacity overflow\n"); return 1; }
    if (sing_s >= 0) { fprintf(stderr, "single-thread singular at %d\n", sing_s); return 1; }

    CK(cudaEventRecord(e0));
    trisolve_kernel<<<1, 1>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                              d_pinv, d_q, d_b, d_xs, d_wi, d_wd);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    t_t_s = kernel_ms(e0, e1);

    /* ============ 路 3：并行 GPU（本里程碑） ============ */
    {
        size_t shb = LU_PAR_SHARED_BYTES(n);
        if (shb > 48 * 1024) {
            CK(cudaFuncSetAttribute(lu_factor_par_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb));
        }
        /* R8: carveout 属性 A/B 结论——默认**不加** carveout=0 属性：
           pg2k Occupancy 16.67%→33.33%（命中目标 ≥33%），factor 232.3ms（<234 目标）；
           carveout=0（偏好 L1）仅快 3.5%（224.4ms）但 Occupancy 减半；pg5k/bandgap 无差异。
           DSH_CARVEOUT_ON=1 恢复旧行为（偏好 L1）。 */
        if (getenv("DSH_CARVEOUT_ON")) {
            cudaFuncSetAttribute(lu_factor_par_kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout, 0);
        }
    }
    CK(cudaEventRecord(e0));
    lu_factor_par_kernel<<<1, 256, LU_PAR_SHARED_BYTES(n)>>>(
        n, d_Ap, d_Ai, d_Ax, d_diagrow,
        d_Lp, d_Li, d_Lx, Lcap,
        d_Up, d_Ui, d_Ux, Ucap,
        d_pinv, d_xi_p, NULL,
        d_lnnz, d_unnz, d_sing);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    t_f_p = kernel_ms(e0, e1);
    CK(cudaMemcpy(&lnnz_p, d_lnnz, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&unnz_p, d_unnz, sizeof(int), cudaMemcpyDeviceToHost));
    int sing_p = 0;
    CK(cudaMemcpy(&sing_p, d_sing, sizeof(int), cudaMemcpyDeviceToHost));
    if (sing_p == -2) { fprintf(stderr, "par capacity overflow\n"); return 1; }
    if (sing_p == -3) { fprintf(stderr, "par structure overflow (frontier/jlist cap)\n"); return 1; }
    if (sing_p >= 0) { fprintf(stderr, "par singular at %d\n", sing_p); return 1; }

    CK(cudaEventRecord(e0));
    trisolve_par_kernel<<<1, 256>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                    d_pinv, d_q, d_b, d_xp, d_wi, d_wd_p);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    t_t_p = kernel_ms(e0, e1);

    double *xs = (double *)malloc((size_t)n * sizeof(double));
    double *xp = (double *)malloc((size_t)n * sizeof(double));
    CK(cudaMemcpy(xs, d_xs, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xp, d_xp, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

    /* ---- 验证 ---- */
    double l2_ps, l2_pc, r_p, r_c;
    if (R_sc) {
        double *x_orig_p = (double*)malloc((size_t)n*sizeof(double));
        double *x_orig_s = (double*)malloc((size_t)n*sizeof(double));
        double *xc_orig = (double*)malloc((size_t)n*sizeof(double));
        for (int i = 0; i < n; i++) {
            x_orig_p[i] = C_sc[i]*xp[i]; x_orig_s[i] = C_sc[i]*xs[i];
            xc_orig[i] = C_sc[i]*x_cpu[i];
        }
        r_p = residual(&M_orig, x_orig_p);
        r_c = residual(&M_orig, xc_orig);
        l2_ps = rel_l2(n, x_orig_p, x_orig_s);
        l2_pc = rel_l2(n, x_orig_p, xc_orig);
        free(x_orig_p); free(x_orig_s); free(xc_orig);
    } else {
        r_p = residual(&M, xp);
        r_c = residual(&M, x_cpu);
        l2_ps = rel_l2(n, xp, xs);
        l2_pc = rel_l2(n, xp, x_cpu);
    }

    printf("\n---- 时间对比（cudaEvent, ms）----\n");
    printf("  factor:  单thread %8.3f ms | 并行 %8.3f ms | 加速 %8.1fx\n",
           t_f_s, t_f_p, t_f_s / t_f_p);
    printf("  trisolve:单thread %8.3f ms | 并行 %8.3f ms | 加速 %8.1fx\n",
           t_t_s, t_t_p, t_t_s / t_t_p);

    printf("\n---- 正确性 ----\n");
    printf("  并行 vs CPU 参考解:  相对L2 = %.3e   %s\n", l2_pc,
           l2_pc <= 1e-6 ? "PASS (L2<=1e-6)" : "FAIL");
    printf("  并行 vs 单thread 解: 相对L2 = %.3e   %s\n", l2_ps,
           l2_ps <= 1e-6 ? "PASS (L2<=1e-6)" : "FAIL");
    printf("  并行残差 ||b-Ax||/||b|| = %.3e\n", r_p);
    printf("  CPU  残差 ||b-Ax||/||b|| = %.3e\n", r_c);
    printf("  因子结构: L 单thread %d / 并行 %d / CPU %ld | "
           "U 单thread %d / 并行 %d / CPU %ld\n",
           lnnz_s, lnnz_p, lnnz_cpu, unnz_s, unnz_p, unnz_cpu);

    /* ---- Occupancy ---- */
    {
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, 0));
        int nbF = 0, nbT = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbF,
            lu_factor_par_kernel, 256, LU_PAR_SHARED_BYTES(n)));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbT,
            trisolve_par_kernel, 256, 0));
        printf("\n---- Occupancy（cudaOccupancyMaxActiveBlocksPerMultiprocessor）----\n");
        printf("  %s %d SMs, maxThreads/SM=%d\n",
               prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
        printf("  lu_factor_par_kernel:  maxActiveBlocks/SM=%d (256 thr) => %.2f%%\n",
               nbF, 100.0 * nbF * 256 / prop.maxThreadsPerMultiProcessor);
        printf("  trisolve_par_kernel:   maxActiveBlocks/SM=%d (256 thr) => %.2f%%\n",
               nbT, 100.0 * nbT * 256 / prop.maxThreadsPerMultiProcessor);
    }

    int pass = (l2_pc <= 1e-6);
    printf("\n== %s ==\n", pass ? "PASS" : "FAIL");

    cudaFree(d_Ap); cudaFree(d_Ai); cudaFree(d_Ax); cudaFree(d_diagrow);
    cudaFree(d_q);
    cudaFree(d_Lp); cudaFree(d_Li); cudaFree(d_Lx);
    cudaFree(d_Up); cudaFree(d_Ui); cudaFree(d_Ux);
    cudaFree(d_pinv); cudaFree(d_xi_s); cudaFree(d_xi_p); cudaFree(d_x);
    cudaFree(d_b); cudaFree(d_xs); cudaFree(d_xp); cudaFree(d_wi);
    cudaFree(d_wd); cudaFree(d_wd_p);
    cudaFree(d_lnnz); cudaFree(d_unnz); cudaFree(d_sing);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    free(Ap); free(Ai); free(Ax); free(diagrow); free(q);
    free(x_cpu); free(xs); free(xp);
    free(M.row); free(M.col); free(M.val); free(M.b);
    return pass ? 0 : 1;
}
