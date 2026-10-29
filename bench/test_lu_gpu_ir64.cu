// ============================================================================
// test_lu_gpu_ir64.cu — R10: fp64 因子 + GMRES-IR 病态矩阵验证（M4 病态 GPU 解）
//
// 目标：3 病态矩阵（neural/flash/sc-dac）GPU 精确解——M1 处方（fp64 因子 + Ruiz
//   + GMRES-IR）的 GPU 落地；M4 原用 fp32 因子（NaN），本驱动用 fp64 单 thread
//   因子（lu_gpu.cu lu_factor_kernel）+ fp64 并行 trisolve（trisolve_par_kernel）
//   + host SpMV + fp64 GMRES-IR（右预条件，同 test_lu_gpu_f32 IR 循环）。
//
// 编译:
//   nvcc -O3 -fmad=false -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src \
//        -I /home/gravity/n46-mix/src/include -o test_lu_gpu_ir64 \
//        src/solver/test_lu_gpu_ir64.cu \
//        /home/gravity/n46-mix/src/maths/KLU/colamd.c \
//        /home/gravity/n46-mix/src/maths/KLU/colamd_global.c
// 用法: test_lu_gpu_ir64 <s0.mtx> <s0.rhs> [pin] [flt] [shift]   （DSH_RUIZ=1 启用 Ruiz）
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>

#include "cuda_compat.h"
#include "ngspice/colamd.h"
#include "lu_gpu.cu"        /* lu_factor_kernel（fp64 单 thread） */
#include "lu_gpu_par.cu"    /* trisolve_par_kernel（fp64 并行） */

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

static int build_csc_q(const s0mat *M, const int *q,
                       int **Ap_out, int **Ai_out, double **Ax_out)
{
    int n = M->n;
    int *invq = NULL;
    if (q) {
        invq = (int*)malloc((size_t)n * sizeof(int));
        for (int j = 0; j < n; j++) invq[q[j]] = j;
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
    *Ap_out = Ap; *Ai_out = Ai; *Ax_out = Ax;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [pin] [flt] [shift]\n", argv[0]); return 2; }
    s0mat M;
    if (read_s0(argv[1], argv[2], &M)) return 1;
    double pinv = (argc >= 4) ? atof(argv[3]) : 0.0;
    double shift = (argc >= 6) ? atof(argv[5]) : 0.0;
    printf("n=%d nnz=%d pin=%.1e shift=%.1e\n", M.n, M.nnz, pinv, shift);

    /* .flt 浮动节点 pin 注入 */
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

    /* Ruiz 行/列均衡（DSH_RUIZ=1；参考 lu_tool.c ruiz 30 迭代） */
    if (getenv("DSH_RUIZ")) {
        double *R_sc = (double*)malloc((size_t)M.n * sizeof(double));
        double *C_sc = (double*)malloc((size_t)M.n * sizeof(double));
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
        for (int i = 1; i <= M.n; i++) M.b[i] *= R_sc[i-1];
        free(rowmax); free(colmax); free(R_sc); free(C_sc);
        printf("ruiz applied\n");
    }

    /* COLAMD */
    int *q = NULL;
    if (!getenv("DSH_NATURAL")) {
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
    } else { printf("natural order\n"); }

    int *Ap, *Ai; double *Ax;
    build_csc_q(&M, q, &Ap, &Ai, &Ax);
    int n = M.n;
    long cap = (long)M.nnz * 24 + 1024;
    int *d_Ap, *d_Ai; double *d_Ax;
    int *d_Lp, *d_Li; double *d_Lx;
    int *d_Up, *d_Ui; double *d_Ux;
    int *d_pinv, *d_xi; double *d_x;
    int *d_lnnz, *d_unnz, *d_sing;
    DSH_CHECK(DSH_Malloc((void**)&d_Ap, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ai, (size_t)M.nnz*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ax, (size_t)M.nnz*sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_Lp, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Li, cap*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Lx, cap*sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_Up, (n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ui, cap*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_Ux, cap*sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_pinv, n*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_xi, (2*n+1)*sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_x, n*sizeof(double)));
    DSH_CHECK(DSH_Malloc((void**)&d_lnnz, sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_unnz, sizeof(int)));
    DSH_CHECK(DSH_Malloc((void**)&d_sing, sizeof(int)));
    DSH_CHECK(DSH_Memcpy(d_Ap, Ap, (n+1)*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_Ai, Ai, (size_t)M.nnz*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
    DSH_CHECK(DSH_Memcpy(d_Ax, Ax, (size_t)M.nnz*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));

    /* shift 因子（仅因子化）：对角 + shift*|diag| */
    if (shift > 0) {
        for (int i = 0; i < M.n; i++)
            for (int k = 0; k < M.nnz; k++)
                if (M.row[k]-1 == i && M.col[k]-1 == i) { Ax[k] += shift * fabs(Ax[k]); break; }
        DSH_CHECK(DSH_Memcpy(d_Ax, Ax, (size_t)M.nnz*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE));
    }

    int *diagrow = (int*)malloc((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) diagrow[k] = q ? q[k] : k;
    int *d_diag;
    DSH_CHECK(DSH_Malloc((void**)&d_diag, (size_t)n*sizeof(int)));
    DSH_CHECK(DSH_Memcpy(d_diag, diagrow, (size_t)n*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    lu_factor_kernel<<<1, 1>>>(n, d_Ap, d_Ai, d_Ax, d_diag,
                               d_Lp, d_Li, d_Lx, cap,
                               d_Up, d_Ui, d_Ux, cap,
                               d_pinv, d_xi, d_x,
                               d_lnnz, d_unnz, d_sing);
    DSH_CHECK(DSH_DeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double factor_ms = (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    int lnnz, unnz, sing;
    DSH_CHECK(DSH_Memcpy(&lnnz, d_lnnz, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    DSH_CHECK(DSH_Memcpy(&unnz, d_unnz, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    DSH_CHECK(DSH_Memcpy(&sing, d_sing, sizeof(int), DSH_MEMCPY_DEVICE_TO_HOST));
    printf("GPU LU(fp64): lnnz=%d unnz=%d singular=%d factor_ms=%.1f\n", lnnz, unnz, sing, factor_ms);
    if (sing >= 0) { printf("FACTOR SINGULAR at %d\n", sing); return 1; }

    /* ---- GMRES-IR（fp64 因子 + fp64 并行 trisolve 预条件 + host SpMV）---- */
    double *xcur = (double*)calloc((size_t)n, sizeof(double));
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

        int *d_q = NULL;
        if (q) {
            DSH_CHECK(DSH_Malloc((void**)&d_q, (size_t)n*sizeof(int)));
            DSH_CHECK(DSH_Memcpy(d_q, q, (size_t)n*sizeof(int), DSH_MEMCPY_HOST_TO_DEVICE));
        }
        double *d_b, *d_xv; int *d_work_i; double *d_work_d;
        DSH_CHECK(DSH_Malloc((void**)&d_b, (size_t)n*sizeof(double)));
        DSH_CHECK(DSH_Malloc((void**)&d_xv, (size_t)n*sizeof(double)));
        DSH_CHECK(DSH_Malloc((void**)&d_work_i, (size_t)n*sizeof(int)));
        DSH_CHECK(DSH_Malloc((void**)&d_work_d, (size_t)(2*n)*sizeof(double)));

        int total_iters = 0;
        double final_rel = 0.0;
        for (int outer = 0; outer < 7; outer++) {
            for (int i = 0; i < n; i++) r0[i] = rhs[i];
            for (int i = 0; i < M.nnz; i++) {
                int rr = M.row[i]-1, cc = M.col[i]-1;
                r0[rr] -= M.val[i] * xcur[cc];
            }
            double beta = 0.0;
            for (int i = 0; i < n; i++) beta += r0[i]*r0[i];
            beta = sqrt(beta);
            fprintf(stderr, "[gmres64] outer=%d res=%g\n", outer, beta/norm_b);
            if (beta/norm_b <= 1e-10) { final_rel = beta/norm_b; break; }
            for (int i = 0; i < n; i++) V[0][i] = r0[i]/beta;
            g[0] = beta;
            int j = 0;
            for (j = 0; j < m; j++) {
                DSH_Memcpy(d_b, V[j], (size_t)n*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
                trisolve_par_kernel<<<1, 256>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                                d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
                DSH_DeviceSynchronize();
                DSH_Memcpy(z, d_xv, (size_t)n*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
                for (int i = 0; i < n; i++) w[i] = 0.0;
                for (int i = 0; i < M.nnz; i++) {
                    int rr = M.row[i]-1, cc = M.col[i]-1;
                    w[rr] += M.val[i] * z[cc];
                }
                for (int pass = 0; pass < 2; pass++) {
                    for (int i = 0; i <= j; i++) {
                        double h = 0.0;
                        for (int k2 = 0; k2 < n; k2++) h += V[i][k2] * w[k2];
                        H[i*m + j] += h;
                        for (int k2 = 0; k2 < n; k2++) w[k2] -= h * V[i][k2];
                    }
                }
                double hn = 0.0;
                for (int k2 = 0; k2 < n; k2++) hn += w[k2]*w[k2];
                hn = sqrt(hn);
                if (hn < 1e-14) { j++; break; }
                H[(j+1)*m + j] = hn;
                for (int k2 = 0; k2 < n; k2++) V[j+1][k2] = w[k2]/hn;
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
            double *y = (double*)calloc((size_t)j, sizeof(double));
            for (int i = j-1; i >= 0; i--) {
                double s = g[i];
                for (int k2 = i+1; k2 < j; k2++) s -= H[i*m+k2]*y[k2];
                y[i] = s / H[i*m+i];
            }
            for (int i = 0; i < n; i++) z[i] = 0.0;
            for (int k2 = 0; k2 < j; k2++)
                for (int i = 0; i < n; i++) z[i] += y[k2]*V[k2][i];
            DSH_Memcpy(d_b, z, (size_t)n*sizeof(double), DSH_MEMCPY_HOST_TO_DEVICE);
            trisolve_par_kernel<<<1, 256>>>(n, d_Lp, d_Li, d_Lx, d_Up, d_Ui, d_Ux,
                                            d_pinv, d_q, d_b, d_xv, d_work_i, d_work_d);
            DSH_DeviceSynchronize();
            DSH_Memcpy(z, d_xv, (size_t)n*sizeof(double), DSH_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < n; i++) xcur[i] += z[i];
            free(y);
            for (int i = 0; i < n; i++) r0[i] = rhs[i];
            for (int i = 0; i < M.nnz; i++) { int rr=M.row[i]-1, cc=M.col[i]-1; r0[rr] -= M.val[i]*xcur[cc]; }
            double nr = 0.0;
            for (int i = 0; i < n; i++) nr += r0[i]*r0[i];
            final_rel = sqrt(nr)/norm_b;
            if (final_rel <= 1e-10) break;
        }
        for (int j2 = 0; j2 <= m; j2++) free(V[j2]);
        free(V); free(H); free(cs); free(sn); free(g); free(w); free(z); free(r0); free(rhs);
        {
            double num = 0, den = 0;
            double *bh = (double*)calloc((size_t)n, sizeof(double));
            for (int i = 0; i < M.nnz; i++) { int rr=M.row[i]-1, cc=M.col[i]-1; bh[rr] += M.val[i]*xcur[cc]; }
            for (int i = 0; i < n; i++) { double d = bh[i]-M.b[i+1]; num += d*d; den += M.b[i+1]*M.b[i+1]; }
            free(bh);
            double rel = sqrt(num/den);
            printf("GPU fp64 GMRES-IR: inner_total=%d final_res=%.3e  %s\n", total_iters, rel,
                   rel <= 1e-6 ? "PASS (S1: L2<=1e-6)" : "FAIL");
        }
        DSH_Free(d_q); DSH_Free(d_b); DSH_Free(d_xv); DSH_Free(d_work_i); DSH_Free(d_work_d);
    }
    free(xcur);
    DSH_Free(d_Ap); DSH_Free(d_Ai); DSH_Free(d_Ax);
    DSH_Free(d_Lp); DSH_Free(d_Li); DSH_Free(d_Lx);
    DSH_Free(d_Up); DSH_Free(d_Ui); DSH_Free(d_Ux);
    DSH_Free(d_pinv); DSH_Free(d_xi); DSH_Free(d_x);
    DSH_Free(d_lnnz); DSH_Free(d_unnz); DSH_Free(d_sing); DSH_Free(d_diag);
    free(diagrow); free(Ap); free(Ai); free(Ax); free(q);
    free(M.row); free(M.col); free(M.val); free(M.b);
    return 0;
}
