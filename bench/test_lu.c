// ============================================================================
// test_lu.c — M2 CPU 正确性验证：dsh_symbolic → dsh_numeric → dsh_trisolve
// 验证：解 Ax=b vs 残差口径 + 符号复用计数（同矩阵两次 numeric 仅一次 symbolic）
// 编译：gcc -O2 -DDSH_PLATFORM_HOST -I src/solver -I /home/gravity/g3-ws/src \
//       -o test_lu src/solver/test_lu.c src/solver/lu_static.c
// 用法：./test_lu <s0.mtx> <s0.rhs> [pin]
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lu_static.h"
#include "splu.h"
#include "level.h"
#include "ngspice/colamd.h"

/* ---- s0 格式读取：COO + rhs ---- */
typedef struct {
    int n, nnz;
    int *row, *col;
    double *val;
    double *b;
} s0mat;

static int read_s0(const char *mtx_path, const char *rhs_path, s0mat *M)
{
    FILE *f = fopen(mtx_path, "r");
    char line[512];
    long cap = 0;
    int i;
    if (!f) { fprintf(stderr, "open fail %s\n", mtx_path); return 1; }
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    cap = 1024;
    M->row = malloc(cap * sizeof(int)); M->col = malloc(cap * sizeof(int));
    M->val = malloc(cap * sizeof(double));
    M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        if (M->nnz >= cap) { cap *= 2; M->row = realloc(M->row, cap*sizeof(int));
            M->col = realloc(M->col, cap*sizeof(int)); M->val = realloc(M->val, cap*sizeof(double)); }
        sscanf(line, "A %d %d %lf", &M->row[M->nnz], &M->col[M->nnz], &M->val[M->nnz]);
        M->nnz++;
    }
    fclose(f);
    M->b = calloc((size_t)M->n + 1, sizeof(double));
    if (rhs_path) {
        f = fopen(rhs_path, "r");
        if (f) {
            while (fgets(line, sizeof line, f)) {
                int idx; double v;
                if (sscanf(line, "RHS %d %lf", &idx, &v) == 2 && idx >= 1 && idx <= M->n)
                    M->b[idx] = v;
            }
            fclose(f);
        }
    }
    for (i = 1; i <= M->n; i++) if (M->b[i] == 0) M->b[i] = 1.0;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [pin] [flt]\n", argv[0]); return 2; }
    s0mat M;
    if (read_s0(argv[1], argv[2], &M)) return 1;
    double pinv = (argc >= 4) ? atof(argv[3]) : 0.0;
    printf("n=%d nnz=%d pin=%.1e\n", M.n, M.nnz, pinv);

    /* .flt 浮动节点 → COO 对角加 pin（有对角加值、无对角新增；参考 lu_tool read_mtx） */
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
                        M.row = realloc(M.row, (M.nnz + 1) * sizeof(int));
                        M.col = realloc(M.col, (M.nnz + 1) * sizeof(int));
                        M.val = realloc(M.val, (M.nnz + 1) * sizeof(double));
                        M.row[M.nnz] = i + 1; M.col[M.nnz] = i + 1; M.val[M.nnz] = pinv; M.nnz++;
                    }
                }
            }
            fclose(ff);
            printf("flt pin applied (%d -> nnz=%d)\n", pinv > 0 ? 1 : 0, M.nnz);
        }
    }

    dsh_lu_static F;
    dsh_lu_static_init(&F);
    F.pin = 0.0;   /* pin 已在 COO 层注入 */

    /* 生成 COLAMD 列序（参考 lu_tool.c；pg 系自然序 LU 数值不稳定） */
    int *q = NULL;
    {
        long Alen = colamd_recommended(M.nnz, M.n, M.n);
        if (Alen < 2L * M.nnz + 16L * M.n + 16) Alen = 2L * M.nnz + 16L * M.n + 16;
        int *Ai_copy = malloc((size_t)Alen * sizeof(int));
        /* COO -> CSC（1-based 索引） */
        int *cp = calloc((size_t)M.n + 1, sizeof(int));   /* colamd 列指针（输出列序） */
        int *Ap_t = calloc((size_t)M.n + 1, sizeof(int));
        for (int i = 0; i < M.nnz; i++) Ap_t[M.col[i]]++;
        for (int i = 0; i < M.n; i++) { Ap_t[i + 1] += Ap_t[i]; cp[i] = Ap_t[i]; }
        cp[M.n] = M.nnz;
        int *cur = malloc((size_t)M.n * sizeof(int));
        memcpy(cur, Ap_t, (size_t)M.n * sizeof(int));
        for (int i = 0; i < M.nnz; i++) { int p = cur[M.col[i] - 1]++; Ai_copy[p] = M.row[i] - 1; }
        free(cur);
        /* colamd 要求每列行号升序：对每列 [Ap_t[c], Ap_t[c+1]) 排序 */
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
        if (!ok) { fprintf(stderr, "colamd failed stats[0]=%d stats[1]=%d stats[2]=%d stats[3]=%d\n",
                           stats[0], stats[1], stats[2], stats[3]);
                   colamd_report(stats); return 1; }
        q = malloc((size_t)M.n * sizeof(int));
        for (int j = 0; j < M.n; j++) q[j] = cp[j];
        free(cp); free(Ai_copy); free(Ap_t);
        printf("colamd ok\n");
    }

    /* 1. 符号分解（一次，带 COLAMD 列序） */
    int rc = dsh_symbolic(M.n, M.nnz, M.row, M.col, q, &F);
    printf("symbolic rc=%d count=%ld lnnz_est=%ld unnz_est=%ld\n",
           rc, dsh_symbolic_count(), F.lnnz, F.unnz);

    /* 2. 数值分解（两次，验证符号复用计数） */
    rc = dsh_numeric(&F, M.row, M.col, M.val, M.nnz);
    printf("numeric#1 rc=%d min_diag_U=%.3e lnnz=%ld unnz=%ld count=%ld\n",
           rc, F.min_diag_U, F.lnnz, F.unnz, dsh_symbolic_count());
    if (rc != 0) return 1;
    rc = dsh_numeric(&F, M.row, M.col, M.val, M.nnz);
    printf("numeric#2 rc=%d count=%ld (期望仍=1: 符号分解未重复)\n",
           rc, dsh_symbolic_count());

    /* 2.5 U 列长分布（splu.h 因子） */
    if (F.factor_cache) {
        lu_factors *fac = (lu_factors *)F.factor_cache;
        int cnt[8]={0};
        for (int k=0;k<M.n;k++){ int len=fac->U.Up[k+1]-fac->U.Up[k]; cnt[len<=1?0:len<=2?1:len<=4?2:len<=8?3:len<=16?4:len<=32?5:len<=64?6:7]++; }
        { FILE *lf=fopen("/tmp/l_cpu.txt","w");
          for (int k=0;k<M.n;k++){ fprintf(lf,"%d %d |", k, fac->L.Lp[k+1]-fac->L.Lp[k]);
            for (int p=fac->L.Lp[k];p<fac->L.Lp[k+1];p++) fprintf(lf," (%d,%.6g)", fac->L.Li[p], fac->L.Lx[p]); fprintf(lf,"\n"); }
          fclose(lf); }
        { FILE *uf=fopen("/tmp/u_cpu.txt","w");
          for (int k=0;k<M.n;k++){ fprintf(uf,"%d %d |", k, fac->U.Up[k+1]-fac->U.Up[k]);
            for (int p=fac->U.Up[k];p<fac->U.Up[k+1];p++) fprintf(uf," (%d,%.6g)", fac->U.Ui[p], fac->U.Ux[p]); fprintf(uf,"\n"); }
          fclose(uf); }
        printf("CPU U 列长分布 [1,2,4,8,16,32,64,>64]: %d %d %d %d %d %d %d %d\n", cnt[0],cnt[1],cnt[2],cnt[3],cnt[4],cnt[5],cnt[6],cnt[7]);
    }

    /* 2.6 level 分布（位置序 L，factor 并行化潜力） */
    if (F.factor_cache) {
        lu_factors *fac = (lu_factors *)F.factor_cache;
        int *lvl = (int*)malloc((size_t)M.n * sizeof(int));
        int nlevels;
        dsh_level_compute(M.n, fac->U.Up, fac->U.Ui, lvl, &nlevels);
        int *hist = (int*)calloc((size_t)nlevels, sizeof(int));
        dsh_level_hist(M.n, lvl, nlevels, hist);
        printf("LEVEL: depth=%d max_wave=%d (level %d)\n", nlevels, hist[0], 0);
        if (nlevels > 1) { int mw=0, mi=0; for (int i=0;i<nlevels;i++) if (hist[i]>mw){mw=hist[i];mi=i;} printf("LEVEL: max_wave=%d at level %d (深度 %d)\n", mw, mi, nlevels); }
        free(lvl); free(hist);
    }

    /* 3. 三角求解 */
    double *x = malloc((size_t)M.n * sizeof(double));
    double *b1 = malloc((size_t)(M.n + 1) * sizeof(double));
    for (int i = 1; i <= M.n; i++) b1[i] = M.b[i];
    rc = dsh_trisolve(&F, b1 + 1, x);
    printf("trisolve rc=%d\n", rc);

    /* 4. 残差验证 ||b - A x||/||b|| */
    {
        double num = 0, den = 0;
        for (int i = 0; i < M.nnz; i++) {
            int r = M.row[i] - 1, c = M.col[i] - 1;
            num += 0; /* 下面用累加 */
        }
        /* 重算 b_hat = A x */
        double *bh = calloc((size_t)M.n, sizeof(double));
        for (int i = 0; i < M.nnz; i++) {
            int r = M.row[i] - 1, c = M.col[i] - 1;
            bh[r] += M.val[i] * x[c];
        }
        for (int i = 0; i < M.n; i++) {
            double d = bh[i] - b1[i + 1];
            num += d * d;
            den += b1[i + 1] * b1[i + 1];
        }
        free(bh);
        double rel = sqrt(num / den);
        printf("RESIDUAL ||b-Ax||/||b|| = %.3e  %s\n", rel,
               rel <= 1e-6 ? "PASS (S1: L2<=1e-6)" : "FAIL");
    }

    dsh_lu_static_free(&F);
    free(x); free(b1);
    free(M.row); free(M.col); free(M.val); free(M.b);
    return 0;
}
