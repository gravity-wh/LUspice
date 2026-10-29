// ============================================================================
// stats_lu.c — M3 并行化设计用统计：复刻 lu_gpu.cu 单 thread 算法（host 版），
// 收集每列可达集大小 |R_k|、可达集内已主元列数、散射项数、依赖深度（level），
// 用于评估"可达集裁剪 + level 排序"并行前代方案的收益。
// 编译：
//   gcc -O2 -I src/solver -I /home/gravity/g3-ws/src -I /home/gravity/n46-mix/src/include \
//       -o stats_lu src/solver/stats_lu.c \
//       /home/gravity/n46-mix/src/maths/KLU/colamd.c \
//       /home/gravity/n46-mix/src/maths/KLU/colamd_global.c -lm
// 用法：./stats_lu <s0.mtx> [pin] [flt]
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ngspice/colamd.h"

typedef struct { int n, nnz; int *row, *col; double *val; } s0mat;

static int read_s0(const char *mtx_path, s0mat *M)
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
    return 0;
}

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
    for (int c = 0; c < M->n; c++) {
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
    if (!ok) { fprintf(stderr, "colamd failed\n"); return 1; }
    *q_out = (int *)malloc((size_t)M->n * sizeof(int));
    for (int j = 0; j < M->n; j++) (*q_out)[j] = cp[j];
    return 0;
}

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

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <s0.mtx> [pin] [flt]\n", argv[0]); return 2; }
    s0mat M;
    if (read_s0(argv[1], &M)) return 1;
    double pin_val = (argc >= 3) ? atof(argv[2]) : 0.0;
    apply_flt_pin(&M, (argc >= 4) ? argv[3] : NULL, pin_val);
    printf("== %s == n=%d nnz=%d pin=%.1e\n", argv[1], M.n, M.nnz, pin_val);

    int *q = NULL;
    if (run_colamd(&M, &q)) return 1;
    int *Ap, *Ai, *diagrow;
    double *Ax;
    build_csc_q(&M, q, &Ap, &Ai, &Ax, &diagrow);
    int n = M.n;

    long Lcap = (long)n * (n + 1) / 2 + 16;
    int *Lp = (int *)calloc((size_t)n + 1, sizeof(int));
    int *Li = (int *)malloc((size_t)Lcap * sizeof(int));
    double *Lx = (double *)malloc((size_t)Lcap * sizeof(double));
    int *Up = (int *)calloc((size_t)n + 1, sizeof(int));
    int *Ui = (int *)malloc((size_t)Lcap * sizeof(int));
    double *Ux = (double *)malloc((size_t)Lcap * sizeof(double));
    int *pinv = (int *)malloc((size_t)n * sizeof(int));
    int *rowat = (int *)malloc((size_t)n * sizeof(int));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    char *inR = (char *)calloc((size_t)n, 1);       /* 可达标记 */
    int  *level = (int *)malloc((size_t)n * sizeof(int)); /* level[position] */
    int  *Rlist = (int *)malloc((size_t)n * sizeof(int));

    long lnz = 0, unz = 0;
    long tot_scatter_entries = 0;
    long tot_scatter_cols = 0;
    long tot_reach = 0;
    long max_reach = 0, max_level = 0, max_scatter_cols = 0, max_sc_entries = 0, max_lcol = 0;
    long tot_bfs_rounds = 0, max_bfs_rounds = 0;
    long tot_solve_rounds = 0, max_solve_rounds = 0;
    long tot_solve_level_sum = 0;

    for (int i = 0; i < n; i++) { x[i] = 0.0; pinv[i] = -1; rowat[i] = -1; }

    for (int k = 0; k < n; k++) {
        Lp[k] = (int)lnz;
        Up[k] = (int)unz;

        /* x = A(:,k) */
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];

        /* ---- 结构 BFS：从 A 列 k 种子出发沿 L 图计算可达集 R_k（含值 0 行）---- */
        int nR = 0;
        for (int p = Ap[k]; p < Ap[k + 1]; p++) {
            int i = Ai[p];
            if (!inR[i]) { inR[i] = 1; Rlist[nR++] = i; }
        }
        int head = 0, rounds = 0;
        while (head < nR) {
            rounds++;
            int tail = nR;
            for (int t = head; t < tail; t++) {
                int i = Rlist[t];
                int j = pinv[i];
                if (j >= 0 && j < k) {          /* 已主元行 → 散射 L 列 j */
                    for (int p = Lp[j] + 1; p < Lp[j + 1]; p++) {
                        int ii = Li[p];
                        if (!inR[ii]) { inR[ii] = 1; Rlist[nR++] = ii; }
                    }
                }
            }
            head = tail;
        }
        tot_bfs_rounds += rounds;
        if (rounds > max_bfs_rounds) max_bfs_rounds = rounds;
        tot_reach += nR;
        if (nR > max_reach) max_reach = nR;

        /* ---- level（求解轮数）：对可达列 j 按 j 降序算 level[j] ---- */
        long sc_cols = 0, sc_entries = 0, maxlv = 0;
        for (int j = k - 1; j >= 0; j--) {
            int r = rowat[j];
            if (!inR[r]) { level[j] = 0; continue; }
            int lv = 0;
            for (int p = Lp[j] + 1; p < Lp[j + 1]; p++) {
                int i = Li[p];
                if (inR[i] && pinv[i] >= 0 && pinv[i] < k) {
                    int l2 = level[pinv[i]];
                    if (l2 > lv) lv = l2;
                }
            }
            level[j] = lv + 1;
            if (lv + 1 > maxlv) maxlv = lv + 1;
        }
        tot_solve_rounds += maxlv;
        if (maxlv > max_solve_rounds) max_solve_rounds = maxlv;
        tot_solve_level_sum += maxlv;

        /* ---- 实际数值前代（稠密 j-loop 计散射统计）---- */
        for (int j = 0; j < k; j++) {
            int r = rowat[j];
            double xr = x[r];
            if (xr != 0.0) {
                sc_cols++;
                sc_entries += (Lp[j + 1] - Lp[j] - 1);
                for (int p = Lp[j] + 1; p < Lp[j + 1]; p++)
                    x[Li[p]] -= Lx[p] * xr;
            }
        }
        tot_scatter_cols += sc_cols;
        tot_scatter_entries += sc_entries;
        if (sc_entries > max_sc_entries) max_sc_entries = sc_entries;
        if (k > 0 && Lp[k] - Lp[k-1] - 1 > max_lcol) max_lcol = Lp[k] - Lp[k-1] - 1;
        if (sc_cols > max_scatter_cols) max_scatter_cols = sc_cols;

        /* 主元扫描 */
        int ipiv = -1;
        double a = -1.0;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0) {
                double t = fabs(x[i]);
                if (t > a) { a = t; ipiv = i; }
            } else if (x[i] != 0.0) {
                Ui[unz] = pinv[i]; Ux[unz] = x[i]; unz++;
            }
        }
        if (ipiv == -1 || a <= 0.0) { fprintf(stderr, "singular col %d\n", k); return 1; }
        int ck = diagrow[k];
        if (pinv[ck] < 0 && fabs(x[ck]) >= a) ipiv = ck;
        double pivot = x[ipiv];
        Ui[unz] = k; Ux[unz] = pivot; unz++;
        pinv[ipiv] = k; rowat[k] = ipiv;

        Li[lnz] = ipiv; Lx[lnz] = 1.0; lnz++;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0 && x[i] != 0.0) {
                Li[lnz] = i; Lx[lnz] = x[i] / pivot; lnz++;
            }
            x[i] = 0.0;
            inR[i] = 0;   /* 清可达标记 */
        }
    }
    Lp[n] = (int)lnz;
    Up[n] = (int)unz;

    printf("L nnz=%ld  U nnz=%ld  avg L col len=%.1f\n", lnz, unz, (double)lnz / n);
    printf("散射列总数 = %ld (avg/col %.2f, max %ld)\n",
           tot_scatter_cols, (double)tot_scatter_cols / n, max_scatter_cols);
    printf("散射项总数 = %ld (avg/col %.2f, max/col %ld)\n",
           tot_scatter_entries, (double)tot_scatter_entries / n, max_sc_entries);
    printf("可达集 |R_k|: 总和 %ld (avg %.2f, max %ld)\n",
           tot_reach, (double)tot_reach / n, max_reach);
    printf("BFS 轮数:  总和 %ld (avg %.2f, max %ld)\n",
           tot_bfs_rounds, (double)tot_bfs_rounds / n, max_bfs_rounds);
    printf("max 单 L 列长 = %ld\n", max_lcol);
    printf("求解 level: 总和 %ld (avg %.2f, max %ld)\n",
           tot_solve_rounds, (double)tot_solve_rounds / n, max_solve_rounds);
    printf("估算：稠密 j-loop 总迭代 = %ld；可达裁剪后求解迭代(round)总和 = %ld\n",
           (long)n * n / 2, tot_solve_rounds);
    return 0;
}
