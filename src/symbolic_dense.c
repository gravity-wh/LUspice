// ============================================================================
// symbolic_dense.c — M3 wave factor 的符号分解（与 wave kernel 主元逻辑一致）
// 全行扫描主元（非 splu 可达集）——保证符号结构 = wave kernel 数值结构
// 逻辑 = lu_gpu_wave.cu 的 factor_wave_kernel 的 host 镜像（diagrow 自然主元）
// 输出：U 结构（依赖图）+ L 结构（静态预分配）+ 每列 lnnz/unnz
// 编译：gcc -O2 -c symbolic_dense.c
// ============================================================================
#include <stdlib.h>
#include <string.h>
#include <math.h>

int symbolic_dense(int n, const int *Ap, const int *Ai, const double *Ax,
                   const int *diagrow,
                   int **Lp_out, int **Li_out, double **Lx_out, long *lnnz_out,
                   int **Up_out, int **Ui_out, double **Ux_out, long *unnz_out,
                   int **pinv_out, int **rowat_out)
{
    long cap = 16l * Ap[n] + 16l * n + 64;
    int *Lp = (int*)calloc((size_t)n+1, sizeof(int));
    int *Up = (int*)calloc((size_t)n+1, sizeof(int));
    int *Li = (int*)malloc((size_t)cap*sizeof(int));
    int *Ui = (int*)malloc((size_t)cap*sizeof(int));
    double *Lx = (double*)malloc((size_t)cap*sizeof(double));
    double *Ux = (double*)malloc((size_t)cap*sizeof(double));
    int *pinv = (int*)malloc((size_t)n*sizeof(int));
    int *rowat = (int*)malloc((size_t)n*sizeof(int));
    double *x = (double*)calloc((size_t)n, sizeof(double));
    long lnz = 0, unz = 0;
    for (int i = 0; i < n; i++) { pinv[i] = -1; rowat[i] = -1; }

    for (int k = 0; k < n; k++) {
        Lp[k] = (int)lnz; Up[k] = (int)unz;
        for (int i = 0; i < n; i++) x[i] = 0.0;
        for (int p = Ap[k]; p < Ap[k+1]; p++) x[Ai[p]] = Ax[p];
        /* 前代（稠密全列扫描，与 wave kernel 一致） */
        for (int j = 0; j < k; j++) {
            int r = rowat[j];
            double xr = x[r];
            if (xr != 0.0)
                for (int p = Lp[j]+1; p < Lp[j+1]; p++) x[Li[p]] -= Lx[p]*xr;
        }
        /* 主元（全行扫描） */
        int ipiv = -1; double a = -1.0;
        for (int i = 0; i < n; i++)
            if (pinv[i] < 0) { double v = fabs(x[i]); if (v > a) { a = v; ipiv = i; } }
        for (int i = 0; i < n; i++)
            if (pinv[i] >= 0 && x[i] != 0.0) { Ui[unz] = pinv[i]; Ux[unz++] = x[i]; }
        if (ipiv < 0 || a <= 0.0) { free(Lp);free(Up);free(Li);free(Ui);free(Lx);free(Ux);free(pinv);free(rowat);free(x); return k; }
        int ck = diagrow[k];
        if (pinv[ck] < 0 && fabs(x[ck]) >= a) ipiv = ck;
        double pivot = x[ipiv];
        Ui[unz] = k; Ux[unz++] = pivot;
        pinv[ipiv] = k; rowat[k] = ipiv;
        Li[lnz] = ipiv; Lx[lnz++] = 1.0;
        for (int i = 0; i < n; i++)
            if (pinv[i] < 0 && x[i] != 0.0) { Li[lnz] = i; Lx[lnz++] = x[i]/pivot; }
    }
    Lp[n] = (int)lnz; Up[n] = (int)unz;
    *Lp_out = Lp; *Li_out = Li; *Lx_out = Lx; *lnnz_out = lnz;
    *Up_out = Up; *Ui_out = Ui; *Ux_out = Ux; *unnz_out = unz;
    *pinv_out = pinv; *rowat_out = rowat;   /* 固定主元（所有权转移） */
    free(x);
    return -1;
}
