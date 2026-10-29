// ============================================================================
// symbolic_u.c — M3 level scheduling 的 CPU 符号分解（C 编译，供 CUDA 驱动 extern "C" 调用）
// 输出：U 结构（Up/Ui 所有权转调用方）+ 每列 unnz（静态预分配）+ 总 unnz
// 输入：COLAMD 重排后的 CSC（q 序，q=NULL 即当前序）
// 编译：gcc -O2 -I /home/gravity/g3-ws/src -c symbolic_u.c
// ============================================================================
#include <stdlib.h>
#include <string.h>
#include "splu.h"

int symbolic_u(int n, const int *Ap, const int *Ai, const double *Ax,
               int **Up_out, int **Ui_out, long *unnz_out,
               int **Lp_out, int **Li_out, long *lnnz_out)
{
    lu_factors F;
    memset(&F, 0, sizeof(F));
    F.n = n;
    F.pinv = (int *)malloc((size_t)n * sizeof(int));
    F.L = *lu_csc_alloc(n, 16l * Ap[n] + 16l * n + 64);   /* 容量 = 16*nnz+n（M2 经验上限） */
    F.U = *lu_csc_alloc(n, 16l * Ap[n] + 16l * n + 64);
    int *xi = (int *)malloc((size_t)(2 * n + 1) * sizeof(int));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    int rc = lu_factor(n, Ap, Ai, Ax, NULL, &F, 1.0, xi, x);
    if (rc >= 0) {   /* singular */
        free(F.pinv); free(F.L.Lp); free(F.L.Li); free(F.L.Lx);
        free(F.U.Up); free(F.U.Ui); free(F.U.Ux);
        free(xi); free(x);
        return rc;
    }
    *Up_out = F.U.Up;    /* 所有权转移 */
    *Ui_out = F.U.Ui;
    *unnz_out = F.U.unnz;
    *Lp_out = F.L.Lp;    /* L 结构（静态预分配用） */
    *Li_out = F.L.Li;
    *lnnz_out = F.L.lnnz;
    free(F.pinv); free(F.L.Lx);
    free(xi); free(x);
    return -1;
}
