// ============================================================================
// cpu_ref_lu.c — M3 GPU 验证用 CPU 参考（splu.h 稀疏左视 LU 封装，C 编译）
//
// 复用 third_party/splu.h 的 lu_factor + lu_solve_perm（已交叉验证：
// bandgap 6.956e-12 / pg2k 3.164e-15）。本文件只做参数/内存胶水，不改算法。
// splu.h 的 malloc 无显式转型，必须在 C 模式下编译（nvcc 对 .c 文件走 host
// C 编译器），故封装成 extern "C" 供 GPU 侧基准调用。
//
// 编译时需 -I src/third_party（splu.h 为 header-only，无独立编译单元）。
// ============================================================================
#include <stdlib.h>
#include "splu.h"
#include "cpu_ref_lu.h"

int cpu_lu_ref(int n, const int *Ap, const int *Ai, const double *Ax,
               const int *q, const double *b, double tol,
               double *x, long *lnnz, long *unnz)
{
    lu_factors F;
    lu_csc *L = &F.L, *U = &F.U;
    int rc, k;
    long init_cap = (long)n * (long)n + 4096;   /* 上界足够，grow 兜底 */
    int *xi;
    double *xw;

    if (!n || n < 1) return -2;

    F.pinv = (int *)malloc((size_t)n * sizeof(int));
    L->n = U->n = n;
    L->Lp = (int *)malloc((size_t)(n + 1) * sizeof(int));
    L->Li = (int *)malloc((size_t)init_cap * sizeof(int));
    L->Lx = (double *)malloc((size_t)init_cap * sizeof(double));
    U->Up = (int *)malloc((size_t)(n + 1) * sizeof(int));
    U->Ui = (int *)malloc((size_t)init_cap * sizeof(int));
    U->Ux = (double *)malloc((size_t)init_cap * sizeof(double));
    L->Lcap = U->Ucap = init_cap;
    L->lnnz = U->unnz = 0;
    xi = (int *)malloc((size_t)(2 * n + 1) * sizeof(int));
    xw = (double *)malloc((size_t)n * sizeof(double));

    rc = lu_factor(n, Ap, Ai, Ax, q, &F, tol, xi, xw);
    if (rc >= 0) {
        free(F.pinv); free(L->Lp); free(L->Li); free(L->Lx);
        free(U->Up); free(U->Ui); free(U->Ux);
        free(xi); free(xw);
        return rc;                 /* 奇异列 */
    }
    lu_solve_perm(n, &F, q, b, x);

    if (lnnz) *lnnz = L->lnnz;
    if (unnz) *unnz = U->unnz;
    (void)k;

    free(F.pinv); free(L->Lp); free(L->Li); free(L->Lx);
    free(U->Up); free(U->Ui); free(U->Ux);
    free(xi); free(xw);
    return -1;
}
