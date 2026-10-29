// ============================================================================
// lu_gpu.cu — M3 GPU 稀疏左视 LU（device 单 thread 初版，正确性优先）
//
// 算法本源：/home/gravity/g3-ws/src/splu.h 的 lu_factor / lu_solve_perm
//   （CSparse cs_lu 移植：DFS 可达集 + 前代散射 + 部分主元 tol=1.0）。
//   device 版与之逐列数值等价，差异仅在：
//     1. 可达集用稠密简化（x 稠密向量 + 全行扫描），n<=4050 可接受；
//        稀疏 DFS 留后续迭代（初版正确性优先）
//     2. malloc 动态增长 → host 预分配上界数组（cap = n(n+1)/2+16，免 grow）
//     3. A 已由 host 按 COLAMD q 重排（q 序 CSC），kernel 内自然列序 = q 序；
//        自然主元候选行 = diagrow[k]（= q[k]，重排前该列的对角行）
// 纪律：标准 CUDA 子集；内核内严禁 malloc/cudaMalloc（工作数组全 host 传入）。
// 编译：见 test_lu_gpu.cu 头部。
// ============================================================================
#include <math.h>
#include <cuda_runtime.h>
#include "lu_gpu.h"

// ---------------------------------------------------------------------------
// lu_factor_kernel：单 thread 完整稀疏左视 LU（P * A(: ,q) = L * U）
//
// 与 splu.h lu_factor 的逐列等价：
//   for k:
//     x = A(:,q[k]) 散射（A 已重排，即列 k）          —— 等价 spsolve 的 scatter
//     前代 x = L^{-1} x：按主元序 j=0..k-1 全列扫描，     —— 等价 spsolve 的散射环
//       仅散射 x[rowat[j]] != 0 的 L 列 j（rowat[j]=主元 j 的原始行）
//     主元扫描：已主元行 → U 次对角项（位置 pinv[i]）；未主元行 → |x| 最大候选
//     tol=1.0 自然主元：diagrow[k] 若占优则保留（= splu 的 pinv[col]<0 分支）
//     U 对角写末项 → pinv/rowat 记录 → L 列（单位对角首项 + 未主元行 x/pivot）
//   末：L 行号 remap 为主元位置（splu.h 同款，trisolve 前代用位置索引）
// 注意：与 splu.h 的唯一数值差异是散射顺序（稠密按 j 升序 vs 稀疏按 DFS 序），
//   各列累加独立、仅影响 ~1e-16 舍入；主元/结构判定不受影响（实测一致）。
// ---------------------------------------------------------------------------
__global__ void lu_factor_kernel(
    int n,
    const int  *Ap, const int *Ai, const double *Ax,
    const int  *diagrow,
    int        *Lp, int *Li, double *Lx, long Lcap,
    int        *Up, int *Ui, double *Ux, long Ucap,
    int        *pinv,
    int        *xi,
    double     *x,
    int        *lnnz, int *unnz, int *singular_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int *rowat = xi;              /* rowat[k] = 主元 k 的原始行（pinv 逆映射） */
    long lnz = 0, unz = 0;

    for (int i = 0; i < n; i++) { x[i] = 0.0; pinv[i] = -1; rowat[i] = -1; }
    for (int k = 0; k <= n; k++) { Lp[k] = 0; Up[k] = 0; }

    for (int k = 0; k < n; k++) {
        Lp[k] = (int)lnz;
        Up[k] = (int)unz;

        /* x = A(:,k)（A 已按 q 重排；赋值而非累加，同 splu spsolve） */
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];

        /* 前代 x = L^{-1} x：L 列 j 单位对角（首项=1，行 rowat[j]），
         * 只对 x[rowat[j]] != 0 的列散射（等价稀疏可达集裁剪） */
        for (int j = 0; j < k; j++) {
            int r = rowat[j];
            double xr = x[r];
            if (xr != 0.0) {
                for (int p = Lp[j] + 1; p < Lp[j + 1]; p++)
                    x[Li[p]] -= Lx[p] * xr;
            }
        }

        /* 主元扫描：已主元行（pinv>=0 且非零）→ U 次对角项；未主元行 → 候选 */
        int ipiv = -1;
        double a = -1.0;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0) {
                double t = fabs(x[i]);
                if (t > a) { a = t; ipiv = i; }
            } else if (x[i] != 0.0) {
                if (unz >= Ucap) { *singular_out = -2; return; }
                Ui[unz] = pinv[i]; Ux[unz] = x[i]; unz++;
            }
        }
        if (ipiv == -1 || a <= 0.0) { *singular_out = k; return; }

        /* 部分主元 tol=1.0：自然对角行（重排前的 q[k]）若占优则保留 */
        int ck = diagrow[k];
        if (pinv[ck] < 0 && fabs(x[ck]) >= a) ipiv = ck;
        double pivot = x[ipiv];

        /* U 对角：每列末项 */
        if (unz >= Ucap) { *singular_out = -2; return; }
        Ui[unz] = k; Ux[unz] = pivot; unz++;
        pinv[ipiv] = k; rowat[k] = ipiv;

        /* L 列 k：单位对角首项 + 未主元行 x/pivot（仅非零项） */
        if (lnz >= Lcap) { *singular_out = -2; return; }
        Li[lnz] = ipiv; Lx[lnz] = 1.0; lnz++;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0 && x[i] != 0.0) {
                if (lnz >= Lcap) { *singular_out = -2; return; }
                Li[lnz] = i; Lx[lnz] = x[i] / pivot; lnz++;
            }
            x[i] = 0.0;               /* 全向量清零，防跨列残留 */
        }
    }
    Lp[n] = (int)lnz;
    Up[n] = (int)unz;

    /* L 行号 remap 为主元位置（splu.h 同款，trisolve 前代按位置索引） */
    for (long p = 0; p < lnz; p++) Li[p] = pinv[Li[p]];

    *lnnz = (int)lnz;
    *unnz = (int)unz;
    *singular_out = -1;
}

// ---------------------------------------------------------------------------
// trisolve_kernel：单 thread 前代 + 回代（lu_solve_perm 等价）
//   pb = P b：(P b)[k] = b[rowat[k]]
//   前代 L y = pb（L 单位对角首项，散射到 pb[位置]，位置索引来自 remap 后 Li）
//   回代 U z = y（U 对角末项，散射到 y[位置]）
//   输出 x = Q z：x[q[k]] = z[k]
// ---------------------------------------------------------------------------
__global__ void trisolve_kernel(
    int n,
    const int  *Lp, const int *Li, const double *Lx,
    const int  *Up, const int *Ui, const double *Ux,
    const int  *pinv, const int *q,
    const double *b, double *x,
    int        *work_i, double *work_d)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int *rowat = work_i;
    double *pb = work_d;
    double *y = work_d + n;

    for (int i = 0; i < n; i++) rowat[i] = -1;
    for (int i = 0; i < n; i++) if (pinv[i] >= 0) rowat[pinv[i]] = i;

    /* pb = P b */
    for (int k = 0; k < n; k++)
        pb[k] = (rowat[k] >= 0) ? b[rowat[k]] : 0.0;

    /* 前代 L y = pb */
    for (int k = 0; k < n; k++) {
        y[k] = pb[k] / Lx[Lp[k]];
        for (int p = Lp[k] + 1; p < Lp[k + 1]; p++)
            pb[Li[p]] -= Lx[p] * y[k];
    }

    /* 回代 U z = y */
    for (int k = n - 1; k >= 0; k--) {
        y[k] /= Ux[Up[k + 1] - 1];
        for (int p = Up[k]; p < Up[k + 1] - 1; p++)
            y[Ui[p]] -= Ux[p] * y[k];
    }

    /* x = Q z */
    for (int k = 0; k < n; k++) x[q ? q[k] : k] = y[k];
}
