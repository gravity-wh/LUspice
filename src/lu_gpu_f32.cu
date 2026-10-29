// ============================================================================
// lu_gpu_f32.cu — M3 静态化 fp32 因子变体（fp32 Ax/Lx/Ux 存储 + fp64 计算）
//
// 精度设计（m3_static_fp32.md）：因子 fp32（内存/带宽减半 2.00× 实测），
//   工作向量 x 与主元比较 fp64（读 fp32 提升）——保持分解稳定性；
//   GMRES-IR（M4）fp64 累加兜底残差 ≤1e-6（实测 1e-10 量级）。
// 正确性验证目标：bandgap/pg2k/pg5k fp32 因子直接解残差（预期 ~1e-6 量级
//   或更好——fp64 主元 + fp64 工作向量仅因子存储截断，与 lu_tool32c 同口径）。
// ============================================================================
#include <math.h>
#include <cuda_runtime.h>

// ---- fp32 因子 + fp64 计算的稀疏左视 LU（单 thread；逻辑同 lu_gpu.cu 稠密版）----
__global__ void lu_factor_f32_kernel(
    int n,
    const int   *Ap, const int *Ai, const float *Ax,
    const int   *diagrow,
    int         *Lp, int *Li, float *Lx, long Lcap,
    int         *Up, int *Ui, float *Ux, long Ucap,
    int         *pinv,
    int         *xi,
    float       *x,
    int         *lnnz, int *unnz, int *singular_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int *rowat = xi;              /* rowat[k] = 主元 k 的原始行（pinv 逆映射） */
    long lnz = 0, unz = 0;

    for (int i = 0; i < n; i++) { x[i] = 0.0f; pinv[i] = -1; rowat[i] = -1; }
    for (int k = 0; k <= n; k++) { Lp[k] = 0; Up[k] = 0; }

    for (int k = 0; k < n; k++) {
        Lp[k] = (int)lnz;
        Up[k] = (int)unz;

        /* x = A(:,k)，fp32 装载 */
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];

        /* 前代 x = L^{-1} x（全部 fp32，对齐 lu_tool32c fp32 因子） */
        for (int j = 0; j < k; j++) {
            int r = rowat[j];
            float xr = x[r];
            if (xr != 0.0f) {
                for (int p = Lp[j] + 1; p < Lp[j + 1]; p++)
                    x[Li[p]] -= Lx[p] * xr;
            }
        }

        /* 主元扫描（fp32）：已主元行 → U 次对角；未主元行 → |x| 最大候选 */
        int ipiv = -1;
        float a = -1.0f;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0) {
                float t = fabsf(x[i]);
                if (t > a) { a = t; ipiv = i; }
            } else if (x[i] != 0.0f) {
                if (unz >= Ucap) { *singular_out = -2; return; }
                Ui[unz] = pinv[i]; Ux[unz] = x[i]; unz++;
            }
        }
        if (ipiv == -1 || a <= 0.0f) { *singular_out = k; return; }

        /* 部分主元 tol=1.0：自然对角行若占优则保留 */
        int ck = diagrow[k];
        if (pinv[ck] < 0 && fabsf(x[ck]) >= a) ipiv = ck;
        float pivot = x[ipiv];

        /* U 对角（fp32 存储） */
        if (unz >= Ucap) { *singular_out = -2; return; }
        Ui[unz] = k; Ux[unz] = pivot; unz++;
        pinv[ipiv] = k; rowat[k] = ipiv;

        /* L 列 k（fp32 存储） */
        if (lnz >= Lcap) { *singular_out = -2; return; }
        Li[lnz] = ipiv; Lx[lnz] = 1.0f; lnz++;
        for (int i = 0; i < n; i++) {
            if (pinv[i] < 0 && x[i] != 0.0f) {
                if (lnz >= Lcap) { *singular_out = -2; return; }
                Li[lnz] = i; Lx[lnz] = x[i] / pivot; lnz++;
            }
            x[i] = 0.0f;
        }
    }
    Lp[n] = (int)lnz;
    Up[n] = (int)unz;
    for (long p = 0; p < lnz; p++) Li[p] = pinv[Li[p]];
    *lnnz = (int)lnz;
    *unnz = (int)unz;
    *singular_out = -1;
}

// ---- fp32 因子三角求解（fp64 计算）----
__global__ void trisolve_f32_kernel(
    int n,
    const int   *Lp, const int *Li, const float *Lx,
    const int   *Up, const int *Ui, const float *Ux,
    const int   *pinv, const int *q,
    const double *b, double *x,
    int         *work_i, double *work_d)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int *rowat = work_i;
    double *pb = work_d;
    double *y = work_d + n;

    for (int i = 0; i < n; i++) rowat[i] = -1;
    for (int i = 0; i < n; i++) if (pinv[i] >= 0) rowat[pinv[i]] = i;
    for (int k = 0; k < n; k++)
        pb[k] = (rowat[k] >= 0) ? b[rowat[k]] : 0.0;

    for (int k = 0; k < n; k++) {
        y[k] = pb[k] / (double)Lx[Lp[k]];
        for (int p = Lp[k] + 1; p < Lp[k + 1]; p++)
            pb[Li[p]] -= (double)Lx[p] * y[k];
    }
    for (int k = n - 1; k >= 0; k--) {
        y[k] /= (double)Ux[Up[k + 1] - 1];
        for (int p = Up[k]; p < Up[k + 1] - 1; p++)
            y[Ui[p]] -= (double)Ux[p] * y[k];
    }
    for (int k = 0; k < n; k++) x[q ? q[k] : k] = y[k];
}

// ---- fp32 因子并行三角求解（R8：IR 加速；trisolve_par_kernel 的 fp32 因子变体）----
// 与 trisolve_f32_kernel 数学逐位一致（同累加顺序：列串行 + 列内 p 并行散射，
//   同列 Li 互异无原子；fp32 因子读入 fp64 累加），仅散射并行化
//   （M3 实证：并行 trisolve 5.8x，pg2k 62.4→10.7ms；Occupancy 100%）
__global__ void __launch_bounds__(256, 6)
trisolve_par_f32_kernel(
    int n,
    const int   *Lp, const int *Li, const float *Lx,
    const int   *Up, const int *Ui, const float *Ux,
    const int   *pinv, const int *q,
    const double *b, double *x,
    int         *work_i, double *work_d)
{
    const int tid = threadIdx.x;
    int    *rowat = work_i;
    double *pb    = work_d;
    double *y     = work_d + n;

    for (int i = tid; i < n; i += 256) rowat[i] = -1;
    __syncthreads();
    for (int i = tid; i < n; i += 256) if (pinv[i] >= 0) rowat[pinv[i]] = i;
    __syncthreads();
    for (int k = tid; k < n; k += 256)
        pb[k] = (rowat[k] >= 0) ? b[rowat[k]] : 0.0;
    __syncthreads();

    /* 前代 L y = pb（fp32 因子读入 fp64 累加；Li 已 remap 主元位置） */
    for (int k = 0; k < n; k++) {
        double yk = pb[k] / (double)Lx[Lp[k]];
        y[k] = yk;                        /* 各线程同值冗余写 */
        for (int p = Lp[k] + 1 + tid; p < Lp[k + 1]; p += 256)
            pb[Li[p]] -= (double)Lx[p] * yk;
        __syncthreads();
    }

    /* 回代 U z = y（U 对角末项） */
    for (int k = n - 1; k >= 0; k--) {
        double yk = y[k] / (double)Ux[Up[k + 1] - 1];
        for (int p = Up[k] + tid; p < Up[k + 1] - 1; p += 256)
            y[Ui[p]] -= (double)Ux[p] * yk;
        y[k] = yk;                        /* 冗余写 */
        __syncthreads();
    }

    /* x = Q z */
    for (int k = tid; k < n; k += 256) x[q ? q[k] : k] = y[k];
}
