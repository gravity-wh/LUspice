// ============================================================================
// lu_gpu.h — M3 GPU 稀疏左视 LU：kernel 声明（device 单 thread 初版）
//
// 逻辑 = splu.h lu_factor / lu_solve_perm 的预分配移植：
//   - P * A(: ,q) = L * U，A 由 host 按 COLAMD q 重排后上传（q 序 CSC）
//   - L：CSC，单位对角存每列首项，行号先为原始行、分解后 remap 为主元位置
//   - U：CSC，对角存每列末项（位置 k）
//   - 部分主元 fp64 tol=1.0；奇异/容量溢出经 singular_out 返回
//   - 内核零动态分配：全部工作数组由 host 预分配传入
// ============================================================================
#ifndef DSH_LU_GPU_H
#define DSH_LU_GPU_H

// ---------------------------------------------------------------------------
// lu_factor_kernel：单 thread 完整稀疏左视 LU（稠密简化可达集）
//   输入  A（q 序 CSC，原始行号）+ diagrow[k]=q[k]（自然主元候选行）
//   输出  Lp/Li/Lx、Up/Ui/Ux（CSC 因子）、pinv、lnnz、unnz、singular_out
//   singular_out：-1 成功；>=0 奇异列 k；-2 容量溢出（不应发生）
//   工作区：pinv[n]（初值 -1）、xi[2n+1]（前 n 作 rowat = pinv 逆映射）、
//           x[n] 稠密工作向量（每列末清零）
//   容量：Lcap/Ucap 由 host 按上界 n(n+1)/2+16 预分配（保证不溢出，免 grow）
// ---------------------------------------------------------------------------
__global__ void lu_factor_kernel(
    int n,
    const int  *Ap, const int *Ai, const double *Ax,   // A 按 q 重排后的 CSC
    const int  *diagrow,                                // diagrow[k] = q[k]
    int        *Lp, int *Li, double *Lx, long Lcap,     // L 因子工作区
    int        *Up, int *Ui, double *Ux, long Ucap,     // U 因子工作区
    int        *pinv,                                   // [n] 行置换
    int        *xi,                                     // [2n+1] 工作区
    double     *x,                                      // [n] 稠密工作向量
    int        *lnnz, int *unnz, int *singular_out);    // 输出计数/状态

// ---------------------------------------------------------------------------
// trisolve_kernel：单 thread 前代 + 回代（lu_solve_perm 等价）
//   pb = P b → L y = pb（前代，L 单位对角首项）→ U z = y（回代，U 对角末项）
//   → x = Q z：x[q[k]] = z[k]（q=NULL 视作自然序）
//   工作区：work_i[n]（rowat）、work_d[2n]（pb + y）
// ---------------------------------------------------------------------------
__global__ void trisolve_kernel(
    int n,
    const int  *Lp, const int *Li, const double *Lx,
    const int  *Up, const int *Ui, const double *Ux,
    const int  *pinv, const int *q,                    // q=NULL 自然序
    const double *b, double *x,                        // 输入 b、输出 x
    int        *work_i, double *work_d);

#endif /* DSH_LU_GPU_H */
