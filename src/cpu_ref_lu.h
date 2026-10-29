// ============================================================================
// cpu_ref_lu.h — M3 GPU 验证用 CPU 参考（splu.h 稀疏左视 LU 的 C 封装）
// 供 test_lu_gpu.cu（C++/CUDA）以 extern "C" 调用；splu.h 本体留 C 编译
// （其 malloc 无显式转型，C++ 编译会报错）。
// ============================================================================
#ifndef CPU_REF_LU_H
#define CPU_REF_LU_H

#ifdef __cplusplus
extern "C" {
#endif

// 稀疏左视 LU（splu.h lu_factor + lu_solve_perm 完整链路）：
//   P * A(: ,q) = L * U；A 为原始序 CSC（0-based）；q 为 COLAMD 列序（可为 NULL）
//   b 0-based（n 个 double）；x 输出（原始序，Q 已还原）
// 返回 -1 成功；>=0 奇异列 k
int cpu_lu_ref(int n, const int *Ap, const int *Ai, const double *Ax,
               const int *q, const double *b, double tol,
               double *x, long *lnnz, long *unnz);

#ifdef __cplusplus
}
#endif

#endif /* CPU_REF_LU_H */
