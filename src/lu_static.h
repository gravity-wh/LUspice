// ============================================================================
// lu_static.h — M2 静态化稀疏 LU 接口（符号分解复用 + 零动态分配数值分解）
//
// 设计依据：design/lu_static_storage.md（goal R9）
//   - 符号分解（CPU）一次性产出含 fill-in 的列结构 → GPU 预分配
//   - 数值分解（GPU）只写 Ax（fp32），结构不变 → 零动态分配
//   - 主元比较用 fp64（读 fp32 Ax 提升），对齐 fpgmres.c 部分主元 tol=1.0
//   - 平台无关：只依赖 cuda_compat.h 的 DSH_* 抽象（CUDA/MXMACA 双平台）
//
// 迁移纪律：标准 CUDA 子集 + DSH_* 接口；不用 cuSparse/cuSOLVER
// ============================================================================
#ifndef DSH_LU_STATIC_H
#define DSH_LU_STATIC_H

#include "cuda_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---- 静态 LU 数据结构（M2 核心）----
typedef struct {
    int     n;          // 矩阵阶
    long    cnnz;       // 原矩阵非零元（CS 模式）
    long    lnnz;       // 符号分解后 L 非零元（精确，由符号分解给出）
    long    unnz;       // U 非零元
    long    cap;        // 分配容量 = lnnz + unnz + 余量(<1%)

    // 列结构（符号分解填充一次，数值分解只读）
    int    *Ap;         // 列指针（n+1，含 fill-in 的列结构）[host + device]
    int    *Ai;         // 行索引（预填充 fill-in 位置）[device]
    float  *Ax;         // fp32 数值（L/U 合并存储，数值分解每因子重写）[device]

    // 置换
    int    *P;          // 行置换（部分主元 tol=1.0）[device]
    int    *Q;          // 列置换（COLAMD 序，CPU 侧产生）[host 传入]

    // 诊断（S₁ 病态策略输入）
    double  min_diag_U;   // 最小 |U 对角|
    double  pivot_growth; // 主元增长因子

    // 设备指针管理（cuda_compat.h 抽象层分配）
    int     dev_allocated; // 1 = device 内存已分配

    // 内部因子缓存（初版：指向 splu.h 的 lu_factors；M3 静态化时替换为
    // 预分配的 L/U 静态布局）。仅 CPU 正确性版使用。
    void   *factor_cache;
    double  pin;       // 对角扰动（电源网格锚定浮动节点，如 pg 系 1e-8/1e-7）
} dsh_lu_static;

// ---- 符号分解（CPU，模式变化时执行一次；赛题允许异构）----
// 输入：COO 行/列/值（或 CSC），COLAMD 列序
// 输出：Ap/Ai（含 fill-in 结构）、lnnz/unnz
// 实现提示：可先复用 CPU 侧 splu 的 LNNZ 估计（M0 实测比值 1.6-20.6），
//          后续换精确符号分解（cs_lu 符号阶段，n46-mix splu.h 有参考）
int dsh_symbolic(int n, long cnnz,
                 const int *coo_row, const int *coo_col,   // 0-based
                 const int *colamd_q,                       // 列置换（可为 NULL=自然序）
                 dsh_lu_static *F);

// ---- 数值分解（GPU，每 Newton 因子执行；零动态分配）----
// 输入：COO 数值 + 静态结构（Ap/Ai 已定）
// 输出：Ax（fp32 L/U 数值）、P（行置换）、诊断
int dsh_numeric(dsh_lu_static *F,
                const int *coo_row, const int *coo_col,
                const double *coo_val, int nnz);

// ---- 三角求解（GPU，前代+回代；M4 深化为 level 化）----
// x = U^{-1} L^{-1} P b（Q 序已含在结构中）
int dsh_trisolve(const dsh_lu_static *F, const double *b, double *x);

// ---- GMRES-IR 外壳（M4：复用 fpgmres.c 的 IR 逻辑，SpMV 走 SELL-C-σ/CSR）----
int dsh_gmres_ir(dsh_lu_static *F, const double *b, double *x,
                 int max_outer, int max_inner, double rtol,
                 double *residual_seq, int *iters);

// ---- 生命周期 ----
int dsh_lu_static_init(dsh_lu_static *F);       // 零初始化
int dsh_lu_static_alloc(dsh_lu_static *F);      // device 分配（按 cap）
void dsh_lu_static_free(dsh_lu_static *F);      // device 释放

// ---- 诊断查询（喂 S₁ 病态策略）----
const char *dsh_lu_status_str(int rc);           // rc 含义：0=OK 1=SINGULAR 2=ALLOC 3=ARG

// ---- 符号复用计数（M2 验收：100 因子仅 1 次符号分解）----
long dsh_symbolic_count(void);                   // 累计符号分解次数（进程级）

#ifdef __cplusplus
}
#endif

#endif /* DSH_LU_STATIC_H */
