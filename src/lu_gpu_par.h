// ============================================================================
// lu_gpu_par.h — M3 GPU 稀疏左视 LU：并行版 kernel 声明
//
// 与 lu_gpu.cu（单 thread 初版）逐列数值等价，列内并行化：
//   - A 散射 / 可达集 BFS / 前代散射 / 主元归约 / U/L 构建 / x 清零 全部并行
//   - 列间依赖（列 k 需 L 列 <k）由单 block 顺序列循环 + __syncthreads 满足
//   - 前代用可达集裁剪 + warp 专用加载器流水（见 lu_gpu_par.cu 头注释）
// 纪律：标准 CUDA 子集；内核零动态分配（动态共享由 host 传字节数）。
// ============================================================================
#ifndef DSH_LU_GPU_PAR_H
#define DSH_LU_GPU_PAR_H

// 并行前代/可达集容量上界（实测 pg5k max |R_k|=746、max 散射列=679；1024 余量）
#define LU_PAR_CAP 1024

// 动态共享内存字节数（block=256）——R8 修正：与 v8 kernel 实际布局对齐
//   （旧宏 12n+28864 描述 ring_v[8][256]/jlist/s_pinv/vready 旧设计，v8 已把
//     frontier/jlist 移全局 xi、归约改 red8[8]——旧宏导致 pg2k 多分配 28KB
//     → L1 carveout 缩水 + Occupancy 限 1 block/SM）
//   v8 实际布局：x[n](8B) + red8_v[8](8B) + red8_i[8](4B) + mark[n](4B) +
//                cnt[16](4B) = 12n + 160 B（+96B 余量对齐）
#define LU_PAR_SHARED_BYTES(n) \
    ((size_t)(n) * 12 + (size_t)256)

// ---------------------------------------------------------------------------
// lu_factor_par_kernel：并行稀疏左视 LU（P * A(: ,q) = L * U）
//   网格：1 block × 256 threads（单 block 顺序列循环；列内并行）
//   输入  A（q 序 CSC，原始行号）+ diagrow[k]=q[k]
//   输出  Lp/Li/Lx、Up/Ui/Ux（CSC 因子）、pinv、lnnz、unnz、singular_out
//   工作区（全局）：pinv[n]（-1 初值）、xi[n+2*CAP]（[0,n)=rowat、
//                   [n,n+2*CAP)=frontier 双缓冲）、xg[n] 未用（兼容，可为 NULL）
//   动态共享：LU_PAR_SHARED_BYTES(n) 字节（见 .cu 内布局）
//   singular_out：-1 成功；>=0 奇异列 k；-2 容量溢出；-3 结构溢出（不应发生）
// ---------------------------------------------------------------------------
__global__ void lu_factor_par_kernel(
    int n,
    const int  *Ap, const int *Ai, const double *Ax,
    const int  *diagrow,
    int        *Lp, int *Li, double *Lx, long Lcap,
    int        *Up, int *Ui, double *Ux, long Ucap,
    int        *pinv,
    int        *xi,                                  // [n+2*CAP]：rowat+front
    double     *xg,                                  // 未用（兼容），可为 NULL
    int        *lnnz, int *unnz, int *singular_out);

// ---------------------------------------------------------------------------
// trisolve_par_kernel：并行前代 + 回代（lu_solve_perm 等价）
//   pb = P b → L y = pb（前代，L 单位对角首项）→ U z = y（回代，U 对角末项）
//   → x = Q z；L 列内散射并行（Li 为 remap 后位置）
//   工作区：work_i[n]（rowat）、work_d[2n]（pb + y）
// ---------------------------------------------------------------------------
__global__ void trisolve_par_kernel(
    int n,
    const int  *Lp, const int *Li, const double *Lx,
    const int  *Up, const int *Ui, const double *Ux,
    const int  *pinv, const int *q,                 // q=NULL 自然序
    const double *b, double *x,
    int        *work_i, double *work_d);

#endif /* DSH_LU_GPU_PAR_H */
