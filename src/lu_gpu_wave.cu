// ============================================================================
// lu_gpu_wave.cu — M3 level-scheduling factor（每列 block + 符号 U 依赖前代）
//
// 设计：design/m3_level_sched.md
// 关键：符号分解（CPU）预给每列 lnnz/unnz → host 预计算 Lp/Up（累积、静态）
//   → kernel 每列写固定段（无 Lp/Up 竞争）；前代按符号 U 依赖散射（wave 间
//   launch 同步保证依赖列 L 完整）；x 用每列私有段（x_all[wave_cnt][n] 或
//   单列 kernel 内共享——本版用 device 全局 x + 每列 block 独占行域，wave
//   内列 A 行可能重叠——先用"每列一段 x"（x_all 传该列段指针））
// ============================================================================
#include <cuda_runtime.h>
#include <math.h>

#define WAVE_BLOCK 256

// 每列 block：前代（符号 U 依赖散射）+ 主元（t==0 串行）+ U/L 写固定段
__global__ void factor_wave_kernel(
    const int *cols, int n,
    const int *Ap, const int *Ai, const double *Ax,
    const int *symUp, const int *symUi,      /* 符号 U 结构（依赖图） */
    const int *sym_pinv, const int *sym_rowat,  /* 固定主元（符号分解输出） */
    int *Lp, int *Li, double *Lx, long Lcap,
    int *Up, int *Ui, double *Ux, long Ucap,
    double *x_seg)                            /* 每列私有 x 段 */
{
    int t = threadIdx.x;
    int k = cols[blockIdx.x];
    /* R9: x_seg = 每 block 私有段（driver 传 x_all[wave_cnt][n]，block b 用段 b）——
         原实现所有 block 共享同一 x[0..n) → wave 内列 A 装载/前代互相覆盖（值错乱） */
    double *x_seg_self = x_seg + (long)blockIdx.x * n;
    /* 1. 清零 + 装载 x_seg = A(:,k)（并行） */
    for (int i = t; i < n; i += blockDim.x) x_seg_self[i] = 0.0;
    __syncthreads();
    for (int p = Ap[k] + t; p < Ap[k + 1]; p += blockDim.x) x_seg_self[Ai[p]] = Ax[p];
    __syncthreads();
    /* 2. 前代：符号 U 依赖 j（R9: 升序排序后散射——symUi 是行序写入，
          与 symbolic_dense 的 j=0..k-1 升序全列扫描 fp 顺序一致；否则舍入路径
          不同 → U 结构漂移/NaN。加 xr!=0 守卫对齐 dense 语义） */
    __shared__ int sdeps[1024];
    __shared__ int snd;
    if (t == 0) {
        int nd = 0;
        for (int p = symUp[k]; p < symUp[k + 1] - 1; p++) {
            int j = symUi[p];
            if (j >= 0 && j < k) sdeps[nd++] = j;
        }
        snd = nd;
    }
    __syncthreads();
    {
        int nd = snd;
        int P = 1; while (P < nd) P <<= 1;
        for (int e = nd + t; e < P; e += blockDim.x) sdeps[e] = 0x7fffffff;
        __syncthreads();
        for (int kk = 2; kk <= P; kk <<= 1)
            for (int jj = kk >> 1; jj > 0; jj >>= 1) {
                for (int i = t; i < P; i += blockDim.x) {
                    int pi = i ^ jj;
                    if (pi > i) {
                        int a = sdeps[i], b = sdeps[pi];
                        if (((i & kk) == 0) ? (a > b) : (a < b)) { sdeps[i] = b; sdeps[pi] = a; }
                    }
                }
                __syncthreads();
            }
        for (int tt = 0; tt < nd; tt++) {
            int j = sdeps[tt];
            int r = sym_rowat[j];
            double xr = x_seg_self[r];
            __syncthreads();
            if (xr != 0.0)
                for (int q = Lp[j] + 1 + t; q < Lp[j + 1]; q += blockDim.x)
                    x_seg_self[Li[q]] -= Lx[q] * xr;
            __syncthreads();
        }
    }
    /* 3. 固定主元（符号）→ U/L 写（R9: 并行化——原 t==0 串行全行扫描 2×n 是 642ms 瓶颈；
          共享原子追加，值逐位一致、列内条目顺序可为任意（位置标记，trisolve 与顺序无关）） */
    __shared__ int scnt[2];
    double pivot = x_seg_self[sym_rowat[k]];
    if (sym_rowat[k] < 0) return;
    if (t == 0) { scnt[0] = 0; scnt[1] = 0; }
    __syncthreads();
    /* U 次对角：已主元行（sym_pinv[i] < k）且 x != 0 */
    for (int i = t; i < n; i += blockDim.x) {
        if (sym_pinv[i] >= 0 && sym_pinv[i] < k && x_seg_self[i] != 0.0) {
            int s = atomicAdd(&scnt[0], 1);
            if (Up[k] + s < Ucap) { Ui[Up[k] + s] = sym_pinv[i]; Ux[Up[k] + s] = x_seg_self[i]; }
        }
    }
    __syncthreads();
    if (t == 0) {
        long unz = Up[k] + scnt[0];
        if (unz < Ucap) { Ui[unz] = k; Ux[unz] = pivot; }
        if (Lp[k] < Lcap) { Li[Lp[k]] = sym_rowat[k]; Lx[Lp[k]] = 1.0; }
    }
    __syncthreads();
    /* L 次对角：未主元行（sym_pinv[i] > k）且 x != 0 */
    for (int i = t; i < n; i += blockDim.x) {
        if (sym_pinv[i] > k && x_seg_self[i] != 0.0) {
            int s = atomicAdd(&scnt[1], 1);
            if (Lp[k] + 1 + s < Lcap) { Li[Lp[k] + 1 + s] = i; Lx[Lp[k] + 1 + s] = x_seg_self[i] / pivot; }
        }
    }
    __syncthreads();
    if (t == 0) { Up[k + 1] = (int)(Up[k] + scnt[0] + 1); Lp[k + 1] = (int)(Lp[k] + 1 + scnt[1]); }
}
