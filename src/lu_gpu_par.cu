// ============================================================================
// lu_gpu_par.cu — M3 GPU 稀疏左视 LU 并行版 v7（L1 优化：共享 <48KB）
//
// 算法 = lu_gpu.cu（单 thread 初版）逐列等价，列内并行：
//   - 网格 1 block × 256 threads，顺序列循环（列 k 依赖 L 列 <k）
//   - 每列阶段：
//     1. x 清零（共享）+ mark 清零（共享）
//     2. A 散射（并行 over 列项）+ 可达集种子（atomicCAS 去重）
//     3. 结构 BFS 求可达集 R_k（rounds；快路径检查 + atomicCAS 去重 → jlist；
//        批量预取 Li 流水化，消除 L2 延迟链）
//     4. bitonic 排序 jlist（升序 = 主元序；jlist 在全局，排序读写全局）
//     5. 前代：jlist 升序散射（列内并行 over p；L 数据 L1 常驻）
//     6. 主元扫描：并行归约 + U 次对角共享原子追加
//     7. 自然主元（diagrow[k] 占优保留）+ U 对角末项 + pinv/rowat + L 对角首项
//     8. L 列并行构建（未主元行 x/pivot）+ 计数
//   末：L 行号 remap 为主元位置（与单 thread 版同款）
// 性能关键：共享 ≤48KB → L1 carveout 提升到 72KB+ → BFS/solve 的 L 工作集
//   （Li+Lp+pinv ~58KB）全部 L1 常驻；实测 BFS 从 102ms 降到 ~13ms。
//   为压共享：jlist 移到全局 xi 尾部（排序/散射读全局，量小可接受）。
// 数值差异仅来自列内散射顺序（并行 vs 顺序）→ ~1e-16 舍入，验收 L2<=1e-6。
// 纪律：标准 CUDA 子集；内核零动态分配（动态共享由 host 传字节数）。
// ============================================================================
#include <math.h>
#include <cuda_runtime.h>
#include "lu_gpu_par.h"

// 动态共享布局（host 传 LU_PAR_SHARED_BYTES）：
//   x[n](8B) + mark[n](4B) + red_v[256](8B) + red_i[256](4B) + cnt[16](4B)

// cnt 索引
#define CNT_CUR 0   /* 当前 frontier 计数 */
#define CNT_NXT 1   /* 下一轮 frontier 计数 */
#define CNT_NJ  2   /* jlist 计数 */
#define CNT_UC  3   /* U 追加计数（共享原子） */
#define CNT_LC  4   /* L 追加计数（共享原子） */
#define CNT_OVF 5   /* 0=ok 1=容量溢出(-2) 2=奇异(k) 3=结构溢出(-3) */

// 全局工作区（位于 xi）：
//   [0,n) rowat；[n, n+2*CAP) frontier 双缓冲；[n+2*CAP, n+4*CAP) jlist
#define GFRONT_BASE (n)
#define GFRONT(i)   xi[GFRONT_BASE + (i)]
#define JLIST_BASE  (n + 2 * LU_PAR_CAP)
#define JLIST(i)    xi[JLIST_BASE + (i)]

// ---------------------------------------------------------------------------
// lu_factor_par_kernel：并行稀疏左视 LU
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 6)
lu_factor_par_kernel(
    int n,
    const int  *Ap, const int *Ai, const double *Ax,
    const int  *diagrow,
    int        *Lp, int *Li, double *Lx, long Lcap,
    int        *Up, int *Ui, double *Ux, long Ucap,
    int        *pinv,
    int        *xi,
    double     *xg,
    int        *lnnz, int *unnz, int *singular_out)
{
    const int tid = threadIdx.x;
    int *rowat = xi;          /* [n] 主元行逆映射 */

    extern __shared__ char smem[];
    double *x      = (double *)smem;                              /* [n] */
    double *red8_v = x + n;                                      /* [8] */
    int *red8_i    = (int *)(red8_v + 8);                        /* [8] */
    int *mark      = red8_i + 8;                                 /* [n] */
    int *cnt       = mark + n;                                   /* [16] */

    /* 初始化 */
    for (int i = tid; i < n; i += 256) {
        pinv[i] = -1; rowat[i] = -1; mark[i] = 0; x[i] = 0.0;
    }
    __syncthreads();

    long lnz = 0, unz = 0;

    for (int k = 0; k < n; k++) {
        const int gen = k + 1;
        if (tid == 0) {
            Lp[k] = (int)lnz; Up[k] = (int)unz;
            cnt[CNT_CUR] = 0; cnt[CNT_NXT] = 0; cnt[CNT_NJ] = 0;
            cnt[CNT_UC] = (int)unz; cnt[CNT_LC] = 0; cnt[CNT_OVF] = 0;
        }
        __syncthreads();

        /* 1. x 清零（共享）+ mark 清零（共享） */
        for (int i = tid; i < n; i += 256) { x[i] *= 0.0; mark[i] = 0; }
        __syncthreads();

        /* 2. A 散射（赋值，同单 thread 版）+ 可达集种子 */
        for (int p = Ap[k] + tid; p < Ap[k + 1]; p += 256) {
            int i = Ai[p];
            x[i] = Ax[p];
            if (atomicCAS(&mark[i], 0, gen) == 0) {   /* 新种子 */
                int s = atomicAdd(&cnt[CNT_CUR], 1);
                if (s < LU_PAR_CAP) GFRONT(s) = i;
                else cnt[CNT_OVF] = 3;
                int j = pinv[i];
                if (j >= 0 && j < k) {
                    int s2 = atomicAdd(&cnt[CNT_NJ], 1);
                    if (s2 < LU_PAR_CAP) JLIST(s2) = j; else cnt[CNT_OVF] = 3;
                }
            }
        }
        __syncthreads();

        /* 3. 结构 BFS：快路径检查 + CAS 去重；Li 批量预取流水化 */
        {
            int curcnt = cnt[CNT_CUR];
            int curbase = 0, nxtbase = LU_PAR_CAP;
            while (curcnt > 0) {
                for (int t = tid; t < curcnt; t += 256) {
                    int i = GFRONT(curbase + t);
                    int j = pinv[i];
                    if (j >= 0 && j < k) {
                        int l0 = Lp[j] + 1, nent = Lp[j + 1] - l0;
                        for (int e = 0; e < nent; e += 16) {
                            int ii0[16];
                            #pragma unroll
                            for (int u = 0; u < 16; u++)
                                ii0[u] = (e + u < nent) ? Li[l0 + e + u] : -1;
                            #pragma unroll
                            for (int u = 0; u < 16; u++) {
                                int ii = ii0[u];
                                if (ii >= 0 && mark[ii] != gen &&
                                    atomicCAS(&mark[ii], 0, gen) == 0) {
                                    int s = atomicAdd(&cnt[CNT_NXT], 1);
                                    if (s < LU_PAR_CAP) GFRONT(nxtbase + s) = ii;
                                    else cnt[CNT_OVF] = 3;
                                    int jj = pinv[ii];
                                    if (jj >= 0 && jj < k) {
                                        int s2 = atomicAdd(&cnt[CNT_NJ], 1);
                                        if (s2 < LU_PAR_CAP) JLIST(s2) = jj;
                                        else cnt[CNT_OVF] = 3;
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();
                int tcur = curbase; curbase = nxtbase; nxtbase = tcur;
                curcnt = cnt[CNT_NXT];
                cnt[CNT_NXT] = 0;
                __syncthreads();
            }
        }
        __syncthreads();
        if (cnt[CNT_OVF] == 3) { if (tid == 0) *singular_out = -3; return; }

        /* 4. bitonic 排序 JLIST[0..nj) 升序（= 主元序；全局数组） */
        {
            int nj = cnt[CNT_NJ];
            int P = 1; while (P < nj) P <<= 1;
            for (int e = nj + tid; e < P; e += 256) JLIST(e) = 0x7fffffff;
            __syncthreads();
            for (int kk = 2; kk <= P; kk <<= 1) {
                for (int jj = kk >> 1; jj > 0; jj >>= 1) {
                    for (int i = tid; i < P; i += 256) {
                        int pi = i ^ jj;
                        if (pi > i) {
                            int a = JLIST(i), b = JLIST(pi);
                            if (((i & kk) == 0) ? (a > b) : (a < b)) {
                                JLIST(i) = b; JLIST(pi) = a;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
        }
        __syncthreads();

        /* 5. 前代：JLIST 升序散射（列内并行 over p；L 数据 L1 常驻）
             （R9 A/B：去 __ldcg → 242.4ms（-4.5%）被否，保留 __ldcg——L2 缓存更优） */
        {
            int nj = cnt[CNT_NJ];
            for (int t = 0; t < nj; t++) {
                int j = JLIST(t);
                double xr = x[rowat[j]];
                if (xr != 0.0) {
                    for (int p = Lp[j] + 1 + tid; p < Lp[j + 1]; p += 256)
                        x[__ldcg(&Li[p])] -= __ldcg(&Lx[p]) * xr;
                }
                __syncthreads();
            }
        }
        __syncthreads();

        /* 6. 主元扫描：未主元行 → |x| max 归约（最小行号 tie-break）；
             已主元行 x!=0 → U 次对角共享原子追加 */
        {
            double maxv = -1.0; int maxi = -1;
            for (int i = tid; i < n; i += 256) {
                int pi = pinv[i];
                if (pi < 0) {
                    double t = fabs(x[i]);
                    if (t > maxv || (t == maxv && i < maxi)) { maxv = t; maxi = i; }
                } else if (x[i] != 0.0) {
                    int s = atomicAdd(&cnt[CNT_UC], 1);
                    if (s < Ucap) { Ui[s] = pi; Ux[s] = x[i]; }
                    else cnt[CNT_OVF] = 1;
                }
            }
            /* warp shuffle 归约 + 8 槽跨 warp 归约（省共享 → L1 扩大） */
            #pragma unroll
            for (int s2 = 16; s2 > 0; s2 >>= 1) {
                double v2 = __shfl_down_sync(0xffffffffu, maxv, s2);
                int i2 = __shfl_down_sync(0xffffffffu, maxi, s2);
                if (v2 > maxv || (v2 == maxv && i2 < maxi)) { maxv = v2; maxi = i2; }
            }
            if ((tid & 31) == 0) { red8_v[tid >> 5] = maxv; red8_i[tid >> 5] = maxi; }
            __syncthreads();
            if (tid == 0) {
                maxv = -1.0; maxi = -1;
                for (int w = 0; w < 8; w++) {
                    double v2 = red8_v[w]; int i2 = red8_i[w];
                    if (v2 > maxv || (v2 == maxv && i2 < maxi)) { maxv = v2; maxi = i2; }
                }
                red8_v[0] = maxv; red8_i[0] = maxi;
            }
            __syncthreads();
        }
        __syncthreads();

        /* 7. 主元选择（tol=1.0 自然主元）+ U 对角末项 + pinv/rowat + L 对角首项 */
        if (tid == 0) {
            double a = red8_v[0]; int ipiv = red8_i[0];
            if (ipiv == -1 || a <= 0.0) {
                cnt[CNT_OVF] = 2;                 /* 奇异列 k */
            } else {
                int ck = diagrow[k];
                if (pinv[ck] < 0 && fabs(x[ck]) >= a) ipiv = ck;
                double pivot = x[ipiv];
                int s = atomicAdd(&cnt[CNT_UC], 1);   /* U 对角（末项） */
                if (s < Ucap) { Ui[s] = k; Ux[s] = pivot; }
                else cnt[CNT_OVF] = 1;
                pinv[ipiv] = k; rowat[k] = ipiv;
                if (Lp[k] < Lcap) { Li[Lp[k]] = ipiv; Lx[Lp[k]] = 1.0; }
                else cnt[CNT_OVF] = 1;
                cnt[CNT_LC] = Lp[k] + 1;          /* L 次对角追加起点 */
            }
        }
        __syncthreads();
        if (cnt[CNT_OVF] == 2) { if (tid == 0) *singular_out = k; return; }
        if (cnt[CNT_OVF] == 1) { if (tid == 0) *singular_out = -2; return; }

        /* 8. L 列 k 并行构建（未主元行 x/pivot；对角首项已写） */
        {
            double pivot = x[rowat[k]];
            for (int i = tid; i < n; i += 256) {
                if (pinv[i] < 0 && x[i] != 0.0) {
                    int s = atomicAdd(&cnt[CNT_LC], 1);
                    if (s < Lcap) { Li[s] = i; Lx[s] = x[i] / pivot; }
                    else cnt[CNT_OVF] = 1;
                }
            }
            __syncthreads();
            if (cnt[CNT_OVF] == 1) { if (tid == 0) *singular_out = -2; return; }
            lnz = cnt[CNT_LC]; unz = cnt[CNT_UC];
            if (tid == 0) { Lp[k + 1] = (int)lnz; Up[k + 1] = (int)unz; }
            __syncthreads();
        }
    }

    /* L 行号 remap 为主元位置（同单 thread 版） */
    for (long p = tid; p < lnz; p += 256) Li[p] = pinv[Li[p]];
    __syncthreads();
    if (tid == 0) { *lnnz = (int)lnz; *unnz = (int)unz; *singular_out = -1; }
}

// ---------------------------------------------------------------------------
// trisolve_par_kernel：并行前代 + 回代（lu_solve_perm 等价）
//   pb = P b → L y = pb（列散射，列内并行）→ U z = y（列散射）→ x = Q z
//   工作区：work_i[n]（rowat）、work_d[2n]（pb + y）
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 6)
trisolve_par_kernel(
    int n,
    const int  *Lp, const int *Li, const double *Lx,
    const int  *Up, const int *Ui, const double *Ux,
    const int  *pinv, const int *q,
    const double *b, double *x,
    int        *work_i, double *work_d)
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

    /* 前代 L y = pb（L 单位对角首项；Li 已 remap 为主元位置） */
    for (int k = 0; k < n; k++) {
        double yk = pb[k] / Lx[Lp[k]];
        y[k] = yk;                        /* 各线程同值，冗余写 */
        for (int p = Lp[k] + 1 + tid; p < Lp[k + 1]; p += 256)
            pb[Li[p]] -= Lx[p] * yk;
        __syncthreads();
    }

    /* 回代 U z = y（U 对角末项） */
    for (int k = n - 1; k >= 0; k--) {
        double yk = y[k] / Ux[Up[k + 1] - 1];
        for (int p = Up[k] + tid; p < Up[k + 1] - 1; p += 256)
            y[Ui[p]] -= Ux[p] * yk;
        y[k] = yk;                        /* 冗余写 */
        __syncthreads();
    }

    /* x = Q z */
    for (int k = tid; k < n; k += 256) x[q ? q[k] : k] = y[k];
}
