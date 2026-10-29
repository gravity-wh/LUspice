// ============================================================================
// multiblock_lu.cu — P3 多 block 波次任务图调度稀疏 LU（v1）
//
// 设计（design/m3_multiblock_sched.md）：
//   - 主机：COLAMD 列序 → 主机 ST 扫描得到 pinv/rowat/L模式/每列容量/参考解
//           → 由 L 模式 + rowat 派生 deps(每列依赖列表) 与 level(列层级)
//   - 设备：按 level 波次发射 grid（每 block 一列），波次间内核边界隐式同步；
//           pinv/rowat 为只读输入（消除同波次列间主元竞态）；
//           每列: A散射 → deps 前代(屏障数=|deps|) → 写 L/U 列(静态槽位)
//           → 全部列完成后再 remap Li 为主元位置
//   - 验收：x 相对误差 vs cpu_lu_ref ≤1e-6；相对单 block PAR 加速比
//
// 编译（race7 根）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -I src/solver \
//     -I /home/gravity/g3-ws/src -I /home/gravity/n46-mix/src/include \
//     -o multiblock_lu_bench src/solver/mp_bench/multiblock_lu.cu \
//     /home/gravity/n46-mix/src/maths/KLU/colamd.c \
//     /home/gravity/n46-mix/src/maths/KLU/colamd_global.c
// 用法：./multiblock_lu_bench <s0.mtx> <s0.rhs> [reps]
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "ngspice/colamd.h"
#include "cpu_ref_lu.h"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

typedef struct { int n, nnz; int *row, *col; double *val; double *b; } s0mat;

static int read_s0(const char *mtx, const char *rhs, s0mat *M)
{
    FILE *f = fopen(mtx, "r"); char line[512];
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    long cap = 1024;
    M->row = (int*)malloc(cap * sizeof(int));
    M->col = (int*)malloc(cap * sizeof(int));
    M->val = (double*)malloc(cap * sizeof(double));
    M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (M->nnz >= cap) { cap *= 2; M->row = (int*)realloc(M->row, cap*4);
                             M->col = (int*)realloc(M->col, cap*4);
                             M->val = (double*)realloc(M->val, cap*8); }
        M->row[M->nnz] = i - 1; M->col[M->nnz] = j - 1; M->val[M->nnz] = v; M->nnz++;
    }
    fclose(f);
    M->b = (double*)calloc((size_t)M->n, sizeof(double));
    f = fopen(rhs, "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (line[0] != 'R' && line[0] != 'n') continue;
            int i; double v;
            if (sscanf(line, "RHS %d %lf", &i, &v) == 2 && i >= 1 && i <= M->n) M->b[i - 1] = v;
        }
        fclose(f);
    }
    return 0;
}

/* ---------- COLAMD + 列重排（与 mp_lu_bench 同款） ---------- */
static int run_colamd(const s0mat *M, int **q_out)
{
    int n = M->n;
    long Alen = colamd_recommended(M->nnz, n, n);
    if (Alen < 2L * M->nnz + 16L * n + 16)
        Alen = 2L * M->nnz + 16L * n + 16;
    int *Ai_copy = (int*)malloc((size_t)Alen * sizeof(int));
    int *cp = (int*)calloc((size_t)(n + 1), sizeof(int));
    int *Ap_t = (int*)calloc((size_t)n + 1, sizeof(int));
    for (int i = 0; i < M->nnz; i++) Ap_t[M->col[i] + 1]++;   /* 1 基偏移 → cp[0]=0 (colamd 约定) */
    for (int i = 0; i < n; i++) Ap_t[i + 1] += Ap_t[i];
    for (int j = 0; j <= n; j++) cp[j] = Ap_t[j];
    int *cur = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cur, Ap_t, (size_t)n * sizeof(int));
    for (int i = 0; i < M->nnz; i++) { int p = cur[M->col[i]]++; Ai_copy[p] = M->row[i]; }
    free(cur);
    for (int c = 0; c < n; c++) {
        int lo = Ap_t[c], hi = Ap_t[c + 1];
        for (int a = lo + 1; a < hi; a++) {
            int key = Ai_copy[a], bb = a - 1;
            while (bb >= lo && Ai_copy[bb] > key) { Ai_copy[bb + 1] = Ai_copy[bb]; bb--; }
            Ai_copy[bb + 1] = key;
        }
    }
    double knobs[COLAMD_KNOBS] = {0};
    int stats[COLAMD_STATS] = {0};
    int ok = colamd(n, n, (int)Alen, Ai_copy, cp, knobs, stats);
    free(Ap_t); free(Ai_copy);
    if (!ok) { fprintf(stderr, "colamd fail stats=%d %d %d %d\n", stats[0], stats[1], stats[2], stats[3]); free(cp); return 1; }
    *q_out = (int*)malloc((size_t)n * sizeof(int));
    for (int j = 0; j < n; j++) (*q_out)[j] = cp[j];
    free(cp);
    return 0;
}

static void build_csc_q(const s0mat *M, const int *q,
                        int **Ap_out, int **Ai_out, double **Ax_out, int **diagrow_out)
{
    int n = M->n;
    int *invq = NULL;
    if (q) { invq = (int*)malloc((size_t)n * 4); for (int j = 0; j < n; j++) invq[q[j]] = j; }
    int *Ap = (int*)calloc((size_t)n + 1, 4);
    for (int i = 0; i < M->nnz; i++) { int c = q ? invq[M->col[i]] : M->col[i]; Ap[c + 1]++; }
    for (int i = 0; i < n; i++) Ap[i + 1] += Ap[i];
    int *Ai = (int*)malloc((size_t)M->nnz * 4);
    double *Ax = (double*)malloc((size_t)M->nnz * 8);
    int *cur = (int*)malloc((size_t)n * 4);
    memcpy(cur, Ap, (size_t)n * 4);
    for (int i = 0; i < M->nnz; i++) {
        int c = q ? invq[M->col[i]] : M->col[i];
        int p = cur[c]++;
        Ai[p] = M->row[i]; Ax[p] = M->val[i];
    }
    free(cur); free(invq);
    *Ap_out = Ap; *Ai_out = Ai; *Ax_out = Ax;
    if (diagrow_out) { int *dg = (int*)malloc((size_t)n * 4);
        for (int k = 0; k < n; k++) dg[k] = q ? q[k] : k; *diagrow_out = dg; }
}

/* ---------- 主机 ST 扫描（镜像 lu_gpu.cu 单 thread 语义）----------
 * 输出: pinv/rowat、Lp/Li/Lx(原始行)、Up/Ui/Ux、每列容量、lnnz/unnz */
typedef struct { int n; int *pinv, *rowat; int *Lp, *Up;
                 int *Li, *Ui; double *Lx, *Ux; long lnz, unz; } host_lu_t;

static int host_st_lu(int n, const int *Ap, const int *Ai, const double *Ax,
                      const int *diagrow, host_lu_t *H)
{
    long cap = (long)n * (n + 1) / 2 + 64;
    H->n = n;
    H->pinv = (int*)malloc((size_t)n * 4);
    H->rowat = (int*)malloc((size_t)n * 4);
    H->Lp = (int*)malloc((size_t)(n + 1) * 4);
    H->Up = (int*)malloc((size_t)(n + 1) * 4);
    H->Li = (int*)malloc((size_t)cap * 4);
    H->Ui = (int*)malloc((size_t)cap * 4);
    H->Lx = (double*)malloc((size_t)cap * 8);
    H->Ux = (double*)malloc((size_t)cap * 8);
    double *x = (double*)calloc((size_t)n, 8);
    long lnz = 0, unz = 0;
    for (int i = 0; i < n; i++) H->pinv[i] = -1;
    for (int k = 0; k < n; k++) {
        H->Lp[k] = (int)lnz; H->Up[k] = (int)unz;
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];
        for (int j = 0; j < k; j++) {
            int r = H->rowat[j];
            double xr = x[r];
            if (xr != 0.0)
                for (int p = H->Lp[j] + 1; p < H->Lp[j + 1]; p++) x[H->Li[p]] -= H->Lx[p] * xr;
        }
        int ipiv = -1; double a = -1.0;
        for (int i = 0; i < n; i++) {
            if (H->pinv[i] < 0) { double t = fabs(x[i]); if (t > a) { a = t; ipiv = i; } }
            else if (x[i] != 0.0) { H->Ui[unz] = H->pinv[i]; H->Ux[unz] = x[i]; unz++; }
        }
        if (ipiv == -1 || a <= 0.0) return k;
        int ck = diagrow[k];
        if (H->pinv[ck] < 0 && fabs(x[ck]) >= a) ipiv = ck;
        double pivot = x[ipiv];
        H->Ui[unz] = k; H->Ux[unz] = pivot; unz++;
        H->pinv[ipiv] = k; H->rowat[k] = ipiv;
        H->Li[lnz] = ipiv; H->Lx[lnz] = 1.0; lnz++;
        for (int i = 0; i < n; i++) {
            if (H->pinv[i] < 0 && x[i] != 0.0) { H->Li[lnz] = i; H->Lx[lnz] = x[i] / pivot; lnz++; }
            x[i] = 0.0;
        }
    }
    H->Lp[n] = (int)lnz; H->Up[n] = (int)unz;
    H->lnz = lnz; H->unz = unz;
    free(x);
    return -1;
}

/* ---------- deps + level（由主机前代实际乘子记录派生） ----------
 * deps(k) = {j<k : 列 k 前代中 x[rowat[j]] != 0}（精确依赖，非 L 模式近似）
 * 两遍扫描：第 1 遍计数，第 2 遍填列 */
typedef struct { int *dep_ptr, *dep_list, *level; int maxlv; } deps_t;

static void record_deps_levels(int n, const int *Ap, const int *Ai, const double *Ax,
                               const host_lu_t *H, deps_t *D)
{
    double *x = (double*)calloc((size_t)n, 8);
    int *dcnt = (int*)calloc((size_t)n, 4);
    /* pass 1: 计数 */
    for (int k = 0; k < n; k++) {
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];
        for (int j = 0; j < k; j++) {
            double xr = x[H->rowat[j]];
            if (xr != 0.0) dcnt[k]++;
            for (int p = H->Lp[j] + 1; p < H->Lp[j + 1]; p++) x[H->Li[p]] -= H->Lx[p] * xr;
        }
        for (int i = 0; i < n; i++) x[i] = 0.0;
    }
    D->dep_ptr = (int*)calloc((size_t)n + 1, 4);
    for (int k = 0; k < n; k++) D->dep_ptr[k + 1] = D->dep_ptr[k] + dcnt[k];
    D->dep_list = (int*)malloc((size_t)(D->dep_ptr[n] + 1) * 4);
    int *cur = (int*)malloc((size_t)n * 4);
    memcpy(cur, D->dep_ptr, (size_t)n * 4);
    /* pass 2: 填列 */
    for (int k = 0; k < n; k++) {
        for (int p = Ap[k]; p < Ap[k + 1]; p++) x[Ai[p]] = Ax[p];
        for (int j = 0; j < k; j++) {
            double xr = x[H->rowat[j]];
            if (xr != 0.0) D->dep_list[cur[k]++] = j;
            for (int p = H->Lp[j] + 1; p < H->Lp[j + 1]; p++) x[H->Li[p]] -= H->Lx[p] * xr;
        }
        for (int i = 0; i < n; i++) x[i] = 0.0;
    }
    free(cur); free(dcnt); free(x);
    /* level：k 升序（deps 均 < k） */
    D->level = (int*)malloc((size_t)n * 4);
    D->maxlv = 0;
    for (int k = 0; k < n; k++) {
        int lv = 1;
        for (int d = D->dep_ptr[k]; d < D->dep_ptr[k + 1]; d++)
            if (D->level[D->dep_list[d]] + 1 > lv) lv = D->level[D->dep_list[d]] + 1;
        D->level[k] = lv;
        if (lv > D->maxlv) D->maxlv = lv;
    }
}

/* ---------- 设备：每列一个 block 的波次内核 ---------- */
__global__ void lu_col_kernel(
    int n, int w,
    const int *wave_cols, const int *wave_off,
    const int *dep_ptr, const int *dep_list,
    const int *rowat, const int *pinv,          /* 只读（主机提供） */
    const int *Ap, const int *Ai, const double *Ax,
    const int *Lp, int *Li, double *Lx, long Lcap,
    const int *Up, int *Ui, double *Ux, long Ucap,
    int *col_status, double *col_piv, const int *full_fwd)
{
    const int k = wave_cols[wave_off[w - 1] + blockIdx.x];
    extern __shared__ double x[];
    int tid = threadIdx.x;
    if (k < 0 || k >= n) { if (tid == 0) col_status[0] = -500 - k; return; }
    for (int i = tid; i < n; i += blockDim.x) x[i] = 0.0;
    __syncthreads();
    if (Ap[k] < 0 || Ap[k + 1] < Ap[k] || Ap[k + 1] > Ap[n]) {
        if (tid == 0) col_status[0] = -600; return;
    }
    for (int p = Ap[k] + tid; p < Ap[k + 1]; p += blockDim.x) x[Ai[p]] = Ax[p];
    __syncthreads();
    /* deps 前代（升序；屏障数 = |deps|，而非 k）；DSH_FULLFWD=1 时按主机方式遍历全部 j<k（调试） */
    if (__ldg(full_fwd)) {
        for (int j = 0; j < k; j++) {
            double xr = x[rowat[j]];
            if (xr != 0.0)
                for (int p = Lp[j] + 1 + tid; p < Lp[j + 1]; p += blockDim.x)
                    x[Li[p]] -= Lx[p] * xr;
            __syncthreads();
        }
    } else {
        for (int d = dep_ptr[k]; d < dep_ptr[k + 1]; d++) {
            int j = dep_list[d];
            double xr = x[rowat[j]];
            if (xr != 0.0)
                for (int p = Lp[j] + 1 + tid; p < Lp[j + 1]; p += blockDim.x)
                    x[Li[p]] -= Lx[p] * xr;
            __syncthreads();
        }
    }
    double pivot = x[rowat[k]];
    if (tid == 0) col_piv[k] = pivot;
    if (pivot == 0.0) { if (tid == 0) col_status[k] = k; return; }
    __shared__ int ucnt;
    if (tid == 0) ucnt = 0;
    __syncthreads();
    /* U 次对角：已主元行且 x!=0（pinv 只读 → 同波次列间无竞态）
     * 注意排除本列主元行 rowat[k]（host pinv 已含其占位 pinv[rowat[k]]=k，
     * 但主机扫描时该行尚未占位 → 这里必须手动排除，否则与 U 对角重复且计数+1） */
    for (int i = tid; i < n; i += blockDim.x)
        if (i != rowat[k] && pinv[i] < k && x[i] != 0.0) {   /* pinv[i]=占位列; <k=已被更早列占位 */
            int s = atomicAdd(&ucnt, 1);
            if ((long)(Up[k] + s) < Ucap) { Ui[Up[k] + s] = pinv[i]; Ux[Up[k] + s] = x[i]; }
        }
    __syncthreads();
    if (tid == 0) {
        int off = Up[k] + ucnt;
        if ((long)off < Ucap) { Ui[off] = k; Ux[off] = pivot; }
        if ((long)Lp[k] < Lcap) { Li[Lp[k]] = rowat[k]; Lx[Lp[k]] = 1.0; }
    }
    /* L 次对角：未主元行且 x!=0 */
    __shared__ int lcnt;
    if (tid == 0) lcnt = 0;
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x)
        if (pinv[i] > k && x[i] != 0.0) {    /* pinv==k 是本列主元行(单位对角); >k=留给更晚列 → L */
            int s = atomicAdd(&lcnt, 1);
            if ((long)(Lp[k] + 1 + s) < Lcap) { Li[Lp[k] + 1 + s] = i; Lx[Lp[k] + 1 + s] = x[i] / pivot; }
        }
    if (tid == 0) col_status[k] = -1;
}

/* 全部列完成后：L 行号 remap 为主元位置 */
__global__ void remap_kernel(long lnz, int *Li, const int *pinv)
{
    long p = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (p < lnz) Li[p] = pinv[Li[p]];
}

/* 比较器（逐列值对比调试用） */
typedef struct { int row; double val; } ev_t;
static int cmp_ev(const void *a, const void *b) { return ((ev_t*)a)->row - ((ev_t*)b)->row; }

/* ---------- 主机三角求解（验证用；镜像 lu_gpu.cu trisolve） ---------- */
static void host_trisolve(int n, const host_lu_t *H, const double *b, double *x)
{
    double *pb = (double*)malloc((size_t)n * 8);
    for (int k = 0; k < n; k++) pb[k] = b[H->rowat[k]];
    for (int j = 0; j < n; j++) {
        double yj = pb[j];
        for (int p = H->Lp[j] + 1; p < H->Lp[j + 1]; p++) pb[H->Li[p]] -= H->Lx[p] * yj;
    }
    for (int j = n - 1; j >= 0; j--) {
        double acc = pb[j];
        for (int p = H->Up[j]; p < H->Up[j + 1]; p++) {
            int r = H->Ui[p];
            if (r != j) acc -= H->Ux[p] * x[r];
            else pb[j] = acc / H->Ux[p];
        }
        x[j] = pb[j];
    }
    free(pb);
}

static double rel_err(int n, const double *a, const double *b)
{
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
    return sqrt(num / (den + 1e-300));
}

/* ---------- 波次分组 ---------- */
typedef struct { int *wave_cols, *wave_off; int nwaves; } wave_t;

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [reps]\n", argv[0]); return 2; }
    s0mat M;
    if (read_s0(argv[1], argv[2], &M)) return 1;
    int reps = argc > 3 ? atoi(argv[3]) : 5;
    int n = M.n;

    int *q = NULL;
    if (run_colamd(&M, &q)) { fprintf(stderr, "colamd fail\n"); return 1; }
    int *Ap, *Ai, *diagrow; double *Ax;
    build_csc_q(&M, q, &Ap, &Ai, &Ax, &diagrow);

    /* 主机 ST 扫描（符号+数值，供 deps/level/容量/参考） */
    host_lu_t H;
    int rc = host_st_lu(n, Ap, Ai, Ax, diagrow, &H);
    if (rc >= 0) { fprintf(stderr, "host ST singular at col %d\n", rc); return 1; }
    deps_t D;
    record_deps_levels(n, Ap, Ai, Ax, &H, &D);
    printf("== multiblock_lu %s ==\nn=%d nnz=%d | L+U=%ld fill=%.1fx | levels=%d avg_deps=%.1f\n",
           argv[1], n, M.nnz, H.lnz + H.unz, (double)(H.lnz + H.unz) / M.nnz,
           D.maxlv, (double)D.dep_ptr[n] / n);

    /* 波次分组 */
    int *wcnt = (int*)calloc((size_t)D.maxlv + 1, 4);
    for (int k = 0; k < n; k++) wcnt[D.level[k]]++;
    wave_t W;
    W.nwaves = D.maxlv;
    W.wave_off = (int*)malloc((size_t)(D.maxlv + 2) * 4);
    W.wave_off[0] = 0;
    for (int w = 1; w <= D.maxlv; w++) W.wave_off[w] = W.wave_off[w - 1] + wcnt[w];
    W.wave_cols = (int*)malloc((size_t)n * 4);
    int *wc = (int*)malloc((size_t)(D.maxlv + 1) * 4);
    memcpy(wc, W.wave_off, (size_t)(D.maxlv + 1) * 4);
    for (int k = 0; k < n; k++) W.wave_cols[wc[D.level[k] - 1]++] = k;   /* 0 基波次索引: level w → 槽 [wave_off[w-1], wave_off[w]) */
    free(wcnt); free(wc);

    /* 设备分配 */
    int *d_Ap, *d_Ai, *d_diag, *d_Lp, *d_Li, *d_Up, *d_Ui, *d_dep_ptr, *d_dep_list, *d_rowat, *d_pinv, *d_status;
    int *d_wcols, *d_woff, *d_fullfwd; double *d_piv;
    double *d_Ax, *d_Lx, *d_Ux;
    CK(cudaMalloc(&d_Ap, (size_t)(n + 1) * 4)); CK(cudaMalloc(&d_Ai, (size_t)M.nnz * 4));
    CK(cudaMalloc(&d_Ax, (size_t)M.nnz * 8)); CK(cudaMalloc(&d_diag, (size_t)n * 4));
    CK(cudaMalloc(&d_Lp, (size_t)(n + 1) * 4)); CK(cudaMalloc(&d_Li, (size_t)(H.lnz + 1) * 4));
    CK(cudaMalloc(&d_Lx, (size_t)(H.lnz + 1) * 8));
    CK(cudaMalloc(&d_Up, (size_t)(n + 1) * 4)); CK(cudaMalloc(&d_Ui, (size_t)(H.unz + 1) * 4));
    CK(cudaMalloc(&d_Ux, (size_t)(H.unz + 1) * 8));
    CK(cudaMalloc(&d_dep_ptr, (size_t)(n + 1) * 4)); CK(cudaMalloc(&d_dep_list, (size_t)(D.dep_ptr[n] + 1) * 4));
    CK(cudaMalloc(&d_rowat, (size_t)n * 4)); CK(cudaMalloc(&d_pinv, (size_t)n * 4));
    CK(cudaMalloc(&d_status, (size_t)n * 4));
    CK(cudaMalloc(&d_piv, (size_t)n * 8));
    CK(cudaMalloc(&d_wcols, (size_t)n * 4)); CK(cudaMalloc(&d_woff, (size_t)(D.maxlv + 2) * 4));
    CK(cudaMalloc(&d_fullfwd, 4));
    { int ff = getenv("DSH_FULLFWD") ? 1 : 0; CK(cudaMemcpy(d_fullfwd, &ff, 4, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(d_Ap, Ap, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Ai, Ai, (size_t)M.nnz * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Ax, Ax, (size_t)M.nnz * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_diag, diagrow, (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Lp, H.Lp, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Up, H.Up, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dep_ptr, D.dep_ptr, (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dep_list, D.dep_list, (size_t)(D.dep_ptr[n] + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rowat, H.rowat, (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pinv, H.pinv, (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wcols, W.wave_cols, (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_woff, W.wave_off, (size_t)(D.maxlv + 2) * 4, cudaMemcpyHostToDevice));

    long Lcap = H.lnz + 1, Ucap = H.unz + 1;
    int block = 256;
    if (getenv("DSH_BLOCK1")) block = 1;
    size_t smem = (size_t)n * 8;
    if (smem > 48 * 1024) { fprintf(stderr, "n=%d 需 smem %.1fKB>48KB，v1 目标 n<=5000\n", n, smem / 1024.0); return 1; }

    /* 波次循环：每波 grid=列数，block b 处理列 k = wave_cols[wave_off[w-1]+b] */
    auto run_factor = [&]() {
        CK(cudaMemset(d_Li, 0, (size_t)(H.lnz + 1) * 4));
        CK(cudaMemset(d_Lx, 0, (size_t)(H.lnz + 1) * 8));
        CK(cudaMemset(d_Ui, 0, (size_t)(H.unz + 1) * 4));
        CK(cudaMemset(d_Ux, 0, (size_t)(H.unz + 1) * 8));
        CK(cudaMemset(d_status, 0, (size_t)n * 4));
        for (int w = 1; w <= D.maxlv; w++) {
            int nb = W.wave_off[w] - W.wave_off[w - 1];
            if (nb == 0) continue;
            lu_col_kernel<<<nb, block, smem>>>(
                n, w, d_wcols, d_woff, d_dep_ptr, d_dep_list, d_rowat, d_pinv,
                d_Ap, d_Ai, d_Ax, d_Lp, d_Li, d_Lx, Lcap,
                d_Up, d_Ui, d_Ux, Ucap, d_status, d_piv, d_fullfwd);
            CK(cudaGetLastError());
        }
        if (!getenv("DSH_NOREMAP"))
            remap_kernel<<<(H.lnz + block - 1) / block, block>>>((long)H.lnz, d_Li, d_pinv);
        CK(cudaGetLastError());
    };
    run_factor();   /* warmup */
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(e0));
        run_factor();
        CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    printf("多block 波次LU(设备, 含remap): %.3f ms (reps=%d 取最优)\n", best, reps);

    /* 状态检查 */
    int *h_status = (int*)malloc((size_t)n * 4);
    CK(cudaMemcpy(h_status, d_status, (size_t)n * 4, cudaMemcpyDeviceToHost));
    int bad = 0; for (int k = 0; k < n; k++) if (h_status[k] != -1) bad++;
    printf("列状态: %d/%d 正常 %s\n", n - bad, n, bad ? "❌" : "✅");

    /* 正确性：设备 L/U（拷回）→ 散射式三角求解 → 与 cpu_lu_ref 对比 */
    int *h_Li = (int*)malloc((size_t)(H.lnz + 1) * 4);
    double *h_Lx = (double*)malloc((size_t)(H.lnz + 1) * 8);
    int *h_Ui = (int*)malloc((size_t)(H.unz + 1) * 4);
    double *h_Ux = (double*)malloc((size_t)(H.unz + 1) * 8);
    CK(cudaMemcpy(h_Li, d_Li, (size_t)(H.lnz + 1) * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_Lx, d_Lx, (size_t)(H.lnz + 1) * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_Ui, d_Ui, (size_t)(H.unz + 1) * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_Ux, d_Ux, (size_t)(H.unz + 1) * 8, cudaMemcpyDeviceToHost));

    auto trisolve = [&](const int *Li_, const double *Lx_, const int *Ui_, const double *Ux_,
                        const double *b, double *x) {
        double *pb = (double*)malloc((size_t)n * 8);
        double *y = (double*)malloc((size_t)n * 8);
        for (int k = 0; k < n; k++) pb[k] = b[H.rowat[k]];
        for (int k = 0; k < n; k++) {
            y[k] = pb[k] / Lx_[H.Lp[k]];
            for (int p = H.Lp[k] + 1; p < H.Lp[k + 1]; p++) pb[Li_[p]] -= Lx_[p] * y[k];
        }
        for (int k = n - 1; k >= 0; k--) {
            y[k] /= Ux_[H.Up[k + 1] - 1];
            for (int p = H.Up[k]; p < H.Up[k + 1] - 1; p++) y[Ui_[p]] -= Ux_[p] * y[k];
        }
        for (int k = 0; k < n; k++) x[k] = y[k];
        free(pb); free(y);
    };
    double *x_dev = (double*)malloc((size_t)n * 8);
    double *x_cpu = (double*)malloc((size_t)n * 8);
    trisolve(h_Li, h_Lx, h_Ui, h_Ux, M.b, x_dev);
    {
        int *Ap0, *Ai0; double *Ax0;
        build_csc_q(&M, NULL, &Ap0, &Ai0, &Ax0, NULL);
        long ln2 = 0, un2 = 0;
        int rc = cpu_lu_ref(n, Ap0, Ai0, Ax0, q, M.b, 1.0, x_cpu, &ln2, &un2);
        free(Ap0); free(Ai0); free(Ax0);
        if (rc >= 0) { fprintf(stderr, "cpu_lu_ref singular at %d\n", rc); return 1; }
    }
    double err = rel_err(n, x_dev, x_cpu);
    double res = 0, nrmb = 0;
    {
        double *r = (double*)calloc((size_t)n, 8);
        for (int i = 0; i < M.nnz; i++) r[M.row[i]] += M.val[i] * x_dev[M.col[i]];
        for (int i = 0; i < n; i++) { double d = r[i] - M.b[i]; res += d * d; nrmb += M.b[i] * M.b[i]; }
        free(r);
    }
    printf("正确性: 设备L/U解 vs cpu_lu_ref rel_err=%.2e  %s\n", err, err < 1e-6 ? "✅" : "❌");
    printf("残差: ||A·x_dev − b||/||b|| = %.2e\n", sqrt(res / (nrmb + 1e-300)));

    /* 同路径参考：主机 ST 因子（remap 后）求解 vs 设备求解 —— 两者应 ~1e-11（排除条件数下限） */
    {
        int *hL2 = (int*)malloc((size_t)(H.lnz + 1) * 4);
        for (long p = 0; p < H.lnz; p++) hL2[p] = H.pinv[H.Li[p]];
        double *x_hst = (double*)malloc((size_t)n * 8);
        trisolve(hL2, H.Lx, H.Ui, H.Ux, M.b, x_hst);
        double e2 = rel_err(n, x_dev, x_hst);
        printf("同路径校验: 设备 vs 主机ST(同序) rel_err=%.2e  %s\n", e2, e2 < 1e-6 ? "✅" : "❌");
        free(hL2); free(x_hst);
    }

    /* L 模式一致性（顺序无关） */
    {
        int *invp = (int*)malloc((size_t)n * 4);
        for (int i = 0; i < n; i++) invp[H.pinv[i]] = i;
        int mismatch = 0;
        for (int j = 0; j < n; j++) {
            int lo = H.Lp[j], hi = H.Lp[j + 1], len = hi - lo;
            ev_t *ea = (ev_t*)malloc((size_t)(len + 1) * sizeof(ev_t));
            ev_t *eb = (ev_t*)malloc((size_t)(len + 1) * sizeof(ev_t));
            for (int t = 0; t < len; t++) { ea[t].row = invp[h_Li[lo + t]]; eb[t].row = H.Li[lo + t];
                                            ea[t].val = h_Lx[lo + t]; eb[t].val = H.Lx[lo + t]; }
            qsort(ea, len, sizeof(ev_t), cmp_ev);
            qsort(eb, len, sizeof(ev_t), cmp_ev);
            for (int t = 0; t < len; t++) if (ea[t].row != eb[t].row) mismatch++;
            free(ea); free(eb);
        }
        free(invp);
        printf("L 模式一致性: 行集差异 %d 项 %s\n", mismatch, mismatch == 0 ? "✅" : "❌");
    }
    free(h_status); free(h_Li); free(h_Lx); free(h_Ui); free(h_Ux);
    free(x_dev); free(x_cpu);
    printf("\n基准对照（单 block PAR，来自 mp_lu_bench 同机同构实测）:\n");
    printf("  多block波次 %.2f ms vs 单block PAR（见验收报告实测值）\n", best);
    return 0;
}
