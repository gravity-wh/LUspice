// ============================================================================
// test_lu_wave.cu — M3 level-scheduling factor 验证驱动
// 流程：s0 读取 → COLAMD → q 序 CSC → symbolic_u（符号 U/L 结构）→ level
//   → x_all 分段 → wave launch（每列 block）→ 单 thread trisolve → 残差
// 编译：nvcc -O3 -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src \
//       -I /home/gravity/n46-mix/src/include -o test_lu_wave test_lu_wave.cu \
//       lu_gpu.cu lu_gpu_wave.cu symbolic_u.c level.c cpu_ref_lu.c \
//       /home/gravity/n46-mix/src/maths/KLU/colamd.c \
//       /home/gravity/n46-mix/src/maths/KLU/colamd_global.c
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "cuda_compat.h"
#include "ngspice/colamd.h"
#include "lu_gpu.cu"        /* 单 thread trisolve_kernel（验证用） */
#include "lu_gpu_wave.cu"   /* factor_wave_kernel */
#include "level.h"

extern "C" int symbolic_dense(int n, const int *Ap, const int *Ai, const double *Ax,
                             const int *diagrow,
                             int **Lp, int **Li, double **Lx, long *lnnz,
                             int **Up, int **Ui, double **Ux, long *unnz,
                             int **pinv_out, int **rowat_out);

typedef struct { int n, nnz; int *row, *col; double *val; double *b; } s0mat;

static int read_s0(const char *mtx, const char *rhs, s0mat *M) {
    FILE *f = fopen(mtx, "r"); char line[512]; long cap = 1024;
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    M->row = (int*)malloc(cap*sizeof(int)); M->col = (int*)malloc(cap*sizeof(int));
    M->val = (double*)malloc(cap*sizeof(double)); M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        if (M->nnz >= cap) { cap *= 2; M->row = (int*)realloc(M->row, cap*sizeof(int));
            M->col = (int*)realloc(M->col, cap*sizeof(int)); M->val = (double*)realloc(M->val, cap*sizeof(double)); }
        sscanf(line, "A %d %d %lf", &M->row[M->nnz], &M->col[M->nnz], &M->val[M->nnz]);
        M->nnz++;
    }
    fclose(f);
    M->b = (double*)calloc((size_t)M->n+1, sizeof(double));
    if (rhs) { f = fopen(rhs, "r");
        if (f) { while (fgets(line, sizeof line, f)) { int i; double v;
            if (sscanf(line, "RHS %d %lf", &i, &v) == 2 && i >= 1 && i <= M->n) M->b[i] = v; }
            fclose(f); } }
    for (int i = 1; i <= M->n; i++) if (M->b[i] == 0) M->b[i] = 1.0;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <s0.mtx> <s0.rhs> [pin] [flt]\n", argv[0]); return 2; }
    s0mat M; if (read_s0(argv[1], argv[2], &M)) return 1;
    double pinv = (argc >= 4) ? atof(argv[3]) : 0.0;
    int n = M.n;
    printf("n=%d nnz=%d pin=%.1e\n", n, M.nnz, pinv);

    /* .flt pin */
    if (argc >= 5 && pinv > 0) {
        FILE *ff = fopen(argv[4], "r");
        if (ff) { char rb[256];
            while (fgets(rb, sizeof rb, ff)) { int row;
                if (sscanf(rb, "F %d", &row) == 1) {
                    int i = row-1, found = 0;
                    for (int k = 0; k < M.nnz; k++)
                        if (M.row[k]-1 == i && M.col[k]-1 == i) { M.val[k] += pinv; found = 1; break; }
                    if (!found) {
                        M.row = (int*)realloc(M.row, (M.nnz+1)*sizeof(int));
                        M.col = (int*)realloc(M.col, (M.nnz+1)*sizeof(int));
                        M.val = (double*)realloc(M.val, (M.nnz+1)*sizeof(double));
                        M.row[M.nnz]=i+1; M.col[M.nnz]=i+1; M.val[M.nnz]=pinv; M.nnz++;
                    }
                }
            }
            fclose(ff);
        }
    }

    /* COLAMD（参考 test_lu.c） */
    int *q = (int*)malloc((size_t)n*sizeof(int));
    {
        int *cp = (int*)calloc((size_t)n+1, sizeof(int));
        long Alen = colamd_recommended(M.nnz, n, n);
        if (Alen < 2L*M.nnz + 16L*n + 16) Alen = 2L*M.nnz + 16L*n + 16;
        int *Ai_copy = (int*)malloc((size_t)Alen*sizeof(int));
        int *Ap_t = (int*)calloc((size_t)n+1, sizeof(int));
        for (int i = 0; i < M.nnz; i++) Ap_t[M.col[i]]++;
        for (int i = 0; i < n; i++) { Ap_t[i+1] += Ap_t[i]; cp[i] = Ap_t[i]; }
        cp[n] = M.nnz;
        int *cur = (int*)malloc((size_t)n*sizeof(int));
        memcpy(cur, Ap_t, (size_t)n*sizeof(int));
        for (int i = 0; i < M.nnz; i++) { int p = cur[M.col[i]-1]++; Ai_copy[p] = M.row[i]-1; }
        free(cur);
        for (int c_ = 0; c_ < n; c_++) { int lo = Ap_t[c_], hi = Ap_t[c_+1];
            for (int a = lo+1; a < hi; a++) { int key = Ai_copy[a], b = a-1;
                while (b >= lo && Ai_copy[b] > key) { Ai_copy[b+1] = Ai_copy[b]; b--; } Ai_copy[b+1] = key; } }
        double knobs[COLAMD_KNOBS] = {0}; int stats[COLAMD_STATS] = {0};
        int ok = colamd(n, n, (int)Alen, Ai_copy, cp, knobs, stats);
        if (!ok) { fprintf(stderr, "colamd failed %d\n", stats[0]); return 1; }
        for (int j = 0; j < n; j++) q[j] = cp[j];
        free(cp); free(Ai_copy); free(Ap_t);
    }

    /* q 序 CSC（COLAMD 重排） */
    int *invq = (int*)malloc((size_t)n*sizeof(int));
    for (int j = 0; j < n; j++) invq[q[j]] = j;
    int *Ap = (int*)calloc((size_t)n+1, sizeof(int));
    for (int i = 0; i < M.nnz; i++) Ap[invq[M.col[i]-1]+1]++;
    for (int i = 0; i < n; i++) Ap[i+1] += Ap[i];
    int *Ai = (int*)malloc((size_t)M.nnz*sizeof(int));
    double *Ax = (double*)malloc((size_t)M.nnz*sizeof(double));
    int *cur = (int*)malloc((size_t)n*sizeof(int));
    memcpy(cur, Ap, (size_t)n*sizeof(int));
    for (int i = 0; i < M.nnz; i++) { int c = invq[M.col[i]-1]; int p = cur[c]++;
        Ai[p] = M.row[i]-1; Ax[p] = M.val[i]; }
    free(cur);
    /* 每列行号升序（主元 tie-break 与 CPU 一致） */
    for (int c_ = 0; c_ < n; c_++) { int lo = Ap[c_], hi = Ap[c_+1];
        for (int a = lo+1; a < hi; a++) { int key = Ai[a], b = a-1;
            while (b >= lo && Ai[b] > key) { Ai[b+1] = Ai[b]; b--; } Ai[b+1] = key; } }
    free(invq);

    int *h_diag = (int*)malloc((size_t)n*sizeof(int));
    for (int k = 0; k < n; k++) h_diag[k] = q[k];
    /* 符号分解（CPU，稠密镜像 wave kernel 主元）→ U/L 结构 + level */
    int *symUp, *symUi, *symLp, *symLi; double *symLx, *symUx; long symUnnz, symLnnz;
    int *sym_pinv, *sym_rowat;
    int src = symbolic_dense(n, Ap, Ai, Ax, h_diag, &symLp, &symLi, &symLx, &symLnnz,
                             &symUp, &symUi, &symUx, &symUnnz, &sym_pinv, &sym_rowat);
    if (src >= 0) { fprintf(stderr, "symbolic singular col %d\n", src); return 1; }
    int *lvl = (int*)malloc((size_t)n*sizeof(int));
    int nlevels;
    dsh_level_compute(n, symUp, symUi, lvl, &nlevels);
    printf("symbolic lnnz=%ld unnz=%ld levels=%d\n", symLnnz, symUnnz, nlevels);
    printf("lvl[0..5]=%d %d %d %d %d %d\n", lvl[0], lvl[1], lvl[2], lvl[3], lvl[4], lvl[5]);
    printf("symLp[0..3]=%d %d %d %d symUp[0..3]=%d %d %d %d\n",
           symLp[0],symLp[1],symLp[2],symLp[3], symUp[0],symUp[1],symUp[2],symUp[3]);
    {
        int cnt[8]={0};
        for (int k=0;k<n;k++){ int len=symUp[k+1]-symUp[k]; cnt[len<=1?0:len<=2?1:len<=4?2:len<=8?3:len<=16?4:len<=32?5:len<=64?6:7]++; }
        printf("SYM U 分布 [1,2,4,8,16,32,64,>64]: %d %d %d %d %d %d %d %d\n", cnt[0],cnt[1],cnt[2],cnt[3],cnt[4],cnt[5],cnt[6],cnt[7]);
    }

    /* GPU 分配（静态：Lp/Up 用符号累积） */
    long cap = symLnnz + symUnnz + 16;
    int *d_Ap, *d_Ai, *d_Lp, *d_Li, *d_Up, *d_Ui, *d_pinv, *d_rowat, *d_diag;
    double *d_Ax, *d_Lx, *d_Ux;
    cudaMalloc((void**)&d_Ap, (size_t)(n+1)*sizeof(int));
    cudaMalloc((void**)&d_Ai, (size_t)M.nnz*sizeof(int));
    cudaMalloc((void**)&d_Ax, (size_t)M.nnz*sizeof(double));
    cudaMalloc((void**)&d_Lp, (size_t)(n+1)*sizeof(int));
    cudaMalloc((void**)&d_Li, (size_t)cap*sizeof(int));
    cudaMalloc((void**)&d_Lx, (size_t)cap*sizeof(double));
    cudaMalloc((void**)&d_Up, (size_t)(n+1)*sizeof(int));
    cudaMalloc((void**)&d_Ui, (size_t)cap*sizeof(int));
    cudaMalloc((void**)&d_Ux, (size_t)cap*sizeof(double));
    cudaMalloc((void**)&d_pinv, (size_t)n*sizeof(int));
    cudaMalloc((void**)&d_rowat, (size_t)n*sizeof(int));
    cudaMalloc((void**)&d_diag, (size_t)n*sizeof(int));
    cudaMemcpy(d_Ap, Ap, (size_t)(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai, Ai, (size_t)M.nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax, Ax, (size_t)M.nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lp, symLp, (size_t)(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Up, symUp, (size_t)(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_diag, h_diag, (size_t)n*sizeof(int), cudaMemcpyHostToDevice);
    /* symUp/symUi 上传（依赖图） */
    int *d_symUp, *d_symUi;
    cudaMalloc((void**)&d_symUp, (size_t)(n+1)*sizeof(int));
    cudaMalloc((void**)&d_symUi, (size_t)cap*sizeof(int));
    cudaMemcpy(d_symUp, symUp, (size_t)(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_symUi, symUi, (size_t)cap*sizeof(int), cudaMemcpyHostToDevice);
    int *d_sym_pinv, *d_sym_rowat;
    cudaMalloc((void**)&d_sym_pinv, (size_t)n*sizeof(int));
    cudaMalloc((void**)&d_sym_rowat, (size_t)n*sizeof(int));
    cudaMemcpy(d_sym_pinv, sym_pinv, (size_t)n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sym_rowat, sym_rowat, (size_t)n*sizeof(int), cudaMemcpyHostToDevice);

    /* wave launch：每 level 一个 wave，wave 内每列 block + 私有 x 段 */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int *d_pinv_i = d_pinv, *d_rowat_i = d_rowat;
    /* pinv/rowat：pinv 每列段（wave 内私有，段初始 = 全局快照），全局 pinv 初始 -1 */
    cudaMemset(d_pinv_i, -1, (size_t)n*sizeof(int));
    cudaMemset(d_rowat_i, -1, (size_t)n*sizeof(int));
    /* R9: 预分配 wave 缓冲（max cnt × n）——原每 wave cudaMalloc/Free 705 次开销大 */
    int maxcnt = 0;
    for (int L = 0; L < nlevels; L++) {
        int c = 0;
        for (int k = 0; k < n; k++) if (lvl[k] == L) c++;
        if (c > maxcnt) maxcnt = c;
    }
    int *d_cols; double *d_xall;
    cudaMalloc((void**)&d_cols, (size_t)maxcnt*sizeof(int));
    cudaMalloc((void**)&d_xall, (size_t)((long)maxcnt*n)*sizeof(double));
    for (int L = 0; L < nlevels; L++) {
        int cnt = 0;
        for (int k = 0; k < n; k++) if (lvl[k] == L) cnt++;
        if (cnt == 0) continue;
        int *cols = (int*)malloc((size_t)cnt*sizeof(int));
        int ci = 0;
        for (int k = 0; k < n; k++) if (lvl[k] == L) cols[ci++] = k;
        if (L == 0) { printf("wave0 cnt=%d cols[0..4]=%d %d %d %d %d\n", cnt, cols[0], cols[1], cols[2], cols[3], cols[4]); }
        cudaMemcpy(d_cols, cols, (size_t)cnt*sizeof(int), cudaMemcpyHostToDevice);
        factor_wave_kernel<<<cnt, WAVE_BLOCK>>>(d_cols, n, d_Ap, d_Ai, d_Ax,
            d_symUp, d_symUi, d_sym_pinv, d_sym_rowat,
            d_Lp, d_Li, d_Lx, cap, d_Up, d_Ui, d_Ux, cap, d_xall);
        cudaError_t err = cudaDeviceSynchronize();
        cudaError_t err2 = cudaGetLastError();
        if (err != cudaSuccess || err2 != cudaSuccess)
            printf("WAVE kernel error L=%d: %s / %s\n", L, cudaGetErrorString(err), cudaGetErrorString(err2));
        free(cols);
    }
    cudaFree(d_cols); cudaFree(d_xall);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float wms; cudaEventElapsedTime(&wms, t0, t1);
    printf("WAVE factor: %.3f ms\n", wms);
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, factor_wave_kernel, WAVE_BLOCK, 0);
        printf("Occupancy factor_wave_kernel: %d blocks/SM (256 thr) => %.1f%% (SM threads)\n",
               nb, 100.0 * nb * WAVE_BLOCK / prop.maxThreadsPerMultiProcessor);
    }

    /* 下载 L/U/pinv 用于 trisolve（单 thread kernel 复用 test_lu_gpu 方式——此处用 host 简单验证残差） */
    /* 简化验证：host 前代+回代（用下载的 L/U） */
    int *h_Lp = (int*)malloc((size_t)(n+1)*sizeof(int));
    int *h_Li = (int*)malloc((size_t)cap*sizeof(int));
    double *h_Lx = (double*)malloc((size_t)cap*sizeof(double));
    int *h_Up = (int*)malloc((size_t)(n+1)*sizeof(int));
    int *h_Ui = (int*)malloc((size_t)cap*sizeof(int));
    double *h_Ux = (double*)malloc((size_t)cap*sizeof(double));
    int *h_pinv = (int*)malloc((size_t)n*sizeof(int));
    cudaMemcpy(h_Lp, d_Lp, (size_t)(n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Li, d_Li, (size_t)cap*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Lx, d_Lx, (size_t)cap*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Up, d_Up, (size_t)(n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ui, d_Ui, (size_t)cap*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ux, d_Ux, (size_t)cap*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pinv, d_pinv, (size_t)n*sizeof(int), cudaMemcpyDeviceToHost);
    long wlnnz = h_Lp[n], wunnz = h_Up[n];
    printf("WAVE L nnz=%ld (sym %ld), U nnz=%ld (sym %ld)\n", wlnnz, symLnnz, wunnz, symUnnz);
    /* 找 Lx/Ux 的第一个 NaN/异常 */
    {
        int badL = -1, badU = -1;
        for (long p = 0; p < wlnnz; p++) if (!(h_Lx[p] == h_Lx[p])) { badL = (int)p; break; }
        for (long p = 0; p < wunnz; p++) if (!(h_Ux[p] == h_Ux[p])) { badU = (int)p; break; }
        printf("NaN: L[%d] U[%d]\n", badL, badU);
        printf("U diag 对比 (wave vs sym) 前5列: ");
        for (int k = 0; k < 5 && k < n; k++)
            printf("[%d: %g vs %g] ", k, h_Ux[h_Up[k+1]-1], symUx[symUp[k+1]-1]);
        printf("\n");
        printf("Li[0..4]=%d %d %d %d %d Ui[0..4]=%d %d %d %d %d\n",
               h_Li[0],h_Li[1],h_Li[2],h_Li[3],h_Li[4], h_Ui[0],h_Ui[1],h_Ui[2],h_Ui[3],h_Ui[4]);
        printf("Lx 对比 (wave vs sym) 前5: ");
        for (int k = 0; k < 5; k++) printf("%g/%g ", h_Lx[k], symLx[k]);
        printf("\n");
        if (badL >= 0) printf("  L 样例: Lx[0..5]=%g %g %g %g %g %g, 对应 sym: %g %g %g %g %g %g\n",
            h_Lx[0],h_Lx[1],h_Lx[2],h_Lx[3],h_Lx[4],h_Lx[5],
            symLx[0],symLx[1],symLx[2],symLx[3],symLx[4],symLx[5]);
    }
    /* 逐列对比 U 列长（wave vs symbolic）找第一差异 */
    {
        int first = -1;
        for (int k = 0; k < n; k++)
            if (h_Up[k+1]-h_Up[k] != symUp[k+1]-symUp[k]) { first = k; break; }
        printf("U 第一差异列: %d (wave %d vs sym %d)\n", first,
               first>=0 ? h_Up[first+1]-h_Up[first] : -1,
               first>=0 ? symUp[first+1]-symUp[first] : -1);
        if (first >= 0) {
            printf("  wave U col %d:", first);
            for (int p = h_Up[first]; p < h_Up[first+1]; p++) printf(" (%d,%.4g)", h_Ui[p], h_Ux[p]);
            printf("\n  sym  U col %d:", first);
            for (int p = symUp[first]; p < symUp[first+1]; p++) printf(" (%d,%.4g)", symUi[p], symUx[p]);
            printf("\n");
        }
        /* R9: 第一数值差异（Lx/Ux 逐项，结构相同前提） */
        int fv = -1; double fd = 0;
        for (long p = 0; p < wlnnz && p < symLnnz; p++) {
            double d = fabs(h_Lx[p] - symLx[p]);
            if (d > fd && !(h_Lx[p]==h_Lx[p] && symLx[p]==symLx[p])) { }
            if (d > 0.0 && h_Lx[p] == h_Lx[p] && symLx[p] == symLx[p]) { fv = (int)p; fd = d; break; }
        }
        printf("Lx 第一数值差异: %s (%d: wave %.17g vs sym %.17g)\n",
               fv < 0 ? "无" : "有", fv, fv>=0?h_Lx[fv]:0, fv>=0?symLx[fv]:0);
        fv = -1;
        for (long p = 0; p < wunnz && p < symUnnz; p++) {
            double d = fabs(h_Ux[p] - symUx[p]);
            if (d > 0.0 && h_Lx[p] == h_Lx[p] && symLx[p] == symLx[p]) { fv = (int)p; fd = d; break; }
        }
        printf("Ux 第一数值差异: %s (%d: wave %.17g vs sym %.17g)\n",
               fv < 0 ? "无" : "有", fv, fv>=0?h_Ux[fv]:0, fv>=0?symUx[fv]:0);
        /* L 逐列结构对比（U 已比，L 未比——缺失贡献可能来自 L 列分布差异） */
        int lfirst = -1;
        for (int k = 0; k < n; k++)
            if (h_Lp[k+1]-h_Lp[k] != symLp[k+1]-symLp[k]) { lfirst = k; break; }
        printf("L 逐列结构第一差异: %d (wave %d vs sym %d)\n", lfirst,
               lfirst>=0 ? h_Lp[lfirst+1]-h_Lp[lfirst] : -1,
               lfirst>=0 ? symLp[lfirst+1]-symLp[lfirst] : -1);
        /* Lx[224] 所在列定位 */
        for (int k = 0; k < n; k++) {
            if (h_Lp[k] <= 224 && 224 < h_Lp[k+1]) {
                printf("Lx[224] 在列 %d (wave 段 [%d,%d)) sym 列长 %d；该列 deps: ",
                       k, h_Lp[k], h_Lp[k+1], symLp[k+1]-symLp[k]);
                for (int p = symUp[k]; p < symUp[k+1]-1 && p < symUp[k+1]; p++) printf("%d ", symUi[p]);
                printf("\n");
                break;
            }
        }
        /* L 列 7 逐项对比（第一 1-ulp 差异所在列） */
        if (n > 7) {
            int k21 = -1;
            for (int k = 0; k < n; k++) if (h_Lp[k] <= 21 && 21 < h_Lp[k+1]) { k21 = k; break; }
            printf("Lx[21] 在列 %d（wave [%d,%d) sym [%d,%d)）\n", k21,
                   h_Lp[k21], h_Lp[k21+1], symLp[k21], symLp[k21+1]);
            printf("L col %d wave: ", k21);
            for (int p = h_Lp[k21]; p < h_Lp[k21+1]; p++) printf("(%d,%.17g) ", h_Li[p], h_Lx[p]);
            printf("\nL col %d sym : ", k21);
            for (int p = symLp[k21]; p < symLp[k21+1]; p++) printf("(%d,%.17g) ", symLi[p], symLx[p]);
            printf("\n");
        }
    }
    /* host 前代+回代（LU 解，q 序）——R9 修正：wave kernel 不输出 pinv（固定符号主元），
       L 行重映射必须用 sym_pinv（原驱动用从未写入的 d_pinv → 全 -1 → 残差 1.0）
       DSH_SYM_SOLVE=1：用 sym 因子走同一 host solve（harness 自检） */
    double *y = (double*)calloc((size_t)n, sizeof(double));
    double *xh = (double*)calloc((size_t)n, sizeof(double));
    double *pb = (double*)calloc((size_t)n, sizeof(double));
    int *rowat = (int*)malloc((size_t)n*sizeof(int));
    int *sol_Li; double *sol_Lx, *sol_Ux; int *sol_Ui; long sol_lnnz;
    double *sol_Lp_d, *sol_Up_d;
    int use_sym = getenv("DSH_SYM_SOLVE") != NULL;
    if (use_sym) { printf("[sym-solve] using symbolic factors\n"); }
    /* host 统一重映射 L 行（用符号主元 sym_pinv；sym_pinv[i]=i 行主元列） */
    for (long p = 0; p < wlnnz; p++) if (h_Li[p] >= 0 && h_Li[p] < n) h_Li[p] = sym_pinv[h_Li[p]];
    for (int i = 0; i < n; i++) rowat[i] = -1;
    for (int i = 0; i < n; i++) if (sym_pinv[i] >= 0) rowat[sym_pinv[i]] = i;
    if (use_sym) {
        for (long p = 0; p < symLnnz; p++) if (symLi[p] >= 0 && symLi[p] < n) symLi[p] = sym_pinv[symLi[p]];
    }
    for (int k = 0; k < n; k++) pb[k] = rowat[k] >= 0 ? M.b[rowat[k]+1] : 0.0;
    for (int k = 0; k < n; k++) {
        int Lp0 = use_sym ? symLp[k] : h_Lp[k];
        int Up0 = use_sym ? symUp[k] : h_Up[k];
        int Lp1 = use_sym ? symLp[k+1] : h_Lp[k+1];
        int Up1 = use_sym ? symUp[k+1] : h_Up[k+1];
        y[k] = pb[k] / (use_sym ? symLx[Lp0] : h_Lx[Lp0]);
        for (int p = Lp0+1; p < Lp1; p++) pb[(use_sym ? symLi : h_Li)[p]] -= (use_sym ? symLx : h_Lx)[p]*y[k];
        (void)Up0;
    }
    for (int k = n-1; k >= 0; k--) {
        int Up0 = use_sym ? symUp[k] : h_Up[k];
        int Up1 = use_sym ? symUp[k+1] : h_Up[k+1];
        y[k] /= (use_sym ? symUx : h_Ux)[Up1-1];
        for (int p = Up0; p < Up1-1; p++) y[(use_sym ? symUi : h_Ui)[p]] -= (use_sym ? symUx : h_Ux)[p]*y[k];
    }
    for (int k = 0; k < n; k++) xh[q[k]] = y[k];
    /* 残差 */
    {
        double num = 0, den = 0;
        double *bh = (double*)calloc((size_t)n, sizeof(double));
        for (int i = 0; i < M.nnz; i++) { int r = M.row[i]-1, c = M.col[i]-1; bh[r] += M.val[i]*xh[c]; }
        for (int i = 0; i < n; i++) { double d = bh[i]-M.b[i+1]; num += d*d; den += M.b[i+1]*M.b[i+1]; }
        free(bh);
        printf("WAVE RESIDUAL = %.3e  %s\n", sqrt(num/den),
               sqrt(num/den) <= 1e-6 ? "PASS (S1)" : "FAIL");
    }
    return 0;
}
