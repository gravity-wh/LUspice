// ============================================================================
// fusion_bench.cu — IR 迭代算子融合基准（策略②）
//
// 对比"IR 单迭代"的两种实现（fp64 残差 + fp32 因子，与 GMRES-IR 语义一致）：
//   未融合：3 次内核启动（残差 SpMV / 前代 L / 回代 U）
//   融合  ：1 次启动（同一内核内分阶段 + __syncthreads）
// 三角求解用"层级波前"（依赖层级由宿主从 L/U 结构算出），两版本算法一致，
// 因此时间差 ≈ 启动开销差（融合省 2 次启动）。
//
// 输入（python 预生成，见 /tmp/fusion_inputs/）：
//   <base>_A.mtx  (fp64, s0)  <base>_L32.mtx / _U32.mtx (fp32, s0)
//   <base>_lv_f.txt / _lv_b.txt (每行一个整数层级)  <base>_b.txt (RHS)
//
// 编译（race7 根）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -o fusion_bench \
//     src/solver/mp_bench/fusion_bench.cu
// 用法：./fusion_bench <base_name> [iters]
// 输出：未融合 vs 融合 每迭代耗时 + 正确性（两版本输出相对差）
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

typedef struct { int n, nnz; int *rowptr, *col; float *val; double *dval; } s0f;

static int read_s0f(const char *path, s0f *M, int fp32)
{
    FILE *f = fopen(path, "r"); char line[512];
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    long cap = 1024;
    int *cnt = (int*)calloc(M->n + 1, sizeof(int));
    M->col = (int*)malloc(cap * sizeof(int));
    M->val = (float*)malloc(cap * sizeof(float));
    M->dval = (double*)malloc(cap * sizeof(double));
    M->nnz = 0;
    rewind(f); fgets(line, sizeof line, f);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (M->nnz >= cap) { cap *= 2; M->col = (int*)realloc(M->col, cap*sizeof(int));
                             M->val = (float*)realloc(M->val, cap*sizeof(float));
                             M->dval = (double*)realloc(M->dval, cap*sizeof(double)); }
        M->col[M->nnz] = j - 1; M->val[M->nnz] = (float)v; M->dval[M->nnz] = v; M->nnz++;
        cnt[i - 1]++;
    }
    fclose(f);
    M->rowptr = (int*)malloc((size_t)(M->n + 1) * sizeof(int));
    M->rowptr[0] = 0;
    for (int i = 0; i < M->n; i++) M->rowptr[i + 1] = M->rowptr[i] + cnt[i];
    free(cnt);
    return 0;
}

/* x += r（IR 的校正步：x_new = x_old + dx） */
__global__ void k_axpy(const int n, double *x, const double *r)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] += r[i];
}

/* ---------- 层级波前三角求解（单 block，两版本共用） ---------- */

// 残差 SpMV（独立内核，未融合版 stage1）
__global__ void k_spmv(const int n, const int *rp, const int *col, const double *val,
                       const double *x, const double *b, double *r)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = b[i];
        for (int j = rp[i]; j < rp[i + 1]; j++) acc -= val[j] * x[col[j]];
        r[i] = acc;
    }
}

// 前代 L z = r（原位; L 单位下三角; 层级 lv_f 升序波前）
__global__ void k_fwd(const int n, const int maxlv, const int *lv, const int *rp,
                      const int *col, const float *val, double *z)
{
    for (int w = 1; w <= maxlv; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv[i] == w) {
                double acc = z[i];
                for (int j = rp[i]; j < rp[i + 1]; j++) {
                    int c = col[j];
                    if (c < i) acc -= (double)val[j] * z[c];
                }
                z[i] = acc;
            }
        }
        __syncthreads();
    }
}

// 回代 U dx = z（原位; U 对角非 1; 依赖 j>i 有更小 lv → 低层级先算, 升序波前）
__global__ void k_bwd(const int n, const int maxlv, const int *lv, const int *rp,
                      const int *col, const float *val, double *z)
{
    for (int w = 1; w <= maxlv; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv[i] == w) {
                double acc = z[i], diag = 1.0;
                for (int j = rp[i]; j < rp[i + 1]; j++) {
                    int c = col[j];
                    if (c > i) acc -= (double)val[j] * z[c];
                    else if (c == i) diag = val[j];
                }
                z[i] = acc / diag;
            }
        }
        __syncthreads();
    }
}

// 融合版：一次启动完成 残差SpMV → 前代 → 回代
__global__ void k_fused(const int n, const int maxlv_f, const int *lv_f,
                        const int *rpA, const int *colA, const double *valA,
                        const int *rpL, const int *colL, const float *valL,
                        const int maxlv_b, const int *lv_b,
                        const int *rpU, const int *colU, const float *valU,
                        const double *b, double *x, double *r)
{
    /* stage 1: r = b - A x */
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = b[i];
        for (int j = rpA[i]; j < rpA[i + 1]; j++) acc -= valA[j] * x[colA[j]];
        r[i] = acc;
    }
    __syncthreads();
    /* stage 2: 前代 L r = r */
    for (int w = 1; w <= maxlv_f; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv_f[i] == w) {
                double acc = r[i];
                for (int j = rpL[i]; j < rpL[i + 1]; j++) {
                    int c = colL[j];
                    if (c < i) acc -= (double)valL[j] * r[c];
                }
                r[i] = acc;
            }
        }
        __syncthreads();
    }
    /* stage 3: 回代 U x = r（对角在 valU 的 c==i 项; 依赖 j>i 更小 lv → 升序波前） */
    for (int w = 1; w <= maxlv_b; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv_b[i] == w) {
                double acc = r[i], diag = 1.0;
                for (int j = rpU[i]; j < rpU[i + 1]; j++) {
                    int c = colU[j];
                    if (c > i) acc -= (double)valU[j] * r[c];
                    else if (c == i) diag = valU[j];
                }
                r[i] = acc / diag;
            }
        }
        __syncthreads();
    }
    /* 注：r 在 stage2/3 后即 dx；stage4 由宿主 k_axpy 完成 x += dx */
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <base_name>\n", argv[0]); return 1; }
    char path[512];
    s0f A, L, U;
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_A.mtx", argv[1]);
    if (read_s0f(path, &A, 0)) { fprintf(stderr, "read %s fail\n", path); return 1; }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_L32.mtx", argv[1]);
    if (read_s0f(path, &L, 1)) { fprintf(stderr, "read %s fail\n", path); return 1; }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_U32.mtx", argv[1]);
    if (read_s0f(path, &U, 1)) { fprintf(stderr, "read %s fail\n", path); return 1; }
    int n = A.n;
    /* 层级与 RHS */
    int *lv_f = (int*)malloc((size_t)n * 4), *lv_b = (int*)malloc((size_t)n * 4);
    double *b = (double*)malloc((size_t)n * 8);
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_lv_f.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%d", &lv_f[i]); fclose(f); }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_lv_b.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%d", &lv_b[i]); fclose(f); }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_b.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%lf", &b[i]); fclose(f); }
    int maxlv_f = 0, maxlv_b = 0;
    for (int i = 0; i < n; i++) { if (lv_f[i] > maxlv_f) maxlv_f = lv_f[i]; if (lv_b[i] > maxlv_b) maxlv_b = lv_b[i]; }

    int *d_rpA, *d_colA, *d_rpL, *d_colL, *d_rpU, *d_colU, *d_lv_f, *d_lv_b;
    double *d_valA, *d_b, *d_x, *d_r, *d_x2;
    float *d_valL, *d_valU;
    CK(cudaMalloc(&d_rpA, (size_t)(n+1)*4)); CK(cudaMalloc(&d_colA, (size_t)A.nnz*4));
    CK(cudaMalloc(&d_rpL, (size_t)(n+1)*4)); CK(cudaMalloc(&d_colL, (size_t)L.nnz*4));
    CK(cudaMalloc(&d_rpU, (size_t)(n+1)*4)); CK(cudaMalloc(&d_colU, (size_t)U.nnz*4));
    CK(cudaMalloc(&d_lv_f, (size_t)n*4)); CK(cudaMalloc(&d_lv_b, (size_t)n*4));
    CK(cudaMalloc(&d_valA, (size_t)A.nnz*8)); CK(cudaMalloc(&d_valL, (size_t)L.nnz*4));
    CK(cudaMalloc(&d_valU, (size_t)U.nnz*4));
    CK(cudaMalloc(&d_b, (size_t)n*8)); CK(cudaMalloc(&d_x, (size_t)n*8));
    CK(cudaMalloc(&d_x2, (size_t)n*8)); CK(cudaMalloc(&d_r, (size_t)n*8));
    CK(cudaMemcpy(d_rpA, A.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_colA, A.col, (size_t)A.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rpL, L.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_colL, L.col, (size_t)L.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rpU, U.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_colU, U.col, (size_t)U.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv_f, lv_f, (size_t)n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv_b, lv_b, (size_t)n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valA, A.dval, (size_t)A.nnz*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valL, L.val, (size_t)L.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valU, U.val, (size_t)U.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, b, (size_t)n*8, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_x, 0, (size_t)n*8)); CK(cudaMemset(d_x2, 0, (size_t)n*8));
    CK(cudaMemset(d_r, 0, (size_t)n*8));

    int iters = argc > 2 ? atoi(argv[2]) : 1;
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    /* 未融合：3 启动 */
    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++) {
        k_spmv<<<1, 256>>>(n, d_rpA, d_colA, d_valA, d_x, d_b, d_r);
        CK(cudaGetLastError());
        k_fwd <<<1, 256>>>(n, maxlv_f, d_lv_f, d_rpL, d_colL, d_valL, d_r);
        CK(cudaGetLastError());
        k_bwd <<<1, 256>>>(n, maxlv_b, d_lv_b, d_rpU, d_colU, d_valU, d_r);  /* 原位: r=U^-1 r */
        CK(cudaGetLastError());
        k_axpy<<<1, 256>>>(n, d_x, d_r);                                     /* x += U^-1 L^-1 (b-Ax) */
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); double t_unfused = ms / iters;
    double *hr = (double*)malloc((size_t)n*8);
    CK(cudaMemcpy(hr, d_r, (size_t)n*8, cudaMemcpyDeviceToHost));
    printf("debug: r[0:3]=%.3e %.3e %.3e\n", hr[0], hr[1], hr[2]);
    free(hr);

    /* 融合：1 启动 */
    CK(cudaMemset(d_x2, 0, (size_t)n*8)); CK(cudaMemset(d_r, 0, (size_t)n*8));
    CK(cudaEventRecord(t0));
    for (int it = 0; it < iters; it++) {
        k_fused<<<1, 256>>>(n, maxlv_f, d_lv_f, d_rpA, d_colA, d_valA,
                            d_rpL, d_colL, d_valL, maxlv_b, d_lv_b,
                            d_rpU, d_colU, d_valU, d_b, d_x2, d_r);
        k_axpy<<<1, 256>>>(n, d_x2, d_r);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1)); double t_fused = ms / iters;

    /* 正确性：融合 vs 未融合 输出 x 相对差 */
    double *hx1 = (double*)malloc((size_t)n*8), *hx2 = (double*)malloc((size_t)n*8);
    CK(cudaMemcpy(hx1, d_x, (size_t)n*8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hx2, d_x2, (size_t)n*8, cudaMemcpyDeviceToHost));
    double err = 0, nrm = 0;
    for (int i = 0; i < n; i++) { double d = hx1[i]-hx2[i]; err += d*d; nrm += hx1[i]*hx1[i]; }
    err = sqrt(err / (nrm + 1e-300));
    printf("debug: hx1[0:4]=%.3e %.3e %.3e %.3e  hx2[0:4]=%.3e %.3e %.3e %.3e  nrm=%.3e\n",
           hx1[0], hx1[1], hx1[2], hx1[3], hx2[0], hx2[1], hx2[2], hx2[3], nrm);

    printf("== fusion_bench %s n=%d (levels f=%d b=%d) ==\n", argv[1], n, maxlv_f, maxlv_b);
    printf("未融合(3启动): %.4f ms/iter\n融合  (1启动): %.4f ms/iter\n", t_unfused, t_fused);
    printf("融合/未融合: %.3fx  | 正确性(融合vs未融合): %.2e %s\n",
           t_fused / t_unfused, err, err < 1e-9 ? "✅" : "❌");
    return 0;
}
