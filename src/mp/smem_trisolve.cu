// ============================================================================
// smem_trisolve.cu — 三角求解的共享内存缓存版 + smem 预算扫描（策略③）
//
// 把波前三角求解的 z 向量（前代/回代反复读写的热数据）驻留共享内存，
// 对比不同 smem 预算（0=关 / 8/16/32/48KB）下的时间与带宽；
// 并报告 smem 预算对"可并发 block 数"的约束（多 block 化时的权衡曲线）。
//
// 输入复用 /tmp/fusion_inputs/（与 fusion_bench 同源，见其数据生成说明）。
//
// 编译（race7 根）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -o smem_trisolve_bench \
//     src/solver/mp_bench/smem_trisolve.cu
// 用法：./smem_trisolve_bench <base_name> [iters]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

typedef struct { int n, nnz; int *rowptr, *col; float *val; } s0f;
static int read_s0f(const char *path, s0f *M)
{
    FILE *f = fopen(path, "r"); char line[512];
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    long cap = 1024;
    int *cnt = (int*)calloc(M->n + 1, sizeof(int));
    M->col = (int*)malloc(cap * sizeof(int));
    M->val = (float*)malloc(cap * sizeof(float));
    M->nnz = 0;
    rewind(f); fgets(line, sizeof line, f);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (M->nnz >= cap) { cap *= 2; M->col = (int*)realloc(M->col, cap*sizeof(int));
                             M->val = (float*)realloc(M->val, cap*sizeof(float)); }
        M->col[M->nnz] = j - 1; M->val[M->nnz] = (float)v; M->nnz++;
        cnt[i - 1]++;
    }
    fclose(f);
    M->rowptr = (int*)malloc((size_t)(M->n + 1) * sizeof(int));
    M->rowptr[0] = 0;
    for (int i = 0; i < M->n; i++) M->rowptr[i + 1] = M->rowptr[i] + cnt[i];
    free(cnt);
    return 0;
}

// ---- 混合访问器：z 元素 i 若 < smem_n 驻留共享内存，否则走全局 ----
#define ZRD(i) (((i) < smem_n) ? zsm[i] : z_global[i])
#define ZWR(i, v) do { if ((i) < smem_n) zsm[i] = (v); else z_global[i] = (v); } while (0)

// ---- 前代：z 可驻留共享内存（动态 smem, extern；部分驻留时混合访问）----
__global__ void k_fwd(const int n, const int maxlv, const int *lv,
                      const int *rp, const int *col, const float *val,
                      double *z_global, const int smem_n)
{
    extern __shared__ double zsm[];
    if (smem_n > 0) for (int i = threadIdx.x; i < n && i < smem_n; i += blockDim.x) zsm[i] = z_global[i];
    __syncthreads();
    for (int w = 1; w <= maxlv; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv[i] == w) {
                double acc = ZRD(i);
                for (int j = rp[i]; j < rp[i + 1]; j++) {
                    int c = col[j];
                    if (c < i) acc -= (double)val[j] * ZRD(c);
                }
                ZWR(i, acc);
            }
        }
        __syncthreads();
    }
    if (smem_n > 0) for (int i = threadIdx.x; i < n && i < smem_n; i += blockDim.x) z_global[i] = zsm[i];
}

// ---- 回代：同样支持 smem 驻留 ----
__global__ void k_bwd(const int n, const int maxlv, const int *lv,
                      const int *rp, const int *col, const float *val,
                      double *z_global, const int smem_n)
{
    extern __shared__ double zsm[];
    if (smem_n > 0) for (int i = threadIdx.x; i < n && i < smem_n; i += blockDim.x) zsm[i] = z_global[i];
    __syncthreads();
    for (int w = 1; w <= maxlv; w++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            if (lv[i] == w) {
                double acc = ZRD(i), diag = 1.0;
                for (int j = rp[i]; j < rp[i + 1]; j++) {
                    int c = col[j];
                    if (c > i) acc -= (double)val[j] * ZRD(c);
                    else if (c == i) diag = val[j];
                }
                ZWR(i, acc / diag);
            }
        }
        __syncthreads();
    }
    if (smem_n > 0) for (int i = threadIdx.x; i < n && i < smem_n; i += blockDim.x) z_global[i] = zsm[i];
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <base_name> [iters]\n", argv[0]); return 1; }
    char path[512];
    s0f L, U;
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_L32.mtx", argv[1]);
    if (read_s0f(path, &L)) { fprintf(stderr, "read %s fail\n", path); return 1; }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_U32.mtx", argv[1]);
    if (read_s0f(path, &U)) { fprintf(stderr, "read %s fail\n", path); return 1; }
    int n = L.n, iters = argc > 2 ? atoi(argv[2]) : 50;
    int *lv_f = (int*)malloc((size_t)n*4), *lv_b = (int*)malloc((size_t)n*4);
    double *b = (double*)malloc((size_t)n*8);
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_lv_f.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%d", &lv_f[i]); fclose(f); }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_lv_b.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%d", &lv_b[i]); fclose(f); }
    snprintf(path, sizeof path, "/tmp/fusion_inputs/%s_b.txt", argv[1]);
    { FILE *f = fopen(path, "r"); for (int i = 0; i < n; i++) fscanf(f, "%lf", &b[i]); fclose(f); }
    int maxlv_f = 0, maxlv_b = 0;
    for (int i = 0; i < n; i++) { if (lv_f[i] > maxlv_f) maxlv_f = lv_f[i]; if (lv_b[i] > maxlv_b) maxlv_b = lv_b[i]; }

    int *d_rpL, *d_colL, *d_rpU, *d_colU, *d_lv_f, *d_lv_b;
    float *d_valL, *d_valU; double *d_z, *d_ref;
    CK(cudaMalloc(&d_rpL, (size_t)(n+1)*4)); CK(cudaMalloc(&d_colL, (size_t)L.nnz*4));
    CK(cudaMalloc(&d_rpU, (size_t)(n+1)*4)); CK(cudaMalloc(&d_colU, (size_t)U.nnz*4));
    CK(cudaMalloc(&d_lv_f, (size_t)n*4)); CK(cudaMalloc(&d_lv_b, (size_t)n*4));
    CK(cudaMalloc(&d_valL, (size_t)L.nnz*4)); CK(cudaMalloc(&d_valU, (size_t)U.nnz*4));
    CK(cudaMalloc(&d_z, (size_t)n*8)); CK(cudaMalloc(&d_ref, (size_t)n*8));
    CK(cudaMemcpy(d_rpL, L.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_colL, L.col, (size_t)L.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rpU, U.rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_colU, U.col, (size_t)U.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv_f, lv_f, (size_t)n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lv_b, lv_b, (size_t)n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valL, L.val, (size_t)L.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valU, U.val, (size_t)U.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_z, b, (size_t)n*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ref, b, (size_t)n*8, cudaMemcpyHostToDevice));
    double *d_ref_b; CK(cudaMalloc(&d_ref_b, (size_t)n*8));
    CK(cudaMemcpy(d_ref_b, b, (size_t)n*8, cudaMemcpyHostToDevice));

    int smem_caps[] = {0, 8, 16, 32, 48};
    double bytes_total = (double)(L.nnz + U.nnz) * 8.0;   /* z 读写口径近似 */
    printf("== smem_trisolve %s n=%d (levels f=%d b=%d) iters=%d ==\n", argv[1], n, maxlv_f, maxlv_b, iters);
    printf("%-6s | %-12s | %-12s | %-12s | %-14s\n", "smemKB", "fwd+bwd(ms)", "GB/s", "z驻留数", "maxBlocks/SM");
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    for (int ci = 0; ci < 5; ci++) {
        int cap_kb = smem_caps[ci];
        int smem_n = (cap_kb > 0) ? (int)((size_t)cap_kb * 1024 / 8) : 0;   /* 可驻留的 z 元素数 */
        if (smem_n > n) smem_n = n;
        size_t smem_bytes = (size_t)smem_n * 8;
        CK(cudaMemset(d_z, 0, (size_t)n*8));
        CK(cudaMemcpy(d_z, b, (size_t)n*8, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(t0));
        for (int it = 0; it < iters; it++) {
            CK(cudaMemcpy(d_z, d_ref_b, (size_t)n*8, cudaMemcpyDeviceToDevice));
            k_fwd<<<1, 256, smem_bytes>>>(n, maxlv_f, d_lv_f, d_rpL, d_colL, d_valL, d_z, smem_n);
            k_bwd<<<1, 256, smem_bytes>>>(n, maxlv_b, d_lv_b, d_rpU, d_colU, d_valU, d_z, smem_n);
        }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        double t = ms / iters;
        /* 占用率视角: 若多 block 化, smem 预算决定的每 SM 可并发 block 数上限 (RTX3060 每SM 100KB) */
        int maxb = (cap_kb > 0) ? (100 / cap_kb) : 999;
        printf("%-6d | %-12.4f | %-12.1f | %-12d | %-14d\n", cap_kb, t,
               bytes_total / (t * 1e-3) / 1e9, smem_n, maxb);
        if (ci == 0) CK(cudaMemcpy(d_ref, d_z, (size_t)n*8, cudaMemcpyDeviceToDevice));
    }
    /* 正确性: 各 smem 档 vs 无 smem 档（d_ref 保存了 ci=0 的结果） */
    double *href = (double*)malloc((size_t)n*8), *hz = (double*)malloc((size_t)n*8);
    CK(cudaMemcpy(href, d_ref, (size_t)n*8, cudaMemcpyDeviceToHost));
    double nrm = 0; for (int i = 0; i < n; i++) nrm += href[i]*href[i];
    for (int ci = 1; ci < 5; ci++) {
        int cap_kb = smem_caps[ci];
        int smem_n = (int)((size_t)cap_kb * 1024 / 8); if (smem_n > n) smem_n = n;
        CK(cudaMemcpy(d_z, d_ref_b, (size_t)n*8, cudaMemcpyDeviceToDevice));
        k_fwd<<<1, 256, (size_t)smem_n*8>>>(n, maxlv_f, d_lv_f, d_rpL, d_colL, d_valL, d_z, smem_n);
        k_bwd<<<1, 256, (size_t)smem_n*8>>>(n, maxlv_b, d_lv_b, d_rpU, d_colU, d_valU, d_z, smem_n);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hz, d_z, (size_t)n*8, cudaMemcpyDeviceToHost));
        double err = 0; for (int i = 0; i < n; i++) { double d = hz[i]-href[i]; err += d*d; }
        int d1 = -1; for (int i = 0; i < n && d1 < 0; i++) if (fabs(hz[i]-href[i]) > 1e-9 * (fabs(href[i])+1)) d1 = i;
        printf("正确性(smem=%dKB): %.2e %s  首分歧@%d (href=%.3e hz=%.3e)\n", cap_kb, sqrt(err/(nrm+1e-300)),
               sqrt(err/(nrm+1e-300)) < 1e-12 ? "✅" : "❌", d1,
               d1 >= 0 ? href[d1] : 0.0, d1 >= 0 ? hz[d1] : 0.0);
    }
    return 0;
}
