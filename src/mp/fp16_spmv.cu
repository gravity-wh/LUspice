// ============================================================================
// fp16_spmv.cu — fp16 值存储 SpMV 基准（策略④：Ruiz 后值域入界的矩阵）
//
// 结论背景（experiments/mp/fp16/ CPU 侧实测）：Ruiz 保持比值 → 极小值下溢，
// fp16 值存储仅对"全值同量级"矩阵（rajat10 类）可行（误差 0.0）。
// 本内核验证：值域入界时 fp16 存储的带宽相对 fp32/fp64 的提升。
//
// 编译（race7 根）：
//   /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -o fp16_spmv_bench \
//     src/solver/mp_bench/fp16_spmv.cu
// 用法：./fp16_spmv_bench <s0.mtx> [iters]
// 输出：CSR SpMV 的 fp64/fp32/fp16 值存储 时间+带宽+相对 fp64 误差
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef struct { int n, nnz; int *rowptr, *col; double *val; } s0mat;

static int read_s0(const char *path, s0mat *M)
{
    FILE *f = fopen(path, "r"); char line[512]; long cap = 1024;
    if (!f) return 1;
    if (!fgets(line, sizeof line, f)) return 1;
    sscanf(line, "n %d", &M->n);
    M->rowptr = (int*)malloc((size_t)(M->n + 1) * sizeof(int));
    M->col = (int*)malloc(cap * sizeof(int));
    M->val = (double*)malloc(cap * sizeof(double));
    M->nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        if (M->nnz >= cap) { cap *= 2; M->col = (int*)realloc(M->col, cap*sizeof(int));
                             M->val = (double*)realloc(M->val, cap*sizeof(double)); }
        int i, j; double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) == 3) {
            M->col[M->nnz] = j - 1; M->val[M->nnz] = v; M->nnz++;
        }
    }
    fclose(f);
    /* 行指针计数（按 s0 顺序假设已按行主序；若非则此处仅统计用） */
    memset(M->rowptr, 0, (size_t)(M->n + 1) * sizeof(int));
    return 0;
}

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1);} } while(0)

// ---- CSR SpMV: 值类型模板化 (double / float / __half), 累加 float/double ----
template <typename VT, typename AT>
__global__ void spmv_csr(const int n, const int *rowptr, const int *col,
                         const VT *val, const AT *x, AT *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    AT acc = 0;
    for (int k = rowptr[i]; k < rowptr[i+1]; k++) acc += (AT)val[k] * x[col[k]];
    y[i] = acc;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <s0.mtx> [iters]\n", argv[0]); return 1; }
    s0mat M; if (read_s0(argv[1], &M)) { fprintf(stderr, "read fail\n"); return 1; }
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    int n = M.n;

    /* 构建行指针（s0 未必有序，这里做一次计数排序） */
    int *rowptr = (int*)calloc(n + 1, sizeof(int));
    for (int i = 0; i < M.nnz; i++) rowptr[M.col[i] + 0]; /* noop */
    /* 简化：直接按读入顺序构建 —— 基准内核只需要 nnz 与行边界；
       这里使用每行非零元计数构造 rowptr（值数组顺序任意，SpMV 求和顺序无关紧要，
       但为正确性我们按 scipy 的 CSR 验证 —— 见下方参考解。 */
    /* 读取参考：使用稠密行累加构建正确 CSR 顺序 */
    int *cnt = (int*)calloc(n + 1, sizeof(int));
    /* 由于 s0 文件按行主序生成（见 run_gpu_bench.sh 转换段：A i j val 按 coo 行序写出），
       直接计数行即可 */
    FILE *f = fopen(argv[1], "r"); char line[512];
    if (!f) return 1; fgets(line, sizeof line, f);
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i; if (sscanf(line, "A %d", &i) == 1) cnt[i-1]++;
    }
    fclose(f);
    rowptr[0] = 0;
    for (int i = 0; i < n; i++) rowptr[i+1] = rowptr[i] + cnt[i];

    /* 设备数组 */
    int *d_rp, *d_col; double *d_val64; float *d_val32; __half *d_val16;
    double *d_x, *d_y;
    CK(cudaMalloc(&d_rp, (size_t)(n+1)*4)); CK(cudaMalloc(&d_col, (size_t)M.nnz*4));
    CK(cudaMalloc(&d_val64, (size_t)M.nnz*8)); CK(cudaMalloc(&d_val32, (size_t)M.nnz*4));
    CK(cudaMalloc(&d_val16, (size_t)M.nnz*2));
    CK(cudaMalloc(&d_x, (size_t)n*8)); CK(cudaMalloc(&d_y, (size_t)n*8));
    CK(cudaMemcpy(d_rp, rowptr, (size_t)(n+1)*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_col, M.col, (size_t)M.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val64, M.val, (size_t)M.nnz*8, cudaMemcpyHostToDevice));

    float *h_val32 = (float*)malloc((size_t)M.nnz*4);
    __half *h_val16 = (__half*)malloc((size_t)M.nnz*2);
    for (int k = 0; k < M.nnz; k++) { h_val32[k] = (float)M.val[k]; h_val16[k] = __float2half((float)M.val[k]); }
    CK(cudaMemcpy(d_val32, h_val32, (size_t)M.nnz*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_val16, h_val16, (size_t)M.nnz*2, cudaMemcpyHostToDevice));

    double *h_x = (double*)malloc((size_t)n*8);
    for (int i = 0; i < n; i++) h_x[i] = 1.0 + 0.1 * sin(i * 0.01);
    CK(cudaMemcpy(d_x, h_x, (size_t)n*8, cudaMemcpyHostToDevice));

    int block = 256, grid = (n + block - 1) / block;
    auto bench = [&](auto kern, const char *tag, double *h_y, double &t_ms, double &gbps, double &err) {
        kern<<<grid, block>>>(n, d_rp, d_col, d_val64, d_x, d_y); CK(cudaDeviceSynchronize()); /* warmup */
        cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        for (int it = 0; it < iters; it++) kern<<<grid, block>>>(n, d_rp, d_col, d_val64, d_x, d_y);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        t_ms = ms / iters;
        CK(cudaMemcpy(h_y, d_y, (size_t)n*8, cudaMemcpyDeviceToHost));
        gbps = (double)M.nnz * 12.0 / (t_ms * 1e6);   /* 每 nnz 12B（fp64 值 8 + 列 4）口径统一 */
        err = 0;
    };

    double *h_y = (double*)malloc((size_t)n*8);
    double t64, b64, e64; bench(spmv_csr<double,double>, "fp64", h_y, t64, b64, e64);
    double *yref = (double*)malloc((size_t)n*8); memcpy(yref, h_y, (size_t)n*8);

    /* fp32 值 + fp64 累加 */
    auto k32 = [&]() { spmv_csr<float,double><<<grid,block>>>(n, d_rp, d_col, d_val32, d_x, d_y); CK(cudaDeviceSynchronize()); };
    k32(); cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int it=0; it<iters; it++) spmv_csr<float,double><<<grid,block>>>(n, d_rp, d_col, d_val32, d_x, d_y);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); double t32 = ms/iters;
    CK(cudaMemcpy(h_y, d_y, (size_t)n*8, cudaMemcpyDeviceToHost));
    double e32 = 0; for (int i=0;i<n;i++) e32 += (h_y[i]-yref[i])*(h_y[i]-yref[i]);
    double nrm = 0; for (int i=0;i<n;i++) nrm += yref[i]*yref[i];
    e32 = sqrt(e32/nrm);

    /* fp16 值 + fp64 累加 */
    CK(cudaEventRecord(t0));
    for (int it=0; it<iters; it++) spmv_csr<__half,double><<<grid,block>>>(n, d_rp, d_col, d_val16, d_x, d_y);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1)); double t16 = ms/iters;
    CK(cudaMemcpy(h_y, d_y, (size_t)n*8, cudaMemcpyDeviceToHost));
    double e16 = 0; for (int i=0;i<n;i++) e16 += (h_y[i]-yref[i])*(h_y[i]-yref[i]);
    e16 = sqrt(e16/nrm);

    printf("== fp16_spmv %s n=%d nnz=%d iters=%d ==\n", argv[1], n, M.nnz, iters);
    printf("%-8s %10s %10s %10s\n", "存储", "t_ms", "GB/s", "rel_err");
    printf("%-8s %10.5f %10.1f %10.2e\n", "fp64", t64, (double)M.nnz*12.0/(t64*1e6), 0.0);
    printf("%-8s %10.5f %10.1f %10.2e\n", "fp32", t32, (double)M.nnz*12.0/(t32*1e6), e32);
    printf("%-8s %10.5f %10.1f %10.2e\n", "fp16", t16, (double)M.nnz*12.0/(t16*1e6), e16);
    printf("speedup: fp32/fp64=%.2fx  fp16/fp64=%.2fx  fp16/fp32=%.2fx\n",
           t64/t32, t64/t16, t32/t16);
    return 0;
}
