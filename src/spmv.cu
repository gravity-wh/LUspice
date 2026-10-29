// ============================================================================
// spmv.cu — CUDA CSR SpMV v0（M0 阶段：验证工具链 + 第一个可用内核）
//
// 赛题七（沐曦 MXMACA GPU 稀疏求解）：本文件刻意只用标准 CUDA 子集，
// 每个 CUDA API 旁注明迁移到 MXMACA 时的对应 API，便于逐行移植。
//   不使用: cudaGraph / 动态并行 / 纹理内存 / cuBLAS / cuSparse
//
// 构建: /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 -o spmv spmv.cu
//       (RTX 3060 = Ampere sm_86; 迁移目标 MXMACA 时改 -arch 与 BLOCK)
//
// 用法: ./spmv <s0.mtx> <s0.rhs|ones> [--out out.txt] [--iters N]
//   <s0.mtx>      s0 格式矩阵: 第一行 "n <n>"，其后 "A i j val"（1-based）
//   <s0.rhs|ones> RHS 文件（"RHS i val"，1-based）或字面量 "ones"（x=1）
//   --out FILE    把 y=A*x 写入文件（每行一个值，%.17g）
//   --iters N     计时循环次数（默认 2000）
//
// 输出（供脚本解析）:
//   n=<n> nnz=<nnz>
//   iters=<N> avg_kernel_ms=<x>      单次内核平均毫秒
//   bandwidth_gbps=<x>               有效带宽 = 2*nnz*8B / 内核时间
//   util_vs_360gbps_pct=<x>          相对 RTX 3060 理论带宽 ~360 GB/s 的百分比
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "spmv.h"

// 线程块大小：集中为宏，方便迁移时按 MXMACA 平台特性（每块最大线程数等）调整。
#define BLOCK 256

// 理论带宽参考（RTX 3060 GDDR6，192-bit @ 15 Gbps ≈ 360 GB/s）
#define THEO_BW_GBPS 360.0

// ----------------------------------------------------------------------------
// s0 格式矩阵读取（host 侧，纯 C）
// 两遍文件扫描：pass1 数 A 行数得 nnz；pass2 读 COO 三元组。
// 再在内存中 计数->前缀和->游标填充，得到按行连续的规范 CSR。
// ----------------------------------------------------------------------------

int read_s0_to_csr(const char* path, CSRMatrix* A) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[spmv] error: cannot open %s\n", path); return 1; }

    char line[512];
    int n = 0;
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }
    // 头行: "n <n>"（本数据集无第二个数字；即使有 "n <n> <nnz>" 也忽略 nnz，以实际计数为准）
    if (sscanf(line, "n %d", &n) != 1 || n <= 0) {
        fprintf(stderr, "[spmv] error: bad header line: %s", line);
        fclose(f);
        return 1;
    }

    // pass1: 数 A 行数 -> nnz（s0 格式头行不含 nnz，必须计数）
    long long nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'A') nnz++;
    }
    if (nnz == 0) { fclose(f); return 1; }

    // 临时 COO 数组
    int* ri = (int*)malloc((size_t)nnz * sizeof(int));
    int* rj = (int*)malloc((size_t)nnz * sizeof(int));
    double* rv = (double*)malloc((size_t)nnz * sizeof(double));

    // pass2: 读三元组（1-based -> 0-based）
    rewind(f);
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }  // 跳过头行
    long long k = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'A') continue;
        int i, j;
        double v;
        if (sscanf(line, "A %d %d %lf", &i, &j, &v) != 3) continue;
        if (i < 1 || i > n || j < 1) continue;
        ri[k] = i - 1;
        rj[k] = j - 1;
        rv[k] = v;
        k++;
    }
    fclose(f);
    if (k != nnz) {
        fprintf(stderr, "[spmv] warning: pass1=%lld vs pass2=%lld entries\n", nnz, k);
        nnz = k;
    }

    // 行计数 -> 前缀和 -> 行指针
    int* rowptr = (int*)calloc((size_t)n + 1, sizeof(int));
    for (long long t = 0; t < nnz; t++) rowptr[ri[t] + 1]++;
    for (int i = 1; i <= n; i++) rowptr[i] += rowptr[i - 1];

    // 游标填充（colidx/values 按行连续）
    int* colidx = (int*)malloc((size_t)nnz * sizeof(int));
    double* values = (double*)malloc((size_t)nnz * sizeof(double));
    int* cursor = (int*)malloc((size_t)n * sizeof(int));
    memcpy(cursor, rowptr, (size_t)n * sizeof(int));
    for (long long t = 0; t < nnz; t++) {
        int p = cursor[ri[t]]++;
        colidx[p] = rj[t];
        values[p] = rv[t];
    }

    free(ri); free(rj); free(rv); free(cursor);

    A->n = n;
    A->nnz = nnz;
    A->rowptr = rowptr;
    A->colidx = colidx;
    A->values = values;
    return 0;
}

void csr_free(CSRMatrix* A) {
    if (!A) return;
    free(A->rowptr);
    free(A->colidx);
    free(A->values);
    A->rowptr = A->colidx = NULL;
    A->values = NULL;
    A->n = 0;
    A->nnz = 0;
}

// ----------------------------------------------------------------------------
// s0 格式 RHS 读取: "n <n>" 头行 + "RHS <i> <val>"（1-based）
// ----------------------------------------------------------------------------

int read_rhs(const char* path, int n, double* x) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[spmv] error: cannot open %s\n", path); return 1; }
    char line[512];
    if (!fgets(line, sizeof line, f)) { fclose(f); return 1; }  // "n <n>"
    int cnt = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'R') continue;  // "RHS i val"
        int i;
        double v;
        if (sscanf(line, "RHS %d %lf", &i, &v) != 2) continue;
        if (i >= 1 && i <= n) { x[i - 1] = v; cnt++; }
    }
    fclose(f);
    return cnt > 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------
// CSR SpMV kernel —— v0：每行一个线程
// 迁移说明：MXMACA 用 __mxc__global__ 修饰 kernel；blockIdx/gridDim 语义相同；
//           __restrict__ 提示编译器可别名分析（MXMACA 编译器同样支持）。
// 长行（nnz/row 很大的行）在 v0 不做分块，后续 v1 再处理。
// ----------------------------------------------------------------------------

__global__ void spmv_csr_kernel(const int n,
                                const int* __restrict__ rowptr,
                                const int* __restrict__ colidx,
                                const double* __restrict__ values,
                                const double* __restrict__ x,
                                double* __restrict__ y) {
    const int row = blockIdx.x * BLOCK + threadIdx.x;
    if (row < n) {
        double sum = 0.0;
        const int j0 = rowptr[row];
        const int j1 = rowptr[row + 1];
        // 内层循环无依赖，编译器可展开/流水；x 按 colidx 收集（v0 不做合并优化）
        for (int k = j0; k < j1; k++) {
            sum += values[k] * x[colidx[k]];
        }
        y[row] = sum;
    }
}

// ----------------------------------------------------------------------------
// 设备侧缓冲分配 + 纯 launch（spmv_host 与 bench 路径共用）
// ----------------------------------------------------------------------------

int csr_alloc_device(const CSRMatrix* A,
                     int** d_rowptr, int** d_colidx, double** d_values) {
    // cudaMalloc        <-> mxMalloc
    // cudaMemcpy H2D    <-> mxMemcpy(mxMemcpyHostToDevice)
    const size_t nb_rp = sizeof(int) * ((size_t)A->n + 1);
    const size_t nb_ci = sizeof(int) * (size_t)A->nnz;
    const size_t nb_vl = sizeof(double) * (size_t)A->nnz;
    if (cudaMalloc((void**)d_rowptr, nb_rp) != cudaSuccess) return 1;
    if (cudaMalloc((void**)d_colidx, nb_ci) != cudaSuccess) return 1;
    if (cudaMalloc((void**)d_values, nb_vl) != cudaSuccess) return 1;
    if (cudaMemcpy(*d_rowptr, A->rowptr, nb_rp, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(*d_colidx, A->colidx, nb_ci, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(*d_values, A->values, nb_vl, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    return 0;
}

void spmv_launch(int n,
                 const int* d_rowptr, const int* d_colidx, const double* d_values,
                 const double* d_x, double* d_y, void* stream) {
    // 迁移说明：<<<grid, block, 0, stream>>> 与 MXMACA 启动语法对应
    const int grid = (n + BLOCK - 1) / BLOCK;
    spmv_csr_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(
        n, d_rowptr, d_colidx, d_values, d_x, d_y);
}

// ----------------------------------------------------------------------------
// host 封装：完整一次 y = A*x 设备往返（正确性路径）
// ----------------------------------------------------------------------------

int spmv_host(const CSRMatrix* A, const double* x, double* y, double* time_ms) {
    const int n = A->n;
    int *d_rowptr = NULL, *d_colidx = NULL;
    double *d_values = NULL, *d_x = NULL, *d_y = NULL;

    if (csr_alloc_device(A, &d_rowptr, &d_colidx, &d_values)) return 1;
    if (cudaMalloc((void**)&d_x, sizeof(double) * (size_t)n) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&d_y, sizeof(double) * (size_t)n) != cudaSuccess) return 1;
    // cudaMemcpy H2D <-> mxMemcpy
    if (cudaMemcpy(d_x, x, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    // warmup（避免首次 launch 的上下文初始化进入计时）
    spmv_launch(n, d_rowptr, d_colidx, d_values, d_x, d_y, 0);
    // cudaDeviceSynchronize <-> mxDeviceSynchronize
    cudaDeviceSynchronize();

    // cudaEvent 计时 <-> mxEventRecord / mxEventElapsedTime
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    spmv_launch(n, d_rowptr, d_colidx, d_values, d_x, d_y, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    if (time_ms) *time_ms = (double)ms;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 结果回拷 D2H <-> mxMemcpy(mxMemcpyDeviceToHost)
    if (cudaMemcpy(y, d_y, sizeof(double) * (size_t)n, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;

    // cudaFree <-> mxFree
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

// ----------------------------------------------------------------------------
// bench 计时循环：warmup + N 次 launch 取平均（结果留在 d_y）
// ----------------------------------------------------------------------------

static double bench_loop(int n,
                         const int* d_rowptr, const int* d_colidx, const double* d_values,
                         const double* d_x, double* d_y, int iters) {
    for (int i = 0; i < 10; i++) spmv_launch(n, d_rowptr, d_colidx, d_values, d_x, d_y, 0);
    cudaDeviceSynchronize();  // 排空 warmup

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++) {
        spmv_launch(n, d_rowptr, d_colidx, d_values, d_x, d_y, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms / (double)iters;  // 单次内核平均毫秒
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

static void usage(const char* prog) {
    fprintf(stderr,
            "usage: %s <s0.mtx> <s0.rhs|ones> [--out out.txt] [--iters N]\n"
            "  y = A*x; x 来自 RHS 文件或 ones\n",
            prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    const char* mtx_path = argv[1];
    const char* rhs_arg = argv[2];
    const char* out_path = NULL;
    int iters = 2000;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { fprintf(stderr, "[spmv] unknown arg: %s\n", argv[i]); return 1; }
    }
    if (iters < 1) iters = 1;

    // --- host 侧：读矩阵 + 读 x ---
    CSRMatrix A;
    memset(&A, 0, sizeof A);
    if (read_s0_to_csr(mtx_path, &A)) return 1;

    double* x = (double*)malloc(sizeof(double) * (size_t)A.n);
    if (!strcmp(rhs_arg, "ones")) {
        for (int i = 0; i < A.n; i++) x[i] = 1.0;
    } else {
        if (read_rhs(rhs_arg, A.n, x)) {
            fprintf(stderr, "[spmv] error: cannot read rhs %s\n", rhs_arg);
            csr_free(&A);
            free(x);
            return 1;
        }
    }
    double* y = (double*)malloc(sizeof(double) * (size_t)A.n);

    // --- 设备侧：分配 + 拷贝 + launch + 回拷 ---
    int *d_rowptr = NULL, *d_colidx = NULL;
    double *d_values = NULL, *d_x = NULL, *d_y = NULL;
    if (csr_alloc_device(&A, &d_rowptr, &d_colidx, &d_values)) return 1;
    if (cudaMalloc((void**)&d_x, sizeof(double) * (size_t)A.n) != cudaSuccess) return 1;
    if (cudaMalloc((void**)&d_y, sizeof(double) * (size_t)A.n) != cudaSuccess) return 1;
    if (cudaMemcpy(d_x, x, sizeof(double) * (size_t)A.n, cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    // 正确性 launch（结果在 d_y）
    spmv_launch(A.n, d_rowptr, d_colidx, d_values, d_x, d_y, 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[spmv] CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // 计时循环（x 不变，最后一次 timed launch 的结果即正确结果）
    double avg_ms = bench_loop(A.n, d_rowptr, d_colidx, d_values, d_x, d_y, iters);

    // 回拷结果
    if (cudaMemcpy(y, d_y, sizeof(double) * (size_t)A.n, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;

    // --- 汇报：任务定义的带宽 = 2*nnz*8B / 内核时间（读 A 值 + 读 x） ---
    double bytes = 2.0 * (double)A.nnz * 8.0;
    double bw = bytes / (avg_ms * 1e-3) / 1e9;  // GB/s
    printf("n=%d nnz=%lld\n", A.n, (long long)A.nnz);
    printf("iters=%d avg_kernel_ms=%.6f\n", iters, avg_ms);
    printf("bandwidth_gbps=%.1f\n", bw);
    printf("util_vs_360gbps_pct=%.1f\n", 100.0 * bw / THEO_BW_GBPS);

    if (out_path) {
        FILE* f = fopen(out_path, "w");
        if (!f) { fprintf(stderr, "[spmv] error: cannot write %s\n", out_path); return 1; }
        for (int i = 0; i < A.n; i++) fprintf(f, "%.17g\n", y[i]);
        fclose(f);
    }

    // --- 清理 ---
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    csr_free(&A);
    free(x);
    free(y);
    return 0;
}
