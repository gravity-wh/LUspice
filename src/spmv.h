// ============================================================================
// spmv.h — CSR SpMV v0 公共接口（M0 阶段）
//
// 赛题七（沐曦 MXMACA GPU 稀疏求解）迁移纪律：
//   本模块只使用标准 CUDA 子集（__global__ kernel / cudaMalloc / cudaMemcpy /
//   cudaFree / cudaDeviceSynchronize / cudaEvent 计时），不使用 cudaGraph、
//   动态并行、纹理内存、cuBLAS/cuSparse。后续逐 API 映射 MXMACA。
//
// s0 矩阵格式（本地数据统一格式）：
//   第一行 "n <n>"（可选第二个数为期望 nnz，本数据集实际省略）
//   其后每行 "A <i> <j> <val>"，i/j 为 1-based
// RHS 格式：
//   第一行 "n <n>"，其后每行 "RHS <i> <val>"，i 为 1-based
// ============================================================================
#ifndef RACE7_SOLVER_SPMV_H
#define RACE7_SOLVER_SPMV_H

#ifdef __cplusplus
extern "C" {
#endif

// host 侧 CSR 矩阵
typedef struct {
    int n;            // 矩阵维度
    long long nnz;    // 非零元个数
    int* rowptr;      // 行指针，长度 n+1（CSR 行偏移，row i 的元素在 [rowptr[i], rowptr[i+1])）
    int* colidx;      // 列下标，长度 nnz（0-based）
    double* values;   // 非零元值，长度 nnz
} CSRMatrix;

// 读 s0 格式 .mtx 到 CSR（0-based）。成功返回 0；失败返回非 0。
// 内部两遍扫描：第一遍数 A 行数得到 nnz，第二遍填充。
// 注意：调用方负责用 csr_free() 释放。
int read_s0_to_csr(const char* path, CSRMatrix* A);

// 释放 read_s0_to_csr 分配的内存。
void csr_free(CSRMatrix* A);

// 读 s0 格式 RHS 到 x（长度 n，0-based）。成功返回 0；失败返回非 0。
int read_rhs(const char* path, int n, double* x);

// host 封装：完整一次 y = A*x 设备往返（cudaMalloc + cudaMemcpy H2D +
// kernel launch + cudaMemcpy D2H + cudaFree），供正确性/单次调用使用。
// time_ms 非空时返回单次 kernel 时间（毫秒，含一次 warmup）。
// 成功返回 0；失败返回非 0。
int spmv_host(const CSRMatrix* A, const double* x, double* y, double* time_ms);

// 设备侧 CSR 缓冲分配 + 拷入（spmv_host 与 bench 路径共用）。
// 成功返回 0；失败返回非 0。
int csr_alloc_device(const CSRMatrix* A,
                     int** d_rowptr, int** d_colidx, double** d_values);

// 纯 launch：y = A*x，stream 可为 0（默认流）。调用前需先
// csr_alloc_device + 拷入 x；结果留在 d_y，由调用方回拷。
void spmv_launch(int n,
                 const int* d_rowptr, const int* d_colidx, const double* d_values,
                 const double* d_x, double* d_y, void* stream);

#ifdef __cplusplus
}
#endif

#endif // RACE7_SOLVER_SPMV_H
