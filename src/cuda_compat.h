// ============================================================================
// cuda_compat.h — DSH 求解器双平台抽象层（CUDA ↔ MXMACA）
//
// 赛题七：基于沐曦 GPU 的稀疏矩阵求解及 SPICE 加速（国创 & 沐曦）
// 让同一套求解器源码（kernel + host 驱动）可在两套工具链上编译：
//   - 开发期：NVIDIA RTX 3060 + CUDA 12.6（nvcc）
//   - 评测期：沐曦 C500 + MXMACA 软件栈（mxcc，见 §0.3）
//
// 设计意图
// --------
// 1. 统一命名（DSH_*）双映射：CUDA 分支展开为 cudaXxx，MXMACA 分支展开为 mcXxx。
//    依据：MXMACA 运行时 API 是 CUDA 运行时 API 的逐函数改名版
//    （cudaMalloc→mcMalloc, cudaMemcpy→mcMemcpy, cudaDeviceSynchronize→
//    mcDeviceSynchronize, cudaError_t→mcError_t, cudaSuccess→mcSuccess），
//    见《MXMACA 运行时 API 参考》
//    https://developer.metax-tech.com/api/client/document/preview/990/index.html
//    与官方示例 github.com/MetaX-MACA/maca-samples（vectorAdd/unifiedMemory 等）。
// 2. 语言扩展（__global__/__device__/__host__/__shared__/<<<>>>/__syncthreads/
//    gridDim/blockIdx/threadIdx/warpSize）两平台完全一致，无需封装，
//    见《MXMACA C++ 编程指南》
//    https://developer.metax-tech.com/api/client/document/preview/559/C500_MXMACAC%2B%2BProgrammingGuide_CN.html
// 3. 编译方式：
//    - CUDA 分支：nvcc 编译（自动识别 __CUDACC__，免传宏；亦可 -DDSH_PLATFORM_CUDA）。
//    - MXMACA 分支：mxcc -x maca 编译，必须 -DDSH_PLATFORM_MXMACA。
//      mxcc 用法见 https://developer.metax-tech.com/api/client/document/preview/336/C500_mxccCompilerGuide_CN.html
// 4. 使用约束：
//    - 业务代码只调用 DSH_* 接口与两平台一致的 kernel 语法，不直接调 cuda*/mc*，
//      保证迁移零改动。
//    - 平台相关常量（线程束大小等）经 DSH_WARP_SIZE / DSH_PropWaveSize() 访问。
//
// 待环境确认项（不臆造，见 design/cuda_mxmaca_abstraction.md §7）：
//   - mxcc 的 GPU 架构选择机制（官方示例无 -arch 参数）
//   - C500 每 SM 共享内存上限、FP64 峰值算力（支持已确认，数值待测）
//   - mcMemPrefetch / mcMemAdvise 是否存在（统一内存 mcMallocManaged 已确认）
//   - 纹理对象运行时 API（本工程不计划使用纹理）
// ============================================================================
#ifndef DSH_CUDA_COMPAT_H
#define DSH_CUDA_COMPAT_H

// ----------------------------------------------------------------------------
// 0. 平台选择
// ----------------------------------------------------------------------------
#if defined(DSH_PLATFORM_CUDA) && defined(DSH_PLATFORM_MXMACA)
#error "DSH: DSH_PLATFORM_CUDA and DSH_PLATFORM_MXMACA are mutually exclusive"
#endif

#if !defined(DSH_PLATFORM_CUDA) && !defined(DSH_PLATFORM_MXMACA) && !defined(DSH_PLATFORM_HOST)
#if defined(__CUDACC__)
#define DSH_PLATFORM_CUDA   /* nvcc 自动识别 */
#else
#error "DSH: must define DSH_PLATFORM_CUDA / DSH_PLATFORM_HOST, or -DDSH_PLATFORM_MXMACA (MXMACA 分支需 mxcc -DDSH_PLATFORM_MXMACA)"
#endif
#endif

#include <stdio.h>
#include <stdlib.h>

#if defined(DSH_PLATFORM_CUDA)
#include <cuda_runtime.h>
#elif defined(DSH_PLATFORM_MXMACA)
#include <mc_common.h>
#include <mc_runtime.h>
#else /* DSH_PLATFORM_HOST：无 GPU 头，mc* 映射到 stdlib */
#include <string.h>
typedef int mcMemcpyKind;
#define mcMalloc(p, n)      (*(p) = malloc(n), (*(p) ? 0 : -1))
#define mcFree(p)           (free(p), 0)
#define mcMemset(p, v, n)   (memset((p), (v), (n)), 0)
#define mcMemcpy(d, s, n, k) ((void)(k), memcpy((d), (s), (n)), 0)
#define mcMemcpyAsync(d, s, n, k, st) ((void)(k), (void)(st), memcpy((d), (s), (n)), 0)
#define mcMemsetAsync(p, v, n, st)  ((void)(st), memset((p), (v), (n)), 0)
#define mcPeekAtLastError()         0
#define mcMallocManaged(p, n) (*(p) = malloc(n), (*(p) ? 0 : -1))
#define mcStreamCreate(s)     (*(s) = NULL, 0)
#define mcStreamDestroy(s)    ((void)(s), 0)
#define mcStreamSynchronize(s) ((void)(s), 0)
#define mcStreamWaitEvent(s, e, f) ((void)(s), (void)(e), (void)(f), 0)
#define mcGetDeviceCount(c)   (*(c) = 0, 0)
#define mcSetDevice(d)        ((void)(d), 0)
#define mcGetDeviceProperties(p, d) ((void)(p), (void)(d), 0)
#define mcMallocHost(p, n)  (*(p) = malloc(n), (*(p) ? 0 : -1))
#define mcFreeHost(p)       (free(p), 0)
#define mcDeviceSynchronize() 0
#define mcGetLastError()    0
#define mcGetErrorString(e) "host"
#define mcEventCreate(e)    (*(e) = NULL, 0)
#define mcEventDestroy(e)   0
#define mcEventRecord(e, s) ((void)(e), (void)(s), 0)
#define mcEventSynchronize(e) ((void)(e), 0)
#define mcEventElapsedTime(ms, a, b) ((void)(a), (void)(b), *(ms) = 0, 0)
#define mcSuccess           0
#endif

// ----------------------------------------------------------------------------
// 1. 类型别名
// ----------------------------------------------------------------------------
#if defined(DSH_PLATFORM_CUDA)
typedef cudaError_t      dsh_status_t;
typedef cudaStream_t     DSH_Stream_t;
typedef cudaEvent_t      DSH_Event_t;
typedef cudaDeviceProp   DSH_DeviceProp_t;
#define DSH_SUCCESS      cudaSuccess
#elif defined(DSH_PLATFORM_MXMACA)
typedef mcError_t        dsh_status_t;
typedef mcStream_t       DSH_Stream_t;
typedef mcEvent_t        DSH_Event_t;
typedef mcDeviceProp_t   DSH_DeviceProp_t;
#define DSH_SUCCESS      mcSuccess
#else /* DSH_PLATFORM_HOST：CPU 占位 */
typedef int              dsh_status_t;
typedef void*            DSH_Stream_t;
typedef void*            DSH_Event_t;
typedef void*            DSH_DeviceProp_t;
#define DSH_SUCCESS      0
#endif

// ----------------------------------------------------------------------------
// 2. 常量映射（拷贝方向 / 默认流 / 线程束大小）
// ----------------------------------------------------------------------------
#if defined(DSH_PLATFORM_CUDA)
#define DSH_MEMCPY_HOST_TO_DEVICE   cudaMemcpyHostToDevice
#define DSH_MEMCPY_DEVICE_TO_HOST   cudaMemcpyDeviceToHost
#define DSH_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define DSH_DEFAULT_STREAM          cudaStream_t(0)   /* legacy default stream */
#define DSH_WARP_SIZE               32                 /* NVIDIA warp = 32 线程 */
#elif defined(DSH_PLATFORM_MXMACA)
#define DSH_MEMCPY_HOST_TO_DEVICE   mcMemcpyHostToDevice
#define DSH_MEMCPY_DEVICE_TO_HOST   mcMemcpyDeviceToHost
#define DSH_MEMCPY_DEVICE_TO_DEVICE mcMemcpyDeviceToDevice
#define DSH_DEFAULT_STREAM          mcStream_t(0)     /* 对应 mxcc --default-stream legacy */
#define DSH_WARP_SIZE               64                 /* 沐曦 wave = 64 线程（官方确认） */
#else /* HOST */
#define DSH_MEMCPY_HOST_TO_DEVICE   0
#define DSH_MEMCPY_DEVICE_TO_HOST   0
#define DSH_MEMCPY_DEVICE_TO_DEVICE 0
#define DSH_DEFAULT_STREAM          NULL
#define DSH_WARP_SIZE               1
#endif

// ----------------------------------------------------------------------------
// 3. 内存管理（分配 / 释放 / 拷贝 / 清零）
// ----------------------------------------------------------------------------
static inline dsh_status_t DSH_Malloc(void **ptr, size_t size) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMalloc(ptr, size);
#else
  return mcMalloc(ptr, size);
#endif
}

static inline dsh_status_t DSH_Free(void *ptr) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaFree(ptr);
#else
  return mcFree(ptr);
#endif
}

static inline dsh_status_t DSH_MallocManaged(void **ptr, size_t size) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMallocManaged(ptr, size);
#else
  return mcMallocManaged(ptr, size); /* 已确认（官方 unifiedMemory 示例） */
#endif
}

static inline dsh_status_t DSH_MallocHost(void **ptr, size_t size) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMallocHost(ptr, size);
#else
  return mcMallocHost(ptr, size); /* 已确认（官方 pinnedMemory 示例） */
#endif
}

static inline dsh_status_t DSH_Memcpy(void *dst, const void *src, size_t count, int kind) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMemcpy(dst, src, count, (cudaMemcpyKind)kind);
#else
  return mcMemcpy(dst, src, count, (mcMemcpyKind)kind);
#endif
}

static inline dsh_status_t DSH_MemcpyAsync(void *dst, const void *src, size_t count,
                                          int kind, DSH_Stream_t stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMemcpyAsync(dst, src, count, (cudaMemcpyKind)kind, stream);
#else
  return mcMemcpyAsync(dst, src, count, (mcMemcpyKind)kind, stream);
#endif
}

static inline dsh_status_t DSH_Memset(void *ptr, int value, size_t count) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMemset(ptr, value, count);
#else
  return mcMemset(ptr, value, count);
#endif
}

static inline dsh_status_t DSH_MemsetAsync(void *ptr, int value, size_t count,
                                          DSH_Stream_t stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaMemsetAsync(ptr, value, count, stream);
#else
  return mcMemsetAsync(ptr, value, count, stream);
#endif
}

// ----------------------------------------------------------------------------
// 4. 同步
// ----------------------------------------------------------------------------
static inline dsh_status_t DSH_DeviceSynchronize(void) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaDeviceSynchronize();
#else
  return mcDeviceSynchronize();
#endif
}

// ----------------------------------------------------------------------------
// 5. 错误处理
// ----------------------------------------------------------------------------
static inline dsh_status_t DSH_GetLastError(void) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaGetLastError();
#else
  return mcGetLastError();
#endif
}

static inline dsh_status_t DSH_PeekAtLastError(void) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaPeekAtLastError();
#else
  return mcPeekAtLastError();
#endif
}

static inline const char *DSH_ErrorString(dsh_status_t e) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaGetErrorString(e);
#else
  return mcGetErrorString(e);
#endif
}

// 检查一次调用返回码，失败打印并退出（可在任何函数中使用）
#define DSH_CHECK(call)                                                        \
  do {                                                                         \
    dsh_status_t _dsh_err_ = (call);                                           \
    if (_dsh_err_ != DSH_SUCCESS) {                                            \
      fprintf(stderr, "[DSH] %s:%d  %s  failed: %s (code %d)\n", __FILE__,     \
              __LINE__, #call, DSH_ErrorString(_dsh_err_), (int)_dsh_err_);    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// kernel launch 后检查（<<<>>> 异步，错误在 launch 后立即暴露）
#define DSH_CHECK_KERNEL_LAUNCH() DSH_CHECK(DSH_GetLastError())

// ----------------------------------------------------------------------------
// 6. 流 / 事件（Datasweep 批量并发时使用）
// ----------------------------------------------------------------------------
static inline dsh_status_t DSH_StreamCreate(DSH_Stream_t *stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaStreamCreate(stream);
#else
  return mcStreamCreate(stream);
#endif
}

static inline dsh_status_t DSH_StreamDestroy(DSH_Stream_t stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaStreamDestroy(stream);
#else
  return mcStreamDestroy(stream);
#endif
}

static inline dsh_status_t DSH_StreamSynchronize(DSH_Stream_t stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaStreamSynchronize(stream);
#else
  return mcStreamSynchronize(stream);
#endif
}

static inline dsh_status_t DSH_StreamWaitEvent(DSH_Stream_t stream, DSH_Event_t event) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaStreamWaitEvent(stream, event, 0);
#else
  return mcStreamWaitEvent(stream, event, 0);
#endif
}

static inline dsh_status_t DSH_EventCreate(DSH_Event_t *event) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaEventCreate(event);
#else
  return mcEventCreate(event);
#endif
}

static inline dsh_status_t DSH_EventDestroy(DSH_Event_t event) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaEventDestroy(event);
#else
  return mcEventDestroy(event);
#endif
}

static inline dsh_status_t DSH_EventRecord(DSH_Event_t event, DSH_Stream_t stream) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaEventRecord(event, stream);
#else
  return mcEventRecord(event, stream);
#endif
}

static inline dsh_status_t DSH_EventSync(DSH_Event_t event) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaEventSynchronize(event);
#else
  return mcEventSynchronize(event);
#endif
}

static inline dsh_status_t DSH_EventElapsed(float *ms, DSH_Event_t start, DSH_Event_t end) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaEventElapsedTime(ms, start, end);
#else
  return mcEventElapsedTime(ms, start, end);
#endif
}

// ----------------------------------------------------------------------------
// 7. 设备管理（诊断 / 特性查询）
// ----------------------------------------------------------------------------
static inline dsh_status_t DSH_GetDeviceCount(int *count) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaGetDeviceCount(count);
#else
  return mcGetDeviceCount(count);
#endif
}

static inline dsh_status_t DSH_SetDevice(int device) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaSetDevice(device);
#else
  return mcSetDevice(device);
#endif
}

static inline dsh_status_t DSH_GetDeviceProperties(DSH_DeviceProp_t *prop, int device) {
#if defined(DSH_PLATFORM_CUDA)
  return cudaGetDeviceProperties(prop, device);
#else
  return mcGetDeviceProperties(prop, device);
#endif
}

// 线程束（wave）大小：CUDA 恒 32；沐曦由设备属性 waveSize 给出（C500 = 64）。
// 字段名不同：cudaDeviceProp.warpSize vs mcDeviceProp_t.waveSize。
static inline int DSH_PropWaveSize(const DSH_DeviceProp_t *prop) {
#if defined(DSH_PLATFORM_CUDA)
  return (int)prop->warpSize;
#elif defined(DSH_PLATFORM_MXMACA)
  return (int)prop->waveSize;
#else
  (void)prop; return 1; /* HOST */
#endif
}

// ----------------------------------------------------------------------------
// 8. 内核启动语法宏（可选）
//    <<<>>> 语法两平台完全一致；本宏仅作统一入口 + 便于未来挂 launch 检查。
//    用法：DSH_LAUNCH(kernel, grid, block, arg1, arg2);  或 DSH_LAUNCH(k, g, b);
// ----------------------------------------------------------------------------
#define DSH_LAUNCH(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)

// 线程块大小集中配置（迁移时按平台特性调整；沐曦 wave=64，建议取 64 的倍数）
#ifndef DSH_BLOCK_SIZE
#define DSH_BLOCK_SIZE 256
#endif

#endif /* DSH_CUDA_COMPAT_H */
