# M3 GPU 稀疏 LU 初版验证报告（2026-08-15）

## 结论
**bandgap（n=113）、pg2k（n=4050）、pg5k（n=5000）三矩阵全部 PASS**：GPU 解 vs CPU 解（splu.h 参考）相对 L2 ≤ 1e-6，且 GPU 因子结构（lnnz/unnz）与 CPU **逐位一致**。

| 矩阵 | n | nnz | 列序 | pin | GPU lnnz/unnz | CPU lnnz/unnz | GPU vs CPU 相对L2 | GPU 残差 ||b-Ax||/||b|| | CPU 残差 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bandgap | 113 | 616 | COLAMD | 无 | 582 / 777 | 582 / 777 | **1.759e-12** | 4.161e-12 | 6.956e-12 | ✅ PASS |
| pg2k | 4050 | 23406 | COLAMD | .flt 1e-8 | 171725 / 171737 | 171725 / 171737 | **9.189e-10** | 3.145e-15 | 3.164e-15 | ✅ PASS |
| pg5k | 5000 | 28925 | COLAMD | .flt 1e-8 | 230964 / 230979 | 230964 / 230979 | **7.151e-09** | 6.552e-15 | 6.375e-15 | ✅ PASS |

（CPU 残差与 M2 记录一致：bandgap 6.956e-12 / pg2k 3.164e-15 / pg5k 6.375e-15，见 experiments/m2/m2_cpu_verification.md。GPU 因子结构与 CPU 完全相同，GPU 残差不劣于 CPU。）

## 环境与编译
- GPU：NVIDIA GeForce RTX 3060（28 SM，maxThreads/SM=1536，12GB）
- 工具链：CUDA 12.6（/usr/local/cuda-12.6/bin/nvcc），`-O3 -arch=sm_86`
- 编译命令：
```
nvcc -O3 -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src \
     -I /home/gravity/n46-mix/src/include -o test_lu_gpu \
     src/solver/test_lu_gpu.cu src/solver/lu_gpu.cu \
     src/solver/cpu_ref_lu.c \
     /home/gravity/n46-mix/src/maths/KLU/colamd.c \
     /home/gravity/n46-mix/src/maths/KLU/colamd_global.c -lm
```
- 运行：
```
./test_lu_gpu <valdata>/sky130-bandgap-reference-pvt/matrix_conv.mtx \
              data/rhs_generated/sky130-bandgap-reference-pvt.rhs
./test_lu_gpu /home/gravity/g3-ws/data/pg2k.mtx /home/gravity/g3-ws/data/pg2k.rhs 1e-8 /home/gravity/g3-ws/data/pg2k.flt
./test_lu_gpu /home/gravity/g3-ws/data/pg5k.mtx /home/gravity/g3-ws/data/pg5k.rhs 1e-8 /home/gravity/g3-ws/data/pg5k.flt
```

## Kernel 时间（cudaEvent，单 thread kernel）

| 矩阵 | lu_factor_kernel | trisolve_kernel |
|---|---|---|
| bandgap | 4.14 ms | 0.33 ms |
| pg2k | 5463 ms（5.46 s） | 63.4 ms |
| pg5k | 8595 ms（8.60 s） | 84.3 ms |

- 单 thread 串行 LU 属预期量级（任务预案：pg2k 可能数秒）：稠密简化可达集按列全扫描 O(n·lnnz)，主元/清零全行扫描 O(n²)。
- **并行化是后续性能迭代的核心动机**（每列一个 block / level 调度；稀疏 DFS 可达集替代全行扫描）。
- 正确性 > 性能：本里程碑以三矩阵 PASS 验收，时间为基线记录。

## Occupancy（cudaOccupancyMaxActiveBlocksPerMultiprocessor，block=1 thread）

| kernel | maxActiveBlocks/SM | SM 线程占用 |
|---|---|---|
| lu_factor_kernel | 16 | 16/1536 = **1.04%** |
| trisolve_kernel | 16 | 16/1536 = **1.04%** |

单 thread 设计下 Occupancy 理论值极低（预期内）；并行化后按 warp 粒度统计。

## 实现要点
- `src/solver/lu_gpu.cu`
  - `lu_factor_kernel`：单 thread 完整稀疏左视 LU，逻辑 = splu.h lu_factor 预分配版。
    - 可达集**稠密简化**：x 稠密向量 + 按主元序全列扫描前代（等价 spsolve 散射环），每列末全向量清零防跨列残留
    - 部分主元 **fp64 tol=1.0**；自然主元候选行 = `diagrow[k]=q[k]`（host 按 q 重排后的对角行），与 splu.h `pinv[col]` 语义一致
    - L：单位对角首项、行号先为原始行、分解末 remap 为主元位置；U：对角末项
    - **零动态分配**：全部工作数组 host 预分配（容量 = 上界 n(n+1)/2+16，免 grow）
  - `trisolve_kernel`：单 thread 前代 + 回代（= lu_solve_perm：rowat 逆映射 P b → L y → U z → **Q z 散射**）
- `src/solver/test_lu_gpu.cu`：host 驱动（复用 test_lu.c 的 s0 读取 / COLAMD cp 初始化+行号升序 / .flt pin 注入），q 序 CSC 上传 → launch → 下载 x → L2 + 残差 + 时间 + Occupancy
- `src/solver/cpu_ref_lu.c`：CPU 参考（splu.h lu_factor+lu_solve_perm 的 extern "C" 封装；splu.h 仅 C 编译）
- 数据通路：s0 矩阵 → .flt pin（COO 层）→ COLAMD q → q 序 CSC → device → 因子 → 解

## 零动态分配代码审查（验收项 2）
- `__global__` 内调用白名单：仅 `fabs`（device 数学），无 malloc/cudaMalloc/new/realloc/free
- L/U 容量 = n(n+1)/2+16（稠密上界，任何主元/填充下不溢出；实测 pg2k 使用 171725/8.2M，pg5k 230964/10.0M）
- 内核入口单 thread 保护：`if (threadIdx.x != 0 || blockIdx.x != 0) return;`

## 修复链（本里程碑）
1. **host 指针误入 kernel**：trisolve 的 `q` 须先 cudaMalloc 上传（d_q），kernel 不能解引用 host 指针（非法访问）
2. **CPU 参考双重列置换**：build_csc_q 已重排 A，CPU 参考必须用**原始序 CSC + q**（传重排 A + q 会再置换一次，lnnz 582→2219）
3. **colamd cp 使用后释放（use-after-free）**：q 拷贝必须在 free(cp) 之前
4. 继承上一工作版：invq 逆映射列序、cp 列指针初始化、全行扫描主元替代 DFS 标记恢复（消除上一版 U 结构差 229 的问题——本版 U 结构与 CPU 完全一致）

## 产物清单
- `src/solver/lu_gpu.cu` / `src/solver/lu_gpu.h` —— GPU kernel（因子 + 三角求解）
- `src/solver/test_lu_gpu.cu` —— host 驱动与验收
- `src/solver/cpu_ref_lu.c` / `cpu_ref_lu.h` —— CPU 参考封装
- `test_lu_gpu` —— 可执行（工作区根目录）
- 本报告：`experiments/m3/m3_gpu_verification.md`

## 未完成项（后续迭代）
- ⏳ 并行化：lu_factor 每列 block / level 调度、trisolve 每行一 thread（Occupancy 与时间随并行化提升）
- ⏳ 稀疏 DFS 可达集替代稠密全行扫描（O(n·lnnz) → 近线性，pg2k 因子时间 5.46s → 预期 <0.1s 量级）
- ⏳ 静态化 fp32 Ax（符号分解复用 + fp64 主元比较；fp32 因子存储已由 lu_gpu_f32.cu 验证，见本目录 lu_gpu_f32 相关工作）
- ⏳ MXMACA 平台移植（cuda_compat.h DSH_* 抽象层映射）
