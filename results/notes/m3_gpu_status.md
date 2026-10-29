# M3 GPU LU 调试状态（goal R36）

## 已确认事实
- GPU/CPU L 结构：列 0-4 完全一致（含值）；**列 5 起主元不同**（同一 fillin 值在 GPU 挂行 10、CPU 挂行 28——主元被不同列选中，系统性差 18 列）
- U 结构：列 10 起不同（对角符号反，级联自 L）
- dlu_dfs/dlu_reach/dlu_spsolve 与 splu.h 逐行对比一致
- 已排除：x 残留（每列全清无效）、CSC 行序（排序后无变化）、FMA（-fmad=false 无效）
- lnnz 总数一致（582）但列内分布不同

## 下一步方向（优先级）
1. **host/device 双跑同一 dlu_* 代码**（__host__ __device__ 宏）——host 跑若与 splu.h 一致 → device 环境问题（内存/编译）；不一致 → 移植逻辑 bug
2. 打印列 5 的 dlu_spsolve 输入（A 列 5 + L 列 0-4）与输出 x，与 CPU 单步对比
3. 检查 device 内存布局（Lp/Li/Lx 预分配数组的跨列污染——cap 内无越界写但读可能错位）
4. 备选：放弃 DFS 移植，GPU 用稠密 LU（n≤989 的 valdata 矩阵完全可行，pg 系后续再并行稀疏）

## 关键洞察
- 列 0-4 一致 → 核心算法对，bug 是"第 5 列才触发的边界条件"
- 主元系统性差 18 列 → 可达集或 x 值在某处系统性偏置

## 编译
nvcc -O3 -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src -I /home/gravity/n46-mix/src/include -o test_lu_gpu src/solver/test_lu_gpu.cu /home/gravity/n46-mix/src/maths/KLU/colamd.c /home/gravity/n46-mix/src/maths/KLU/colamd_global.c
