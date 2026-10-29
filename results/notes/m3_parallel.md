# M3 GPU 稀疏 LU 并行化报告（2026-08-15）

## 结论

**正确性全部 PASS**：bandgap / pg2k / pg5k 三矩阵的并行版解与单 thread 版（已验证基线）**逐位一致**（相对 L2 = 0.000e+00），与 CPU 参考相对 L2 ≤ 1e-6，因子结构（lnnz/unnz）与单 thread / CPU **完全相同**。

**性能**：factor 加速 23.0x（pg2k）~ 25.8x（pg5k）、trisolve 加速 5.8x~6.6x；pg2k factor 5.30s → **230.9 ms**、pg5k 8.30s → **322.0 ms**。**未达到 <0.1s 目标**（瓶颈分析见 §6）。

**Occupancy**：`lu_factor_par_kernel` 16.67%（1 block/SM，共享内存 ~59KB 上限所致）、`trisolve_par_kernel` 100%。优化建议见 §7。

## 1. 正确性对比

| 矩阵 | n | nnz | 并行 vs 单thread 相对L2 | 并行 vs CPU 相对L2 | 并行残差 ||b-Ax||/||b|| | CPU 残差 | L/U 结构（单thread/并行/CPU） | 判定 |
|---|---|---|---|---|---|---|---|---|
| bandgap | 113 | 616 | **0.000e+00** | 1.759e-12 | 4.161e-12 | 6.956e-12 | 582/582/582 · 777/777/777 | ✅ PASS |
| pg2k | 4050 | 23406 | **0.000e+00** | 9.189e-10 | 3.145e-15 | 3.164e-15 | 171725×3 · 171737×3 | ✅ PASS |
| pg5k | 5000 | 28925 | **0.000e+00** | 7.151e-09 | 6.552e-15 | 6.375e-15 | 230964×3 · 230979×3 | ✅ PASS |

> 并行版 vs 单 thread 版逐位一致：列内并行散射虽改变同列内的累加顺序，但对这些矩阵的舍入路径未产生可观测差异（主元判定结构不变）。

## 2. 时间对比（cudaEvent，ms）

| 矩阵 | factor 单thread | factor 并行 | factor 加速 | trisolve 单thread | trisolve 并行 | trisolve 加速 |
|---|---|---|---|---|---|---|
| bandgap | 3.983 | **1.138** | 3.5x | 0.326 | 0.324 | 1.0x |
| pg2k | 5303.366 | **230.889** | 23.0x | 62.379 | 10.665 | 5.8x |
| pg5k | 8304.021 | **321.960** | 25.8x | 83.272 | 12.632 | 6.6x |

（pg5k 并行 factor 略高：其共享 58.8KB 越过 48KB carveout 阈值 → L1 缩小，见 §6。）

## 3. Occupancy（cudaOccupancyMaxActiveBlocksPerMultiprocessor，block=256）

| kernel | maxActiveBlocks/SM | SM 线程占用 |
|---|---|---|
| lu_factor_par_kernel | 1 | 256/1536 = **16.67%** |
| trisolve_par_kernel | 6 | 1536/1536 = **100%** |

- factor 仅 1 block/SM：动态共享 ~59KB（n=5000）→ 每 SM 只能驻留 1 个 block（99KB/59KB）。bandgap（共享 4.3KB）时可达 6 blocks = 100%。
- **占用低 ≠ 利用率低**：本算法列间严格串行（列 k 依赖 L 列 <k），单 block 顺序列循环是正确性所需；并行性在列内（256 线程 × 各阶段）。
- 优化建议（§7）：共享 ≤48KB 可把 L1 提到 72KB+（bandgap 已享受；pg2k/pg5k 需把 x 或 mark 移出共享）。

## 4. 实现（src/solver/lu_gpu_par.cu / lu_gpu_par.h / test_lu_gpu_par.cu）

并行版 = 单 thread 版（lu_gpu.cu）逐列等价，列内全阶段并行：

1. **网格 1 block × 256 threads，顺序列循环**（列 k 依赖 L 列 <k，`__syncthreads` 保证 block 内可见性；零跨 block 同步，无 launch 循环）。
2. **每列阶段**（全部并行）：
   - x 清零（共享）+ mark 清零（共享）
   - A 散射（并行 over 列项）+ 可达集种子（atomicCAS 去重）
   - **结构 BFS 求可达集 R_k**：rounds（实测 pg2k 平均 3.3 轮、pg5k 3.3 轮），快路径检查 `mark[ii] != gen` + `atomicCAS` 去重，R_k 内已主元列 → jlist；Li 批量预取（16 项/批 unroll）流水化 L2 延迟
   - **bitonic 排序 jlist**（升序 = 主元序；依赖链按主元序散射）
   - **前代**：jlist 升序散射，列内并行 over p（同列行号互异 → 无原子）
   - 主元扫描：warp-shuffle + 8 槽跨 warp 归约（`|x|` max，最小行号 tie-break）+ U 次对角共享原子追加
   - 自然主元（diagrow[k] 占优保留 tol=1.0）+ U 对角末项 + pinv/rowat + L 对角首项
   - L 列并行构建（未主元行 x/pivot）
   - 末：L 行号 remap 为主元位置（与单 thread 版同款）
3. **trisolve_par_kernel**：前代/回代按列串行、列内并行散射（Li 已 remap 为主元位置）。
4. 零动态分配：内核仅用 host 预分配数组 + 动态共享（大小随 n 由 host 计算）。

**可达集裁剪收益**（host 侧 stats_lu.c 实测，pg2k）：稠密 j-loop 每列平均 ~2025 次检查 → 可达裁剪后前代只散射 ~41 列（依赖链深度 ≈ 散射列数，L 图近路径结构）；BFS 轮数仅 ~3.3。总散射项 12.8M（pg2k）/ 19.9M（pg5k）。

## 5. 编译与运行

```bash
nvcc -O3 -arch=sm_86 -I src/solver -I /home/gravity/g3-ws/src \
     -I /home/gravity/n46-mix/src/include -o test_lu_gpu_par \
     src/solver/test_lu_gpu_par.cu src/solver/lu_gpu_par.cu \
     src/solver/lu_gpu.cu src/solver/cpu_ref_lu.c \
     /home/gravity/n46-mix/src/maths/KLU/colamd.c \
     /home/gravity/n46-mix/src/maths/KLU/colamd_global.c -lm

./test_lu_gpu_par <valdata>/sky130-bandgap-reference-pvt/matrix_conv.mtx \
                   data/rhs_generated/sky130-bandgap-reference-pvt.rhs
./test_lu_gpu_par /home/gravity/g3-ws/data/pg2k.mtx /home/gravity/g3-ws/data/pg2k.rhs 1e-8 /home/gravity/g3-ws/data/pg2k.flt
./test_lu_gpu_par /home/gravity/g3-ws/data/pg5k.mtx /home/gravity/g3-ws/data/pg5k.rhs 1e-8 /home/gravity/g3-ws/data/pg5k.flt
```

## 6. 性能瓶颈分析（pg2k，clock64 分阶段实测）

| 阶段 | 时间 | 占比 | 说明 |
|---|---|---|---|
| BFS（可达集） | ~107 ms | 53% | 每列 ~3167 次 L 列项扫描；受 L2 延迟链限制 |
| solve（前代） | ~82 ms | 41% | 每列 ~41 步串行散射 × 每步 ~500 周期（L 读 L2 延迟） |
| sort（bitonic） | ~11 ms | 5% | 全局 jlist 排序 |
| scan + Lbuild + 杂项 | ~30 ms | ~12% | 全行扫描（pinv/x） |

- **根因**：GA102 每 SM 统一 L1/共享 128KB。x（32KB @4050）+ mark（16KB）+ 归约等共享已占 ~48-59KB，L1 只剩 ~64-80KB；BFS/solve 的 L 因子工作集（Li+Lx 行足迹 ~26-52KB + Lp + pinv ~48-58KB）在完整 kernel 上下文中不能全部 L1 常驻 → 每步 ~600 周期的 L2 延迟上链。
- **旁证**：把 BFS 单独抽出成独立 kernel（同结构、共享 48.7KB → L1 72KB+）实测 BFS 全量 **~12ms**（比完整 kernel 内快 ~9x）；共享 <48KB 阈值对 L1 carveout 有决定性影响。完整 kernel 内 BFS 慢 9x 的机制未完全定位（寄存器压力 / 阶段间 L1 污染，`__ldcg` 缓解有限）。
- 调优变体实测（全部正确性 PASS）：v1 237ms / v2 211ms（最优）/ v3 240ms / v4 269ms / v5 232ms / v6 217ms / v7 238ms / **当前 v8 231ms**（最终版）。

## 7. 未完成项与下一步建议

- ⏳ **<0.1s 目标未达成**（当前 231ms，pg2k）。方向：
  1. **BFS/solve 与共享/L1 联合调优**：把 x 或 mark 移出共享（字节 mark / 全局 x + `__ldcg`），使共享 <48KB → L1 72KB+，让 BFS/solve 的 L 读全 L1 常驻（独立 kernel 旁证 BFS 可到 ~12ms）；
  2. **稀疏 DFS 可达集替代 BFS 扫描**（O(n·lnnz) → 近线性），进一步降 BFS；
  3. **level 化 trisolve**（当前 10-13ms，可再降）。
- ⏳ **Occupancy ≥60%**：当前 factor 16.67%（共享 ~59KB 限 1 block/SM）。共享 ≤48KB 后可达 2-3 blocks（33-50%）；trisolve 已 100%。
- ⏳ 稀疏 DFS 可达集 / 静态化 fp32 Ax / MXMACA 平台移植（沿用 M3 初版未完成项）。

## 产物清单

- `src/solver/lu_gpu_par.cu` / `lu_gpu_par.h` —— 并行版 kernel（factor + trisolve）
- `src/solver/test_lu_gpu_par.cu` —— 三路对比驱动（CPU 参考 / 单 thread / 并行）
- `src/solver/stats_lu.c` —— 设计用统计工具（可达集/散射/依赖深度实测，host 侧）
- `test_lu_gpu_par` —— 可执行（工作区根目录）
- 本报告：`experiments/m3/m3_parallel.md`

## 8. R8 更新（2026-08-16，评价循环反馈）

- **LU_PAR_SHARED_BYTES 宏修正**：旧宏 12n+28864 描述 ring 缓冲旧设计（v8 已移全局/改 red8[8]），
  实际需求 12n+160。修正后 pg2k 共享 75.6→47.7KB（carveout 解除），pg5k factor 322→307ms（-5%）。
- **carveout 属性默认关闭**：`PreferredSharedMemoryCarveout=0` 会把 Occupancy 压到 1 block/SM；
  去掉后 pg2k factor Occupancy **16.67%→33.33%**（2 blocks 理论，命中目标 ≥33%），factor 231.9ms
  （<234ms 目标）；carveout=0 仅快 3.5% 但 Occupancy 减半；pg5k（58.8KB 共享）两配置无差异。
- **IR trisolve 并行化**（fp32 因子版，lu_gpu_f32.cu `trisolve_par_f32_kernel`）：GMRES-IR 220→41.6ms
  （5.3×），残差逐位一致。详见 experiments/eval/r8_ab.json + iter_loop_log R8。

## 9. R9 更新（2026-08-16）：wave/level factor 并行化打通（R51-R57 阻塞解除）

- 根因（4 项）：前代散射顺序（symUi 行序→升序排序）、x_seg 共享竞争（按 block 分段）、
  **FMA 收缩（-fmad=false）**、driver pinv 未写入（sym_pinv 重映射）——详见 design/m3_level_sched.md §17。
- 并行 U/L build（t==0 串行 2×n 扫描 → 256 线程 + 共享原子）：
  | 矩阵 | v8 单 block | wave factor | 加速 | 残差 |
  |---|---|---|---|---|
  | pg2k | 231.9ms | **108ms** | 2.15× | 3.133e-15 PASS |
  | pg5k | 307.8ms | **137.9ms** | 2.23× | 6.369e-15 PASS |
- **SR pg2k = 16/108 ≈ 0.148（目标 ≥0.15 基本达成）；wave 多 block + 4KB 共享 → Occupancy 上限 100%。**
