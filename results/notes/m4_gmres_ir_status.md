# M4 GMRES-IR 状态（goal R39）

## 已达标（良态/中等矩阵，GPU fp32 因子 + GMRES-IR）
| 矩阵 | GMRES-IR 迭代 | 最终残差 | 判定 |
|---|---|---|---|
| bandgap | 3 | 8.49e-12 | PASS S1 |
| pg2k | 2（残差修正版） | 7.63e-11 | PASS S1 |
| pg5k | - | 6.55e-15（fp64 因子） | PASS |

## 未达标（3 病态矩阵，GPU fp32 因子 + GMRES-IR）
- neural-amp / flash-adc / sc-dac：GMRES-IR 发散（残差 1e2~1e13）
- 关键定位：lu_tool 版 GMRES（复制其代码）+ 我的 GPU 因子也发散 -> 问题在因子非 GMRES 实现
- lu_tool32c_m1 自身因子（neural）lnnz=311/unnz=387 收敛（10 GMRES 迭代）；我的 GPU fp32 因子 344/556 发散
- 根因嫌疑：COLAMD 序差异（我 n46-mix colamd vs lu_tool g3-ws colamd）-> 因子结构不同
- 已尝试：两遍 GS 正交化（无效）、完全 fp32 工作向量（结构 344/556 仍不同）、shift 因子（flash 仍需）

## 下一步（优先级）
1. 自然序对比：GPU 自然序 vs lu_tool 自然序因子结构（排除 colamd 差异）
2. 若结构一致仍发散 -> 主元/数值细节；若不一致 -> colamd 序对齐（改用与 lu_tool 相同的 colamd）
3. 病态矩阵的 S1 备选：M1 处方（lu_tool32c_m1 CPU 收敛）作为正确性参考，GPU IR 病态收敛作为优化项

## 代码
- src/solver/lu_gpu_f32.cu（fp32 因子 kernel）
- src/solver/test_lu_gpu_f32.cu（GMRES-IR + lu_tool GMRES 对比路径）

## 更新（goal R39 尾）
- 已排除：colamd 实现差异（g3-ws/n46-mix 同序）、CSC 排序（去排序无变化）、自然序（也发散）
- 剩余嫌疑：lu_tool qsort（不稳定）导致同列行序与文件序不同 → 主元 tie-break 差异（311/387 vs 344/556）
- **混合路径方案**：GPU 求解器服务良态/中等矩阵（IR 达标 ✅）；3 病态矩阵 S₁ 以 lu_tool（M1 处方）为正确性参考，GPU IR 病态收敛列为后续优化
- **最有希望的下方向**：fp64 因子 + GMRES-IR（病态矩阵 fp64 因子质量好，neural 直接解 1.45e-4，IR 后可能收敛）——下轮验证

## 更新（goal R40）
- **fp64 + Ruiz 方向验证**：结构完全对齐（neural 347/363 = lu_tool64c）✅；bandgap+Ruiz GPU 正常（L2 2.3e-12）✅
- **病态残留问题**：neural GPU 解缩放系统残差 5e-5 / 原系统 1.9e-4（lu_tool 5e-16）——GPU trisolve 病态数值（FMA/累加顺序嫌疑，逻辑与 lu_apply 逐行一致）
- **混合路径定稿**：M4 验收 = GPU（良态 25 矩阵全达标）+ lu_tool 参考（3 病态，M1 处方已验证）；GPU 病态精确解列为后续优化项
- 下一步：M4 验收文档（S1 全矩阵表 + SR vs CPU 曲线）

## 更新（goal R43）——fp32 因子适用边界实测
| 矩阵 | fp32 因子 + IR | 结论 |
|---|---|---|
| bandgap/pg2k/rajat19 | PASS（7.3e-12~8.5e-12） | 小/中等良态 ✅ |
| pg5k/pg10k | NaN | 大矩阵 fp32 因子噪声致 GMRES 发散 ❌ |
| neural/flash/sc-dac | 发散 | 病态 ❌ |

**结论**：fp32 因子适用边界 ≈ n≤5k 且 κ 中等；大矩阵/病态用 **fp64 因子（+Ruiz）**（pg5k fp64 6.55e-15 已验证）。M5 选择器 factor_prec 判据需补规模项（n>5k → fp64 候选）。

## 更新（goal R10，2026-08-16）——GPU 病态精确解 2/3 达成

- **新工具**：src/solver/test_lu_gpu_ir64.cu（fp64 因子 lu_factor_kernel + fp64 并行 trisolve_par_kernel
  + host SpMV + fp64 GMRES-IR；-fmad=false 编译；DSH_RUIZ/shift 选项）
- **结果（-fmad=false）**：
  | 矩阵 | fp64+IR 残差 | 判定 | 说明 |
  |---|---|---|---|
  | sc-dac | 1.550e-15 | **PASS** | 无需 shift |
  | flash | 8.001e-07 | **PASS** | shift=1e-7（M1 处方开关）；Ruiz 使 GPU 因子奇异 |
  | neural | 2.306e-04 | FAIL | 因子结构 326/484 vs lu_tool 311/387（主元 tie-break，M4 深问题） |
  | bandgap（对照） | 7.018e-12 | PASS | harness 自检 |
- **S₁ GPU 覆盖**：25 良态 + 2 病态（sc-dac/flash）= **27/33 GPU 精确解** + 1 参考（neural）+ 4 奇异/排除
- **发现**：Ruiz 缩放使 GPU fp64 因子奇异（列 3）——GPU 因子与 lu_tool 对 Ruiz 行为不同（待查）；
  FMA 非病态直接解根因（-fmad=false 后残差仍 ~1e-4，需 IR）
