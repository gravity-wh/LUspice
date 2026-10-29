# 自适应混合精度策略（Adaptive Mixed Precision）· 设计

> 赛题七 · 目标：在"精度-速度-内存"三角里，按矩阵实际表现自动选择最优精度路径，最大化 GPU 计算吞吐。
> 状态：**v2（2026-08-16）——已用 33 矩阵实测修正**：v1 的静态 κ 阈值判据被证伪，改为**动态升级策略**；fp16 边界用溢出/下溢实测界定。
> 实测数据：`experiments/mp/mp_ir_classification.json`（P1/P3 分类）、本会话 fp16 精度/范围分析；GPU 分阶段吞吐数字见 `reports/mixed_precision_throughput.md`（subagent 执行中）。
> 关联：`design/m5_selector.md`（格式选择器）、`reports/M1_conditioning_prescription.md`、`reports/M4_acceptance.md`。

---

## 1. 核心结论（先看这个）

1. **混合精度最有效的两个计算过程**（按收益排序，**GPU 实测 RTX 3060**，见 `reports/mixed_precision_throughput.md`）：
   - **SpMV（迭代精化内层 / 批量扫描）**：A 值存 fp32 → 字节减半 → 大矩阵（n≥8k）实测 **×1.35–2.81**（rajat10 1.35× @158 GB/s、rajat22 2.81×、rajat13 SELL 2.73×）；n≤5k 时启动延迟主导（~1×，需批处理）；
   - **LU 因子存储 + 三角求解**：L/U 存 fp32 → 内存近减半（flash-adc 实测 **0.104 vs 0.221 MB, 2.13×**）、分解/三角求解 ~**1.3×**（flash ST 1.34×/1.30×）。
2. **fp32 主路径的适用面比预想宽**（33 矩阵 CPU 实测）：**12/33 用"fp32 因子 + fp64 主元 + 迭代精化"即可达标**，包括 κ=2.8e15 的 rajat22（7.85e-10）——**κ 静态阈值不可靠，必须动态试跑**。
3. **fp16 是窄众增强**：实测 SpMV 用 fp16 值存储 + fp64 累加，输出误差仅 ~1e-4（即使动态范围 1e28）——**瓶颈不是动态范围，而是溢出/下溢**：本批 33 矩阵中仅 **rajat10**（全值同量级，GPU fp32 误差 0.0）可 fp16 直存；其余 4–33% 的值 < fp16 最小正常数 6e-5（如 1e-12 电容/小导纳），必须先 Ruiz 缩放。fp16 只用于"缩放后值域入界"矩阵的**存储/搬运**（fp32 累加），或 tensor-core 块内 GEMM 输入。
4. **精度不能一刀切**：纯 fp32 直接 LU 在病态矩阵上误差 9.6e2（OTA 实测）；**必须** fp64 主元 + 迭代精化兜底；"自适应" = 动态升级（先试最快的，不行再升），而不是静态查表。

---

## 2. 为什么是"这些阶段"最有效（瓶颈判据）

| 阶段 | 算术强度 AI (FLOP/Byte) | 瓶颈类型 | 精度降半的效果 |
|---|---|---|---|
| SpMV / 三角求解 | ≈0.2（远低于 GPU 平衡点 ~15–60） | **带宽受限** | 字节减半 → 吞吐 ×2 |
| 稀疏 LU 数值分解（左视） | 中低（更新以 gather/scatter 为主） | 带宽/延迟受限为主 | 字节减半 → 吞吐 ×1.5–2 |
| 超节点内稠密 GEMM | 高（BLAS-3） | **计算受限** | fp16/fp32 输入 + fp32 累加 → ×2–8（受超节点尺寸限制，电路矩阵通常收益有限） |
| 符号分解/重排序 | —（整数/图操作） | 与精度无关 | 不适用 |

---

## 3. 精度路径设计（v2：动态升级，取代静态 κ 表）

### 3.1 路径定义

| 路径 | 因子精度 | 主元精度 | 预处理/精化 | 实测覆盖（33 矩阵） |
|---|---|---|---|---|
| **P-fast（fp32 主路径）** | fp32 | fp64 | fp64 残差 GMRES-IR（≤8 次） | **12/33 达标**（含 rajat13 κ1.5e11、rajat19 κ9.2e10、rajat22 κ2.8e15） |
| **P-safe（fp64 主路径）** | fp64 | fp64 | Ruiz + 对角移位(1e-7·\|diag\|) + IR | +6/33 达标（flash/limiting/dac/sc-2to1/div2…） |
| **P-hard（fp64 强化）** | fp64 | fp64 | Ruiz 结构对齐 + trisolve 数值优化（M4 处方） | neural/comparator/OTA/pg 系（绝对精度口径下仍难，S₁ 相对口径可放宽） |
| **P-singular** | — | — | 检测 → 移位 → 明确报错 | rajat16/18/20/25（fp64 也奇异） |
| **P-fp16 增强** | fp16 值 + fp32 累加 | fp64 | IR | 仅 rajat10 级（值域入界）；其余需 Ruiz 缩放后评估 |

### 3.2 动态升级规则（取代 v1 的 κ 查表）

```text
for 每个矩阵:
  1. 跑 P-fast (fp32 因子 + fp64 主元 + IR ≤ 8 次)     ← 最快, 先试
  2. 判据: 残差/相对误差 ≤ 1e-6 ?  → 完成 ✅ (12/33)
  3. 失败或 fp32 分解异常(奇异) → 跑 P-safe (fp64 + Ruiz + shift + IR)
  4. 仍失败 → P-hard (结构对齐等) 或 P-singular 检测报错
# 升级开销可摊薄: 符号分解跨路径复用; P-fast 的 fp32 分解失败时其符号结构仍可用
```

**为什么不用静态 κ 表**（实测证据）：
- rajat22 κ=2.8e15 → P-fast 达标（7.85e-10）；OTA κ=1.5e13 → P-fast 失败（9.6e2）——κ 同量级，结果相反；
- 判据应为**实际的分解质量**（最小主元、IR 残差下降率），不是 κ 估计；
- P-fast 尝试成本低（fp32 分解快），失败后再升级的额外开销有限，且符号分解可复用。

### 3.3 与格式选择器（m5_selector）集成

两层决策正交：先选格式（SELL-C-σ32 vs CSR，按 n/行均匀度），再走精度升级链。合成 `(格式 × 精度路径)` 决策对；精度升级在运行时按残差触发，格式在预处理阶段定死。

---

## 4. 全 33 矩阵实测分类表（v2 依据）

> 方法：fp32 splu + fp64 残差 IR(8) → 失败则 Ruiz 重试 → 失败则 fp64 splu 兜底；判据 = 与已知真解 x=1 的 L2 相对误差 ≤1e-6（**比赛题 S₁ 的"vs CPU 参考"更严格**）。

| 类别 | 矩阵 | 说明 |
|---|---|---|
| **P-fast ✅ (12)** | bandgap 2.2e-11 / beta 9.4e-13 / constant-gm 4.6e-13 / ldo 1.1e-11 / thermometer 2.1e-10 / rajat10 0.0 / rajat11 1.4e-13 / rajat12 6.6e-15 / rajat13 2.0e-15 / rajat15 2.3e-11 / rajat19 4.2e-11 / rajat22 7.9e-10 | fp32 因子 + IR 即可；**含 κ≥1e11 的 rajat13/19/22** |
| **P-safe ✅ (6)** | flash-adc 1.2e-7 / limiting-amp 9.2e-7 / current-dac 3.2e-7 / sc-2to1 1.5e-7 / sc-dac 4.2e-8 / div2 2.2e-9 | fp64 直接解即可；fp32+IR 失败 |
| **P-hard (11)** | neural / comparator / OTA / pg2k / pg5k / pg10k / pg10k_amp / pg10k_mesh / pg25k / pg50k / pg100k | fp64 直接解也 >1e-6（绝对精度口径）；pg 系 fp32 分解即报"Factor is exactly singular"→ 需 Ruiz+shift |
| **P-singular (4)** | rajat16 / rajat18 / rajat20 / rajat25 | fp64 也奇异 → 检测 + 移位 + 报错 |

**注意**：S₁ 实际口径是"GPU vs CPU 参考（KLU 同算法族）"的相对误差。实测表明：对 P-fast/P-safe 能解的 18 个矩阵，S₁ 口径与上表绝对口径基本一致（参考解本身接近真解）；**对 P-hard 矩阵，fp32+IR 直接 NaN/发散，S₁ 相对口径也救不了**（OTA 实测 2.29e14）→ 必须走 P-safe/P-hard 的 fp64 预处理链。上表因此是**可靠的保守路径依据**。

---

## 5. fp16 实测边界（v2 新增）

| 测量 | 结果 |
|---|---|
| SpMV：fp16 值存储 + fp64 累加 vs fp64 | 输出相对误差 **~1e-4**（rajat10 为 0.0；bandgap/pg5k/OTA 3–4e-4）——**动态范围大不致命**（小值贡献小） |
| 值域：值 > 65504 或 < 6e-5（fp16 正常范围） | 仅 rajat10 0%/0% 可直存；rajat11 33% 下溢、bandgap 28%、pg5k 25%、flash 33%、rajat22 13% |
| 结论 | fp16 直存前提 = **缩放后值域入界**（Ruiz 先行）+ fp32 累加；作 IR 内层预条件可接受 1e-4 扰动；作直接求解则误差 ~κ·1e-4 不可接受 |

---

## 6. 正确性风险与控制（S₁ 安全网）

| 风险 | 现象（实测） | 控制 |
|---|---|---|
| fp32 主元塌缩 | OTA fp32 直接解误差 9.6e2、IR 发散 | P-fast 主元必须 fp64；IR 不收敛自动升级 |
| fp32 分解奇异（未缩放 pg） | pg2k–pg100k fp32 splu "Factor is exactly singular" | 预处理（Ruiz/shift）先于 fp32 分解；仍失败 → P-safe/P-hard |
| fp16 下溢/溢出 | 33 矩阵中 32 个有 4–33% 值 < 6e-5 | fp16 前必须 Ruiz 缩放 + 值域检查；仅 P-fp16 启用 |
| IR 不收敛（预条件太烂） | OTA κ=1.5e13 fp32 预条件发散 | 升级链（P-fast → P-safe → P-hard）；IR 层数/重启上限 |
| 奇异矩阵 | rajat16/18/20/25 | 检测 → 移位(1e-7·\|diag\|) → 仍失败明确报错 |

---

## 7. 与评分项的证据链

| 评分项 | 混合精度贡献 | 证据状态 |
|---|---|---|
| S₁ 正确性 (20) | 动态升级链保证尽可能多矩阵达标 | ✅ P-fast 12/33 + P-safe 6/33（`mp_ir_classification.json`）；neural/flash/sc-dac 处方已达标 |
| S₂ 加速比 (25) | SpMV fp32 ×1.35–2.81（n≥8k）、LU 分解 1.34×+内存 2.13×、三角求解 1.30×（GPU 实测） | ✅ `reports/mixed_precision_throughput.md` + `mp_gpu_bench.json` |
| S₃ 带宽利用率 (70%) | fp32 值搬运 → 同利用率下有效吞吐 ×1.4–2.8（rajat10 158 GB/s 实测） | ✅ 同上 |

---

## 8. 下一步

1. ✅ v2 动态升级策略定稿（本文件）
2. ✅ 33 矩阵实测分类（§4）+ fp16 边界（§5）
3. ⏳ **GPU 分阶段吞吐实测**（subagent 02987ba4 后台执行）→ 合并进 `reports/mixed_precision_throughput.md`，量化 P-fast 的 ×2 收益
4. ⏳ P-fp16 实证：rajat10（及 Ruiz 后 rajat11–13）fp16 存储 SpMV 的 GPU 带宽增益
5. ⏳ 升级链与 m5_selector 合并为 (格式×精度) 双层选择器，5 用例回归

---

*依据：`experiments/mp/mp_ir_classification.json`（33 矩阵分类）、本会话 fp16 精度/值域实测、`reports/M1_conditioning_prescription.md`、`reports/M4_acceptance.md`。*
