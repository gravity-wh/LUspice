# 04 · 自适应精度升级链

> 核心判断：**精度路径不能靠 κ 查表决定，只能"先试最快的，不行再升"。**
> 本文件给出升级链的规则、33 矩阵的实测分类，以及每条路径的失败模式。

---

## 1. 精度路径定义

| 路径 | 因子精度 | 主元 | 预处理 / 精化 | 实测覆盖 |
|---|---|---|---|---|
| **P-fast** | fp32 | fp64 | fp64 残差 GMRES-IR（≤8 次） | **12/33** 达标 |
| **P-safe** | fp64 | fp64 | Ruiz + 对角移位(1e-7·\|diag\|) + IR | **+6/33** 达标 |
| **P-hard** | fp64 | fp64 | Ruiz 结构对齐 + trisolve 数值优化 | 11 个矩阵，绝对精度口径下仍 >1e-6 |
| **P-singular** | — | — | 检测 → 移位 → 明确报错 | 4 个（rajat16/18/20/25） |
| **P-fp16** | fp16 值 + fp32 累加 | fp64 | Ruiz 缩放 + IR | 仅 rajat10 级（值域天然入界） |

---

## 2. 升级规则

```text
for 每个矩阵:
  1. 走 P-fast（fp32 因子 + fp64 主元 + IR ≤ 8 次）        ← 最快，先试
  2. 判据 ‖r‖/‖b‖ ≤ 1e-6 ?
       是 → 完成 ✅（12/33）
  3. 失败 / fp32 分解报奇异 → 走 P-safe（fp64 + Ruiz + shift + IR）
       是 → 完成 ✅（+6/33）
  4. 仍失败 → P-hard（结构对齐）或 P-singular（检测报错）
```

**成本分析**：P-fast 尝试很便宜（fp32 分解本身快 1.3×，且符号分解在路径间复用）。
失败后的额外开销只是一次 fp64 分解 + 几轮 IR，比"一律 fp64"更划算。

---

## 3. 为什么不能用静态 κ 表（关键实证）

| 矩阵 | κ | P-fast 结果 | 结论 |
|---|---|---|---|
| rajat22 | **2.8e15** | 达标 7.85e-10 | 极端病态却成功 |
| rajat13 | 1.5e11 | 达标 2.0e-15 | 成功 |
| OTA-5T | 1.5e13 | 失败 9.6e2 | 相对"更良态"却失败 |

κ 差两个数量级，结果却相反。原因是 κ 只是**上界估计**，真正决定 IR 收敛的是
**分解后的实际质量**（最小主元、IR 残差下降率、预条件子的谱分布）。
所以判据必须是运行时可观测的量，不是预处理的估计值。

---

## 4. 33 矩阵实测分类

> 方法：fp32 splu + fp64 残差 IR(8) → 失败则 Ruiz 重试 → 失败则 fp64 splu 兜底。
> 判据：与已知真解 x=1 的 L2 相对误差 ≤ 1e-6（**比比赛口径更严格的绝对口径**）。

### P-fast ✅（12）

| 矩阵 | 残差 | 备注 |
|---|---|---|
| bandgap | 2.2e-11 | |
| beta-multiplier | 9.4e-13 | |
| constant-gm | 4.6e-13 | |
| ldo-ota5 | 1.1e-11 | |
| thermometer-trim | 2.1e-10 | |
| rajat10 | 0.0 | 全值同量级，fp16 候选 |
| rajat11 | 1.4e-13 | |
| rajat12 | 6.6e-15 | |
| rajat13 | 2.0e-15 | κ 1.5e11 仍成功 |
| rajat15 | 2.3e-11 | |
| rajat19 | 4.2e-11 | |
| rajat22 | 7.9e-10 | **κ 2.8e15** |

### P-safe ✅（6）

| 矩阵 | 残差 | 说明 |
|---|---|---|
| flash-adc | 1.2e-7 | fp32+IR 失败，fp64 直解可 |
| limiting-amplifier | 9.2e-7 | |
| current-dac | 3.2e-7 | |
| sc-2to1 | 1.5e-7 | |
| sc-dac-6bit | 4.2e-8 | |
| div2 | 2.2e-9 | |

### P-hard（11）· 绝对口径下仍 >1e-6

`neural` / `comparator` / `OTA` / `pg2k` / `pg5k` / `pg10k` / `pg10k_amp` /
`pg10k_mesh` / `pg25k` / `pg50k` / `pg100k`

pg 系的特有问题：**fp32 分解直接报 "Factor is exactly singular"**（CPU 与 GPU 一致），
必须先 Ruiz 缩放 + 对角移位才能进 fp32 路径。

> **口径差异说明**：比赛的实际口径是"GPU 解 vs CPU 参考解"的相对误差。对 P-fast/P-safe 能解的
> 18 个矩阵，两种口径基本一致（参考解本身接近真解）；对 P-hard 矩阵，fp32+IR 会直接 NaN/发散，
> 相对口径也救不回来（OTA 实测 2.29e14）→ 必须走 fp64 预处理链。
> 因此本表是**保守路径依据**，不是悲观结论。

### P-singular（4）

`rajat16` / `rajat18` / `rajat20` / `rajat25` — fp64 也奇异，只能检测 + 移位 + 报错。

---

## 5. 失败模式与控制手段

| 风险 | 实测现象 | 控制 |
|---|---|---|
| fp32 主元塌缩 | OTA fp32 直接解误差 **9.6e2**、IR 发散 | 主元必须 fp64；IR 不收敛自动升级 |
| fp32 分解奇异（未缩放 pg） | pg2k–pg100k 报 "Factor is exactly singular" | 预处理（Ruiz/shift）先于 fp32 分解 |
| fp16 下溢/溢出 | 32/33 矩阵有 4–33% 值 < 6e-5 | Ruiz 缩放 + 值域检查；仅 P-fp16 启用 |
| IR 不收敛 | OTA κ1.5e13 fp32 预条件发散 | 升级链 + 迭代上限 |
| 真奇异 | rajat16/18/20/25 | 检测 → 移位 → 明确报错 |

---

## 6. 与格式选择器的正交关系

两层决策彼此独立：

```
第一层（预处理阶段定死）：格式选择
    n ≥ 10k 且行均匀 → SELL-C-σ32；否则 CSR
第二层（运行时按残差触发）：精度路径
    P-fast → P-safe → P-hard → P-singular
```

合成 `(格式 × 精度路径)` 决策对。格式在预处理定，精度在求解中升。

---

## 7. 状态

- ✅ v2 动态升级策略定稿
- ✅ 33 矩阵实测分类（§4）、fp16 边界（`02_precision_budget.md` §5）
- ⏳ P-fp16 实证：rajat10（及 Ruiz 后的 rajat11–13）fp16 存储 SpMV 的 GPU 带宽增益
- ⏳ 升级链与格式选择器合并为统一双层选择器

---

*上游设计稿：[appendix/adaptive_mixed_precision.md](appendix/adaptive_mixed_precision.md) ·
分类数据：[../results/mp_ir_classification.json](../results/mp_ir_classification.json)*
