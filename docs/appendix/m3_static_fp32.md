# M3 静态化与 fp32 因子设计（m3_static_fp32）

> M3 两大任务衔接：① GPU LU 移植（线程 eac5d43a 进行中）；② 静态化 fp32 Ax + 访存调优
> 本文件：fp32 因子存储的精度预算 + 静态化结构设计（基于 M0/M1/M2 全部实证）
> 更新：2026-08-15（goal R31）

---

## 1. 精度预算（fp32 因子 + fp64 主元 + GMRES-IR）

**实证链**（全部本地实测）：
- fp32 因子内存 = fp64 的 **2.00× 减半**（g3：pg100k 166.1→83.0MB，逐位复现）
- fp32 因子 + GMRES-IR：pg100k 残差 4.6e-6 → **7.1e-10**（M0 baseline）
- M1 处方：病态矩阵（κ≤3e14）增强 IR 参数（outer≥7/inner≥100/rtol1e-10）可达 **4.2e-11~9e-10**
- fp32 边界：κ_ruiz ≤6.5e11 收敛；flash-adc（κ 4.6e12）需 fp64 回退

**精度设计**：
| 环节 | 精度 | 理由 |
|---|---|---|
| 因子存储 Ax（L/U） | **fp32** | 内存/带宽减半（S₂/S₃ 核心） |
| 主元比较（部分主元） | **fp64**（读 fp32 提升） | 保持分解稳定性（g5 建议） |
| 工作向量 x（分解中） | fp64 | 累积误差控制 |
| GMRES-IR 累加 | **fp64**（SpMV fp32 数据+fp64 累加，Kahan 可选） | 残差到 1e-10 量级 |
| 解输出 | fp64 | S₁ 判据口径 |

**S₁ 判据满足**：目标残差 ≤1e-6，实测 1e-10 量级 → 余量 4 个数量级（含 κ 波动缓冲）。

## 2. 静态化结构（M2 骨架的 GPU 化）

```
符号分解（CPU，模式变化时一次）：COLAMD → 完整 L/U 结构（含 fillin）
  → GPU 预分配：Ap/Ai（int32）+ Ax（float32，L 区 + U 区）
数值分解（GPU，每 Newton 因子）：
  → 只写 Ax（fp32）+ P（行置换）——零动态分配（无 malloc/cudaMalloc）
三角求解（GPU，每迭代）：前代+回代（level 化，M3 后期）
GMRES-IR（GPU，M4）：SpMV 走 SELL-C-σ32（L 级）/ CSR（长尾）——M1 已定
```

**零动态分配断言**：dsh_numeric_gpu 内核路径内无 malloc/cudaMalloc——用 `cudaMalloc` 调用点计数（调试钩子）验证 = 0。

## 3. 符号复用语义（对齐 KLU klu_refactor）

- 稳态 Newton 因子：模式恒定（g5 实证 nnz 恒 9042）→ 符号分解只做一次，数值分解复用结构
- 模式变化检测：nnz 或列结构变化 → 重做符号分解（g5 反例：S0 因子模式波动）
- M2 CPU 版已实证：两次 numeric 仅 1 次 symbolic（count=1）

## 4. 访存调优目标（S₃）

| 指标 | 目标 | 实测现状 | 路径 |
|---|---|---|---|
| Occupancy | ≥60% | SpMV v0 = 100%（工具 occ_check.cu 已验证） | LU 内核按 block/共享内存配置 |
| 带宽利用率 | ≥70% | SpMV SELL-C-σ32 = 78.3%（已达标） | LU/三角求解按合并访问设计 |
| 因子内存 | ≤ fp64/2 | 83MB（fp32，M0 实测） | 静态化后保持 |

## 5. M3 验收清单

- [ ] GPU 数值分解：pg5k/pg2k/bandgap GPU vs CPU/参考 L2≤1e-6（线程 eac5d43a）
- [ ] fp32 Ax 静态化：同矩阵 fp32 因子 vs fp64 因子残差对比（≤1e-6，IR 后 ≤1e-8）
- [ ] 零动态分配断言（numeric 内无 cudaMalloc）
- [ ] 符号复用计数（GPU 版）
- [ ] Occupancy 报告（每个 LU kernel）
- [ ] 内存对比（fp32 静态 vs fp64 动态）≤1/2

---

*衔接：M2 CPU 版（已验证）→ M3 GPU 移植+静态化（本文件）→ M4 IR 集成（SpMV 格式已定）→ M5 自适应选择器。*
