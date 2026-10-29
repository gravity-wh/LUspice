# M2 LU 静态化存储设计（lu_static_storage）

> 里程碑 M2：符号分解复用 → 数值分解的格式**预分配、零动态插入**
> 依赖：M1 决策（LU 主格式建议 CSC；若 M1 决策不同需调整）
> 对齐：赛题 §四.2"利用 Newton 迭代中符号分解可复用的特性"；M0 实证（g5：20386 因子 nnz 恒 9042）
> 更新：2026-08-15（goal R9 起草，M1 结果到位后回填）

---

## 1. 设计目标

| 项 | 目标 | 依据 |
|---|---|---|
| 正确性 | pg100k LU 分解 L2≤1e-6（对 fp64 参考） | S₁ 判据 |
| 内存 | 因子存储 ≤ fp64/2（fp32 Ax） | M0 实测 fp32 83.0MB vs fp64 166.1MB |
| 分配 | **数值分解期零动态分配**（结构预分配） | GPU 动态分配是性能杀手 |
| 复用 | 符号分解只做一次，跨 Newton 因子复用 | g5 实证 nnz 恒定；KLU klu_refactor 同语义 |
| 兼容 | 数据流对齐 klusmp.c SMPmatrix（COO→CSC） | 任务二接线不返工 |

## 2. 静态 CSC 数据结构

```c
typedef struct {
    int      n;          // 矩阵阶
    long     cnnz;       // 原矩阵非零元（CS 模式）
    long     lnnz;       // 符号分解后 L 非零元（精确，由符号分解给出）
    long     unnz;       // U 非零元（精确）
    long     cap;        // 分配容量 = lnnz + unnz（+ 少量余量，<1%）
    int     *Ap;         // 列指针（符号分解后含 fill-in 的列结构）
    int     *Ai;         // 行索引（预填充 fill-in 位置，数值分解只写不插）
    float   *Ax;         // fp32 数值（L/U 分块或合并存储，M2 定）
    // 主元/置换
    int     *P;          // 行置换（部分主元 tol=1.0，fpgmres 已验证）
    int     *Q;          // 列置换（COLAMD 序，CPU 侧产生）
    // 诊断（喂 S₁ 病态策略）
    double   min_diag_U; // 最小 |U 对角|（LU 质量判据）
    double   pivot_growth; // 主元增长因子
} dsh_lu_static;
```

要点：
- **Ap/Ai 由符号分解一次性填充**（含所有 fill-in 位置），数值分解期间 Ai 不变、Ax 每因子重写
- **主元比较用 fp64**（读 fp32 Ax 提升为 fp64 比较），保持稳定性（对齐 g5 建议）
- fp32 Ax 存储 = 带宽/内存减半（M0 实证 2.00×）

## 3. 符号分解接口（CPU 侧，赛题允许异构）

```
符号分解（CPU，模式变化时执行一次）：
  dsh_symbolic(A_pattern, COLAMD序) -> {Ap, Ai, lnnz, unnz, Q}
数值分解（GPU，每 Newton 因子执行）：
  dsh_numeric(静态布局, Ax_fp32) -> {L, U 数值, P, 诊断}
三角求解（GPU，每迭代执行）：
  dsh_trisolve(L, U, P, Q, b) -> x     // M4 深化：level 化
```

模式变化检测：nnz 或列结构变化 → 重做符号分解（g5 反例：S0 因子模式波动；稳态 Newton 因子模式恒定）。

## 4. 与 klusmp.c 的对接（任务二预留）

| ngspice 数据 | 静态化映射 |
|---|---|
| SMPmatrix → COO（CKTload 产物） | 首因子：COO→CSC→符号分解→静态布局；后续因子：COO 数值 → 静态布局写入 |
| LoadGmin_CSC | gmin 注入走 Ax 对角更新（不改变结构） |
| fp32gmresFactorKLU / SolveKLU | 替换为 dsh_numeric / dsh_trisolve（GMRES-IR 外壳保留） |
| KLUmatrixNodeCollapsing | 节点折叠映射表传入（模式变化时触发重符号） |

## 5. 验收口径（M2 结束）

1. pg100k：dsh_numeric 输出 vs fp64 参考 L2≤1e-6；内存 ≤83MB（fp32 因子）
2. 零动态分配：数值分解路径无 malloc/cudaMalloc（用分配器计数断言验证）
3. 符号复用：同拓扑 100 个连续因子，符号分解只执行 1 次（计时验证：符号分解时间仅首次出现）
4. 与 CPU 参考（lu_tool64）交叉验证：同一矩阵，GPU 解 vs CPU 解 L2≤1e-6
5. 难度分级矩阵子集（25 个 M0 达标矩阵）全过 S₁ 判据

## 6. 风险与回退

| 风险 | 回退 |
|---|---|
| fill-in 精确值不可得（符号分解实现复杂） | 先用 SuperLU splu 的 LNNZ 估计做容量（M0 实测比值 1.6-20.6），超配 10% |
| fp32 Ax + fp64 主元仍不稳定（κ>6.5e11） | 回退 fp64 因子（内存 2×，M3 再优化）；或引入 Ruiz 缩放（M1b 处方） |
| 静态布局限制并行度（CSC 列序约束） | M3 访存调优时评估列块（panel）重排，重排不改变结构 |

---

*联动：M1 决策表（格式选型）、M1b 处方（预处理链）、format_bench_protocol（指标口径）、cuda_compat.h（平台抽象）。*
