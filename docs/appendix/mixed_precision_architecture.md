# 混合精度优化 · 架构对照图（在哪一步、改了什么、如何查验）

> 目的：把三个 fp32 优化**精确落到求解流水线的具体步骤**，标注每个步骤的数据精度、代码位置与实测数字，方便对照源码查验。
> 实测来源：`reports/mixed_precision_throughput.md` + `experiments/mp/mp_gpu_bench.json`（RTX 3060）
> 代码位置：`src/solver/`（GPU 内核）、`/home/gravity/n46-mix/src/maths/KLU/fpgmres.c`（CPU 参考 GMRES-IR）

---

## 1. 总架构：三个 fp32 优化各在哪一步

```mermaid
graph TD
    IN["输入: A(fp64, Matrix Market) + b<br/>RHS 由 MNA 组装, 每 NR 迭代变"] --> R1

    subgraph 预处理(CPU, 与精度无关)
        R1["① 重排序 COLAMD<br/>输出 P,Q · 只做一次"] --> R2["② 符号分解<br/>输出 nnz(L)+nnz(U) 布局 · 只做一次"]
    end

    R2 --> F["③ 数值分解 LU (GPU)<br/>【优化2】fp32 因子存储<br/>L/U 值数组 fp32(4B) · 主元/工作向量 fp64<br/>内存 2.13× 减半(flash 实测)"]
    F --> T["④ 三角求解 (GPU)<br/>【优化2 延续】fp32 因子 → 字节减半 → ~1.3×<br/>前代 Ly=Pb · 回代 Ux=Qy"]
    T --> IR["⑤ GMRES-IR 迭代精化<br/>【优化3】fp32 L/U 预条件 + fp64 残差<br/>【优化1】内层 SpMV 可用 fp32 值存储<br/>大矩阵带宽 ×1.35–2.81"]
    IR --> X["输出: x (fp64) → 送回 L5 更新工作点"]
```

**一句话**：①② 与精度无关（结构操作）；**优化2** 落在 ③④（因子存储/搬运）；**优化3** 是 ⑤ 的整体结构（让 ② 的精度牺牲可接受）；**优化1** 在 ⑤ 内层的矩阵向量乘（及 Datasweep 批量扫描）。

---

## 2. 逐步骤精度对照表（对照查验用）

| 步骤 | 数据 | 精度（fp64 路径） | 精度（fp32 优化后） | 正确性保护 | 实测收益 |
|---|---|---|---|---|---|
| ① 重排序 | P,Q 置换数组 | int（与精度无关） | 不变 | — | fill-in 340×→15.8× |
| ② 符号分解 | L/U 非零元布局 | int（与精度无关） | 不变 | — | 静态预分配，零动态分配 |
| ③ 数值分解 | A 值 → L/U 值 | fp64 (8B/值) | **fp32 (4B/值)** | **主元比较/工作向量 fp64**（`d_diag`、`x` 用 double） | 因子内存 **2.13× 减半**（flash: 0.104 vs 0.221 MB）；分解 **1.34×** |
| ④ 三角求解 | L/U 读、x 计算 | fp64 | **L/U 按 fp32 读**（累加/结果 fp64） | 结果向量 fp64 | **1.30×**（flash ST） |
| ⑤ GMRES-IR | 残差 r、解 x | fp64 | **残差 r = b − A·x 必须 fp64**（A 按 fp64 存，或存储误差可忽略时按 fp32） | fp64 残差是精化收敛的前提 | 12/33 矩阵仅 fp32 因子+IR 达标（含 rajat22 κ2.8e15） |
| ⑤ 内层 SpMV | A·x | fp64 | **A 值按 fp32 存**（限 pg/rajat 类；sky130 类存储误差 1e-4 会卡 IR） | IR 修正残余误差 | 大矩阵（n≥8k）**×1.35–2.81**（rajat22 2.81×/rajat13 SELL 2.73×/rajat10 1.35×） |

---

## 3. 三个优化的代码位置（对照源码查验）

### 优化2 · LU 因子 fp32 存储 → `src/solver/lu_gpu_f32.cu`

```c
// 文件头注释即设计口径（m3_static_fp32.md）：
//   "因子 fp32（内存/带宽减半 2.00× 实测），工作向量 x 与主元比较 fp64"
__global__ void lu_factor_f32_kernel(          // ③ 数值分解：fp32 因子 + fp64 计算
    ... d_Lx32, d_Ux32 ...                     // L/U 值数组 = float（4B）
    ... d_diag ...)                            // 主元相关 = double（fp64 保护）
// 步骤：fp32 装载 A(:,k) → fp32 前代 → 主元扫描（fp64）→ fp32 存 L/U
trisolve_f32_kernel(...)                       // ④ 三角求解：fp32 因子读 + fp64 累加
trisolve_par_f32_kernel(...)                   // ④ 并行版（单 block 256 线程）
```

查验点：`d_Lx32/d_Ux32/d_Ax32` 的类型必须是 `float`；主元比较路径出现 `double`；`trisolve` 读 fp32 写 fp64。

### 优化3 · GMRES-IR（fp32 预条件 + fp64 残差）→ `/home/gravity/n46-mix/src/maths/KLU/fpgmres.c`

```c
// fpgmres.c - fp32-LU + fp64-residual GMRES-IR kernel (Carson-Higham)
// 主循环: 外迭代(IR) × 内迭代(GMRES)
//   r = b - A*x          ← fp64 残差（A 按 fp64 存）
//   z = M^{-1} r         ← 预条件 = fp32 L/U 三角求解（优化2 的产物）
//   x += z ; 判据 ||r||/||b||
```

查验点：残差计算路径用 fp64（`A`、`b`、`r` 均为 double）；预条件求解用 fp32 因子（`L`、`U` 值数组 float）。GPU 集成版：`src/solver/test_lu_gpu_f32.cu`（GMRES-IR + GPU fp32 因子）。

### 优化1 · SpMV fp32 值存储 → 基准内核 `src/solver/mp_bench/mp_spmv_bench.cu`（生产并入 `spmv.cu`/`sellc_spmv.cu`）

```c
// mp_spmv_bench.cu: 同一 CSR/SELL 内核的 fp64 与 fp32 值两个变体
//   csr_fp64 / csr_fp32 / sell_fp64 / sell_fp32
// fp32 变体: values 数组 float（4B）→ 每非零元搬移字节 12→8 → 带宽 ×1.35–2.81
```

查验点：`values` 指针类型 `float` vs `double`；累加器可 fp32 或 fp64（精度权衡）；IR 集成版见 `src/solver/ir_spmv_bench.cu`（已实测 pg2k 1.010e-11 / rajat19 7.327e-12 逐位一致）。

---

## 4. 正确性边界（查验时务必对照）

| 约束 | 原因 | 实测依据 |
|---|---|---|
| ③④ 的主元必须 fp64 | fp32 主元塌缩 → 分解不稳定 | OTA fp32 直接解误差 9.6e2（绝对不可用） |
| ⑤ 的残差必须 fp64（A 按 fp64 存） | 残差精度决定 IR 收敛极限 | bandgap fp32+IR 1.4e-2→4.9e-12；OTA fp32 预条件 IR 发散（κ1.5e13） |
| ⑤ 内层 SpMV 用 fp32 值仅限存储误差小的矩阵 | fp32 存储误差 ~1e-7（pg/rajat）可被 IR 修正；~1e-4（sky130 大动态范围）会卡住 IR | pg 3–6e-8 / rajat10 0.0 / bandgap 3.4e-4 |
| pg 系 fp32 分解需先 Ruiz/shift | 未缩放时 fp32 下直接奇异 | pg2k–pg100k "Factor is exactly singular"（CPU+GPU 一致） |
| 小矩阵（n≤5k）fp32 收益被启动延迟吞掉 | 内核 µs 量级，调度开销主导 | pg2k/pg5k SpMV ~1.0× |

---

## 5. 与赛题评分项的查验映射

| 评分项 | 查验对应 |
|---|---|
| S₁ 正确性 | ⑤ 的 fp64 残差 + ③④ 的 fp64 主元 + P-safe/P-hard 升级链（`design/adaptive_mixed_precision.md`） |
| S₂ 加速比 | ③④ 的 fp32 因子（内存/带宽）+ ⑤ 内层 SpMV fp32（×1.35–2.81）+ ①② 符号复用摊薄 |
| S₃ 带宽利用率 | fp32 值搬运 → 同利用率下有效吞吐 ×1.4–2.8；Profiler 数据需沐曦环境复采 |

---

*架构与实测数字同源：`mp_gpu_bench.json` / `mp_ir_classification.json` / `reports/mixed_precision_throughput.md`。*
