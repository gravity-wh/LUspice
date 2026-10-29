# 05 · 静态化存储与符号复用

> 两个目的：① 符号分解只做一次（Newton 迭代中矩阵模式恒定）；
> ② 数值分解期**零动态分配**——GPU 上的 malloc 是性能杀手，必须彻底消掉。
> 这两点与精度无关，但决定了混合精度收益能不能落地。

---

## 1. 设计目标

| 项 | 目标 | 依据 |
|---|---|---|
| 正确性 | pg100k 分解 L2 ≤ 1e-6（对 fp64 参考） | 判据口径 |
| 内存 | 因子存储 ≤ fp64/2 | 实测 fp32 83.0 MB vs fp64 166.1 MB |
| 分配 | 数值分解期**零动态分配** | 结构预分配，写不插 |
| 复用 | 符号分解只做一次，跨 Newton 因子复用 | 实测 nnz 恒定 9042 |
| 兼容 | 数据流对齐 KLU 的 `SMPmatrix`（COO→CSC） | 与现有求解器接线不返工 |

---

## 2. 静态 CSC 结构

```c
typedef struct {
    int      n;            // 矩阵阶
    long     cnnz;         // 原矩阵非零元
    long     lnnz;         // 符号分解后 L 非零元（精确）
    long     unnz;         // 符号分解后 U 非零元（精确）
    long     cap;          // 分配容量 = lnnz + unnz（+ <1% 余量）

    int     *Ap;           // 列指针（含 fill-in 的列结构，符号分解后固定）
    int     *Ai;           // 行索引（fill-in 位置已预留，数值分解只写不插）
    float   *Ax;           // fp32 数值（L/U 存储，每因子重写）

    int     *P;            // 行置换（部分主元 tol=1.0）
    int     *Q;            // 列置换（COLAMD 序，CPU 侧产生）

    double   min_diag_U;   // 最小 |U 对角|（分解质量判据）
    double   pivot_growth; // 主元增长因子
} dsh_lu_static;
```

**三个要点**：

1. **`Ap/Ai` 由符号分解一次性填满**（含全部 fill-in 位置）。数值分解期间 `Ai` 不变，`Ax` 每因子重写。
   这是"零动态插入"的结构基础。
2. **主元比较用 fp64**（读 fp32 的 `Ax` 提升为 double 再比），保持稳定性。
3. **`Ax` 用 float** —— 这是混合精度的落点，内存直接减半。

接口定义见 `src/lu_static.h`（`dsh_symbolic` / `dsh_numeric` / `dsh_trisolve` / `dsh_gmres_ir`）。

---

## 3. 三段式流程

```
符号分解（CPU，模式变化时一次）
    dsh_symbolic(A_pattern, COLAMD序) → {Ap, Ai, lnnz, unnz, Q}
                    ↓
数值分解（GPU，每 Newton 因子）
    dsh_numeric(静态布局, Ax_fp32) → {L/U 数值, P, 诊断}
                    ↓
三角求解（GPU，每 IR 迭代）
    dsh_trisolve(L, U, P, Q, b) → x
```

**模式变化检测**：nnz 或列结构变化才重做符号分解。稳态 Newton 迭代中模式恒定
（实测 20386 个因子 nnz 恒为 9042），所以符号分解的摊销成本趋近于零。
反例：S0 阶段的因子模式会波动，此时必须重新符号分解——不能无条件复用。

---

## 4. 与 KLU 侧的对接

| 现有数据 / 函数 | 静态化映射 |
|---|---|
| `SMPmatrix` → COO（load 阶段产物） | 首因子：COO→CSC→符号分解→静态布局；后续因子：只写数值 |
| `LoadGmin_CSC` | gmin 注入走 `Ax` 对角更新，**不改变结构** |
| `fp32gmresFactorKLU` / `SolveKLU` | 替换为 `dsh_numeric` / `dsh_trisolve`，IR 外壳保留 |
| `KLUmatrixNodeCollapsing` | 节点折叠映射表传入；模式变化时触发重符号分解 |

---

## 5. 零动态分配的验证方法

断言不是"看代码没写 malloc"，而是**运行时计数**：

```c
// 调试钩子：替换 cudaMalloc 为带计数的包装
size_t g_malloc_calls = 0;
#define cudaMalloc(p, s) (g_malloc_calls++, cudaMalloc(p, s))
// dsh_numeric 返回后断言
assert(g_malloc_calls == 0);
```

验收时在 `dsh_numeric` 前后取计数差，必须为 0。

---

## 6. 符号复用计数

```c
long dsh_symbolic_count(void);   // 进程级累计符号分解次数
```

实测：对同一矩阵连续两次 `dsh_numeric`，`dsh_symbolic_count()` = **1**。
这是"符号复用生效"的直接证据。

---

## 7. 与后端抽象的配合

`lu_static.c` 只依赖 `src/cuda_compat.h` 的 `DSH_*` 接口，不直接调 `cudaMalloc` 等符号：

| 抽象 | 作用 |
|---|---|
| `DSH_MALLOC` / `DSH_FREE` | 统一显存分配，便于插桩计数 |
| `DSH_MEMCPY` / `DSH_MEMSET` | 统一拷贝方向枚举 |
| `DSH_CHECK` | 错误检查（CUDA 与目标后端错误码映射） |

这样同一份 `lu_static.c` 可在 CUDA 与沐曦 HPCC 栈上编译，不需要条件编译分支。

---

## 8. 验收状态

| 项 | 状态 |
|---|---|
| pg100k 分解 L2 ≤ 1e-6；内存 ≤ 83 MB | ✅ |
| 零动态分配（`cudaMalloc` 计数 = 0） | ✅ |
| 符号复用计数（两次 numeric / 一次 symbolic） | ✅ |
| Occupancy 报告（每个 LU kernel） | ⏳ 单 block 内核 1.04%，是当前性能主瓶颈 |

---

*上游设计稿：[appendix/lu_static_storage.md](appendix/lu_static_storage.md) ·
接口定义：[../src/lu_static.h](../src/lu_static.h)*
