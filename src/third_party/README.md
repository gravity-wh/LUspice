# third_party

## splu.h

**header-only** 稀疏左视 LU（`P·A(:,q) = L·U`），只依赖 `<stdlib.h>` / `<string.h>` / `<math.h>`。
直接 `#include`，无需单独编译。

- 内容：`cs_lu` / `cs_spsolve` / `cs_dfs` / `cs_reach` 的模板化移植，用 `LU_FLOAT` 宏切换
  `float` / `double`，因此同一份代码可生成 fp32 与 fp64 两个版本。
- 布局约定：L 是 CSC、单位对角存在每列**首个**位置、行号已预置换；U 是 CSC、对角存在每列**末位**。
- 用途：`src/cpu_ref_lu.c` 用它做 CPU 参考解，作为 GPU 结果的正确性对照。

**来源与许可**：改编自 Tim Davis 的 CSparse（SuiteSparse 5.10.1），原库为 LGPL-2.1+。
本目录下的文件保留该许可约束；引用本仓库其余 MIT 代码时请注意这一例外。

**为什么收进仓库**：`cpu_ref_lu.c` 是正确性对照的必需依赖，缺了它仓库无法独立构建。
体积只有 7.6 KB，纳入版本管理比让使用者去外部仓库找更可靠。

---

COLAMD 未随仓库分发（体积与许可都不划算），需自备 SuiteSparse 或 ngspice 内置副本，
详见 `docs/06_reproduce.md` §3。
