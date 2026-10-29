# 06 · 复现手册

> 目标：拿到这个仓库的人能在自己的机器上复现 `results/` 下的数字。
> 分三级：只读数据 → 跑单矩阵 → 跑全批。

---

## 1. 环境要求

| 项 | 要求 | 备注 |
|---|---|---|
| GPU | sm_70 及以上（fp64 单元完整） | 本项目实测用 RTX 3060（sm_86） |
| CUDA | ≥ 11.0（实测 12.6） | 沐曦栈用 `htcc`，见 §3.2 |
| 编译器 | `nvcc` / `htcc` | 需要 C++14 |
| 依赖库 | **COLAMD**（SuiteSparse 或 ngspice 内置副本） | 唯一的第三方依赖 |
| Python | 3.8+，`numpy` + `scipy` | 仅用于 Matrix Market → s0 转换与 JSON 聚合 |

---

## 2. 数据准备

**仓库不含矩阵数据**（原始 Matrix Market 共 128 MB，未纳入版本管理）。
需要自行准备两类矩阵：

| 类别 | 例子 | 来源 |
|---|---|---|
| 标准稀疏矩阵 | `rajat10/13/22`、`pg2k/5k/10k/100k` | SuiteSparse Matrix Collection |
| 电路矩阵 | `sky130-*`（bandgap / flash-adc / OTA / neural-amp …） | 用 ngspice 对小尺寸模拟电路做直流工作点提取，导出 MNA 矩阵 |

放置为 `data/matrices/<name>.mtx`（Matrix Market 格式，1-based）。

---

## 3. 编译

### 3.1 CUDA

```bash
nvcc -O3 -arch=sm_86 -std=c++14 \
     -I src -I src/third_party -I /path/to/colamd/inc \
     -o mp_lu_bench \
     src/mp/mp_lu_bench.cu \
     src/cpu_ref_lu.c \
     /path/to/colamd.c /path/to/colamd_global.c
```

三个 `-I` 都不能省：

| 路径 | 为什么需要 |
|---|---|
| `-I src` | `mp_lu_bench.cu` 会 `#include "lu_gpu.cu"` / `"lu_gpu_par.cu"` / `"lu_gpu_f32.cu"`——这三个是内核**实现**文件，不是头文件 |
| `-I src/third_party` | `cpu_ref_lu.c` 依赖 header-only 的 `splu.h` |
| `-I /path/to/colamd/inc` | `mp_lu_bench.cu` 写的是 `#include "ngspice/colamd.h"`，所以该路径下要有 `ngspice/` 这一层目录 |

如果 COLAMD 头路径不同，改 `mp_lu_bench.cu` 顶部的 `#include "ngspice/colamd.h"` 即可。

### 3.2 沐曦 MXMACA / HPCC 栈

```bash
export HPCC_PATH=/opt/hpcc
export PATH=$HPCC_PATH/bin:$HPCC_PATH/htgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=$HPCC_PATH/lib:$LD_LIBRARY_PATH

$HPCC_PATH/htgpu_llvm/bin/htcc -x hpcc -offload-arch native -std=c++17 -O3 \
    -I$HPCC_PATH/include -I src \
    -o mp_lu_bench src/mp/mp_lu_bench.cu src/cpu_ref_lu.c \
    -L$HPCC_PATH/lib -Wl,-rpath,$HPCC_PATH/lib -lhcruntime
```

注意两点（实测踩过的坑）：
- 编译器在 `$HPCC_PATH/htgpu_llvm/bin/htcc`，**不在** `$HPCC_PATH/bin/`；
- 若同时要用 `cu*` 命名的 CUDA 源文件，需走 `$HPCC_PATH/tools/cu-bridge/bin/cucc`，
  直接用 `htcc` 编 `.cu` 会因缺 `__wcuda_version_internal__` 失败。

---

## 4. 运行

### 4.1 单矩阵

```bash
# 先转成 s0 格式（见 §4.3）
./mp_lu_bench /tmp/s0/rajat22.mtx /tmp/s0/rajat22.rhs 1e-8 1 15000 3
#              <mtx>            <rhs>            pin flt max_n_st reps
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `pin` | 对角扰动（电源网格浮动节点锚定） | 0 |
| `flt` | 是否走 fp32 路径 | 0 |
| `max_n_st` | 单线程因子允许的最大阶（超过则跳过记 n/a） | 15000 |
| `reps` | 计时重复次数（取 min） | 3 |

环境变量 `DSH_ST_REPS=n` 覆盖单线程因子的重复次数。

### 4.2 全批

```bash
bash scripts/run_gpu_bench.sh
```

脚本做三件事：① `data/matrices/*.mtx` → s0（`/tmp/mp_s0`）；② 逐个跑基准；
③ 聚合为 `results/mp_gpu_bench.json`。矩阵清单写在脚本内的 `names` 列表里，按需改。

### 4.3 s0 格式

纯文本，两段：

```
n 4050
A 1 1 1.234e-05
A 2 1 -3.1e-06
...
```

```
n 4050
RHS 1 0.0
RHS 2 1.5e-03
...
```

`A` 行 = 三元组（1-based 行、列、值），`RHS` 行 = 右端项。转换代码见
`scripts/run_gpu_bench.sh` 内嵌的 Python。

---

## 5. 计时纪律（重要）

**这台机器噪声很大，单次测量不可信。**

实测：同一二进制连跑 7 次，`Matrix load` 在 4.06 – 10.79 s 之间跳（max/min = **2.66**）。
在这种噪声下，逐例对比会得到"46 快 54 慢"的随机结果。

因此本项目的数字都遵守：

1. **warmup 后再计时**（至少 1 次预热）；
2. **多次取 min**（内核级 `cudaEvent`，取最优而非平均——min 更接近真实能力）；
3. 需要跨版本对比时，**基线版与优化版同时段交替跑**（ABABAB × 7 轮），看配对中位差；
4. 中大规模矩阵**多轮取中位**，单次结果不引用。

---

## 6. 逐项对表（复现后自查）

| 仓库里的数字 | 复现命令 | 对应 JSON 字段 |
|---|---|---|
| fp32/fp64 因子内存比 2.13× | §4.1 跑 flash-adc | `mem_fp32_mb` / `mem_fp64_mb` |
| 分解 1.34× / 三角求解 1.30× | §4.1 跑 flash-adc | `speedup_factor_st` / `speedup_trisolve_st` |
| SpMV 大矩阵 1.35–2.81× | §4.1 跑 rajat10/13/22 | `speedup_csr_fp32` / `speedup_sell_fp32` |
| 33 矩阵残差 ≤1e-6 | 需跑 GMRES-IR 全链 | `results/s1_table.json` 的 `GPU_res` |
| SR 0.15–0.22 | wave/level 版本 | `results/sr_curve.json` |

---

## 7. 已知会遇到的失败

| 现象 | 原因 | 处理 |
|---|---|---|
| pg 系 fp32 报 "Factor is exactly singular" | 未做 Ruiz/shift 预处理 | 先缩放再分解，或走 P-safe |
| 小矩阵（n≤5k）fp32 无加速 | 启动延迟主导 | 正常，不是 bug；需批处理才划算 |
| 病态矩阵 fp32+IR 发散 | 预条件子失效 | 升级到 P-safe（fp64 + shift） |
| GPU 三角求解病态精度 1.9e-4 | FMA / 累加顺序差异 | 已知未解项，见 README §6 |

---

*数据字典见 [../results/README.md](../results/README.md)。*
