# results · 数据字典

所有数字都不是手写的，全部来自这里的 JSON。想质疑哪个结论，就打开对应的文件。

---

## 1. 主数据

| 文件 | 内容 | 对应结论 |
|---|---|---|
| `mp_gpu_bench.json` | 每矩阵一行：fp64/fp32 的分解耗时、三角求解耗时、因子内存、SpMV 带宽、fp32 相对误差 | README §3.1 / §3.2 |
| `mp_ir_classification.json` | 33 矩阵的精度路径分类（P-fast / P-safe / P-hard / P-singular）与残差 | `docs/04_adaptive_policy.md` §4 |
| `s1_table.json` | 33 矩阵的 GPU 残差、CPU 基线耗时、PASS/REF/FAIL 判定 | README §3.3 |
| `sr_curve.json` | wave/level 并行版的 SR = T_cpu / T_gpu | README §3.4 |
| `cpu_baseline.json` | KLU 单线程基线（33 矩阵 T_cpu，S₂ 分母） | 同上 |

### `mp_gpu_bench.json` 字段

每条记录 `{matrix, lu, spmv}`，两个子对象：

**`lu`**

| 字段 | 含义 |
|---|---|
| `n` / `nnz` | 阶数 / 原矩阵非零元 |
| `lnnz_cpu` / `unnz_cpu` | CPU 参考分解得到的 L/U 非零元 |
| `lnnz64` / `unnz64` | GPU fp64 分解的 L/U 非零元（与 CPU 一致性校验） |
| `lnnz32` / `unnz32` | GPU fp32 分解的非零元；**-1 = 未取到**（通常为 fp32 分解报奇异或超时） |
| `mem_fp64_mb` / `mem_fp32_mb` | 因子内存（含索引）；比值即 README 的 2.13× |
| `t_factor_st64_ms` / `t_factor_st32_ms` | 单线程分解耗时；-1 = 未测 |
| `t_factor_par64_ms` | 单 block 并行分解耗时 |
| `t_trisolve_st64_ms` / `t_trisolve_st32_ms` | 三角求解耗时 |
| `speedup_factor_st` / `speedup_trisolve_st` / `speedup_trisolve_par` | fp64→fp32 加速比；-1 = 缺数据无法算 |

**`spmv`**

| 字段 | 含义 |
|---|---|
| `t_csr_fp64_ms` / `t_csr_fp32_ms` | CSR 格式耗时 |
| `t_sell_fp64_ms` / `t_sell_fp32_ms` | SELL-C-σ 格式耗时 |
| `bw_csr_fp64_gbs` / `bw_sell_fp64_gbs` | fp64 带宽（GB/s） |
| `speedup_csr_fp32` / `speedup_sell_fp32` | fp32 值存储的加速比 |
| `rel_err_fp32` | fp32 值存储解的相对误差 |
| `sell_stored` / `sell_pad_pct` | SELL 实际存储量 / 填充率 |

**`-1` 就是 `-1`，是"没取到"，不是"零"或"无限"。** 引用时请标注 n/a。

---

## 2. 目标平台实测（`x201_20261029/`）

| 文件 | 内容 |
|---|---|
| `x201_20261029/ANALYSIS.md` | 沐曦 X201 上 pg2k 的 8 次运行、统计量、与 3060 的对照、失败模式 |
| `x201_20261029/raw/*.jsonl` | 每行一次有效运行（原始 JSON + 挂钟 + 目标卡占用快照） |
| `x201_20261029/raw/*.log` | 人读日志，含失败轮次的完整报错 |

**这是本仓库唯一一组在目标硬件（沐曦 X201）上跑出来的数据**，其余都是 RTX 3060。
引用前务必看 `docs/07_platform_notes.md` 的适用边界——两平台结论不一致。

字段与 `mp_gpu_bench.json` 的 `lu` 子对象同名同义，可直接对照。

---

## 3. 分项实验（`raw/`）

| 目录 | 内容 |
|---|---|
| `raw/batch_lu/` | 批量 LU（多 RHS 一次分解，摊薄启动开销） |
| `raw/fp16/` | fp16 值存储可行性：值域统计 `fp16_cpu_stats.json`、结果 `fp16_results.json`、分析脚本 |
| `raw/fusion/` | 内核融合基准（融合三角求解 + SpMV 的收益） |
| `raw/multiblock/` | 多 block 分解调度 |
| `raw/selector2/` | 格式选择器交叉校验（含 `crosscheck.py`） |
| `raw/sram/` | 共享内存三角求解（bank 冲突与分块） |

这些是探索性实验，结论未全部进入主文档；保留原始数据以便追溯。

---

## 4. 报告与实验记录（`notes/`）

| 文件 | 内容 |
|---|---|
| `REPORT_mixed_precision.md` | 混合精度最有效作用点与自适应策略（完整报告版） |
| `REPORT_m4_acceptance.md` | GMRES-IR 集成与 S₁ 全矩阵判据验收 |
| `notes/m3_gpu_status.md` | GPU LU 移植进度与阻塞项 |
| `notes/m3_gpu_verification.md` | GPU vs CPU 正确性验证明细 |
| `notes/m3_parallel.md` | 并行化尝试记录 |
| `notes/m4_gmres_ir_status.md` | GMRES-IR 集成状态 |
| `notes/acceptance_protocol.md` | 验收协议（怎么判 PASS） |
| `notes/acceptance_summary.md` | 验收汇总 |

`notes/` 是**过程记录**，可能包含已被后续实验推翻的中间结论。引用结论请以
`docs/` 与 `REPORT_*.md` 为准。

---

## 5. 数据口径提醒

1. **平台**：全部 GPU 数字来自 RTX 3060（sm_86）+ CUDA 12.6，**不是沐曦 GPU**。
2. **判据**：`s1_table.json` 的 PASS 与 `mp_ir_classification.json` 的绝对口径判据不同，
   前者是"vs CPU 参考解"，后者是"vs 已知真解 x=1"。两边差 1e-6 的边界情况下结论可能不一致，
   引用时注明用哪套口径。
3. **可复现性**：小矩阵（n<5k）时间在 µs 量级，噪声占比高，单次数字不要引用。
4. **失败集合**：rajat16/18/20/25 是**矩阵本身奇异**（fp64 也解不了），不是实现缺陷。
