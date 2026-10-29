# 策略验收协议（goal b93ce8e1 · 验收方：主 agent）

> 目标：按 `design/ai_infra_playbook.md` 逐个验收 5 个 AI Infra→稀疏LU 策略（①批量常驻内核 ②算子融合 ③SRAM分块 ④fp16存储 ⑤双层选择器）。
> 基线（2026-08-16 15:50）：git HEAD `72952d9`；`mp_gpu_bench.json` 指纹 `7664e20b`；selector.c 86 行；工作区无未提交改动。

## 1. 验收流程（每个策略结算后执行，顺序固定）

```text
① 交付物清单核对 → ② 自行编译 → ③ 复跑数据 → ④ 阈值判定 → ⑤ 交叉核对 → ⑥ 登记/打回
```

## 2. 各策略验收细则

### 策略① 批量 LU 常驻内核（subagent ae04029b）
| 项 | 验收判据 |
|---|---|
| 交付物 | `src/solver/mp_bench/batch_spmv.cu` + `experiments/mp/batch_lu/{batch_results.json, report.md}` |
| 编译 | nvcc 12.6 -arch=sm_86 自行编译通过（不信"他说能编译"） |
| 正确性 | 批量结果 vs 单矩阵内核/scipy：相对误差 ≤1e-6（fp32 累加允许 ≤1e-5 并注明） |
| **核心性能** | **n≤5k 矩阵批处理后 fp32/fp64 收益从 ~1.0× 恢复到 ≥1.3×**；给出 k（批大小）拐点 |
| 对照 | 与 `mp_gpu_bench.json` 的单矩阵基线对比 |

### 策略④ fp16 存储（subagent c6e45abc）
| 项 | 验收判据 |
|---|---|
| 交付物 | `src/solver/mp_bench/fp16_spmv.cu` + `experiments/mp/fp16/{fp16_results.json, report.md}` |
| 编译 | nvcc -arch=sm_86 通过（__half 原生支持） |
| 正确性 | Ruiz 后 fp16 值存储 + fp32 累加 SpMV 误差 **≤1e-6**（逐矩阵报告） |
| **核心性能** | fp16 存储带宽 **≥ fp32 存储**（期望 ~1.5–2×，字节减半） |
| 值域判定 | 给出"可 fp16 存储"的矩阵清单与判定规则（Ruiz 后 min/max 是否入 [6e-5, 65504]） |

### 策略⑤ 双层选择器（subagent c2aaa707）
| 项 | 验收判据 |
|---|---|
| 交付物 | 修改的 `selector.c/h` + `src/solver/test_selector.c` + `experiments/mp/selector2/{selector2_results.json, report.md}` |
| 兼容 | 原格式决策接口不破坏（M1 定稿语义） |
| 正确性 | 5 用例回归（bandgap/rajat22/flash/OTA/rajat16）与 `mp_ir_classification.json` 实测分类一致 |
| 推演 | 33 矩阵 (格式×精度) 决策表与实测分类统计交叉核对 |
| 纪律 | **不用静态 κ 表**——κ 只作初始猜测；判据为试跑残差/分解质量 |

### 策略② 算子融合（第二批）
| 项 | 验收判据 |
|---|---|
| 交付物 | IR 融合内核/多流基准 + `experiments/mp/fusion/` |
| 正确性 | 融合后与未融合逐位一致或误差 ≤1e-6（残差路径不变） |
| 性能 | 有实测时间对比（融合 vs 未融合、单流 vs 多流），注明启动/传输占比 |

### 策略③ SRAM 分块（第二批）
| 项 | 验收判据 |
|---|---|
| 交付物 | panel/smem 分块内核 + `experiments/mp/sram/` |
| 性能 | smem 预算扫描（不同 smem 大小 × 带宽/占用率），有实测对比表 |
| 正确性 | 与未分块版本一致 |

## 3. 交叉核对与登记

- 每个策略通过后：更新 `RESOURCE_INDEX.md`（按里程碑小节登记）+ git commit（信息含实测数字）
- 不合格：打回重做，附具体问题清单（编译错误/阈值不达标/缺数据）
- 全部 5 策略验收完成 → 更新 `design/roadmap.md` 状态标记 → goal complete

## 4. 验收环境（复现口径）

- GPU：RTX 3060 12GB（sm_86）；nvcc `/usr/local/cuda-12.6/bin/nvcc`
- 编译：`nvcc -O3 -arch=sm_86 -I src/solver ...`（各内核文件头有注释）
- 计时：cudaEvent 内核级 / clock_gettime 整批，warmup 后取最优；同一环境对比（基线见上）
- 数据源：`data/matrices/*.mtx`；s0 转换段见 `experiments/mp/run_gpu_bench.sh`
