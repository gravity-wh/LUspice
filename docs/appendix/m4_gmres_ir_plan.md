# M4 GMRES-IR 集成设计（m4_gmres_ir_plan）

> M4：三角求解 + GMRES-IR 集成 → 全矩阵集 S₁ 判据 + SR vs CPU 曲线
> 衔接：M1 决策（SpMV 格式 SELL-C-σ32/CSR）、M2 CPU 正确性、M3 GPU LU+静态化
> 更新：2026-08-15（goal R32）

---

## 1. GMRES-IR 流程（对接 lu_static.h 的 dsh_gmres_ir）

```
输入：A（CSC/COO）、b、静态因子（L/U fp32 + P/Q）
1. x0 = trisolve(L,U,P,Q,b)                    # 初始解（fp32 因子）
2. r = b - A x0                                # fp64 残差（SpMV）
3. 循环 outer ≤ max_outer（默认 7，M1 处方）:
     GMRES(inner ≤ 100, rtol 1e-10) 解 A z = r   # 预条件子 = 静态因子
     x += z; r = b - A x
     若 ||r||/||b|| ≤ 1e-10 退出
4. 输出 x（fp64）
```

**参数默认值（M1 处方最终定稿，goal R33 修正）**：max_outer=1~3、内迭代 m=50/maxiter=300/rtol=1e-9——保守参数。注意：此前"增强 IR（mo=7/m=100/i=600）是关键"的结论**已由完整扫描推翻**——增强 IR 在 flash-adc 上重新 NaN，mo≤3 才稳定。
**移位因子（关键新发现）**：shift=1e-7·|diag| 移位因子分解（仅因子化，matvec 用原矩阵）是 flash-adc 从 NaN→1.7e-10 的唯一开关，对全部 3 个病态矩阵有 10~40× 残差增益——fpgmres.c 已有 use_shift 选项但未默认开启，M4 内核必须默认 shift=1e-7。
**Ruiz 诊断规则**：先算 Ruiz 降幅——>1e2（行列失衡型，如 s0_tb1_ff9 降 1.26e5×）才用；<1e2（固有近奇异型，neural/sc-dac）**跳过 Ruiz**（裸 fp32 反而收敛）。
**fp64 兜底**：fp32 LU min|pivot|<1e-9·max|A| → 因子不可靠，fp64 直解兜底（flash 也可走此路径）。

## 2. SpMV 格式选择（M1 决策落位）

| 矩阵类型 | SpMV 格式 | 带宽实测 |
|---|---|---|
| 行均匀（pg 系，n≥10k） | **SELL-C-σ32** | 78.3%（达标 S₃） |
| 长尾（rajat 系） | CSR | 优于 SELL 43% |
| 小矩阵（n<10k） | CSR（默认） | 启动开销主导 |

实现：`dsh_spmv_dispatch(A, x, y, fmt)`——格式选择由 M5 自适应选择器决定，M4 先用显式参数（按矩阵特征表）。

## 3. 与 lu_static.h 对接

```
dsh_gmres_ir(F, b, x, max_outer, max_inner, rtol, residual_seq, iters)
  ├─ dsh_trisolve_gpu（M3）——每 IR 迭代 1 次（预条件）
  ├─ SpMV：SELL-C-σ32（L 级）/ CSR（长尾）——每迭代 1 次
  └─ fp64 累加（Kahan 可选）——残差 1e-10 目标
```

## 4. S₁ 全矩阵集验收方案

- 矩阵集：M0 baseline 的 25 个达标矩阵（PASS+IR）+ 3 个病态（M1 处方已解决）+ flash-adc（fp64 回退）
- 判据：fp32 因子 + GMRES-IR 后 L2 ≤1e-6（vs lu_tool64c fp64 参考解）
- 输出：experiments/m4/s1_full_matrix_results.json（每矩阵残差 + verdict）+ 汇总表
- **fp64 回退开关**：κ 判据（Ruiz 后 >6.5e11）→ fp64 因子（flash-adc 路径）

## 5. SR vs CPU 曲线方案（S₂ 数据）

- T_cpu：M0 baseline（lu_tool64c 单线程，baseline.json 已有 33 矩阵）
- T_gpu：M3 GPU LU + M4 IR（含传输，S₂ 口径）
- 曲线：SR = T_cpu/T_gpu vs n（42→100352，log-log）
- 输出：experiments/m4/sr_curve.json + 图表（matplotlib）
- 目标：大矩阵（n≥10k）SR > 1 且随规模增长；小矩阵记录启动开销分析

## 6. M4 验收清单

- [ ] GMRES-IR：25+3 矩阵残差 ≤1e-6（fp32 因子）
- [ ] flash-adc fp64 回退达标
- [ ] SR vs CPU 曲线（33 矩阵，全链路口径）
- [ ] dsh_gmres_ir 参数默认值（7/100/1e-10）生效
- [ ] SpMV 格式按矩阵特征选择（SELL/CSR）

---

*衔接：M3（GPU LU + 静态化）→ M4（本文件）→ M5（自适应选择器自动选格式 + 文档）。*
