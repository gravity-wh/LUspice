#!/usr/bin/env python3
# ============================================================================
# cpu_fp16_analysis.py — CPU 侧完整测量：Ruiz 均衡后 fp16 值存储 SpMV 可行性
# 赛题七 R13 · fp16 值存储策略（FP8/低精度量化的稀疏 LU 对应物）
#
# 对候选矩阵（rajat10/11/12, pg2k, pg5k, bandgap）：
#   1. Ruiz 均衡（8 轮行/列最大元素缩放，同 notebooks/03 的 row_max/col_max）
#   2. 统计原始/均衡后值域分布：min/max、<fp16 最小正常数 6.1035e-5 占比、
#      <fp16 最小次正规 5.96e-8 占比（= 会被 flush 到 0）、>65504 占比
#   3. SpMV 误差（vs fp64 参考，inf 范数相对误差）：
#        - fp16 值存储 + fp32 累加（x 取 fp64 与 fp16 两种）
#        - fp16 值存储 + 精确累加（fsum, 分离"存储量化误差"与"累加误差"）
#        - fp32 值存储 + fp32 累加（基线）
#   4. 判定：值域入界（0% 下溢到 flush / 0% 溢出）且误差 ≤1e-6 → "可 fp16 存储"
#   5. 输出均衡后 s0 文件（GPU 基准复用）+ fp16_cpu_stats.json
#
# 用法：python3 experiments/mp/fp16/cpu_fp16_analysis.py
# ============================================================================
import json, math, os
import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.abspath(__file__))
MATDIR = os.path.join(ROOT, "..", "..", "..", "data", "matrices")
S0DIR = os.path.join(ROOT, "s0")
os.makedirs(S0DIR, exist_ok=True)

FP16_MIN_NORMAL = 6.1035e-5      # 2^-14
FP16_MIN_SUBN   = 5.9604645e-8   # 2^-24（更小 → 转换后为 0）
FP16_MAX        = 65504.0

MATS = ["rajat10", "rajat11", "rajat12", "pg2k", "pg5k", "sky130-bandgap-reference-pvt"]
SHORT = {"sky130-bandgap-reference-pvt": "bandgap"}


def row_max(A):
    A = A.tocsr()
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    out = np.zeros(A.shape[0]); np.maximum.at(out, rows, np.abs(A.data)); return out


def col_max(A):
    A = A.tocsc()
    cols = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
    out = np.zeros(A.shape[1]); np.maximum.at(out, cols, np.abs(A.data)); return out


def ruiz(A, iters=8):
    """Ruiz 均衡：交替行/列缩放，让每行每列最大元素≈1。返回 (B, Dr, Dc)。"""
    n = A.shape[0]
    B = A.tocoo().astype(np.float64)
    Dr = np.ones(n); Dc = np.ones(n)
    for _ in range(iters):
        rmax = row_max(B); cminv = 1.0 / np.sqrt(np.maximum(rmax, 1e-300))
        B = sp.diags(cminv) @ B; Dr = Dr * cminv
        cmax = col_max(B); rinv = 1.0 / np.sqrt(np.maximum(cmax, 1e-300))
        B = B @ sp.diags(rinv); Dc = Dc * rinv
    B = B.tocsr(); B.eliminate_zeros()
    return B, Dr, Dc


def range_stats(v):
    a = np.abs(v)
    frac_lt_normal = float(np.mean(a < FP16_MIN_NORMAL))
    frac_lt_sub   = float(np.mean(a < FP16_MIN_SUBN))     # flush 到 0
    frac_gt_max   = float(np.mean(a > FP16_MAX))
    return {
        "min_abs": float(a.min()), "max_abs": float(a.max()),
        "frac_lt_6.1e-5": frac_lt_normal,
        "frac_lt_6e-8_flush": frac_lt_sub,
        "frac_gt_65504": frac_gt_max,
        "in_range_no_overflow": frac_gt_max == 0.0,
        "in_range_no_flush": frac_lt_sub == 0.0,
        "in_range_no_subnormal": frac_lt_normal == 0.0,
    }


def spmv_fp32_accum(A_csr, v_stored, x):
    """CSR SpMV：v_stored(已是 fp32 宽度的存储值) * x，float32 累加（逐行 dot）。"""
    y = np.zeros(A_csr.shape[0], dtype=np.float32)
    idx, ptr = A_csr.indices, A_csr.indptr
    for i in range(A_csr.shape[0]):
        p0, p1 = ptr[i], ptr[i + 1]
        if p1 > p0:
            y[i] = np.dot(v_stored[p0:p1], x[idx[p0:p1]])
    return y


def spmv_exact_accum(A_csr, v_f64, x):
    """CSR SpMV：v_f64 * x，fsum 精确累加（分离存储量化误差）。"""
    y = np.zeros(A_csr.shape[0])
    idx, ptr = A_csr.indices, A_csr.indptr
    for i in range(A_csr.shape[0]):
        p0, p1 = ptr[i], ptr[i + 1]
        if p1 > p0:
            y[i] = math.fsum((v_f64[p0:p1] * x[idx[p0:p1]]).tolist())
    return y


def rel_err(y, yref):
    denom = float(np.max(np.abs(yref)))
    if denom == 0.0:
        return 0.0 if np.max(np.abs(y)) == 0.0 else float("inf")
    return float(np.max(np.abs(y - yref)) / denom)


def write_s0(path, B, name):
    B = B.tocoo()
    with open(path, "w") as f:
        f.write(f"n {B.shape[0]}\n")
        for k in range(B.nnz):
            f.write(f"A {B.row[k] + 1} {B.col[k] + 1} {B.data[k]:.17g}\n")


def analyze(name):
    A = sp.csr_matrix(scipy.io.mmread(os.path.join(MATDIR, f"{name}.mtx")))
    A = A.astype(np.float64); A.eliminate_zeros()
    n, nnz = A.shape[0], A.nnz
    short = SHORT.get(name, name)

    B, Dr, Dc = ruiz(A)

    rec = {"matrix": name, "short": short, "n": n, "nnz": nnz,
           "raw": range_stats(A.data), "ruiz8": range_stats(B.data)}

    # ---- SpMV 误差（x = ones；另加随机 x 稳健性检查）----
    rng = np.random.default_rng(7)
    res = {}
    for xname, x in (("ones", np.ones(n)), ("rand", rng.uniform(-1, 1, n))):
        y64 = B @ x                                    # fp64 参考（scipy fp64 累加）
        v16f32 = B.data.astype(np.float16).astype(np.float32)   # fp16 存储 → fp32 宽度
        v32 = B.data.astype(np.float32)
        x64 = x.astype(np.float64)
        x16f32 = x.astype(np.float16).astype(np.float32)
        y16_f32acc = spmv_fp32_accum(B, v16f32, x64.astype(np.float32))
        y16x16_f32acc = spmv_fp32_accum(B, v16f32, x16f32)
        y16_exact = spmv_exact_accum(B, v16f32.astype(np.float64), x64)
        y32_f32acc = spmv_fp32_accum(B, v32, x64.astype(np.float32))
        res[xname] = {
            "fp16A_fp32acc_x64": rel_err(y16_f32acc, y64),
            "fp16A_fp32acc_x16": rel_err(y16x16_f32acc, y64),
            "fp16A_exactacc":    rel_err(y16_exact, y64),
            "fp32A_fp32acc":     rel_err(y32_f32acc, y64),
        }
    rec["spmv_err"] = res

    # ---- 判定 ----
    r = rec["ruiz8"]
    e = res["ones"]
    rec["verdict"] = {
        "value_range_ok": bool(r["in_range_no_overflow"] and r["in_range_no_flush"]),
        "err_le_1e-6": bool(e["fp16A_fp32acc_x64"] <= 1e-6),
        "fp16_storable": bool(
            r["in_range_no_overflow"] and r["in_range_no_flush"]
            and e["fp16A_fp32acc_x64"] <= 1e-6),
        "note": "rajat10 全值=1 → Ruiz 恒等, fp16 精确" if name == "rajat10" else "",
    }

    # ---- 导出均衡后 s0（GPU 复用）；rajat10 原始=均衡（全值1），仍导出 ---- 
    s0name = f"{short}_ruiz.s0"
    write_s0(os.path.join(S0DIR, s0name), B, short)
    rec["s0_file"] = s0name

    print(f"== {name} (n={n}, nnz={nnz})")
    for tag, rs in (("raw", rec["raw"]), ("ruiz8", rec["ruiz8"])):
        print(f"  {tag:6s}: min={rs['min_abs']:.3e} max={rs['max_abs']:.3e} "
              f"<6.1e-5={rs['frac_lt_6.1e-5']*100:5.1f}% flush={rs['frac_lt_6e-8_flush']*100:5.1f}% "
              f">65504={rs['frac_gt_65504']*100:5.1f}%")
    print(f"  spmv_err(ones): fp16/fp32acc={e['fp16A_fp32acc_x64']:.3e} "
          f"fp16(x16)/fp32acc={res['ones']['fp16A_fp32acc_x16']:.3e} "
          f"fp16/exactacc={res['ones']['fp16A_exactacc']:.3e} fp32/fp32acc={res['ones']['fp32A_fp32acc']:.3e}")
    print(f"  spmv_err(rand): fp16/fp32acc={res['rand']['fp16A_fp32acc_x64']:.3e}")
    print(f"  verdict: value_range_ok={rec['verdict']['value_range_ok']} "
          f"err_le_1e-6={rec['verdict']['err_le_1e-6']} fp16_storable={rec['verdict']['fp16_storable']} {rec['verdict']['note']}")
    return rec


def main():
    recs = [analyze(m) for m in MATS]
    out = os.path.join(ROOT, "fp16_cpu_stats.json")
    json.dump(recs, open(out, "w"), indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
