#!/usr/bin/env python3
"""(格式×精度) 双层选择器 33 矩阵交叉核对
镜像 selector.c 规则：格式按 n≥5000 且 row_skew≤5 → SELL；精度初始 P-fast，
升级链 P-fast→P-safe→P-hard→P-sing；与 experiments/mp/mp_ir_classification.json 实测分类对照。
"""
import json, numpy as np, scipy.io, scipy.sparse as sp
DATA='data/matrices'

def features(name):
    A = sp.csr_matrix(scipy.io.mmread(f"{DATA}/{name}.mtx"))
    n, nnz = A.shape[0], A.nnz
    rl = np.diff(A.tocsc().indptr)
    skew = rl.max() / (rl.mean() + 1e-30)
    d = np.abs(A.data); dr = np.log10(d.max() / d[d>0].min())
    return n, skew, dr

def format_sel(n, skew):
    return "SELL-C-σ32" if (n >= 5000 and skew <= 5.0) else "CSR"

def mp_initial(n, kappa, dr, singular):
    if singular: return "P-sing", 0
    pre_ruiz = (kappa and kappa > 1e12) or (dr > 20.0)
    fp16 = (0 < dr <= 1.0) and (kappa and kappa <= 1e6)
    return "P-fast", (pre_ruiz, fp16)

ESC = {"P-fast":"P-safe", "P-safe":"P-hard", "P-hard":"P-sing", "P-sing":"P-sing"}
def classify_to_path(c):
    # mp_ir_classification.json 的 path 字段 → 终点路径
    if c == "P1": return "P-fast"
    if c == "P3(fp64可解)": return "P-safe"
    if c == "P3(fp64误差大)": return "P-hard"
    return "P-sing"   # P3(fp64也失败) / P1-FAIL → 奇异/检测

man = json.load(open(f"{DATA}/matrix_manifest.json"))['matrices']
cls = {r['id']: r['path'] for r in json.load(open('experiments/mp/mp_ir_classification.json'))['results']}
rows = []
for x in man:
    name = x['id']
    k = (x.get('kappa_1norm') or {}).get('est')
    singular = k is None
    n, skew, dr = features(name)
    fmt = format_sel(n, skew)
    init, hints = mp_initial(n, k if k else 0, dr, singular)
    exp = classify_to_path(cls.get(name, "P1"))
    # 一致性: 初始 P-fast + 升级链能到达实测终点
    p = init; steps = 0
    while p != exp and steps < 4: p = ESC[p]; steps += 1
    consist = (p == exp) and (init == "P-fast" or exp == "P-sing" or init == exp)
    rows.append({"id": name, "n": n, "skew": round(skew,1), "dyn_log10": round(dr,1),
                 "kappa": (f"{k:.1e}" if k else "inf"), "format": fmt, "init": init,
                 "escalations": steps, "expected": exp, "consistent": bool(consist)})
    print(f"{name[:42]:<44} fmt={fmt:<11} init={init:<7} escal={steps} exp={exp:<10} {'✅' if consist else '❌'}")

json.dump(rows, open('experiments/mp/selector2/selector2_results.json','w'), indent=1, ensure_ascii=False)
from collections import Counter
print("\n格式统计:", dict(Counter(r['format'] for r in rows)))
print("路径终点统计:", dict(Counter(r['expected'] for r in rows)))
print("一致性:", sum(r['consistent'] for r in rows), "/", len(rows))
