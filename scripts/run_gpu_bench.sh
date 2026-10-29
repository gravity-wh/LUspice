#!/bin/bash
# ============================================================================
# run_gpu_bench.sh — 混合精度分阶段 GPU 基准（可复现入口）
# 赛题七 R13：fp64 vs fp32 在 稀疏LU分解/三角求解/SpMV 的吞吐对比
#
# 平台：RTX 3060 (sm_86) + CUDA 12.6（沐曦 MXMACA 开发代理）
# 计时：cudaEvent 内核级 wall-clock；warmup 后取多次最优（min）
# 用法：bash experiments/mp/run_gpu_bench.sh
# 产物：experiments/mp/mp_gpu_bench.json（每矩阵一行 JSON）
# 复现：脚本内命令含完整参数；矩阵 s0 转换由脚本内 python 完成
# ============================================================================
set -e
cd "$(dirname "$0")/../.."          # race7 根
OUT=experiments/mp/mp_gpu_bench.json
S0=/tmp/mp_s0
mkdir -p $S0

# ---- 1) MM -> s0（与既有转换器同格式：n 头 + A i j val / RHS i val）----
python3 - <<'EOF'
import numpy as np, scipy.io
import scipy.sparse as sp, os
S0='/tmp/mp_s0'; os.makedirs(S0, exist_ok=True)
names = ["sky130-bandgap-reference-pvt","sky130-flash-adc-3bit-pvt",
         "pg2k","pg5k","pg10k","rajat13","rajat10","rajat22"]
for name in names:
    A = sp.csr_matrix(scipy.io.mmread(f"data/matrices/{name}.mtx"))
    A.eliminate_zeros(); A = A.tocoo()
    with open(f"{S0}/{name}.mtx","w") as f:
        f.write(f"n {A.shape[0]}\n")
        for k in range(A.nnz):
            f.write(f"A {A.row[k]+1} {A.col[k]+1} {A.data[k]:.17g}\n")
    b = sp.csr_matrix(A) @ np.ones(A.shape[0])
    with open(f"{S0}/{name}.rhs","w") as f:
        f.write(f"n {A.shape[0]}\n")
        for i in range(A.shape[0]):
            f.write(f"RHS {i+1} {b[i]:.17g}\n")
    print(f"converted {name}")
EOF

echo "[" > $OUT
TMPJSON=/tmp/mp_gpu_records
mkdir -p $TMPJSON && rm -f $TMPJSON/*
for m in sky130-bandgap-reference-pvt sky130-flash-adc-3bit-pvt pg2k pg5k pg10k rajat13 rajat10 rajat22; do
  # LU 分阶段（bandgap/flash 含 ST；pg/rajat 大矩阵或 fp32 需预处理 → max_n_st=0 只测 PAR）
  case "$m" in
    sky130-bandgap-reference-pvt|sky130-flash-adc-3bit-pvt) MAXST=15000 ;;
    *) MAXST=0 ;;
  esac
  timeout 240 ./mp_lu_bench $S0/$m.mtx $S0/$m.rhs 0 0 $MAXST 3 2>/dev/null \
    | grep -o '"matrix":.*' | tail -1 > $TMPJSON/$m.lu
  timeout 120 ./mp_spmv_bench $S0/$m.mtx 100 4 2>/dev/null \
    | grep -o '"matrix":.*' | tail -1 > $TMPJSON/$m.spmv
  echo "== $m done =="
done
# Python 聚合（健壮：逐条 json.loads，失败置 null 并记录原因）
python3 - "$OUT" "$TMPJSON" <<'PYEOF'
import json, sys, os
out, d = sys.argv[1], sys.argv[2]
recs = []
for f in sorted(os.listdir(d)):
    m = f.rsplit('.', 1)[0]
    lu = sp = None
    for ext, key in (('lu','lu'), ('spmv','spmv')):
        p = os.path.join(d, f"{m}.{ext}")
        if os.path.getsize(p) == 0:
            recs.append({"matrix": m, key: None, key + "_status": "timeout-or-no-output"})
            continue
        try:
            raw = open(p).read().strip()
            obj = json.loads(raw)   # raw 已是完整 JSON 对象（"matrix":...}）
        except Exception as e:
            obj = None
            recs.append({"matrix": m, key: None, key + "_status": f"parse-fail: {str(e)[:60]}"})
            continue
        if key == 'lu':
            lu = obj
        else:
            sp = obj
    if lu or sp:
        recs.append({"matrix": m, "lu": lu, "spmv": sp})
json.dump(recs, open(out, 'w'), indent=1)
print("saved", len(recs), "records ->", out)
PYEOF
