// ============================================================================
// selector.c — M5 自适应 (格式×精度) 双层选择器
//
// 第一层 · 格式（M1 定稿，保留原语义）：
//   n≥5000 且行均匀（max/mean≤5）→ SELL-C-σ32；否则 CSR
//
// 第二层 · 精度路径（v2，design/adaptive_mixed_precision.md 动态升级链）：
//   - 初始 = P-fast（fp32 因子 + fp64 主元 + fp64 残差 IR）——先试最快；
//     κ 只作初始猜测/预算（实测 rajat22 κ2.8e15 过 P-fast、OTA κ1.5e13 失败，
//     κ 静态判据不可靠，升级由运行时试跑信号触发）
//   - 提示：pre_ruiz_hint（κ>1e12 或动态范围>1e20 → 建议先 Ruiz，P-safe 准备）；
//           fp16_candidate（均衡后值域入界：动态范围≤1e1 且 κ≤1e6 → rajat10 类）
//   - 升级链：dsh_mp_escalate 运行时单步升级（P-fast→P-safe→P-hard→P-sing）
//
// 特征输入：dsh_matrix_info（n/nnz/行长度统计/κ 估计/动态范围）——CPU 侧一次 O(nnz) 扫描
// 输出：dsh_config（格式 + 精度路径 + 预处理开关 + IR 参数）
// ============================================================================
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "selector.h"

/* 特征提取：行长度统计 + 行均匀性（一次扫描行指针） */
void dsh_matrix_analyze(int n, long nnz, const int *rowptr /* n+1 */,
                        dsh_matrix_info *info)
{
    memset(info, 0, sizeof(*info));
    info->n = n;
    info->nnz = nnz;
    if (n <= 0) return;
    long sum = 0, maxlen = 0, p50 = 0;
    int *lens = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int len = rowptr[i + 1] - rowptr[i];
        lens[i] = len;
        sum += len;
        if (len > maxlen) maxlen = len;
    }
    /* 中位数（排序） */
    {
        int *tmp = (int *)malloc((size_t)n * sizeof(int));
        memcpy(tmp, lens, (size_t)n * sizeof(int));
        for (int i = 1; i < n; i++) { int k = tmp[i], j = i - 1;
            while (j >= 0 && tmp[j] > k) { tmp[j + 1] = tmp[j]; j--; } tmp[j + 1] = k; }
        p50 = tmp[n / 2];
        free(tmp);
    }
    free(lens);
    info->avg_rowlen = n > 0 ? (double)sum / n : 0.0;
    info->max_rowlen = maxlen;
    info->p50_rowlen = p50;
    info->row_skew = (info->avg_rowlen > 0) ? (double)maxlen / info->avg_rowlen : 0.0;
}

/* ============ 第二层 · 精度路径（v2 动态升级链） ============ */

void dsh_mp_initial(const dsh_matrix_info *info, dsh_config *cfg)
{
    /* 初始一律 P-fast（先试最快；κ 不可靠，不作为静态判据） */
    cfg->mp_path = DSH_MP_PFAST;
    cfg->pre_ruiz_hint = 0;
    cfg->fp16_candidate = 0;

    if (info->singular_flag) {
        cfg->mp_path = DSH_MP_PSING;          /* 已知奇异 → 直接 P-sing（检测→移位→报错） */
        return;
    }
    /* 预算提示（不改变初始路径，只提示预处理准备）：
     *  κ>1e12 或动态范围>1e20 → 建议先 Ruiz（P-safe 准备） */
    if ((info->kappa_ruiz > 1e12) || (info->dyn_range_log10 > 20.0)) {
        cfg->pre_ruiz_hint = 1;
    }
    /* fp16 候选：均衡后值域入界（动态范围≤1e1 且 κ≤1e6）——rajat10 类实测误差 0.0 */
    if (info->dyn_range_log10 > 0 && info->dyn_range_log10 <= 1.0 &&
        info->kappa_ruiz > 0 && info->kappa_ruiz <= 1e6) {
        cfg->fp16_candidate = 1;
    }
}

int dsh_mp_escalate(int cur)
{
    switch (cur) {
    case DSH_MP_PFAST: return DSH_MP_PSAFE;
    case DSH_MP_PSAFE: return DSH_MP_PHARD;
    case DSH_MP_PHARD: return DSH_MP_PSING;
    default:           return DSH_MP_PSING;   /* 终端态 */
    }
}

const char *dsh_mp_name(int p)
{
    switch (p) {
    case DSH_MP_PFAST: return "P-fast";
    case DSH_MP_PSAFE: return "P-safe";
    case DSH_MP_PHARD: return "P-hard";
    case DSH_MP_PSING: return "P-sing";
    default: return "?";
    }
}

/* ============ 双层决策（格式 + 精度初始） ============ */
void dsh_format_select(const dsh_matrix_info *info, dsh_config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    /* 第一层 · SpMV 格式（M1 定稿语义保留） */
    if (info->n >= 5000 && info->row_skew <= 5.0) {
        cfg->spmv_format = DSH_FMT_SELL_C_SIGMA;
        cfg->sigma = 32;
    } else {
        cfg->spmv_format = DSH_FMT_CSR;
        cfg->sigma = 0;
    }
    /* 第二层 · 精度路径初始（v2） */
    dsh_mp_initial(info, cfg);
    /* 因子精度跟随路径：P-fast→fp32 因子（fp64 主元）；P-safe/P-hard→fp64 因子 */
    cfg->factor_prec = (cfg->mp_path == DSH_MP_PFAST) ? DSH_PREC_FP32 : DSH_PREC_FP64;
    /* 预处理链（默认保守；Ruiz 开关由 κ 诊断动态决定，此处给默认） */
    cfg->use_ruiz = cfg->pre_ruiz_hint;     /* 预算提示开 → 预开 Ruiz（P-safe 准备） */
    cfg->use_shift = 1;                     /* shift=1e-7 因子默认开（M1 关键发现） */
    cfg->shift = 1e-7;
    /* IR 参数（M1 处方保守定稿） */
    cfg->ir_outer = 3;
    cfg->ir_inner = 50;
    cfg->ir_rtol = 1e-9;
}

const char *dsh_fmt_name(int fmt)
{
    switch (fmt) {
    case DSH_FMT_CSR: return "CSR";
    case DSH_FMT_SELL_C_SIGMA: return "SELL-C-sigma32";
    case DSH_FMT_ELL: return "ELL";
    default: return "?";
    }
}
