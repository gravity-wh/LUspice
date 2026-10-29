// selector.h — M5 自适应 (格式×精度) 双层选择器接口
#ifndef DSH_SELECTOR_H
#define DSH_SELECTOR_H

typedef struct {
    int n;
    long nnz;
    double avg_rowlen, max_rowlen, p50_rowlen;
    double row_skew;        /* max/mean 行长度（M1 格式决策特征） */
    double kappa_ruiz;      /* Ruiz 后条件数（外部填；<=0 表示未知） */
    double kappa_est;       /* 原始 κ 估计（外部填，仅作初始猜测；<=0 未知） */
    double dyn_range_log10; /* log10(max|A| / min|nonzero|)（fp16 候选判据；<=0 未知） */
    int    singular_flag;   /* 1 = 已知奇异（rajat16 类） */
} dsh_matrix_info;

enum {
    DSH_FMT_CSR = 0,
    DSH_FMT_SELL_C_SIGMA = 1,
    DSH_FMT_ELL = 2
};
enum { DSH_PREC_FP32 = 0, DSH_PREC_FP64 = 1 };

/* 混合精度路径（design/adaptive_mixed_precision.md v2 动态升级链）：
 *   P-fast   : fp32 因子 + fp64 主元 + fp64 残差 IR   —— 先试，最快
 *   P-safe   : fp64 + Ruiz + 移位(1e-7|diag|) + IR     —— 升级 1
 *   P-hard   : fp64 + Ruiz 结构对齐 + trisolve 数值优化 —— 升级 2
 *   P-sing   : 检测 → 移位 → 明确报错                   —— 终端态
 * 纪律：κ 只作初始猜测/预算，不作为静态判据（实测 rajat22 κ2.8e15 过 P-fast、
 *       OTA κ1.5e13 失败）；升级由运行时信号触发（试跑残差/分解质量）。 */
enum {
    DSH_MP_PFAST = 0,
    DSH_MP_PSAFE = 1,
    DSH_MP_PHARD = 2,
    DSH_MP_PSING = 3
};

typedef struct {
    int spmv_format;        /* DSH_FMT_* */
    int sigma;              /* SELL 列块宽 */
    int use_ruiz;           /* 0/1（κ 诊断动态决定） */
    int use_shift;          /* 1 = shift 因子（默认） */
    double shift;           /* 1e-7 */
    int factor_prec;        /* DSH_PREC_*（跟随 mp_path：P-fast→fp32，P-safe/hard→fp64） */
    int ir_outer, ir_inner; /* 保守参数 */
    double ir_rtol;
    /* --- (格式×精度) 双层：精度路径（v2 新增） --- */
    int mp_path;            /* DSH_MP_*（初始路径，运行时经 dsh_mp_escalate 升级） */
    int pre_ruiz_hint;      /* 1 = 建议先 Ruiz（κ 或动态范围超预算，P-safe 准备） */
    int fp16_candidate;     /* 1 = 均衡后值域入界候选（fp16 值存储增强可用） */
} dsh_config;

void dsh_matrix_analyze(int n, long nnz, const int *rowptr, dsh_matrix_info *info);
void dsh_format_select(const dsh_matrix_info *info, dsh_config *cfg);
const char *dsh_fmt_name(int fmt);

/* 精度路径：初始决策（并入 dsh_format_select 内被调用，也可单独调） */
void dsh_mp_initial(const dsh_matrix_info *info, dsh_config *cfg);
/* 运行时单步升级（试跑失败后调用）：P-fast→P-safe→P-hard→P-sing（终端） */
int  dsh_mp_escalate(int cur);
const char *dsh_mp_name(int p);

#endif
