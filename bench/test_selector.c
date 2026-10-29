// ============================================================================
// test_selector.c — (格式×精度) 双层选择器 5 用例回归
//
// 编译：gcc -O2 -o test_selector test_selector.c selector.c
// 运行：./test_selector
//
// 用例（特征取自 data/matrices/matrix_manifest.json + experiments/mp/
//        mp_ir_classification.json 实测分类）：
//   U1 bandgap (P-fast ✅)   U2 rajat22 (P-fast ✅, κ2.8e15 证明κ不可靠)
//   U3 flash-adc (P-safe)    U4 OTA (P-hard)    U5 rajat16 (P-sing 奇异)
// 验证：初始路径 + 升级链序列 与实测分类一致；格式决策与 M1 定稿一致。
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include "selector.h"

static int failures = 0;
#define CHECK(cond, msg) do { if (cond) { printf("  ✅ %s\n", msg); } \
    else { printf("  ❌ %s\n", msg); failures++; } } while (0)

/* 打印一次决策 */
static void dump(const char *name, const dsh_matrix_info *mi)
{
    dsh_config cfg;
    dsh_format_select(mi, &cfg);
    printf("%s: 格式=%s | 初始路径=%s | pre_ruiz=%d fp16=%d factor_prec=%s\n",
           name, dsh_fmt_name(cfg.spmv_format), dsh_mp_name(cfg.mp_path),
           cfg.pre_ruiz_hint, cfg.fp16_candidate,
           cfg.factor_prec == DSH_PREC_FP32 ? "fp32" : "fp64");
}

int main(void)
{
    printf("== (格式×精度) 双层选择器 5 用例回归 ==\n\n");

    /* U1 bandgap: n=113 长尾行, κ≈7.6e7, 动态范围 log10≈28 */
    {
        dsh_matrix_info mi = {113, 616, 5.5, 16, 5, 2.9, 7.6e7, 7.6e7, 28.2, 0};
        dump("U1 bandgap", &mi);
        dsh_config cfg; dsh_format_select(&mi, &cfg);
        CHECK(cfg.spmv_format == DSH_FMT_CSR, "格式=CSR (n<5000)");
        CHECK(cfg.mp_path == DSH_MP_PFAST, "初始路径=P-fast");
        CHECK(cfg.pre_ruiz_hint == 1, "动态范围>1e20 → pre_ruiz_hint=1");
        CHECK(cfg.factor_prec == DSH_PREC_FP32, "因子=fp32 (P-fast)");
        CHECK(dsh_mp_escalate(DSH_MP_PFAST) == DSH_MP_PSAFE, "升级链 P-fast→P-safe");
    }
    /* U2 rajat22: n=39899, κ≈2.8e15(!) — κ 大但实测 P-fast ✅ → 初始仍 P-fast */
    {
        dsh_matrix_info mi = {39899, 195429, 4.9, 60, 4, 12.2, 2.8e15, 2.8e15, 81.7, 0};
        dump("U2 rajat22", &mi);
        dsh_config cfg; dsh_format_select(&mi, &cfg);
        CHECK(cfg.spmv_format == DSH_FMT_CSR, "格式=CSR (行不均 skew=12.2>5, 长尾→CSR 与 M1 一致)");
        CHECK(cfg.mp_path == DSH_MP_PFAST, "初始路径=P-fast（κ 高也不升级——κ 不是静态判据）");
        CHECK(cfg.pre_ruiz_hint == 1, "κ>1e12 → pre_ruiz_hint=1 (预算提示)");
    }
    /* U3 flash-adc: n=989, κ≈1.5e13 — 实测 P-safe（fp32+IR 失败 → 升级） */
    {
        dsh_matrix_info mi = {989, 5798, 5.9, 32, 5, 5.4, 1.5e13, 1.5e13, 46.5, 0};
        dump("U3 flash-adc", &mi);
        dsh_config cfg; dsh_format_select(&mi, &cfg);
        CHECK(cfg.mp_path == DSH_MP_PFAST, "初始=P-fast（先试）");
        CHECK(dsh_mp_escalate(DSH_MP_PFAST) == DSH_MP_PSAFE, "试跑失败 → 升级 P-safe（实测分类一致）");
    }
    /* U4 OTA: n=42, κ≈1.5e13 — 实测 P-hard（fp32+IR 发散、fp64+Ruiz 才可） */
    {
        dsh_matrix_info mi = {42, 210, 5.0, 20, 4, 5.0, 1.5e13, 1.5e13, 27.0, 0};
        dump("U4 OTA", &mi);
        dsh_config cfg; dsh_format_select(&mi, &cfg);
        CHECK(cfg.mp_path == DSH_MP_PFAST, "初始=P-fast（先试）");
        CHECK(dsh_mp_escalate(dsh_mp_escalate(DSH_MP_PFAST)) == DSH_MP_PHARD,
              "两次失败 → P-hard（实测分类一致）");
    }
    /* U5 rajat16: n=94294 — 已知奇异 */
    {
        dsh_matrix_info mi = {94294, 432866, 4.6, 50, 4, 10.9, 0, 0, 82.5, 1};
        dump("U5 rajat16", &mi);
        dsh_config cfg; dsh_format_select(&mi, &cfg);
        CHECK(cfg.mp_path == DSH_MP_PSING, "奇异标志 → P-sing（检测→移位→报错）");
    }
    /* 升级链单调性 */
    {
        int seq[] = {DSH_MP_PFAST, DSH_MP_PSAFE, DSH_MP_PHARD, DSH_MP_PSING};
        int ok = 1;
        for (int i = 0; i < 3; i++)
            if (dsh_mp_escalate(seq[i]) != seq[i + 1]) ok = 0;
        CHECK(ok, "升级链单调: P-fast→P-safe→P-hard→P-sing");
    }

    printf("\n%s (失败 %d)\n", failures == 0 ? "== 全部通过 ==" : "== 有失败 ==", failures);
    return failures == 0 ? 0 : 1;
}
