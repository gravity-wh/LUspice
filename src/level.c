// ============================================================================
// level.c — M3 factor 并行化的 level 调度前置（CPU 符号分析）
//
// level scheduling：稀疏 LU 数值分解中，列 k 依赖 L 列 j（j 在 L 列 k 的
// 非零行集合，即前驱）。同 level 的列可并行分解（无相互依赖）。
//   level[k] = 1 + max(level[j]) for j in pred(k)
// 用途：factor 并行化的任务分组（每 level 一个 wave，wave 内列并行）。
//
// 输入：L 的下三角结构（CSC：Lp/Li，单位对角首项；行索引为原始行号或位置）
// 输出：level 数组 + level 深度 + 每 level 列数
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "level.h"

/* 计算每列 level（用 U 结构：U 列 k 的非对角位置 j<k = 依赖列）
   列 k 的分解需要列 j 的 L 完整（U(j,k)≠0 的 j）——依赖图由 U 结构给出 */
int dsh_level_compute(int n, const int *Up, const int *Ui,
                      int *level_out, int *nlevels_out)
{
    int *level = level_out;
    for (int k = 0; k < n; k++) level[k] = 0;
    int maxl = 0;
    for (int k = 0; k < n; k++) {
        int lk = 0;
        /* U 列 k 的非对角行（位置 <k，对角是末项 Ui[Up[k+1]-1]=k） */
        for (int p = Up[k]; p < Up[k + 1] - 1; p++) {
            int j = Ui[p];               /* 依赖列（位置） */
            if (j >= 0 && j < k) {
                if (level[j] + 1 > lk) lk = level[j] + 1;
            }
        }
        level[k] = lk;
        if (lk > maxl) maxl = lk;
    }
    *nlevels_out = maxl + 1;
    return 0;
}

/* 统计每 level 的列数（并行 wave 大小） */
int dsh_level_hist(int n, const int *level, int nlevels, int *hist_out)
{
    memset(hist_out, 0, (size_t)nlevels * sizeof(int));
    for (int k = 0; k < n; k++) if (level[k] < nlevels) hist_out[level[k]]++;
    return 0;
}
