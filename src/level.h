// level.h — level 调度接口
#ifndef DSH_LEVEL_H
#define DSH_LEVEL_H

#ifdef __cplusplus
extern "C" {
#endif

int dsh_level_compute(int n, const int *Up, const int *Ui, int *level_out, int *nlevels_out);
int dsh_level_hist(int n, const int *level, int nlevels, int *hist_out);

#ifdef __cplusplus
}
#endif

#endif /* DSH_LEVEL_H */
