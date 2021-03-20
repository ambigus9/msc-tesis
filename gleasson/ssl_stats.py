import numpy as np
import pandas as pd

def estimate_stats_scores(df):
    stats_scores = {}
    df_temp = df.copy()
    df_temp['arch_scores_mean'] = df_temp.apply(lambda x: np.mean(list(x['arch_scores'].values())) ,axis=1)
    df_temp['arch_scores_std'] = df_temp.apply(lambda x: np.std(list(x['arch_scores'].values())) ,axis=1)

    scores_mean = df_temp['arch_scores_mean'].mean()
    scores_std = df_temp['arch_scores_std'].mean()

    stats_scores['df'] = df_temp
    stats_scores['scores_mean'] = scores_mean
    stats_scores['scores_std'] = scores_std
    return stats_scores

def label_stats(df_EL, df_LC, pipeline):
    labeling_stats = {}

    if len(df_EL) > 0:
        df_EL_stats = estimate_stats_scores(df_EL)
        df_EL["gth"] = df_EL[pipeline["x_col_name"]].apply(lambda x:x.split('/')[-1].split('_')[-1][0])
        df_EL_TP = df_EL[ ( df_EL[pipeline["y_col_name"]] == df_EL["gth"] ) ]
        df_EL_stats_TP = estimate_stats_scores(df_EL_TP)
        labeling_stats["df_EL_stats"] = df_EL_stats
        labeling_stats["df_EL_stats_TP"] = df_EL_stats_TP
    if len(df_LC) > 0:
        df_LC_stats = estimate_stats_scores(df_LC)
        df_LC["gth"] = df_LC[pipeline["x_col_name"]].apply(lambda x:x.split('/')[-1].split('_')[-1][0])
        df_LC_TP = df_LC[ ( df_LC[pipeline["y_col_name"]] == df_LC["gth"] ) ]
        df_LC_stats_TP = estimate_stats_scores(df_LC_TP)
        labeling_stats["df_LC_stats"] = df_LC_stats
        labeling_stats["df_LC_stats_TP"] = df_LC_stats_TP

    return pd.DataFrame(labeling_stats)
