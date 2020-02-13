# encoding:utf-8

import pandas as pd
import numpy as np
from src.tabular.feature_engineering.utils import retrun_df_list
from src.tabular.feature_engineering.utils import set_default_vale


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def encoder(trn_series=None,
            target=None,
            min_samples_leaf=1,
            smoothing=1,
            noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index

    return add_noise(ft_trn_series, noise_level)


def target_encode(df_list, configger):
    """

    :param df_list: the training dataset with 5 flod.
    :param configger: the json str of config setting
    :return:
    """
    import json
    configger = json.loads(configger)

    data = pd.concat(df_list, axis=1)
    var_list = configger["configger"]
    target_col = configger["target_col"]
    index_col = configger["index_col"]

    y = data[target_col]
    if var_list is None:
        X = data[var_list]
    else:
        X = data.drop([target_col], axis=1)


    min_samples_leaf = set_default_vale("min_samples_leaf",configger,1)
    smoothing = set_default_vale("smoothing",configger,1)
    noise_level = set_default_vale("noise_level",configger,0)
    data.loc[:, "target_encode"] = encoder(trn_series=X,
                                          target=y,
                                          min_samples_leaf=min_samples_leaf,
                                          smoothing=smoothing,
                                          noise_level=noise_level)

    df_list_t = []
    for df in df_list:
        df_list_t.append(df.merge(data[[index_col,"target_encode"]],on=index_col))

    return df_list_t
