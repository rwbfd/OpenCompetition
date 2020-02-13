# encoding:utf-8
from scipy import stats
from collections import Iterable

import numpy as np
import json


def mode(x):
    return stats.mode(x)[0][0]


def entropy(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def to_list(obj):
    if not isinstance(obj, Iterable):
        obj = [obj]
    return obj


def key_value_pairs(df, configger):
    """
    do transform like df.groupby([A, B])[C].agg(func),
    func can in min, max, mean, mode, entropy, std, skew, kurt

    :param df:
    :param config_tuples:
    :return:
    """
    configger = json.loads(configger)

    df_t = df

    groupBy_keys = configger["groupBy_keys"]
    groupBy_value = configger["groupBy_value"]
    method = configger["method"]

    trans_column_name = " ".join(["_".join(to_list(groupBy_keys)), groupBy_value, method])

    df_group = df.groupby(groupBy_keys)[groupBy_value]

    if method == 'min':
        res = df_group.aggregate(np.min).reset_index()

    elif method == 'max':
        res = df_group.aggregate(np.max).reset_index()

    elif method == 'mean':
        res = df_group.aggregate(np.mean).reset_index()

    elif method == 'median':
        res = df_group.aggregate(np.median).reset_index()

    elif method == 'mode':
        res = df_group.aggregate(mode).reset_index()

    elif method == 'entropy':
        res = df_group.aggregate(entropy).reset_index()

    elif method == 'std':
        res = df_group.aggregate(np.std).reset_index()

    elif method == 'skew':
        res = df_group.aggregate(stats.skew).reset_index()

    elif method == 'kurt':
        res = df_group.aggregate(stats.kurtosis).reset_index()

    else:
        raise ValueError("the function value {func} is not be support".format(func=method))


    res.rename(columns={groupBy_value: trans_column_name}, inplace=True)
    df_t = df_t.merge(res, on=groupBy_keys)
    return df_t
