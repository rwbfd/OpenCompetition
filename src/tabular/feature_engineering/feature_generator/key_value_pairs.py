# encoding:utf-8
from scipy import stats
from collections import Iterable

import numpy as np
import warnings


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


import pandas as pd


def key_value_pairs(df: pd.DataFrame, config_tuples):
    """
    do transform like df.groupby([A, B])[C].agg(func),
    func can in min, max, mean, mode, entropy, std, skew, kurt

    Parameters
    ----------

    df : pd.DataFrame.
    config_tuples : list ,list of collections.namedtuple("nt",["groupBy_keys","groupBy_values","method"]) object

    Returns
    -------
    df : object, the df_list after transform.
        the column of transform result named like "A_B C method".
        It sort by index column

    """
    df_t = df
    for config_tuple in config_tuples:
        groupBy_keys = config_tuple.groupBy_keys
        groupBy_value = config_tuple.groupBy_value
        method = config_tuple.method

        print(groupBy_keys)
        print(groupBy_value)

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
        # print(res.columns)
        res.rename(columns={groupBy_value:trans_column_name},inplace=True)
        df_t = df_t.merge(res, on=groupBy_keys)

    return df_t
