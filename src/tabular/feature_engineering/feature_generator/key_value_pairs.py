# encoding:utf-8
from scipy import stats
from collections import Iterable

import pandas as pd
import numpy as np

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

def key_value_pairs(df:pd.DataFrame, index_col, config_tuples):
    """
    do transform like df.groupby([A, B])[C, D].agg(func),
    func can in mean, std, skew, kurtosis, entropy, min, max, median, frequency,size.

    Parameters
    ----------

    df : pd.DataFrame. It must have one column named 'id' for indexing.
    index_col : str, the index column of df_list.
    config_tuples : list ,list of collections.namedtuple("nt",["groupBy_keys","groupBy_values","method"]) object

    Returns
    -------
    df_t : object, the df_list after transform.
        the column of transform result named like "A_B\tC_D\t method".
        It sort by index column

    """
    for config_tuple in config_tuples:
        groupBy_keys = config_tuple.groupBy_keys
        groupBy_values = config_tuple.groupBy_values
        method = config_tuple.method
        trans_column_name =  "\t".join(["_".join(to_list(groupBy_keys)),"_".join(groupBy_values),method])

        df_group = df[groupBy_values].groupby(groupBy_keys)
        if method == 'min':
            df.loc[:, trans_column_name] = df_group.aggregate(np.min)

        elif method == 'max':
            df.loc[:, trans_column_name] = df_group.aggregate(np.max)

        elif method == 'mean':
            df.loc[:, trans_column_name] = df_group.aggregate(np.mean)

        elif method == 'median':
            df.loc[:, trans_column_name] = df_group.aggregate(np.median)

        elif method == 'mode':
            df.loc[:, trans_column_name] = df_group.aggregate(mode)

        elif method == 'entropy':
            df.loc[:, trans_column_name] = df_group.aggregate(entropy)

        elif method == 'std':
            df.loc[:, trans_column_name] = df_group.aggregate(np.std)

        elif method == 'skew':
            df.loc[:, trans_column_name] = df_group.aggregate(stats.skew)

        elif method == 'kurt':
            df.loc[:, trans_column_name] = df_group.aggregate(stats.kurtosis)

    return  df.sort_values("index_col")