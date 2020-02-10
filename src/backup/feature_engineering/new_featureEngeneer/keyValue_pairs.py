import numpy as np
from scipy import stats

def key_value_pairs(df_list, A, B, method):
    '''
    
    :param df_list: 
    :param A: string:列名， df[A].groupby(B).aggregate(func)
    :param B: string
    :param method: 
    :return: 
    '''
    if A not in df_list.cloumns or B not in df_list.cloumns:
         raise TypeError('input valid field')

    if method == 'min':
        return df_list[A].groupby(B).aggregate(np.min)
    elif method == 'max':
        return df_list[A].groupby(B).aggregate(np.max)
    elif method == 'mean':
        return df_list[A].groupby(B).aggregate(np.mean)
    elif method == 'median':
        return df_list[A].groupby(B).aggregate(np.median)
    elif method == 'mode':
        return df_list[A].groupby(B).aggregate(mode)
    elif method == 'entropy':
        return df_list[A].groupby(B).aggregate(entropy)
    elif method == 'std':
        return df_list[A].groupby(B).aggregate(np.std)
    elif method == 'skew':
        return df_list[A].groupby(B).aggregate(stats.skew)
    elif method == 'kurt':
        return df_list[A].groupby(B).aggregate(stats.kurtosis)



def mode(x):
    return stats.mode(x)[0][0]

def entropy(x):

    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def skew(x):
    return stats.skew(x)


def kurtosis(x):
    return stats.kurtosis(x)

