import numpy as np


def mean_encoder(df_list, encoded_vars, target, method):
    '''
    :param df_list: 
    :param encoded_vars: 需要编码的变量
    :param target: 预测变量
    :param method: mean or var
    :return: 
    '''
    for A in encoded_vars:
        if A not in df_list.cloumns or target not in df_list.cloumns:
             raise TypeError('input valid field')

        if method == 'mean':
            return df_list[A].groupby(target).aggregate(np.mean)

        elif method == 'var':
            return df_list[A].groupby(target).aggregate(np.var)

