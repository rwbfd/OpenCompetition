import numpy as np
from scipy import stats


def rescaling(df_list, names, method):
    '''
    :param df_list: 
    :param names: 
    :param method: 
    :return: 
    '''
    if names is None:
        raise NameError('please input colnames that you want to transfromer.')

    if method not in ('log', 'exp', 'standardization', 'box_cox', 'yeo_johnson'):
        raise ValueError("please select methods in \
        ['log','exp','standardization','box_cox','yeo_johnson']")


    flag = column_or_1d(df_list, warn=False)
    if flag:

        if method == 'log':
            return log(df_list)
        elif method == 'exp':
            return exp(df_list)
        elif method == 'standardization':
            return standardization(df_list)
        elif method == 'box_cox':
            return box_cox(df_list)
        elif method == 'yeo_Johnson':
            return Yeo_Johnson(df_list)

    else:
        for name in names:
            if method == 'log':
                return log(df_list[name])
            elif method == 'exp':
                return exp(df_list[name])
            elif method == 'standardization':
                return standardization(df_list[name])
            elif method == 'box_cox':
                return box_cox(df_list[name])
            elif method == 'yeo_Johnson':
                return Yeo_Johnson(df_list[name])





def column_or_1d(y, warn=False):
    """
    Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like

    Returns
    -------
    y : array
    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return True
    if len(shape) == 2 and shape[1] == 1:
        return True

    return False

def log(y):
    assert y > 0
    return np.log(y)

def exp(y):
    return np.exp(y)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def box_cox(y):
    y_scale, lambda_ = stats.boxcox(y)
    return y_scale

def Yeo_Johnson(y):
    y_normal, lmbda_ = stats.yeojohnson(y)
    return y_normal
