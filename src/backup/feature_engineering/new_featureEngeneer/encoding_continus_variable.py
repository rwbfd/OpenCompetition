import numpy as np
from scipy import stats
from sklearn import preprocessing


def encode_continuous_variable(df_list, names, method):
    '''
    
    :param df_list: 
    :param names: 
    :param method: 
    :return: 
    '''
    if names is None:
        raise NameError('please input colnames that you want to transfromer.')

    if method not in ('MinAbs', 'MinMax', 'Normalization', 'Iqr', 'box_cox', 'yeo_johnson','RankGuass', 'icdf'):
        raise ValueError("please select methods in \
            ['MinAbs', 'MinMax', 'Normalization', 'Iqr', 'box_cox', 'yeo_johnson','RankGuass', 'icdf']")

    flag = column_or_1d(df_list, warn=False)
    if flag:

        if method == 'MinMax':
            return MinMax(df_list)
        elif method == 'exp':
            return MinAbs(df_list)
        elif method == 'Normalization':
            return Normalization(df_list)
        elif method == 'Iqr':
            return Iqr(df_list)
        elif method == 'box_cox':
            return box_cox(df_list)
        elif method == 'yeo_Johnson':
            return Yeo_Johnson(df_list)
        elif method == 'RankGuass':
            return RankGuass
        elif method == 'icdf':
            return icdf(df_list)

    else:
        for name in names:
            if method == 'MinMax':
                return MinMax(df_list[name])
            elif method == 'exp':
                return MinAbs(df_list[name])
            elif method == 'Normalization':
                return Normalization(df_list[name])
            elif method == 'Iqr':
                return Iqr(df_list[name])
            elif method == 'box_cox':
                return box_cox(df_list[name])
            elif method == 'yeo_Johnson':
                return Yeo_Johnson(df_list[name])
            elif method == 'RankGuass':
                return RankGuass(df_list[name])
            elif method == 'icdf':
                return icdf(df_list[name])


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


def MinMax(x):
    Min, Max = np.min(x), np.max(x)
    x = (x - Min) / (Max - Min)
    return x

def MinAbs(y):
    maxabsscaler_scaler = preprocessing.MaxAbsScaler() # 建立MaxAbsScaler对象
    df_scale = maxabsscaler_scaler.fit_transform(y)
    return df_scale

def Normalization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def Iqr(y):
    pass

def box_cox(y):
    y_scale, lambda_ = stats.boxcox(y)
    return y_scale

def Yeo_Johnson(y):
    y_normal, lmbda_ = stats.yeojohnson(y)
    return y_normal

def RankGuass(y):
    pass

def icdf(x):
    q = stats.norm.cdf(x)
    return stats.norm.ppf(q)