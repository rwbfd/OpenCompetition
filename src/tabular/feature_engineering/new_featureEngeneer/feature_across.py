import numpy as np

def featureAcross(df_list, names, methods_list):
    '''
    连续变量的分箱、决策树组合特征
    :param df_list: 
    :param methods_list: {'bin':{'names':'houseArea', 'bin_dot':[100,500,1000]},
                            }
    :return: 
    '''
    if method not in ('bin','decision tree'):
        raise ValueError('Input a method in ('bin','decision tree')')

    if method == 'bin':




def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]
