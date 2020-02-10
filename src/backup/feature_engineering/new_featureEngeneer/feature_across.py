import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import new_featureEngeneer.config as config


def featureAcross(df_list, names, method):
    '''
    连续变量的分箱
    :param df_list: 
    :param methods_list: {'bin':{'names':'houseArea', 'bin_dot':[100,500,1000]},
                            }
    :return: 
    '''
    if method not in ('bin','decision tree','cluster'):
        raise ValueError("Input a method in ('bin','decision tree','cluster')")

    if method == 'bin':
        for col in names:
            bin = get_quantile_based_boundaries(df_list[col], num_buckets=5)
            df_list[col + '_cut_frequency'] = pd.cut(df_list[col], bin=bin)

    if method == 'decision_tree':
        for col in names:
            bin = optimal_binning_boundary(df_list[col],df_list.target)
            df_list[col+'_cut_decisiontree'] = pd.cut(df_list[col],bin=bin)

    if method == 'cluster':
        for col in names:
            k = config()
            cluster_label = get_cluter_boundaries(df_list[col],k)
            df_list[col+'_cut_cluster'] = cluster_label

def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary

def get_cluter_boundaries(data,k):
    '''
    聚类分箱
    :param data: 
    :param k: 
    :return: 
    '''
    cluster= KMeans(data, k)
    return cluster.labels_
