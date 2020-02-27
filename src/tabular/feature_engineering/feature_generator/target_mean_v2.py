# coding = 'utf-8'

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from nyaggle.feature.category_encoder import TargetEncoder

# nyaggle refer to https://github.com/nyanp/nyaggle/tree/master/nyaggle/feature/category_encoder

class TargetMeanEncoderConfig:
    def __init__(self, fold=5, smooth_parameter=0.9, N_min=100):
        self.fold = fold
        self.smooth_parameter = smooth_parameter
        self.N_min = N_min



def encode_one_column(dfs, y, target_var):
    """
    :param df:
    :param y: str
    :param target_var:
    :param config:
    :return:
    """
    cat_cols = [c for c in dfs.columns if dfs[c].dtype == np.object]
    if target_var not in cat_cols:
        raise ValueError('target_var must be category type')
    train_x,test_x,train_y,test_y = train_test_split(dfs,test_size=0.2,random_state=42)
    train = pd.concat([train_x,train_y],axis=1)
    test = pd.concat([test_x,test_y],axis=1)
    df_new = pd.concat([train, test]).copy()
    kf = KFold(TargetMeanEncoderConfig.fold)
    te = TargetEncoder(kf.split(train))
    # use fit/fit_transform to train data, then apply transform to test data
    train.loc[:, cat_cols] = te.fit_transform(train[target_var], train[y])
    test.loc[:, cat_cols] = te.transform(test[target_var])

    # ... or just call fit_transform to concatenated data
    df_new.loc[:, cat_cols] = te.fit_transform(all[target_var], all[target_var])
    return df_new



def target_mean_encoder(dfs, y, target_vars):
    """
    :param df:
    :param y:
    :param target_vars:
    :param config:
    :return:
    """
    cat_cols = [c for c in dfs.columns if dfs[c].dtype == np.object]
    # if target_var for  not in cat_cols:
    #     raise ValueError('target_var must be category type')
    train_x, test_x, train_y, test_y = train_test_split(dfs, test_size=0.2, random_state=42)
    train = pd.concat([train_x, train_y], axis=1)
    test = pd.concat([test_x, test_y], axis=1)
    df_new = pd.concat([train, test]).copy()
    kf = KFold(TargetMeanEncoderConfig.fold)
    te = TargetEncoder(kf.split(train))
    # use fit/fit_transform to train data, then apply transform to test data
    train.loc[:, cat_cols] = te.fit_transform(train[target_vars], train[y])
    test.loc[:, cat_cols] = te.transform(test[target_vars])

    # ... or just call fit_transform to concatenated data
    df_new.loc[:, cat_cols] = te.fit_transform(all[target_vars], all[target_vars])
    return df_new



def encode_mean_column(dfs, y, target_vars, config):
    """
    :param df:
    :param y: str
    :param target_var:
    :param config:
    :return:
    """
    cat_cols = [c for c in dfs.columns if dfs[c].dtype == np.object]
    train_x, test_x, train_y, test_y = train_test_split(dfs, test_size=0.2, random_state=42)
    train = pd.concat([train_x, train_y], axis=1)
    test = pd.concat([test_x, test_y], axis=1)
    df_new = pd.concat([train, test]).copy(deep=True)
    kf = KFold(n_splits=config.fold, shuffle=True, random_state=0)
    for i, (dev_index, val_index) in enumerate(kf.split(train.index.values)):
        # split data into dev set and validation set
        dev = train.loc[dev_index].reset_index(drop=True)
        val = train.loc[val_index].reset_index(drop=True)

        feature_cols = []
        for var_name in cat_cols:
            feature_name = '{}_mean'.format(var_name)
            feature_cols.append(feature_name)

            prior_mean = np.mean(dev[target_vars])
            stats = dev[[target_vars, var_name]].groupby(var_name).agg(['sum', 'count'])[target_vars].reset_index()

            ### beta target encoding by Bayesian average for dev set
            df_stats = pd.merge(dev[[var_name]], stats, how='left')
            df_stats['sum'].fillna(value=prior_mean, inplace=True)
            df_stats['count'].fillna(value=1.0, inplace=True)
            N_prior = np.maximum(config.N_min - df_stats['count'].values, 0)  # prior parameters
            dev[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (
            N_prior + df_stats['count'])  # Bayesian mean

            ### beta target encoding by Bayesian average for val set
            df_stats = pd.merge(val[[var_name]], stats, how='left')
            df_stats['sum'].fillna(value=prior_mean, inplace=True)
            df_stats['count'].fillna(value=1.0, inplace=True)
            N_prior = np.maximum(config.N_min - df_stats['count'].values, 0)  # prior parameters
            val[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (
            N_prior + df_stats['count'])  # Bayesian mean

            ### beta target encoding by Bayesian average for test set
            df_stats = pd.merge(test[[var_name]], stats, how='left')
            df_stats['sum'].fillna(value=prior_mean, inplace=True)
            df_stats['count'].fillna(value=1.0, inplace=True)
            N_prior = np.maximum(config.N_min - df_stats['count'].values, 0)  # prior parameters
            test[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (
            N_prior + df_stats['count'])  # Bayesian mean

            # Bayesian mean is equivalent to adding N_prior data points of value prior_mean to the data set.
            del df_stats, stats
    df_new = pd.concat([dev, val, test])
    return df_new



