# encoding:utf-8

"""This Module Performs drop-one-out selections of the variables.
More specifically, we random permute one variable in the training set, and check the decrease in validation accuracy
Currently we only allow for logistic regression.
"""
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import OrderedDict


def leave_one_out_selector(train_df, val_df, target_column, index_col=None, method='logistic', metric='acc'):
    """

    :param train_df:  The training dataset.
    :param val_df:  The validation dataset.
    :param target_column: The target column. For now, we assume it is binary.
    :param index_col: The index_col. Default is none.
    :param method: currently only support logistic.
    :param metric: a collection of metric to choose from. Currently, we only support 'acc', 'precision', 'recall', 'micro_f1', 'macro_f1'.
    :return:
    """

    # TODO Currently we copy the whole dataframe for simplicity, we should only copy part of it.

    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    y_train = train_df_copy[target_column]
    y_val = val_df_copy[target_column]
    x_train = train_df_copy.drop(target_column)
    x_val = train_df_copy.drop(target_column)
    if index_col:
        x_train = x_train.drop(index_col)
        x_val = x_val.drop(index_col)

    clf = fit(y_train, x_train)
    y_val_pred = clf.predict(x_val)
    original_score = get_metric(y_val, y_val_pred, metric)
    result = OrderedDict()
    result['original_score'] = original_score
    for column in x_train.columns.values:
        x_train_copy = x_train.copy()
        x_train_copy[column] = np.random.permutation(x_train_copy[column])
        clf = fit(y_train, x_train_copy)
        y_pred = clf.predict(x_val)
        score = get_metric(y_val, y_pred, metric)
        result[column] = score
        result = sorted(result.items(), key=lambda t: t[1], reverse=True)
    return result


def get_metric(y_true, y_pred, metric='acc'):
    if metric == 'acc':
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    elif metric == 'recall':
        return metrics.recall_score(y_true, y_pred)
    elif metric == 'micro_f1':
        return metrics.f1_score(y_true, y_pred, average='micro')
    elif metric == 'macro_f1':
        return metrics.f1_score(y_true, y_pred, average='macro')
    else:
        raise NotImplementedError()


def fit(y, x):
    clf = LogisticRegression(random_state=0).fit(x, y)
    return clf
