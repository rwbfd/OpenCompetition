# coding = 'utf-8'

"""
In this file, we implemented the so-called `remove one selector.' The idea is to select the most important variable by discovering how much the model
accuracy suffers by randomly permuting a variable.

Currently, there are some other things that needs to be done.

1. Add more metrics for classification cases.
2. Add the settings for regression case.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from _collections import OrderedDict
from tqdm import tqdm, trange
import numpy as np
import operator

def get_metric(y_pred, y_true, metric):  # TODO: Need to add more metric
    """

    :param y_pred: the predicted, must be class not probability!
    :param y_true: the true one
    :param metric: String. Can be 'acc', 'micro_f1', 'macro_f1', 'precision', 'recall'
    :return:
    """

    if metric not in ['acc', 'micro_f1', 'macro_f1', 'precision', 'recall']:
        raise NotImplementedError()

    if metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'micro_f1':
        return f1_score(y_true, y_pred, average='micro')
    elif metric == 'macro_f1':
        return f1_score(y_true, y_pred, average='macro')
    elif metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    else:
        raise NotImplementedError()


def get_classification_result(df_train, df_dev, y_name, excluded_vars=None, **kwargs):
    """
    This function perform the logistic regression and then return the prediction for dev set
    :param df_train: the training data set
    :param df_dev:  the dev data set
    :param y_name:  the y_variable. We assume it to be discrete for the moment.
    :param excluded_vars: These variables should not be permuted, and will not enter the selection. An example of such variable is the ID,
    :param kwargs: Arguments for logistic regression.
    :return: predicted class
    """

    y_train = df_train[y_name]
    x_train = df_train.drop(excluded_vars)
    x_dev = df_dev.drop(excluded_vars)

    cls = LogisticRegression(random_state=0).fit(x_train, y_train, kwargs)
    y_pred = cls.predict(x_dev)
    return y_pred


def get_importance(df_train, df_dev, y_name, metric='acc', excluded_vars=None, **kwargs):
    """
    This function perform the random permutation and return an ordered dict with keys to be the variable names, and metric to be the metric.
    :param df_train: Training dataset.
    :param df_dev: Dev dataset.
    :param y_name: The name of the y variable.
    :param metric: String. Can be 'acc', 'micro_f1', 'macro_f1', 'precision', 'recall'
    :param excluded_vars:  List. These variables should not be permuted, and will not enter the selection. An example of such variable is the ID,
    :param kwargs: Arguments for logistic regression.
    :return: An ordered dict containing the metrics.
    """
    df_train_copy = df_train.copy(deep=True)
    df_dev_copy = df_dev.copy(deep=True)

    y_dev = df_dev_copy[y_name]

    vars_to_permute = set(df_train.columns.values).difference(set(excluded_vars))

    y_pred = get_classification_result(df_train_copy, df_dev, y_name, excluded_vars, **kwargs)
    metric_score = get_metric(y_pred, y_dev, metric)

    result = OrderedDict()
    result['original'] = metric_score

    for var in tqdm(vars_to_permute):
        df_train_permute = df_train_copy.copy(deep=True)
        df_train_permute[var] = np.random.permutation(df_train_permute[var])
        y_pred = get_classification_result(df_train_permute, df_dev, y_name, excluded_vars, **kwargs)
        metric_score = get_metric(y_pred, y_dev, metric)
        result['var'] = metric_score

    result = OrderedDict(sorted(result.items(), key=operator.itemgetter(1)))
    return result
