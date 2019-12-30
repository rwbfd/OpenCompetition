# encoding = 'utf-8'

"""
This file performs the importance weight checking by randomly permute one
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import shuffle
from absl import flags
from absl import app
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
from operator import itemgetter
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", 'lightgbm', ['lightgbm', 'xgboost'],
                  "The booster to check the accuracy against. Currently only support LightGBM and XGBoost")
flags.DEFINE_string("train_file", None, "Train file name. Must be tab delimited.")
flags.DEFINE_string("dev_file", None, "Dev file name. Must be tab delimited.")
flags.DEFINE_string("target", None, "Name of the target variable.")
flags.DEFINE_integer("num_rounds", 50, "Number of rounds of to boost.")
flags.DEFINE_integer("depth", 4, "Depth.")
flags.DEFINE_enum("metric", 'acc', ['acc', 'precision', 'recall', 'f1'], "The evaluation metric.")
flags.DEFINE_string("dump_path", None, "The path to dump the result")


def get_metric(metric, y, y_pred):
    if metric == "acc":
        return accuracy_score(y, y_pred)
    elif metric == 'precision':
        return precision_score(y, y_pred)
    elif metric == 'recall':
        return recall_score(y, y_pred)
    elif metric == 'f1':
        return f1_score(y, y_pred)
    else:
        raise NotImplementedError()


def get_x_y(target, data):
    y = data[target]
    x = data.drop(columns=target)
    return x, y


def fit_xgboost_model(target, train_data):
    x_train, y_train = get_x_y(target, train_data)
    train_data_set = xgb.DMatrix(x_train, label=y_train)
    params = {'max_depth': flags.depth,
              'objective': 'binary:logistic',
              'nthread': multiprocessing.cpu_count(),
              'eval_metric': flags.metric,
              }
    num_rounds = flags.num_rounds
    bst = xgb.train(params, train_data_set, num_rounds)
    return bst


def fit_lightgbm_model(target, train_data):
    x_train, y_train = get_x_y(target, train_data)
    train_data_set = lgb.Dataset(x_train, label=y_train)
    param = {'num_leaves': flags.depth,
             'objective': 'binary',
             'metric': flags.metric,
             }
    num_rounds = flags.num_rounds
    bst = lgb.train(param, train_data_set, num_rounds)
    return bst


def get_accuracy(bst, target, dev_data, method='lightgbm', metric='acc'):
    x_dev, y_dev = get_x_y(target, dev_data)
    if method == 'ligthgbm':
        x_dev_data_set = xgb.DMatrix(x_dev)
    elif method == 'xgboost':
        x_dev_data_set = lgb.Dataset(x_dev)
    else:
        raise Exception("Only supported LightGBM and XgBoost.")
    y_pred = (bst.predict(x_dev_data_set) > 0.5).astype(np.int32)
    return get_metric(metric, y_dev, y_pred)


def main():
    target = flags.target
    train_file = flags.train_file
    dev_file = flags.dev_file

    train_data = pd.read_csv(train_file, sep="\t")
    dev_data = pd.read_csv(dev_file, sep="\t")

    if flags.model == 'lightgbm':
        bst = fit_lightgbm_model(target, train_data)
    elif flags.model == 'xgboost':
        bst = fit_xgboost_model(target, train_data)
    else:
        raise Exception("Only support LightGBM and XgBoost")

    #nrow = dev_data.shape[0]
    #var_list = [i for i in train_data.columns.values if i != target]
    #accuracy_score = OrderedDict()
    accuracyScore = get_accuracy(bst, target, dev_data, method=flags.model, metric=flags.metric)

    features = [i for i in train_data.columns] #drop label
    features_scores_diff = []
    for feature in features:
        train_data_copy = train_data.copy()
        train_data_copy[str(feature)] = shuffle(train_data_copy[str(feature)].values)
        model = input("input your model(xgboost,lightgbm):")
        if model == "xgboost":
            bst_shuffle = fit_xgboost_model(target, train_data_copy)            
        elif model == "lightgbm":
            bst_shuffle = fit_lightgbm_model(target, train_data_copy)
            
        accuracy_score[str(feature)] = get_accuracy(bst_shuffle, target, dev_data, method=flags.model, metric=flags.metric)
        diff = abs(accuracyScore-accuracy_score[str(feature)])
        features_scores_diff.append(diff)
    
    feture_importance = sorted(dict(zip(features,features_scores_diff)).items(),key = lambda x:x[1],reverse = False) #
    #accuracy_score = OrderedDict(sorted(accuracy_score.items(), key=itemgetter(1)))
    file = open(flags.dump_path, 'wb')
    #pickle.dump(accuracy_score, file)
    pickle.dump(feture_importance, file)
    file.close()


if __name__ == '__main__':
    app.run(main)
