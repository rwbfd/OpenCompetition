# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-09-03
Description :
auther : wcy
"""
# import modules
import os, sys
from sklearn import metrics
import pandas as pd
import numpy as np


curr_path = os.getcwd()
sys.path.append(curr_path)

__all__ = []


# define function
def eval_predict(predict_file, true_file):
    """
    :param predict_file:
    :param true_file:
    :return:
    """
    # read predict
    predict_out = pd.read_csv(predict_file, sep="\t", header=None).to_numpy()
    predict_out = np.argmax(predict_out, axis=1)

    # read true
    with open(true_file) as f:
        con = f.readlines()[1:]
    true_out = [eval(co.replace("\n", "").split(",")[1]) for co in con]
    true_out = np.array(true_out)

    # calculate
    acc = metrics.accuracy_score(y_true=true_out, y_pred=predict_out)
    f1_score = metrics.f1_score(y_true=true_out, y_pred=predict_out, average='macro')
    cf_matrix = metrics.confusion_matrix(y_true=true_out, y_pred=predict_out)

    ret = {"acc": acc, "f1_score": f1_score, "cf_matrix": cf_matrix}
    print("predict_file" + predict_file)
    print(ret)

    return ret


# main
if __name__ == '__main__':
    predict_file_name_list = ["temp/ccf_output_dir_4_100_test_results.tsv"]
    true_file_name = "data/ccf_data_zr_deal/dev_1.csv"

    for file in predict_file_name_list:
        ret = eval_predict(predict_file=file, true_file=true_file_name)
