# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-09-03
Description :
auther : wcy
"""
# import modules
import os, sys
import numpy as np

curr_path = os.getcwd()
sys.path.append(curr_path)
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

__all__ = []


# define function
def get_metrics_ops(labels, predictions, num_labels, weights):
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels, weights)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))

    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    precisions = []
    recalls = []
    f1s = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0
        f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)


    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1_mean = sum(f1s) / len(f1s)
    # f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1_mean


# main
if __name__ == '__main__':
    conf_mat = np.array([[0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00,0.00000e+00],
                          [0.00000e+00,2.31751e+05,2.50000e+01,4.70000e+01,8.70000e+01,5.78000e+02,7.20000e+01,1.98000e+02,7.20000e+01,1.26000e+02],
                          [0.00000e+00,3.70000e+01,3.16600e+03,8.00000e+00,2.00000e+00,0.00000e+00,7.00000e+00,1.00000e+01,1.00000e+00,0.00000e+00],
                          [0.00000e+00,6.20000e+01,7.00000e+00,5.71400e+03,0.00000e+00,2.00000e+00,0.00000e+00,3.50000e+01,0.00000e+00,0.00000e+00],
                          [0.00000e+00,3.40000e+01,1.00000e+00,0.00000e+00,1.49900e+03,7.00000e+00,8.50000e+01,0.00000e+00,0.00000e+00,0.00000e+00],
                          [0.00000e+00,1.82000e+02,0.00000e+00,0.00000e+00,2.00000e+00,7.61900e+03,1.20000e+01,1.66000e+02,0.00000e+00,0.00000e+00],
                          [0.00000e+00,4.10000e+01,7.00000e+00,0.00000e+00,6.90000e+01,4.00000e+00,3.74300e+03,4.30000e+01,0.00000e+00,1.00000e+00],
                          [0.00000e+00,1.16000e+02,0.00000e+00,2.40000e+01,1.00000e+00,1.47000e+02,2.60000e+01,6.41700e+03,0.00000e+00,0.00000e+00],
                          [0.00000e+00,5.70000e+01,0.00000e+00,0.00000e+00,1.00000e+00,2.00000e+00,1.00000e+00,0.00000e+00,2.78700e+03,3.00000e+01],
                          [0.00000e+00,6.60000e+01,0.00000e+00,0.00000e+00,0.00000e+00,4.00000e+00,0.00000e+00,0.00000e+00,2.60000e+01,6.26400e+03]])
    ret = get_metrics(conf_mat=conf_mat, num_labels=3)
    print(ret)

    # TURE 0.7404276583412609


