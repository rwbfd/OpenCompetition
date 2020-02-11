# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:30:19 2019

@author: 74293
"""

# true_label = []
# predice_label[]

# 传入两个list 分别是true_idx predict_idx
# 使用get()函数得到一个1*3的列表 其中idx默认为空 如果输入idx 则返回对应的label的precision recall f1 否则返回整体的

import numpy as np


class get_evaluation:
    def __init__(self, true_label, predict_label):
        self.label2idx = self.Label2idx(true_label, predict_label)
        # print (self.label2idx)
        self.confMatrix = np.zeros([len(self.label2idx), len(self.label2idx)], dtype=np.int32)
        for i in range(len(true_label)):
            true_labels_idx = self.label2idx[true_label[i]]
            predict_labels_idx = self.label2idx[predict_label[i]]
            self.confMatrix[true_labels_idx][predict_labels_idx] += 1

    def get(self, idx=None):
        self.prediction = []
        self.recall = []
        self.f1 = []
        for i in self.label2idx:
            self.prediction.append(self.calculate_label_prediction(self.confMatrix, self.label2idx[i]))
            self.recall.append(self.calculate_label_recall(self.confMatrix, self.label2idx[i]))
            self.f1.append(self.calculate_f1(self.prediction[-1], self.recall[-1]))
        if idx == None:
            #  print ((self.prediction, len(self.prediction)))
            self.all_prediction = round(sum(self.prediction) / len(self.prediction), 2)
            self.all_recall = round(sum(self.recall) / len(self.recall), 2)
            self.f1 = round(sum(self.f1) / len(self.f1), 2)
            return ([self.all_prediction, self.all_recall, self.f1])
        else:

            return ([self.prediction[idx], self.recall[idx], self.f1[idx]])

    def Label2idx(self, labels1, labels2):
        label2idx = {}
        for i in labels1:
            if i not in label2idx:
                label2idx[i] = len(label2idx)
        for i in labels2:
            if i not in label2idx:
                label2idx[i] = len(label2idx)
        return label2idx

    def calculate_all_prediction(self, confMatrix):  # 对角线上面所有值除以总数
        total_sum = confMatrix.sum()
        correct_sum = (np.diag(confMatrix)).sum()
        prediction = round(100 * float(correct_sum) / float(total_sum), 2)
        return prediction

    def calculate_label_recall(self, confMatrix, labelidx):  # 计算某一个的召回率
        label_total_sum = confMatrix.sum(axis=1)[labelidx]
        label_correct_sum = confMatrix[labelidx][labelidx]
        recall = 0
        if label_total_sum != 0:
            recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        return recall

    def calculate_label_prediction(self, confMatrix, labelidx):
        # 计算某一个类标预测精度：该类被预测正确的数除以该类的总数
        label_total_sum = confMatrix.sum(axis=0)[labelidx]
        label_correct_sum = confMatrix[labelidx][labelidx]
        prediction = 0
        if label_total_sum != 0:
            prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
        return prediction

    def calculate_f1(self, prediction, recall):
        if (prediction + recall) == 0:
            return 0
        return round(2 * prediction * recall / (prediction + recall), 2)
