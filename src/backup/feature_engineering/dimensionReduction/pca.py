# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用PCA做数据降维
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def generateData():
    """
    随机生成训练数据
    """
    np.random.seed(1001)
    x = np.random.rand(100, 20)
    data = pd.DataFrame(x)
    colnames = ['x'+str(i) for i in range(1,len(x[0])+1)]
    data.columns = colnames
    return data


def pca_components(data):
    """
    使用PCA对数据进行降维
    """
    model = PCA(n_components=5)
    #model.fit(data)
    new_x = model.fit_transform(data)
    return new_x

def featureGeneration(data):
    pca = pca_components(data)
    compents = pd.DataFrame(pca)
    compents.columns = ['new_x' + str(i) for i in range(1, 5 + 1)]
    new_data = pd.concat([data, compents], axis=1)
    return new_data


if __name__ == "__main__":
    data = generateData(200)
    # #print(data)
    # model = trainModel(data)
    # #print(model.shape)
    # compents = pd.DataFrame(model)
    # compents.columns = ['new_x'+str(i) for i in range(1,5+1)]
    # new_data = pd.concat([data,compents],axis=1)
    new_data = featureGeneration(data)
    print(new_data)

