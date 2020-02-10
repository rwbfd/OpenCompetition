#lda降维二分类只能降维到一维

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def generate_y():
    np.random.seed(1001)
    y = np.random.randint(0,2,(100,1))
    return y

def LDA():

    lda = LinearDiscriminantAnalysis()
    X,y = generateData(),generate_y()
    lda.fit(X,y)
    X_new = lda.transform(X)
    return X_new

def featureGeneration():
    lda = LDA()
    compents = pd.DataFrame(lda)
    compents.columns = ['new_x' + str(i) for i in range(1, 1 + 1)]
    new_data = pd.concat([generateData(), compents], axis=1)
    return new_data

if __name__ == '__main__':
    print(featureGeneration())
