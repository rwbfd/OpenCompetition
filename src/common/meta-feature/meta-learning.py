import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class feature_extraction:
    def __init__(self):
        pass

    def fHard(self, data_set,label_Wk):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m))
        knn = KNeighborsClassifier(n_neighbors=3) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        for i in range(data_set.shape[0]):
            if pred[i] == label_Wk[i]:
                K[0][i] = 1
        return K

    def fProb(self, data_set,label_Wk):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m))
        knn = KNeighborsClassifier(n_neighbors=3) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        for i in range(data_set.shape[0]):
            if pred[i] == label_Wk[i]:
                K[0][i] = 1
        return K


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # x = np.array([[0.5,0.4],[0.1,0.2],[0.7,0.8],[0.2,0.1],[0.4,0.6],[0.9,0.9],[1,1]]).reshape(-1,2)
    # y = np.array([0,1,0,1,0,1,1]).reshape(-1,1)
    meta = feature_extraction()
    k = meta.fHard(x,y)
    print('k:',k.shape, "x: ", x.shape)
