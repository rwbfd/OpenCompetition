import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math

class feature_extraction:
    def __init__(self):
        pass

    def fHard(self, data_set,label_Wk, n):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m)) #meta feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        for i in range(data_set.shape[0]):
            if pred[i] == label_Wk[i]:
                K[0][i] = 1
        return K

    def fProb(self, data_set,label_Wk,n):
        #Posterior probability： P（A|B) = P(B|A) * P(A) / P(B) [p(wl | xk)] A = wl and B = xk (p(B|A) * p(A) = p(A and B))
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m))
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set) #predict list
        #doing prob algorithm here
        #creating list to store pwl
        pwl = []
        pa = []
        tmp = 0
        for i in range(data_set.shape[0]):
            #counting for a classifer
            if label_Wk[i] not in pwl:
                pwl.append(label_Wk[i])
        for i in range(len(pwl)):
            for j in range(data_set.shape[0]):
                if label_Wk[j] == pwl[i]:
                    tmp += 1
            pa.append(tmp/data_set.shape[0])
            tmp = 0
        # calculate pb
        pxk = []
        pb = []
        tmp = 0
        for i in range(data_set.shape[0]):
            #counting for a classifer
            if pred[i] not in pxk:
                pxk.append(pred[i])
        for i in range(len(pxk)):
            for j in range(data_set.shape[0]):
                if pred[j] == pxk[i]:
                    tmp += 1
            pb.append(tmp/data_set.shape[0])
            tmp = 0
        #shape of pa and pb are same which is equal to classes number
        pt = (pa + pb) / 2 #pa and pb happen in the same time
        for j in range(len(pwl))
            for i in range(m):
                if pwl[j] == label_Wk[i]
                    K[0][i] = pt[j] / pb[j]
        return K

    def fOverall(self, data_set,label_Wk,n):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = 0 #meta feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        tmp = 0
        Wl = 0
        for i in range(m):
            if pred[i] == label_Wk[i]:
                tmp += 1
        K = tmp / m
        return K

    def fCond(self, data_set,label_Wk,n):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = 0 #meta feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        tmp = 0
        Wl = 0
        for i in range(m):
            if pred[i] == label_Wk[i]:
                tmp += 1
        top = tmp / m
        #calculate bottom
        #get xk number
        xk = []
        wl = []
        tmp = 0
        for i in range(m):
            #counting for all classifer
            if pred[i] not in xk:
                xk.append(pred[i])
        for i in range(m):
            #counting for all wl
            if label_Wk[i] not in wl:
                wl.append(label_Wk[i])
        bottom = len(wl) / len(xk)
        K = top / bottom
        return K

    def fConf(self, data_set,label_Wk,n,Max, Min):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = 0 #meta feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        tmp = 0
        Wl = 0
        for i in range(m):
            if pred[i] == label_Wk[i]:
                tmp += 1
        top = tmp / m
        #calculate bottom
        #get xk number
        xk = []
        wl = []
        tmp = 0
        for i in range(m):
            #counting for all classifer
            if pred[i] not in xk:
                xk.append(pred[i])
        for i in range(m):
            #counting for all wl
            if label_Wk[i] not in wl:
                wl.append(label_Wk[i])
        bottom = len(wl) / len(xk)
        K = top / bottom
        K = (K - Min) / (Max - Min)
        return K

    def fAmb(self, data_set,label_Wk,n,ci):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = 0 #meta feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        xk = []
        for i in range(m):
            #counting for all classifer
            if pred[i] not in xk:
                xk.append(pred[i])
        value_famb = ci(xk)
        srt = value_famb.sort(key=int)
        first_large = srt[0]
        second_large = srt[1]
        K = first_large - second_large
        return K

    def fLog(self, data_set,label_Wk,n,ci):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m)) #meta-feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        xk = []
        for i in range(m):
            #counting for all classifer
            if pred[i] not in xk:
                xk.append(pred[i])
        S = ci(xk)
        for i in range(m):
            K[i] = 2 * (S[i] ** (math.log(2) / math.log(m))) - 1
        return K

    def fPRC(self, data_set,label_Wk,n,prc):
        #data_set shape will be [n_samples,shape] (k, m)
        #label_Wk shape will be [n_smaples, 1]
        m = data_set.shape[0]
        K = np.zeros((1,m)) #meta-feature
        knn = KNeighborsClassifier(n_neighbors=n) #set neighbors to 3
        knn.fit(data_set,label_Wk)
        pred = knn.predict(data_set)
        for i in range(m):
            K[i] = prc(pred[i])
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
