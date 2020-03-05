import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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
        for j in range(len(pwl)):
            for i in range(m):
                if pwl[j] == label_Wk[i]:
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

    def fRankOp(self, data_set,label_Wk,n):
        pass

    def fRank(self, data_set, query_sample, n, label_Wk):
        '''给定query sample，计算所有samples与query sample 的距离，升序。计算连续分类正确的数量'''
        m = data_set.shape[0]
        distance = []
        for k in range(m):
            dist = np.linalg.norm(data_set.lioc[k,:] - query_sample) #Euclidean distance
            distance.append(dist)
        distance_df = pd.DataFrame(distance,index=list(range(m)))
        distance_df.sort()
        knn = KNeighborsClassifier(n_neighbors=n)  # set neighbors to 3
        knn.fit(data_set, label_Wk)
        pred = knn.predict(data_set[distance_df.index])
        for j in range(pred):
            if pred[j] != label_Wk[distance_df.index][j]:
                return j

    def fOp(self, data_set):
        pass
    def fKL(self, data_set, label_Wk, n, n_classifier_pooling):
        m = data_set.shape[0]
        K = np.zeros((1, m))
        knn = KNeighborsClassifier(n_neighbors=n)  # set neighbors to 3
        knn.fit(data_set, label_Wk)
        S_x = knn.predict_proba(data_set)
        n_classes = set(label_Wk).count()
        results = []
        for k in range(m):
            s_x_k = S_x[k]
            f_kl_k = 0
            RC = [1/n_classes for _ in range(n_classifier_pooling)]
            for i in range(1,n_classes+1):
                f_kl_k += s_x_k * np.log(s_x_k/RC)
            results.append(f_kl_k)
        return results


    def fExp(self, data_set, label_Wk, n):
        m = data_set.shape[0]
        K = np.zeros((1, m))
        knn = KNeighborsClassifier(n_neighbors=n)  # set neighbors to 3
        knn.fit(data_set, label_Wk)
        S_x = knn.predict_proba(data_set)
        n_classes = set(label_Wk).count()
        results = []
        for k in range(m):
            s_x_k = S_x[k]
            zeros_list = [0 for i in range(n_classes)]
            zeros_list[label_Wk[k]] = 1
            s_x_l_k = zeros_list
            f_exp_k = 1- 2 ^ (-((n_classes-1)*s_x_l_k)/(1-s_x_l_k))
            results.append(f_exp_k)
        return results

    def fEnt(self, data_set, label_Wk, n):
        m = data_set.shape[0]
        K = np.zeros((1, m))
        knn = KNeighborsClassifier(n_neighbors=n)  # set neighbors to 3
        knn.fit(data_set, label_Wk)
        S_x = knn.predict_proba(data_set)
        n_classes = set(label_Wk).count()
        results = []
        for k in range(m):
            s_x_k = S_x[k]
            sum = 0
            for j in range(n_classes):
                sum += s_x_k[j] * np.log(s_x_k[j])
            f_ent_k = -sum
            results.append(f_ent_k)
        return results

    def fMD(self, data_set, label_Wk, n):
        m = data_set.shape[0]
        K = np.zeros((1, m))
        knn = KNeighborsClassifier(n_neighbors=n)  # set neighbors to 3
        knn.fit(data_set, label_Wk)
        S_x = knn.predict_proba(data_set)
        n_classes = set(label_Wk).count()
        results = []
        for k in range(m):
            s_x_k = S_x[k]
            zeros_list = [0 for i in range(n_classes)]
            zeros_list[label_Wk[k]] = 1
            s_x_l_k = zeros_list
            f_md_k = np.min(s_x_k - s_x_l_k)
            results.append(f_md_k)
        return results

    def fPRC(self, data_set):
        pass




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