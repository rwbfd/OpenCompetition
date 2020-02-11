from rpy2.robjects.packages import importr
stats = importr('stats')
mvoutlier = importr('mvoutlier')
extremevalues = importr('extremevalues')
base = importr('base')

import pandas as pd

class OutlierDetection:
    '''
    To detect outliers with R packages,mvoutlier and extremevalues
    '''
    def __init__(self):
        pass


    def detect_outlier_with_mvoutlier(self, df, col, alpha=0.025):
        '''
        :param df: dataframe
        :param col: to detect the column weather has outlier
        :param alpha: uni.plot function mvoutlier 
        :return: 
        '''
        df.index = range(1,df.shape[0]+1)
        p1 = mvoutlier.uni.plot(x=df[[col]], symb=True, quan=1 / 2, alpha=alpha)
        outlier_index = base.which("p1$outliers == T")  #r index start from 1
        df[str(col)+'_outlier'] = pd.Series([1 if i in outlier_index else 0 for i in df.index])
        return df

    def detect_outlier_with_extremevalues(self, df,col,method='I'):
        '''
        :param df: 
        :param col: to detect the column weather has outlier
        :param method: getOutliers is a wrapper function for getOutliersI and getOutliersII.
        :return: 
        '''
        df.index = range(df.shape[0])
        p1 = extremevalues.getOutliers(df[[col]], method=method)
        outlier_index = base.which("p1$outliers == T")
        df[str(col) + '_outlier'] = pd.Series([1 if i in outlier_index else 0 for i in df.index])
        return df


if __name__ == '__main__':
    import numpy as np
    import rpy2.robjects as robjects

    r = robjects.r
    df = pd.DataFrame({'a':[r.rnorm(800)],'b':[r.rnorm(80, 5, 1)]})
    s = OutlierDetection()
    a = 'a'
    print(s.detect_outlier_with_mvoutlier(df, a, alpha=0.025))










