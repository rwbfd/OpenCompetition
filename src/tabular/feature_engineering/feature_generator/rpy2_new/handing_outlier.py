import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.situation
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.interactive as r

utils = importr("utils")
base = importr('base')
extremevalues = importr('extremevalues')
mvoutlier = importr('mvoutlier')

class OutlierDetection:
    '''
    To detect outliers with R packages,mvoutlier and extremevalues
    '''
    def __init__(self):
        pass


    def detect_outlier_with_extremevalues(self, df, col):
        '''
        :param df: dataframe
        :param col: str, to detect the column weather has outlier
        :param alpha: uni.plot function mvoutlier 
        :return: 
        '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(df)

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(r_from_pd_df)

        rpy2.robjects.r('''
            detect_outlier_with_extremevalues <- function(df,s){
                index <- 1:dim(df)[1]
                cols <- unlist(strsplit(s, split = "-"))
                for (column in cols){
                    p1 <- getOutliers(df[,column],method='I')
                    outlier_index <- as.vector(which(p1$outliers == T))
                    col_outlier <- sample(c(0), dim(df)[1], replace=T)
                    col_outlier[outlier_index] <- 1
                    new_col <- paste(column, 'outlier', sep = "_")
                    df[,new_col] <- col_outlier
                    }
                return(df)
                }
                ''')

        rf = rpy2.robjects.r['detect_outlier_with_extremevalues']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(rf(r_from_pd_df,col))
        return data

    def detect_outlier_with_mvoutlier(self, df, col):
        '''
        :param df: 
        :param col: str, to detect the column weather has outlier
        :param method: getOutliers is a wrapper function for getOutliersI and getOutliersII.
        :return: 
        '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(df)

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(r_from_pd_df)

        rpy2.robjects.r('''
            detect_outlier_with_mvoutlier <- function(df,s){
                index <- 1:dim(df)[1]
                cols <- unlist(strsplit(s, split = "-"))
                for (column in cols){
                    p1 <- uni.plot(x=df[,column], symb=True, quan=1/2, alpha=0.5)
                    outlier_index <- as.vector(which(p1$outliers == T))
                    col_outlier <- sample(c(0), dim(df)[1], replace=T)
                    col_outlier[outlier_index] <- 1
                    new_col <- paste(column, 'outlier', sep = "_")
                    df[,new_col] <- col_outlier
                    }
                return(df)
                }
                    ''')

        rf = rpy2.robjects.r['detect_outlier_with_mvoutlier']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(rf(r_from_pd_df,col))
        return data

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    print(OutlierDetection().detect_outlier_with_extremevalues(df,'V1-V3-V4-V5-V7-V9'))
    print(OutlierDetection().detect_outlier_with_mvoutlier(df,'V1-V3-V4-V5-V7-V9'))