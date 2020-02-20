import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.situation
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
# import rpy2.interactive as r
#
# r.packages.utils.install_packages("mvoutlier")
# r.packages.utils.install_packages("extremevalues")

utils = importr("utils")
base = importr('base')
#extremevalues = importr('extremevalues')
outliers = importr('outliers')

class OutlierDetection:
    '''
    To detect outliers with R packages,mvoutlier and extremevalues
    '''
    def __init__(self):
        pass


    def detect_outlier_with_outliers(self, df, col):
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
            detect_outlier_with_outliers <- function(df,s){
                index <- 1:dim(df)[1]
                cols <- unlist(strsplit(s, split = "-"))
                for (column in cols){
                    p1 <- outlier(df[,column],opposite=TRUE)
                    outlier_index <- as.vector(which(df[,column] == p1))
                    col_outlier <- sample(c(0), dim(df)[1], replace=T)
                    col_outlier[outlier_index] <- 1
                    new_col <- paste(column, 'outlier', sep = "_")
                    df[,new_col] <- col_outlier
                    }
                return(df)
                }

                    ''')

        rf = rpy2.robjects.r['detect_outlier_with_outliers']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data = ro.conversion.rpy2py(rf(r_from_pd_df,col))
        return data

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    print(OutlierDetection().detect_outlier_with_outliers(df,'V1-V3-V4-V5-V7-V9'))
