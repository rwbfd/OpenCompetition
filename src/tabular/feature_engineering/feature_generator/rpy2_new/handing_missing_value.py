import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.situation
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import rpy2.interactive as r

r.packages.utils.install_packages("mice")
utils = importr("utils")
base = importr('base')
mice = importr('mice')

class MissingValue:
    def __init__(self):
        pass

    def missing_value_count(self, df):
        nan_lists = {}
        for col in df.columns:
            nan_counter = 0
            for nan in df[col].isnull():
                if nan:
                    nan_counter += 1
                    nan_lists[col] = nan_counter
        for k, v in nan_lists.items():
            print('feature{},total missing value count{}'.format(k, v))



    def handinng_missing_value(self, df, col, method = 'pmm'):
        '''
        
        :param df: 
        :param col: list,连续变量
        :return: 
        '''

        with localconverter(ro.default_converter + pandas2ri.converter):
          r_from_pd_df = ro.conversion.py2rpy(df)

        with localconverter(ro.default_converter + pandas2ri.converter):
          pd_from_r_df = ro.conversion.rpy2py(r_from_pd_df)

        rpy2.robjects.r('''
              imputing_missing_data <- function(df, cols, methods = 'pmm'){
                df_continus_variable <- df[,cols]
                for (i in 1:dim(df_continus_variable)[1]){
                    if(sum(is.na(df_continus_variable[i,]))/length(df_continus_variable[i,]) > 0.2){
                        df_continus_variable <- df_continus_variable[-i,]
                    }
                }
                for (i in 1:dim(df_continus_variable)[2]){
                    if(sum(is.na(df_continus_variable[,i]))/length(df_continus_variable[,i]) > 0.2){
                        df_continus_variable <- df_continus_variable[,-i]
                    }
              
                impute <- mice(df_continus_variable, m=5, maxit=50, meth=methods, seed=500)
                df_continus_variable <- complete(impute, 1)
                df_other <- df
                df_other[,cols] <- NULL
                df <- cbind(df_continus_variable,df_other)
                df
                
            }
            ''')


        rf = rpy2.robjects.r['imputing_missing_data']
        with localconverter(ro.default_converter + pandas2ri.converter):
            data= ro.conversion.rpy2py(rf(r_from_pd_df,col,methods = method))
        return data

if __name__ == '__main__':
    df = pd.read_csv('data.csv')

    print(MissingValue().missing_value_count(df))
    print(MissingValue().handinng_missing_value(df,['V1','V2','V3','V4','V5','V6','V7','V8','V9'],method = 'pmm'))