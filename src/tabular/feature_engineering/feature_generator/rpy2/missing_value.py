from rpy2.robjects.vectors import DataFrame, StrVector
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

r = robjects.r

base = importr('base')
mice = importr('mice')
stats = importr('stats')

class MissingValue:
    def __init__(self):
        pass

    def missing_value_count(self,df):
        nan_lists = {}
        for col in df.columns:
            nan_counter = 0
            for nan in df[col].isnull():
                if nan:
                    nan_counter += 1
                    nan_lists[col] = nan_counter
        for k,v in nan_lists.items():
            print('feature{},total missing value count{}'.format(k,v))

    def imputing_missing_data(self, data, methods = 'ppm', target=False):
        '''
        use R package(MICE) to impute missing data
        :param data: dataframe
        :param methods: 21.bin, 21.norm, 2lonly.mean, 2lonly.pmm,jomoImpute,logreg,mean,norm,norm.nob,panImpute,pmm,polyreg,rf,sample,
                        2l.lmer,2l.pan,2lonly.norm,cart,lda,logreg.boot,midastouch,norm.boot,norm.predict,passive,polr,quadratic,ri
        :param target: data contains predictive variable, target is a predictive variable
        :return: dataframe
        '''
        #如果缺失值数量大于95%，删除该行或该列
        data.index = range(0,data.shape[0])
        for raw in data.index:
            if data.iloc[raw,:].count()/len(data.iloc[raw,:]) < 0.95:
                data.drop(index=raw,axis=0,inplace=True)
        for col in data.columns:
            if data[[col]].count()/len(data[[col]]) < 0.95:
                data.drop(col,axis=1,inplace=True)

        if not target:
            impute = mice.mice(data, m=5, maxit=50, meth=methods, seed=500)
            data = mice.complete(impute, 1)
            return data

        if target:
            temp = mice.mice(data)
            fit = base.with(temp, stats.lm("target ~ ."))
            pooled = mice.pool(fit)
            data = mice.complete(temp, 1)
            return data







