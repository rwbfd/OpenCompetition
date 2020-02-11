from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder


class CategoryEncoders:
    def __init__(self):
        pass
    #supervised encoders
    def target_encoder(self,X,y,cols=None):
        encoder = TargetEncoder(cols=cols)
        return encoder.fit_transform(X,y)

    def catboost_encoder(self,X,y,cols=None):
        encoder = CatBoostEncoder(cols=cols)
        return encoder.fit_transform(X, y)

    def james_stein_encoder(self,X,y,cols=None):
        encoder = JamesSteinEncoder(cols=cols)
        return encoder.fit_transform(X, y)

    def leaveone_encoder(self,X,y,cols=None):
        encoder = LeaveOneOutEncoder(cols=cols)
        return encoder.fit_transform(X, y)

    def mestimate_encoder(self,X,y,cols=None):
        encoder = MEstimateEncoder(cols=cols)
        return encoder.fit_transform(X, y)

    def weight_of_evidence_encoder(self,X,y,cols=None):
        encoder = WOEEncoder(cols=cols)
        return encoder.fit_transform(X, y)


    #unspervised encoders
    def binary_encoder(self,df, col=None):
        '''
        :param col: a list of columns to encode, if None, all string columns will be encoded.
        :param df: 
        :return: 
        '''
        encoder = BinaryEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def backward_difference_encoder(self,df, col=None):
        '''
        
        :param df: 
        :param col: a list of columns to encode, if None, all string columns will be encoded.
        :return: 
        '''
        encoder = BackwardDifferenceEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def basen_encoder(self,df,col=None):
        '''
        :param df: 
        :param col: a list of columns to encode, if None, all string columns will be encoded.
        :return: 
        '''
        encoder = BaseNEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def hash_encoder(self,df, col=None):
        encoder = HashingEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def helmert_encoder(self,df,col=None):
        encoder = HelmertEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def one_hot_encoder(self,df,col=None):
        encoder = OneHotEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def ordinal_encoder(self,df,col=None):
        encoder = OrdinalEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def polynomial_encoder(self,df,col=None):
        encoder = PolynomialEncoder(cols=col).fit(df)
        return encoder.transform(df)

    def sum_encoder(self,df,col=None):
        encoder = SumEncoder(cols=col).fit(df)
        return encoder.transform(df)



if __name__ == '__main__':
    import pandas as pd
    #from sklearn.datasets import load_breast_cancer
    data = pd.read_csv('transfusion_index.csv')

    # df = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6],
    #                    'RATING': ['G', 'B', 'G', 'B', 'B', 'G'],
    #                    'agsht':['A','E','A','E','E','A']})
    s = CategoryEncoders()
    #encoder = s.binary_encoder(df, ['RATING','agsht'])
    #encoder = s.backward_difference_encoder(df, ['RATING','agsht'])
    #encoder = s.basen_encoder(df, ['RATING', 'agsht'])
    #encoder = s.hash_encoder(df, ['RATING', 'agsht'])
    #encoder = s.helmert_encoder(df, ['RATING', 'agsht'])
    # encoder = s.ordinal_encoder(df, ['RATING', 'agsht'])
    data = data.copy()
    X = data.drop('Bone',axis=1)
    y = data[['Bone']]
    encoder = s.target_encoder(X,y,cols=['Lung','Race'])
    # encoder = s.james_stein_encoder(X,y)
    # encoder = s.weight_of_evidence_encoder(X,y)
    # encoder = s.mestimate_encoder(X,y)
    # encoder = s.leaveone_encoder(X,y)

    print(encoder)



