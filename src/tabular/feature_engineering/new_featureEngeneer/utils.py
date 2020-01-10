# encoding:utf-8

import pandas as pd

def get_continue_feature(data):
    continus_features, discrete_features = [], []
    for col in data.columns:
        if data[col].dtype != object:
            try:
                pd.qcut(data[col], 4)
                continus_features.append(col)
            except ValueError:
                discrete_features.append(col)
    print('Continus features are: ', continus_features)
    print('Discrete features are: ', discrete_features)
    return continus_features, discrete_features
