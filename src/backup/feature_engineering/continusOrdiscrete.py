import pandas as pd


def feature_type(data):
    continus_features,discrete_features = [],[]
    for col in data.columns:
        if data[col].dtype != object:
            try:
                pd.qcut(data[col], 4)
                continus_features.append(col)
            except ValueError:
                discrete_features.append(col)
    print('Continus features are: ', continus_features)
    print('\n')
    print('Discrete features are: ', discrete_features)

if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    train_df = pd.read_csv('train.csv', index_col=0)
    data = train_df.drop('SalePrice', axis=1)
    print(feature_type(data))
