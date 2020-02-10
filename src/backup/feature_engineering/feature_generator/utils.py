# encoding:utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

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


def check_index_col(data, index_col):
    """
    check the index column's values are unique.

    Parameters
    ----------
    data : pd.Dataframe
    config : object,
        the config parameter object.

    Returns
    ----------
    data : pd.Dataframe,
        origin data
    """

    index_col_data = data[index_col]
    if index_col_data.shape[1] == index_col_data.drop_duplicates().shape[1]:
        return data
    else:
        raise ValueError("the index column '{}' values must be unique".format(index_col))


def concat_df_list(df_list, config):
    """
    concat the df_list as one dataframe
    Parameters
    ----------
    df_list : pd.DataFrame or collection of pd.DataFrame
        the origin collection of pd.DataFrame
    config : object
        the object of parameters

    Returns
    -------
    df

    """
    if isinstance(df_list, tuple) or isinstance(df_list, list):
        data = pd.concat(df_list, axis=1)
    elif isinstance(df_list, pd.DataFrame):
        data = df_list
    else:
        raise ValueError("paramter df_list must be the collection of one or more pd.DataFrame")

    df = check_index_col(data, config)
    return df


def retrun_df_list(df_list, data, index_col):
    """

    Parameters
    ----------
    df_list :object
        a collection of one or more pd.DataFrame. they must have one column named 'id' for indexing.
    data : pd.DataFrame
        the DataFrame after discrete
    index_col : str
        the index col of  pd.DataFrame

    Returns
    -------
    df_list_t :
        a collection of one or more pd.DataFrame, they are transformed.
    """
    df_list_t = list()
    if isinstance(df_list, tuple) or isinstance(df_list, list):
        for i in range(len(df_list)):
            df_list_t.append(df_list[i][[index_col]].merge(data, on=index_col))

    elif isinstance(df_list, pd.DataFrame):
        df_list_t.append(df_list[[index_col]].merge(data, on=index_col))
    else:
        raise ValueError("paramter df_list must be the collection of one or more pd.DataFrame")

    return df_list_t


def get_train_data(X, y):
    """
    Parameters
    ----------
    X pd.DataFram
    y pd.DataFram

    Returns
    -------
    X_train
    y_train

    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, y_train