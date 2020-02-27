# coding = 'utf-8'
import numpy as np
import pandas as pd


def split_df(df, fold):
    """
    This function splits the dataframe into n folds, specified by the fold parameters
    :param df: The dataframe to be splitted.
    :param fold: The number of folds.
    :return: A list of dataframes.
    """
    df_copy_shuffled = df.copy(deep=True).sample(frac=1)
    splitted_df = np.array_split(df_copy_shuffled, fold)
    return splitted_df


def concat_df(df_list, index_col=None, shuffle=True):
    """
    This function concats a list of dataframes into a single one.
    :param df_list: A list of dataframes.
    :param index_col: Index column, if None, then assumes that there is no repetition in the dataframes.
    :return:  The concated data frames.
    """
    concat_df = pd.concat(df_list)
    if index_col is None:
        return _util_concat_df(concat_df, shuffle)
    else:
        concat_df = concat_df.drop_duplicates().reset_index(drop=True)
        return _util_concat_df(concat_df, shuffle)


def _util_concat_df(concat_df, shuffle):
    if shuffle:
        return concat_df.sample(frac=1)
    else:
        return concat_df
