import gc
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed


# +
class CategoryEncoder:
    def __columns_type(self):

        feats = self.__feats
        category_columns = self.__category_columns
        numeric_columns = self.__numeric_columns
        dtypes = self.__dtypes
        type_object = type(object)

        for col in feats:
            if dtypes[col] == type_object:
                category_columns.append(col)
            else:
                numeric_columns.append(col)

        for col in category_columns:
            feats.remove(col)

        self.__category_columns = category_columns
        self.__numeric_columns = numeric_columns
        self.__feats = feats

    def __init__(self, train_df, test_df, id_column, target_column):

        self.__train_df = train_df
        self.__test_df = test_df
        self.__category_columns = []
        self.__numeric_columns = []
        self.__LE_columns = []
        self.__columns = train_df.columns.values
        self.__train_shape = train_df.shape
        self.__test_shape = test_df.shape
        self.__dtypes = train_df.dtypes
        self.__id_column = id_column
        self.__target_column = target_column
        columns = list(np.copy(self.__columns))
        columns.remove(id_column)
        columns.remove(target_column)
        self.__feats = columns
        self.__columns_type()

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        self.__logger = logger

    def __drop_columns(self, category_columns):

        train_df = self.__train_df
        test_df = self.__test_df

        for col in category_columns:
            train_df.drop(col, axis=1, inplace=True)
            test_df.drop(col, axis=1, inplace=True)

        gc.collect()

        self.__train_df = train_df
        self.__test_df = test_df

    def get_columns_type(self):

        return self.__numeric_columns, self.__category_columns

    def __fill_nan(self, category_columns):

        train_df = self.__train_df
        test_df = self.__test_df

        for col in category_columns:
            train_df[col].fillna('Nan', inplace=True)
            test_df[col].fillna('Nan', inplace=True)

        self.__train_df = train_df
        self.__test_df = test_df

    def __encoder(self, col):

        feats = []
        LE_columns = []
        train_shape = self.__train_shape
        test_shape = self.__test_shape

        traincol_series = self.__train_df[col]
        testcol_series = self.__test_df[col]

        _train_ce = np.zeros(train_shape[0], dtype=int)
        _test_ce = np.zeros(test_shape[0], dtype=int)

        _train_lce = np.zeros(train_shape[0], dtype=int)
        _test_lce = np.zeros(test_shape[0], dtype=int)

        _train_ce_log = np.zeros(train_shape[0], dtype=int)
        _test_ce_log = np.zeros(test_shape[0], dtype=int)

        _train_fe = np.zeros(train_shape[0], dtype=float)
        _test_fe = np.zeros(test_shape[0], dtype=float)

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        le = LabelEncoder()

        # Label Encoding
        _train = traincol_series.values
        _test = testcol_series.values
        _all = np.r_[_train, _test]

        le.fit(_all)
        _train_le = le.transform(_train)
        _test_le = le.transform(_test)

        col_LE = col + '_LE'
        train_df[col_LE] = _train_le
        test_df[col_LE] = _test_le
        feats.append(col_LE)
        LE_columns.append(col_LE)

        col_LE_log = col_LE + '_log'
        train_df[col_LE_log] = np.log(_train_le + 1)
        test_df[col_LE_log] = np.log(_test_le + 1)
        feats.append(col_LE_log)

        col_FE = col + '_FE'
        train_df[col_FE] = _train_le / train_shape[0]
        test_df[col_FE] = _test_le / test_shape[0]
        feats.append(col_FE)

        # Count Encoding and Label Count Encoding
        value_counts = train_df[col_LE].value_counts()
        _index = value_counts.index.values
        value_counts = value_counts.to_dict()
        rank = pd.Series(np.arange(_index.shape[0], 0, -1), index=_index).to_dict()

        # train
        for iter, value in enumerate(_train_le):
            _train_ce[iter] = value_counts[value]
            _train_lce[iter] = rank[value]

        _set_index = set(_index)
        for iter, value in enumerate(_test_le):
            if set([value]) <= _set_index:
                _test_ce[iter] = value_counts[value]
                _test_lce[iter] = rank[value]
            else:
                _test_ce[iter] = 1
                _test_lce[iter] = 1

        col_CE = col + '_CE'
        train_df[col_CE] = _train_ce
        test_df[col_CE] = _test_ce
        feats.append(col_CE)

        col_LCE = col + '_LCE'
        train_df[col_LCE] = _train_lce
        test_df[col_LCE] = _test_lce
        feats.append(col_LCE)

        return train_df, test_df, feats, LE_columns

    def encoding(self, mycategory_columns=None, n_jobs=-1):

        feats = self.__feats
        train_shape = self.__train_shape
        test_shape = self.__test_shape
        LE_columns = self.__LE_columns

        if mycategory_columns == None:
            category_columns = self.__category_columns
        else:
            category_columns = mycategory_columns

        self.__fill_nan(category_columns)

        train_df = self.__train_df
        test_df = self.__test_df

        train_df = self.__reduce_mem_usage(train_df)
        test_df = self.__reduce_mem_usage(test_df)

        _train_ce = np.zeros(train_shape[0], dtype=int)
        _test_ce = np.zeros(test_shape[0], dtype=int)

        _train_lce = np.zeros(train_shape[0], dtype=int)
        _test_lce = np.zeros(test_shape[0], dtype=int)

        encoder = self.__encoder
        result = (Parallel(n_jobs=n_jobs, verbose=3)([delayed(encoder)(x) for x in category_columns]))

        for iter, r in enumerate(result):
            train_df = pd.concat([train_df, r[0]], axis=1)
            test_df = pd.concat([test_df, r[1]], axis=1)
            feats.extend(r[2])
            LE_columns.extend(r[3])

        train_df = self.__reduce_mem_usage(train_df)
        test_df = self.__reduce_mem_usage(test_df)

        self.__train_df = train_df
        self.__test_df = test_df
        self.__feats = feats
        self.__LE_columns = LE_columns

        self.__drop_columns(category_columns)

    def target_encoding(self, folds=5, n_jobs=-1, verbose=3):

        LE_columns = self.__LE_columns
        self.__folds = folds
        train_df = self.__train_df
        test_df = self.__test_df
        feats = self.__feats

        encoder = self.__target_encoder
        result = (Parallel(n_jobs=n_jobs, verbose=3)([delayed(encoder)(x) for x in LE_columns]))

        for iter, r in enumerate(result):
            train_df = pd.concat([train_df, r[0]], axis=1)
            test_df = pd.concat([test_df, r[1]], axis=1)
            feats.extend(r[2])

        train_df = self.__reduce_mem_usage(train_df)
        test_df = self.__reduce_mem_usage(test_df)

        self.__train_df = train_df
        self.__test_df = test_df
        self.__feats = feats

    def __target_encoder(self, col):

        feats = []
        train_df = self.__train_df
        test_df = self.__test_df
        folds = self.__folds
        target_column = self.__target_column
        train_shape = self.__train_shape
        test_shape = self.__test_shape

        _train_te_mean = np.zeros(train_shape[0], dtype=float)
        _test_te_mean = np.zeros(test_shape[0], dtype=float)

        _train_te_std = np.zeros(train_shape[0], dtype=float)
        _test_te_std = np.zeros(test_shape[0], dtype=float)

        te_train_df = pd.DataFrame()
        te_test_df = pd.DataFrame()

        train_values = train_df[col].values
        test_values = test_df[col].values

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[target_column])):

            train_x = train_df.iloc[train_idx]
            valid_x = train_df.iloc[valid_idx]

            value_counts_mean = train_x.groupby(col)[target_column].mean()
            value_counts_std = train_x.groupby(col)[target_column].std().to_dict()
            _index_set = set(value_counts_mean.index.values)
            value_counts_mean = value_counts_mean.to_dict()

            for index in valid_idx:
                value = train_values[index]
                if set([value]) <= _index_set:
                    _train_te_mean[index] = value_counts_mean[value]
                    _train_te_std[index] = value_counts_std[value]
                else:
                    _train_te_mean[index] = 0.5
                    _train_te_std[index] = 0

        col_te_mean = col[:-3] + '_TE_mean'
        col_te_std = col[:-3] + '_TE_std'
        te_train_df[col_te_mean] = _train_te_mean
        feats.append(col_te_mean)
        te_train_df[col_te_std] = _train_te_std
        feats.append(col_te_std)

        value_counts_mean = train_df.groupby(col)[target_column].mean()
        value_counts_std = train_df.groupby(col)[target_column].std().to_dict()
        _index_set = set(value_counts_mean.index.values)
        value_counts_mean = value_counts_mean.to_dict()

        for iter, value in enumerate(test_values):
            if set([value]) <= _index_set:
                _test_te_mean[iter] = value_counts_mean[value]
                _test_te_std[iter] = value_counts_std[value]
            else:
                _test_te_mean[iter] = 0.5
                _test_te_std[iter] = 0

        te_test_df[col_te_mean] = _test_te_mean
        te_test_df[col_te_std] = _test_te_std

        return te_train_df, te_test_df, feats

    def get_df(self):

        train_df = self.__train_df
        test_df = self.__test_df
        feats = self.__feats

        return train_df, test_df, feats

    def __reduce_mem_usage(self, df):
        start_mem = df.memory_usage().sum() / 1024 ** 2

        for col in df.columns:
            col_type = df[col].dtype
            if col_type != 'object' and col_type != 'datetime64[ns]':
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024 ** 2

        return df


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pandas as pd

    bunch = load_boston()
    y = pd.DataFrame(bunch.target, columns=['Target'])
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    df = pd.concat([X, y], axis=1)
    df['Id'] = range(df.shape[0])

    train_df = df.iloc[:405, :]
    test_df = df.iloc[405:, :]

    ce = CategoryEncoder(train_df, test_df, 'Id', 'Target')
    ce.encoding()
    print(ce.get_df())
