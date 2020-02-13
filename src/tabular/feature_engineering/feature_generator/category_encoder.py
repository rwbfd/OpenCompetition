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
from src.tabular.feature_engineering.utils import set_default_vale
import pandas as pd
import json


class CategoryEncoders:
    def __init__(self):
        pass

    def get_Xy(self, df, configger):
        """
        this function perform the getting the X, y and the encode cols of the encoders needing
        :param df: the training dataset
        :param configger: the config setting json.
        :return: the X, y, encode_col
        """
        target_col = configger["target_col"]
        encode_col = configger["encode_col"]
        if encode_col == '':
            X = df.drop([target_col], axis=1)
            encode_col = None
        else:
            X = df[encode_col]

        y = df[target_col]
        return X, y, encode_col

    # supervised encoders
    def target_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
            min_samples_leaf: int
                minimum samples to take category average into account.
            smoothing: float
                smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
                The value must be strictly bigger than 0.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        min_samples_leaf = set_default_vale("min_samples_leaf", configger, 1)
        smoothing = set_default_vale("smoothing", configger, 1.0)

        encoder = TargetEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                handle_missing=handle_missing,
                                handle_unknown=handle_unknown, min_samples_leaf=min_samples_leaf, smoothing=smoothing)

        res = encoder.fit_transform(X, y)

        return res

    def catboost_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
            sigma: float
                adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                sigma gives the standard deviation (spread or "width") of the normal distribution.
            a: float
                additive smoothing (it is the same variable as "m" in m-probability estimate). By default set to 1.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        random_state = set_default_vale("random_state", configger, None)
        sigma = set_default_vale("sigma", configger, None)
        a = set_default_vale("a", configger, 1)

        encoder = CatBoostEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                  handle_unknown=handle_unknown, handle_missing=handle_missing,
                                  random_state=random_state, sigma=sigma, a=a)

        res = encoder.fit_transform(X, y)

        return res

    def james_stein_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop encoded columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
            handle_unknown: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
            model: str
                options are 'pooled', 'beta', 'binary' and 'independent', defaults to 'independent'.
            randomized: bool,
                adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
            sigma: float
                standard deviation (spread or "width") of the normal distribution.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        model = set_default_vale("model", configger, "independent")
        random_state = set_default_vale("random_state", configger, None)
        randomized = set_default_vale("randomized", configger, False, is_bool=True)
        sigma = set_default_vale("sigma", configger, 0.05)

        encoder = JamesSteinEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                    handle_unknown=handle_unknown, handle_missing=handle_missing, model=model,
                                    random_state=random_state, randomized=randomized, sigma=sigma)

        res = encoder.fit_transform(X, y)

        return res

    def leaveone_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'error', 'return_nan'  and 'value', defaults to 'value', which returns the target mean.
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
            sigma: float
                adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing
                data are untouched). Sigma gives the standard deviation (spread or "width") of the normal distribution.
                The optimal value is commonly between 0.05 and 0.6. The default is to not add noise, but that leads
                to significantly suboptimal results.
        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        random_state = set_default_vale("random_state", configger, None)
        sigma = set_default_vale("sigma", configger, None)

        encoder = LeaveOneOutEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                     handle_unknown=handle_unknown, handle_missing=handle_missing,
                                     random_state=random_state, sigma=sigma)

        res = encoder.fit_transform(X, y)

        return res

    def mestimate_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop encoded columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
            handle_unknown: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which returns the prior probability.
            randomized: bool,
                adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
            sigma: float
                standard deviation (spread or "width") of the normal distribution.
            m: float
                this is the "m" in the m-probability estimate. Higher value of m results into stronger shrinking.
                M is non-negative.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        random_state = set_default_vale("random_state", configger, None)
        randomized = set_default_vale("randomized", configger, False, is_bool=True)
        sigma = set_default_vale("sigma", configger, 0.05)
        m = set_default_vale("m", configger, 1.0)

        encoder = MEstimateEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                   handle_unknown=handle_unknown, handle_missing=handle_missing,
                                   random_state=random_state, randomized=randomized, sigma=sigma, m=m)

        res = encoder.fit_transform(X, y)

        return res

    def weight_of_evidence_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_missing: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
            handle_unknown: str
                options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
            randomized: bool,
                adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
            sigma: float
                standard deviation (spread or "width") of the normal distribution.
            regularization: float
                the purpose of regularization is mostly to prevent division by zero.
                When regularization is 0, you may encounter division by zero.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        random_state = set_default_vale("random_state", configger, None)
        randomized = set_default_vale("randomized", configger, False, is_bool=True)
        sigma = set_default_vale("sigma", configger, 0.05)
        regularization = set_default_vale("regularization", configger, 1.0)

        encoder = WOEEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                             handle_unknown=handle_unknown, handle_missing=handle_missing,
                             random_state=random_state, randomized=randomized, sigma=sigma,
                             regularization=regularization)

        res = encoder.fit_transform(X, y)

        return res

    def binary_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_unknown: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has unknown categories.  This can cause
                unexpected changes in dimension in some cases.
            handle_missing: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has nan values.  This can cause
                unexpected changes in dimension in some cases.
        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = BinaryEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def backward_difference_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_unknown: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has unknown categories.  This can cause
                unexpected changes in dimension in some cases.
            handle_missing: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has nan values.  This can cause
                unexpected changes in dimension in some cases.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = BackwardDifferenceEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                            handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def basen_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            base: int
                when the downstream model copes well with nonlinearities (like decision tree), use higher base.
            handle_unknown: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has unknown categories.  This can cause
                unexpected changes in dimension in some cases.
            handle_missing: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has nan values.  This can cause
                unexpected changes in dimension in some cases.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        base = set_default_vale("base", configger, 2)

        encoder = BaseNEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True, base=base,
                               handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def hash_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            hash_method: str
                which hashing method to use. Any method from hashlib works.
            max_process: int
                how many processes to use in transform(). Limited in range(1, 64).
                By default, it uses half of the logical CPUs.
                For example, 4C4T makes max_process=2, 4C8T makes max_process=4.
                Set it larger if you have a strong CPU.
                It is not recommended to set it larger than is the count of the
                logical CPUs as it will actually slow down the encoding.
            max_sample: int
                how many samples to encode by each process at a time.
                This setting is useful on low memory machines.
                By default, max_sample=(all samples num)/(max_process).
                For example, 4C8T CPU with 100,000 samples makes max_sample=25,000,
                6C12T CPU with 100,000 samples makes max_sample=16,666.
                It is not recommended to set it larger than the default value.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        max_process = set_default_vale("max_process", configger, 0)
        max_sample = set_default_vale("max_sample", configger, 0)
        n_components = set_default_vale("n_components", configger, 8)
        hash_method = set_default_vale("hash_method", configger, "md5")

        encoder = HashingEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                 max_process=max_process, max_sample=max_sample, n_components=n_components,
                                 hash_method=hash_method)

        res = encoder.fit_transform(X, y)

        return res

    def helmert_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            handle_unknown: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has unknown categories.  This can cause
                unexpected changes in dimension in some cases.
            handle_missing: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has nan values.  This can cause
                unexpected changes in dimension in some cases.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = HelmertEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                 handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def one_hot_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            use_cat_names: bool
                if True, category values will be included in the encoded column names. Since this can result in duplicate column names, duplicates are suffixed with '#' symbol until a unique name is generated.
                If False, category indices will be used instead of the category values.
            handle_unknown: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has unknown categories.  This can cause
                unexpected changes in dimension in some cases.
            handle_missing: str
                options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
                an extra column will be added in if the transform matrix has nan values.  This can cause
                unexpected changes in dimension in some cases.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")
        use_cat_names = set_default_vale("use_cat_names", configger, False, is_bool=True)

        encoder = OneHotEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                use_cat_names=use_cat_names,
                                handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def ordinal_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            mapping: list of dict
                a mapping of class to label to use for the encoding, optional.
                the dict contains the keys 'col' and 'mapping'.
                the value of 'col' should be the feature name.
                the value of 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
                example mapping: [{'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}}]
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
            handle_missing: str
                options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
                or -2 at transform time if nan is not a category during fit.
        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = OrdinalEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                 handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def polynomial_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            mapping: list of dict
                a mapping of class to label to use for the encoding, optional.
                the dict contains the keys 'col' and 'mapping'.
                the value of 'col' should be the feature name.
                the value of 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
                example mapping: [{'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}}]
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
            handle_missing: str
                options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
                or -2 at transform time if nan is not a category during fit.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = PolynomialEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                                    handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res

    def sum_encoder(self, df, configger):
        """

        :param df: the train dataset.
        :param configger: the json str of configger setting, the params means:
            verbose: int
                integer indicating verbosity of the output. 0 for none.
            cols: list
                a list of columns to encode, if None, all string columns will be encoded.
            drop_invariant: bool
                boolean for whether or not to drop columns with 0 variance.
            return_df: bool
                boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
            mapping: list of dict
                a mapping of class to label to use for the encoding, optional.
                the dict contains the keys 'col' and 'mapping'.
                the value of 'col' should be the feature name.
                the value of 'mapping' should be a dictionary of 'original_label' to 'encoded_label'.
                example mapping: [{'col': 'col1', 'mapping': {None: 0, 'a': 1, 'b': 2}}]
            handle_unknown: str
                options are 'error', 'return_nan' and 'value', defaults to 'value', which will impute the category -1.
            handle_missing: str
                options are 'error', 'return_nan', and 'value, default to 'value', which treat nan as a category at fit time,
                or -2 at transform time if nan is not a category during fit.

        :return: the transform result
        """
        X, y, encode_col = self.get_Xy(df, configger)

        drop_invariant = set_default_vale("drop_invariant", configger, False, is_bool=True)
        handle_missing = set_default_vale("handle_missing", configger, "value")
        handle_unknown = set_default_vale("handle_unknown", configger, "value")

        encoder = SumEncoder(verbose=1, cols=encode_col, drop_invariant=drop_invariant, return_df=True,
                             handle_unknown=handle_unknown, handle_missing=handle_missing)

        res = encoder.fit_transform(X, y)

        return res


def name_feature(df, method):
    ori_names = list(df.columns)
    new_names = []
    for name in ori_names:
        new_names.append(name + "_after_" + method + "_encode")

    df.columns = new_names


def category_encode(df, configger):
    """
    :param df: the training dataset.
    :param configger: the json str of configger setting.
    :return:df_t: the dataset with transform result.The column after trans named like ori_name + "_after_" + method + "_encode"
    """
    configger = json.loads(configger)

    method = configger["method"]
    if method not in ['Target', 'BaseN', 'Binary', 'CatBoost', 'Hash', 'Helmert', 'JamesStein', 'LeaveOneOut',
                      'MEstimate', 'OneHot', 'Ordinal', 'Polynomial', 'Sum', 'WOE']:
        raise ValueError(
            "the method value must in ['Target','BaseN','Binary','CatBoost','Hash','Helmert','JamesStein','LeaveOneOut','MEstimate','OneHot','Ordinal','Polynomial','Sum','WOE'], "
            "'{method}' is not be support.".format(
                method=method))

    encoder = CategoryEncoders()
    if method == "Target":
        res = encoder.target_encoder(df, configger)
    elif method == "BaseN":
        res = encoder.basen_encoder(df, configger)
    elif method == "Binary":
        res = encoder.binary_encoder(df, configger)
    elif method == "CatBoost":
        res = encoder.catboost_encoder(df, configger)
    elif method == "Hash":
        res = encoder.hash_encoder(df, configger)
    elif method == "Helmert":
        res = encoder.helmert_encoder(df, configger)
    elif method == "JamesStein":
        res = encoder.james_stein_encoder(df, configger)
    elif method == "LeaveOneOut":
        res = encoder.leaveone_encoder(df, configger)
    elif method == "MEstimate":
        res = encoder.mestimate_encoder(df, configger)
    elif method == "OneHot":
        res = encoder.one_hot_encoder(df, configger)
    elif method == "Ordinal":
        res = encoder.ordinal_encoder(df, configger)
    elif method == "Polynomial":
        res = encoder.polynomial_encoder(df, configger)
    elif method == "Sum":
        res = encoder.sum_encoder(df, configger)
    else:
        res = encoder.weight_of_evidence_encoder(df, configger)

    name_feature(res, method)

    df_t = pd.concat([df, res], axis=1)
    return df_t
