import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, PowerTransformer
from scipy.special import erfinv
from scipy.stats import kstest, norm


def check_distribution(input_df):
    distribution_types = ["norm", "uniform"]
    p_values = {}
    for distribution_type in distribution_types:
        statistic, p_value = kstest(rvs=input_df, cdf=distribution_type)
        p_values[distribution_type] = p_value

    if max(p_values.values()) < 0.05:
        return "others"

    else:
        if p_values["norm"] >= p_values["uniform"]:
            return "norm"
        else:
            return "uniform"


class GaussRankScaler():

    def __init__(self):
        self.epsilon = 1e-9
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform(self, X):
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        j_range = len(j) - 1

        transformed = j / (j_range / self.range)
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed


def encode_continuous_variable(df, configgers):
    """
    encoder of continuous variables , the method can be in
    ["MinMax","MinAbs","Normalize","Robust","BoxCox","Yeo-Johnson","RankGuass","icdf"]

    Parameters
    ----------
    df: pd.DataFrame, the input dataframe.
    configgers: str ,the config setting json str of encoding continuous variables
        like {"${encode_col}":{"methods": "${method_value}" ,"arg_key of method":arg_value}}

        If you choose method is "MinMax", there are more Fields of config.
            feature_range : tuple [min, max] default=[0, 1]
                Desired range of transformed data.

        If you choose method is "MinMax",  there are more Fields of config.
            norm_method : 'l1', 'l2', or 'max', optional ('l2' by default)
                The norm_method to use to normalize each non zero sample.

        If you choose method is "Robust",  there are more Fields of config.
            quantile_range : tuple [q_min, q_max], 0.0 < q_min < q_max < 100.0
                Default: [25.0, 75.0] = (1st quantile, 3rd quantile) = IQR

        If you choose method is "BoxCox" or "Yeo-Johnson", there are more Fields of config.
            standardize : int, default=1, will trans it to boolean
                Set to True to apply zero-mean, unit-variance normalization to the
                transformed output.

    Returns
    -------
    df_t: pd.DataFrame, the DataFrame after transform, the result columns after transform named like "{origin_column_name}_{method}"
    """
    df_t = df
    configgers = json.loads(configgers)

    for encode_col in configgers.keys:
        method = configgers[encode_col]["method"]
        distribution_type = check_distribution(df[encode_col])

        if distribution_type == "uniform":
            if method == "MinMax":
                try:
                    feature_range = configgers[encode_col]["feature_range"]
                    assert feature_range is not None
                except (AttributeError, AssertionError):
                    feature_range = (0, 1)

                res = MinMaxScaler(feature_range=feature_range).fit_transform(df[[encode_col]])

            elif method == "MinAbs":
                res = MaxAbsScaler().fit_transform(df[[encode_col]])

            elif method == "Normalize":

                try:
                    norm_method = configgers[encode_col]["norm_method"]
                    assert norm_method is not None
                except (AttributeError, AssertionError):
                    norm_method = "l2"

                res = Normalizer(norm=norm_method).fit_transform(df[[encode_col]])

            elif method == "Robust":

                try:
                    quantile = configgers[encode_col]["quantile"]
                    assert quantile is not None
                    with_scaling = False
                except (AttributeError, AssertionError):
                    quantile = (25.0, 75.0)
                    with_scaling = True

                res = RobustScaler(with_centering=True, with_scaling=with_scaling,
                                   quantile_range=quantile).fit_transform(
                    df[[encode_col]])

            else:
                raise ValueError(
                    """The column '{}' most likely a uniformly distribution. So, the method value must be in ["MinMax", "MinAbs", "Normalize", "Robust"]""".format(
                        encode_col))

        elif distribution_type == "norm":
            if method == "BoxCox":
                try:
                    standardize = configgers[encode_col]["standardize"]
                    assert standardize is not None
                    standardize = bool(int(standardize))
                except (AttributeError, AssertionError):
                    standardize = True

                res = PowerTransformer(method="box-cox", standardize=standardize).fit_transform(df[[encode_col]])

            elif method == "Yeo-Johnson":
                try:
                    standardize = configgers[encode_col]["standardize"]
                    assert standardize is not None
                    standardize = bool(int(standardize))
                except (AttributeError, AssertionError):
                    standardize = True

                res = PowerTransformer(method="yeo-johnson", standardize=standardize).fit_transform(df[[encode_col]])

            elif method == "RankGuass":
                res = GaussRankScaler().fit_transform(df[[encode_col]])

            else:
                raise ValueError(
                    """The column '{}' most likely a normal distribution. So, the method value must be in ["BoxCox", "Yeo-Johnson", "Normalize"]""".format(
                        encode_col))

        else:
            if method == "ICDF":
                #TODO add ICDF
                pass
            else:
                raise ValueError(
                    """The column '{}' maybe is others distribution. So, the method value must be in ["BoxCox", "Yeo-Johnson", "Normalize"]""".format(
                        encode_col))

        df_t.loc[:, "_".join([encode_col, method])] = res

    return df_t
