# coding = 'utf-8'
import numpy as np
from sklearn.utils import check_array
from sklearn.preprocessing import KBinsDiscretizer


def discretizer(df_list, names, method_list):
    """
    Parameters
    ----------
    df_list: pd.DataFrame type, the dataframe need to split.
    names: list of column names
    method_list: a dictionary contains the methods as key, and parameters as values.
    key must in ['isometric','quantile','kmeans']
    Like {"isometric":[n_bins]},{"quantile":[n_bins]},{"kmeans":[n_bins]}

    Returns
    -------
    discretizers : list type. the discretizers of all column
    data: the np.array after trans
    """
    if names is None:
        data = check_array(df_list)
    else:
        data = check_array(df_list[names])

    if len(method_list.keys) > 1:
        raise ValueError("method_list only can has 1 key")

    method = list(method_list.keys)[0]
    if method not in ("isometric", "quantile", "kmeans"):
        raise ValueError("`method` must be 'isometric','quantile' or 'kmeans'")

    discretizers = []
    if method == "isometric":
        for column in range(np.shape(data)[0]):
            discretizer = KBinsDiscretizer(n_bins=method_list[method], encode="ordinal", strategy="uniform")
            fit_encoder(column, data, discretizer, discretizers)

    elif method == "quantile":
        for column in range(np.shape(data)[0]):
            discretizer = KBinsDiscretizer(n_bins=method_list[method], encode="ordinal", strategy="quantile")
            fit_encoder(column, data, discretizer, discretizers)
    else:
        for column in range(np.shape(data)[0]):
            discretizer = KBinsDiscretizer(n_bins=method_list[method], encode="ordinal", strategy="kmeans")
            fit_encoder(column, data, discretizer, discretizers)

    return discretizers, data


def fit_encoder(column, data, discretizer, discretizers):
    discretizer.fit(data[:, column])
    data[:, column] = discretizer.transform(data[:, column])
    discretizers.append(discretizer)
