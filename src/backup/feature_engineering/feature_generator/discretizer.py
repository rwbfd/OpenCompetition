# coding = 'utf-8'
from sklearn.preprocessing import KBinsDiscretizer
from src.tabular.feature_engineering.feature_generator.utils import concat_df_list, retrun_df_list, \
    get_continue_feature


class dis_configure:
    """
    The config object of discretizer. It saves the parameters of discretizer and check their validity.

    Parameters
    ----------
    method : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D k-means
            cluster.

    n_bins : int, default=5
    index_col : str, default='id'
        the col of df_list's DataFrame index col
    """

    method = None
    n_bins = None
    index_col = None

    def _check(self):
        if self.method is None:
            self.method = "quantile"
        elif self.method not in ['uniform', 'quantile', 'kmeans']:
            raise ValueError(
                "the method value {} is Invalid! It must be in ['uniform', 'quantile', 'kmeans'], "
                "default is 'quantile'".format(
                    str(self.method)))

        if self.n_bins is None:
            self.n_bins = 5
        elif not isinstance(self.n_bins, int):
            raise ValueError(
                "the n_bins value {} is Invalid! It must be Int value "
                "default is 5".format(
                    str(self.n_bins)))

        if self.index_col is None:
            self.index_col = "id"
        elif not isinstance(self.index_col, str):
            raise ValueError(
                "the index_col value {} is Invalid! It must be str value "
                "default is 'id'".format(
                    str(self.index_col)))

    def __init__(self, method, n_bins, index_col):
        self.method = method
        self.n_bins = n_bins
        self.index_col = index_col
        self._check()


def discretizer(df_list, names, config):
    """
    concat the df_list as one pd.DataFrame then using sklean.KBinsDiscretizer depart it,

    Parameters
    ----------
    df_list: object
        a collection of one or more pd.DataFrame. they must have one column named 'id' for indexing.
    names : list
        a list of the continuous variable column's name. If it's none,checking if all columns are continuous and
        discrete the continuous variable columns.
 
    config : object
        the object of parameters

    Returns
    ----------
    df_list_t : object
        the df_list after trans ,still is the collection of pd.DataFrame
    """

    data = concat_df_list(df_list, config.index_col)

    if names is None:
        names, _ = get_continue_feature(data)

    for name in names:
        kbdis = KBinsDiscretizer(n_bins=config.n_bins, encode="ordinal", strategy=config.method)
        kbdis.fit(data[name])
        data.loc[:, name + "_discred"] = kbdis.transform(data[name])

    df_list_t = retrun_df_list(df_list, data, config)
    return df_list_t
