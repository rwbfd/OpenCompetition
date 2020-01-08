# coding = 'utf-8'
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


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
    """

    method = None
    n_bins = None

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

    def __init__(self, method, n_bins):
        self.method = method
        self.n_bins = n_bins
        self._check()


def discretizer(df_list, names, config):
    """
    concat the df_list as one pd.DataFrame then using sklean.KBinsDiscretizer depart it,

    Parameters
    ----------
    df_list : object
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
    pass

