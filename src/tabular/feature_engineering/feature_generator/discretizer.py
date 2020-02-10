import numpy as np
import json
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class DecisionTreeDiscretizer():
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit_transform(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,shuffle=True)
        clf = DecisionTreeClassifier(max_leaf_nodes=self.n_bins)
        clf.fit(X_train, y_train)
        pred = clf.predict(X)
        return pred

def discretizer(df, configgers):
    """
    Parameters
    ----------
    df: pd.DataFrame, the input DataFrame.
    configgers: list of namedtuple, the config setting of encoding continuous variables like namedtuple("config",["encode_col","method","n_bins"])
        encode_col: list,the column names of the columns need discretize
        method: str,must in ["isometric", "quantile", "KMeans", "trees"]
            if method choose tree, the field "target_col" can't be None.
        n_bins: int
        target_col: str, the target col's name

    Returns
    df_t: pd.DataFrame, the result DataFrame, the new feature column named like '"_".join(encode_col + [method, "discretize"])'
    -------
    """
    df_t = df
    configgers = json.loads(configgers)

    for encode_col in configgers.keys:
        method = configgers[encode_col]["method"]
        n_bins = configgers[encode_col]["n_bins"]

        if method == "isometric":
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")

        elif method == "quantile":
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")

        elif method == "KMeans":
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")

        elif method == "trees":
            discretizer = DecisionTreeDiscretizer(n_bins=n_bins)

        else:
            raise ValueError(
                """
                The method value {func} is not be support for discretizer. 
                It must be in ["isometric", "quantile", "KMeans", "trees"]""".format(
                    func=method))

        if method == "trees":
            target_col = configgers[encode_col]["target_col"]
            res = discretizer.fit_transform(X=df[encode_col], y=df[target_col])
        else:
            res = discretizer.fit_transform(X=df[encode_col])

        df_t.loc[:, "_".join(encode_col + [method, "discretize"])] = res

    return df_t
