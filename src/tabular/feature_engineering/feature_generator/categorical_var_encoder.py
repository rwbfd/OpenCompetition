# coding = 'utf-8'
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import check_array

def cat_encoder(df_list, names, method_list):
    """
    Parameters
    ----------
        df_list : pandas.DataFrame type, the Input pandas DataFrame which need to encode and transform.
        names: str or List[str], which columns need to encode trans.
        method_list: dict type, key must in ['onehot', 'ordinal','hashmap']
        like {'onehot': None}, {'ordinal': None},{'hashmap' : args*}

    Returns
    -------
        encoders: list type, the dict of encoder objects,.
        data : pandas.DataFrame type, the Dataframe after encoder transform.
    """
    if names is None:
        data = check_array(df_list)
    else:
        data = check_array(df_list[names])

    if len(method_list.keys) > 1:
        raise ValueError("method_list only can has 1 key")

    method = list(method_list.keys)[0]
    if method not in ("onehot", "ordinal", "hashmap"):
        raise ValueError("`method` must be 'onehot','ordinal' or 'hashmap'")

    encoders = []
    if method == "onehot":
        for column in range(np.shape(data)[0]):
            encoder = OneHotEncoder()
            data[:, column] = encoder.fit(data[:, column])
            encoders.append(encoder)

    elif method == "ordinal":
        for column in range(np.shape(data)[0]):
            encoder = OrdinalEncoder()
            data[:, column] = encoder.fit(data[:, column])
            encoders.append(encoder)

    return encoders, data
