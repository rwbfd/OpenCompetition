# encoding:utf-8

class named_tuple():

    def __init__(self, groupBy_keys, groupBy_values, method):
        """
        Parameters
        ----------
        groupBy_keys : object, the collection of groupBy key column names. It can be str of list of str
        groupBy_values : object, the collection of groupBy value column names. It can be str of list of str
        method : str,default is "mean", can be in ['min','max','mean','median','mode','entropy','std','skew','kurt']
        """
        self.groupBy_keys = groupBy_keys
        self.groupBy_values = groupBy_values
        self.method = method


def key_value_pairs(df_list, index_col, named_tuples):
    """
    do transform like df.groupby([A, B])[C, D].agg(func),
    func can in mean, std, skew, kurtosis, entropy, min, max, median, frequency,size.

    Parameters
    ----------

    df_list : object,  a collection of one or more pd.DataFrame. they must have one column named 'id' for indexing.
    index_col : str, the index column of df_list.
    named_tuples : list ,list of named_tuple object

    Returns
    -------
    df_list_t : object, the df_list after transform.
        the column of transform result named like "A_B\tC_D\t method".
        It sort by index column

    """
