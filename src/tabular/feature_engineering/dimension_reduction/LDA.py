import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


def lda_reduce(df, configger):
    """
    Parameters
    ----------
    df: pd.DataFrame. the input DataFrame.
    configger: collections.namedtuple. the configger object like
        namedtuple("config",["reduce_col","target_col","n_components","solver","shrinkage","priors","store_covariance","tol"])
        reduce_col: list,default is None.
            The feature columns to reduce. If it's None,reduce all features.

        target_col: str
            The target column name.

        solver : string, optional
            Solver to use, possible values:
              - 'svd': Singular value decomposition (default).
                Does not compute the covariance matrix, therefore this solver is
                recommended for data with a large number of features.
              - 'lsqr': Least squares solution, can be combined with shrinkage.
              - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.

        shrinkage : string or float, optional
            Shrinkage parameter, possible values:
              - None: no shrinkage (default).
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

            Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

        priors : array, optional, shape (n_classes,)
            Class priors.

        n_components : int, optional (default=None)
            Number of components (<= min(n_classes - 1, n_features)) for
            dimensionality reduction. If None, will be set to
            min(n_classes - 1, n_features).

        store_covariance : bool, optional
            Additionally compute class covariance matrix (default False), used
            only in 'svd' solver.

        tol : float, optional, (default 1.0e-4)
            Threshold used for rank estimation in SVD solver.

    Returns
    -------
    df_t: pd.DataFrame. The result columns named like 'LDA_component_${n}'
    """
    reduce_col = configger.reduce_col
    target_col = configger.target_col
    solver = configger.solver
    shrinkage = configger.shrinkage
    priors = configger.priors
    n_components = configger.n_components
    store_covariance = configger.store_covariance
    tol = configger.tol

    if reduce_col is None:
        reduce_col = list(df.columns)
        reduce_col.remove(target_col)

    lda = LinearDiscriminantAnalysis(solver = solver,shrinkage = shrinkage,priors =priors,store_covariance= store_covariance,n_components=n_components,tol = tol)
    X = df[reduce_col]
    y = df[target_col]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True)
    lda.fit(X_train, y_train)

    res = lda.transform(X=X)
    names = ("LDA_component_" + str(i) for i in range(res.shape[1]))

    res = pd.DataFrame(res, columns=names)
    df_t = pd.concat([df, res], axis=1)

    return df_t
