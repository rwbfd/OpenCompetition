# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn.decomposition import PCA

def pca_reduce(df, configger):
    """

    Parameters
    ----------
    df
    configger

        n_components : int, float, None or str
                Number of components to keep.
                if n_components is not set all components are kept::

                    n_components == min(n_samples, n_features)

                If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
                MLE is used to guess the dimension. Use of ``n_components == 'mle'``
                will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

                If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
                number of components such that the amount of variance that needs to be
                explained is greater than the percentage specified by n_components.

                If ``svd_solver == 'arpack'``, the number of components must be
                strictly less than the minimum of n_features and n_samples.

                Hence, the None case results in::

                    n_components == min(n_samples, n_features) - 1

            whiten : bool, optional (default False)
                When True (False by default) the `components_` vectors are multiplied
                by the square root of n_samples and then divided by the singular values
                to ensure uncorrelated outputs with unit component-wise variances.

                Whitening will remove some information from the transformed signal
                (the relative variance scales of the components) but can sometime
                improve the predictive accuracy of the downstream estimators by
                making their data respect some hard-wired assumptions.

            svd_solver : str {'auto', 'full', 'arpack', 'randomized'}
                If auto :
                    The solver is selected by a default policy based on `X.shape` and
                    `n_components`: if the input data is larger than 500x500 and the
                    number of components to extract is lower than 80% of the smallest
                    dimension of the data, then the more efficient 'randomized'
                    method is enabled. Otherwise the exact full SVD is computed and
                    optionally truncated afterwards.
                If full :
                    run exact full SVD calling the standard LAPACK solver via
                    `scipy.linalg.svd` and select the components by postprocessing
                If arpack :
                    run SVD truncated to n_components calling ARPACK solver via
                    `scipy.sparse.linalg.svds`. It requires strictly
                    0 < n_components < min(X.shape)
                If randomized :
                    run randomized SVD by the method of Halko et al.

            tol : float >= 0, optional (default .0)
                Tolerance for singular values computed by svd_solver == 'arpack'.

            iterated_power : int >= 0, or 'auto', (default 'auto')
                Number of iterations for the power method computed by
                svd_solver == 'randomized'.

            random_state : int, RandomState instance or None, optional (default None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    Returns
    -------
    df_t: the result dataFrame. the new feature named like "pca_component_${n}"

    """

    n_components = configger.n_components
    whiten = configger.whiten
    svd_solver = configger.svd_solver
    tol = configger.tol
    iterated_power = configger.iterated_power
    random_state = configger.random_state

    pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power,
              random_state=random_state)
    pca_res = pca.fit_transform(df)
    names = ["pca_components_" + str(i) for i in range(pca_res.shape[1])]
    res = pd.DataFrame(pca_res, columns=names)
    df_t = pd.concat([df, res], axis=1)
    return df_t
