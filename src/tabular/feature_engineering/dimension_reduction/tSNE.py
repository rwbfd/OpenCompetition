from sklearn.manifold import TSNE
import pandas as pd


def tsne_reduce(df, configger):
    """

    Parameters
    ----------
    df: pd.DataFrame. the input dataFrame.
    configger:
    the config setting of tSNE like namedtuple("config",
    ["n_components","perplexity","early_exaggeration","learning_rate",
        "n_iter","n_iter_without_progress","min_grad_norm","metric","init","verbose","random_state","method","angle","n_jobs"]).
        n_components : int, optional (default: 2)
            Dimension of the embedded space.

        perplexity : float, optional (default: 30)
            The perplexity is related to the number of nearest neighbors that
            is used in other manifold learning algorithms. Larger datasets
            usually require a larger perplexity. Consider selecting a value
            between 5 and 50. Different values can result in significanlty
            different results.

        early_exaggeration : float, optional (default: 12.0)
            Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them. For
            larger values, the space between natural clusters will be larger
            in the embedded space. Again, the choice of this parameter is not
            very critical. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high.

        learning_rate : float, optional (default: 200.0)
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers. If the cost function gets stuck in a bad local
            minimum increasing the learning rate may help.

        n_iter : int, optional (default: 1000)
            Maximum number of iterations for the optimization. Should be at
            least 250.

        n_iter_without_progress : int, optional (default: 300)
            Maximum number of iterations without progress before we abort the
            optimization, used after 250 initial iterations with early
            exaggeration. Note that progress is only checked every 50 iterations so
            this value is rounded to the next multiple of 50.

            parameter *n_iter_without_progress* to control stopping criteria.

        min_grad_norm : float, optional (default: 1e-7)
            If the gradient norm is below this threshold, the optimization will
            be stopped.

        metric : string or callable, optional
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scipy.spatial.distance.pdist for its metric parameter, or
            a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            If metric is "precomputed", X is assumed to be a distance matrix.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from X as input and return a value indicating
            the distance between them. The default is "euclidean" which is
            interpreted as squared euclidean distance.

        init : string or numpy array, optional (default: "random")
            Initialization of embedding. Possible options are 'random', 'pca',
            and a numpy array of shape (n_samples, n_components).
            PCA initialization cannot be used with precomputed distances and is
            usually more globally stable than random initialization.

        verbose : int, optional (default: 0)
            Verbosity level.

        random_state : int, RandomState instance or None, optional (default: None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.  Note that different initializations might result in
            different local minima of the cost function.

        method : string (default: 'barnes_hut')
            By default the gradient calculation algorithm uses Barnes-Hut
            approximation running in O(NlogN) time. method='exact'
            will run on the slower, but exact, algorithm in O(N^2) time. The
            exact algorithm should be used when nearest-neighbor errors need
            to be better than 3%. However, the exact method cannot scale to
            millions of examples.

               Approximate optimization *method* via the Barnes-Hut.

        angle : float (default: 0.5)
            Only used if method='barnes_hut'
            This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
            'angle' is the angular size (referred to as theta in [3]) of a distant
            node as measured from a point. If this size is below 'angle' then it is
            used as a summary node of all points contained within it.
            This method is not very sensitive to changes in this parameter
            in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
            computation time and angle greater 0.8 has quickly increasing error.

        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run for neighbors search. This parameter
            has no impact when ``metric="precomputed"`` or
            (``metric="euclidean"`` and ``method="exact"``).
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

    Returns
    -------
    df_t: the result dataFrame. the new feature named like "tsne_component_${n}"

    """
    n_components = configger.n_components
    perplexity = configger.perplexity
    early_exaggeration = configger.early_exaggeration
    learning_rate = configger.learning_rate
    n_iter = configger.n_iter
    n_iter_without_progress = configger.n_iter_without_progress
    min_grad_norm = configger.min_grad_norm
    metric = configger.metric
    init = configger.init
    verbose = configger.verbose
    random_state = configger.random_state
    method = configger.method
    angle = configger.angle
    n_jobs = configger.n_jobs

    tsne = TSNE(n_components=n_components, init=init, random_state=random_state, perplexity=perplexity, method=method,
                n_iter=n_iter, verbose=verbose, learning_rate=learning_rate, early_exaggeration=early_exaggeration,
                n_iter_without_progress=n_iter_without_progress, min_grad_norm=min_grad_norm, metric=metric,
                angle=angle, n_jobs=n_jobs)
    X_tsne = tsne.fit_transform(df)
    names = ["tsne_component_" + str(i) for i in range(n_components)]
    res = pd.DataFrame(X_tsne, columns=names)
    df_t = pd.concat([df, res], axis=1)
    return df_t
