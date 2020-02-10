# encoding:utf-8
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split


class GPConfig:
    def __init__(self, feature_cols, target_col, generation=1000, population_size=5000, hall_of_fame=100,
                 n_components=10,
                 parsimony_coefficient=0.0005, max_samples=0.9):
        self.generation = generation
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']
        self.feature_cols = feature_cols
        self.target_col = target_col


def symbolicLearning(df, gp_config):
    """

    Parameters
    ----------
    df: pd.DataFrame,the input dataFrame.
    gp_config: GPConfig object, the config object of gplearn.SymbolicTransformer.

    Returns
    -------
    df_t: pd.DataFrame, df with the features of SymbolicTransformer trans.
        The new features named like 'symbolic_component_{0 to n}'(n is the n_components)
    """

    gp = SymbolicTransformer(generations=gp_config.generation, population_size=gp_config.population_size,
                             hall_of_fame=gp_config.hall_of_fame, n_components=gp_config.n_components,
                             function_set=gp_config.function_set,
                             parsimony_coefficient=gp_config.parsimony_coefficient,
                             max_samples=gp_config.max_samples, verbose=1,
                             random_state=0, n_jobs=3)

    X = df[gp_config.feature_cols]
    y = df[gp_config.target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    gp.fit(X_train, y_train)
    names = ["symbolic_component_" + str(i) for i in range(gp_config.n_components)]
    res = pd.DataFrame(gp.transform(X),columns=names)
    df_t = pd.concat([df,res],axis=1)
    return df_t


