# encoding:utf-8
"""
@author: sugang
@time: 2020/1/21 3:15 下午
@desc:
"""
import pandas as pd
from sklearn import datasets
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split



class GPConfig:
    def __init__(self, generation=1000, population_size=5000, hall_of_fame=100, n_components=10,
                 parsimony_coefficient=0.0005, max_samples=0.9):
        self.generation = generation
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']


def gp_feature_generator(df_train, df_test, y_name, var_list, gp_config):
    gp = SymbolicTransformer(generations=gp_config.generation, population_size=gp_config.population_size,
                             hall_of_fame=gp_config.hall_of_fame, n_components=gp_config.n_components,
                             function_set=gp_config.function_set,
                             parsimony_coefficient=gp_config.parsimony_coefficient,
                             max_samples=gp_config.max_samples, verbose=1,
                             random_state=0, n_jobs=3)

    y_train = df_train[[y_name]]
    if var_list is None:
        X_train = df_train.drop(y_name, axis=1)
        X_test = df_test.drop(y_name, axis=1)
    else:
        X_train = df_train[var_list]
        X_test = df_train[var_list]

    gp.fit(X_train, y_train)

    return gp.transform(X_test)
