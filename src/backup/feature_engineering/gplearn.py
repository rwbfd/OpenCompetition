import pandas as pd
from sklearn import datasets
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split


# def data_prepare():
#     boston = datasets.load_boston()
#     boston_feature = pd.DataFrame(boston.data, columns=boston.feature_names)
#     boston_label = pd.Series(boston.target).to_frame("TARGET")
#     boston = pd.concat([boston_label, boston_feature], axis=1)
#     return boston
#
# data = data_prepare()
#
# function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']
#
# gp1 = SymbolicTransformer(generations=10, population_size=1000,
#                          hall_of_fame=100, n_components=10,
#                          function_set=function_set,
#                          parsimony_coefficient=0.0005,
#                          max_samples=0.9, verbose=1,
#                          random_state=0, n_jobs=3)
#
# x_train, x_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.2, random_state=10)
#
# gp_train_feature = gp1.transform(x_train)
# gp_test_feature = gp1.transform(x_test)
#
# train_idx = x_train.index
# test_idx = x_test.index
#
# new_feature_name = [str(i)+'V' for i in range(1, 11)]
# train_new_feature = pd.DataFrame(gp_train_feature, columns=new_feature_name, index=train_idx)
# test_new_feature = pd.DataFrame(gp_test_feature, columns=new_feature_name, index=test_idx)
# x_train_0 = pd.concat([x_train, train_new_feature], axis=1)
# x_test_0 = pd.concat([x_test, test_new_feature], axis=1)
#
# new_x_data = pd.concat([x_train_0, x_test_0], axis=0)
# new_data = pd.concat([data['TARGET'], new_x_data], axis=1)
# #print(new_data.columns)

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
    pass
