import pandas as pd
from gplearn.genetic import SymbolicTransformer


def symbolicLearning(df_list):
    '''
    
    :param df_list: 
    :return: 
    '''
    df_list = pd.DataFrame(df_list)
    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']

    gp = SymbolicTransformer(generations=10, population_size=1000,
                              hall_of_fame=100, n_components=10,
                              function_set=function_set,
                              parsimony_coefficient=0.0005,
                              max_samples=0.9, verbose=1,
                              random_state=0, n_jobs=3)
    gp_feature = gp.transform(df_list)
    new_feature_name = [str(i) + 'V' for i in range(1, len(function_set)+1)]
    new_feature = pd.DataFrame(gp_feature, columns=new_feature_name)
    return new_feature
