import pandas as pd

def missingvalue_countcolumns(data):
    id = data.index
    arr = data.isnull().sum(axis=1)
    count_list = []
    for i in arr:
        count_list.append(i)
    missingvalue = pd.DataFrame({'missing_count': count_list}, index=id)
    newdata = pd.concat([data, missingvalue], axis=1)
    return newdata

if __name__ == '__main__':
    import numpy as np

    train_df = pd.read_csv('train.csv', index_col=0)
    print(missingvalue_countcolumns(data))

