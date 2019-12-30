# encoding = 'utf-8'

import pandas as pd
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file_name', None, 'The input file name. Supposed to be tab delimited file')
flags.DEFINE_string('output_file_name', None, 'The output file name.')
flags.DEFINE_list('var_names', None, 'The name(s) of variable to ont-hot encode.')
flags.DEFINE_bool('drop_original', True, 'Whether to drop the original variable. Default is TRUE')


def encode_one_column(df, column_name, drop_original=True):
    column_to_encode = df[column_name]
    column_name_ordinal = column_name + "_ordinal"
    value_counts = column_to_encode.value_counts()
    df[column_name_ordinal] = df[column_name]

    for i in range(value_counts.shape[0]):
        df.column_name_ordinal[df[column_name] == value_counts.index[i]] = i
    if drop_original:
        df = df.drop(columns=column_name)
    return df


def main():
    input_file_name = flags.input_file_name
    output_file_name = flags.output_file_name
    drop_original = flags.drop_original
    var_names = flags.var_names
    df = pd.read_csv(input_file_name, sep="\t")
    for var in var_names:
        df = encode_one_column(df, var, drop_original)
    df.to_csv(output_file_name, index=False, sep="\t", encode='utf-8')


if __name__ == '__main__':
    app.run(main)
