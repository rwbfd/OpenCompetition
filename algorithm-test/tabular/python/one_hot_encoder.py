# encoding = 'utf-8'

import pandas as pd
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file_name', None, 'The input file name. Supposed to be tab delimited file')
flags.DEFINE_string('output_file_name', None, 'The output file name.')
flags.DEFINE_list('var_names', None, 'The name(s) of variable to ont-hot encode.')
flags.DEFINE_bool('drop_original', True, 'Whether to drop the original variable. Default is TRUE')
flags.DEFINE_bool('drop_last', False, 'Whether to drop the last column. Default is FALSE')


def encode_one_column(df, column_name, drop_original=True, drop_last=False):
    one_hot = pd.get_dummies(df[column_name])
    for name in one_hot.columns.values:
        one_hot.rename(columns={name: column_name + "_" + name})
    if drop_last:
        one_hot = one_hot.drop(columns=df.columns.values[df.shape[1] - 1])
    df.join(one_hot)
    if drop_original:
        df = df.drop(columns=[column_name])
    return df


def main():
    input_file_name = flags.input_file_name
    output_file_name = flags.output_file_name
    drop_original = flags.drop_original
    drop_last = flags.drop_last
    var_names = flags.var_names

    data = pd.read_csv(input_file_name, sep='\t')
    for var in var_names:
        data = encode_one_column(data, var, drop_original, drop_last)
    data.to_csv(output_file_name, index=False, sep="\t")


if __name__ == '__main__':
    app.run(main)
