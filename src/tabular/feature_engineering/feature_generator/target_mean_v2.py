# coding = 'utf-8'

import pandas as pd
import numpy as np
import sklearn


class TargetMeanEncoderConfig:
    def __init__(self, fold=5, smooth_parameter=0.9):
        self.fold = fold
        self.smooth_parameter = smooth_parameter


def encode_one_column(dfs, y, target_var, config):
    """

    :param df:
    :param y:
    :param target_var:
    :param config:
    :return:
    """


def target_mean_encoder(df, y, target_vars, config):
    """

    :param df:
    :param y:
    :param target_vars:
    :param config:
    :return:
    """
