import logging
import math
import os

import torch
from torch import nn
from src.common.layer.normalize_linear_layer import NormedLinearLayer

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class MLPConfig:
    def __init__(self, neuron_list, use_norm=True, use_dropout=True, dropout_rate=0.05, type_norm='pixel_norm',
                 type_linear='norm_linear', activation='mish'):
        assert isinstance(neuron_list, list)
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.dropout_rate = 0.0
        if use_dropout:
            self.dropout_rate = dropout_rate
        if type_norm not in {'pixel_norm', 'batch_norm', 'none'}:
            raise NotImplementedError()
        self.type_norm = type_norm

        if type_linear not in {'norm_linear', 'linear'}:
            raise NotImplementedError()
        self.type_linear = type_linear

        if activation not in {'relu', 'gelu', 'gelu_bert', 'mish'}:
            raise NotImplementedError
        self.activation = activation


class MLP(nn.Module):
    def __init__(self, mlp_config):
        super(MLP, self).__init__()
        self.mlp_config = mlp_config
        self.layers = list()
        if self.mlp_config.type_lienar == 'linear':
            for layer in range(len(self.mlp_config.neuron_list) - 1):
                in_neuron = self.mlp_config.neuron_list[layer]
                out_neuron = self.mlp_config.neuron_list[layer + 1]
                self.layers.append(nn.Linear(in_neuron, out_neuron))
        else:
            pass  # TODO Finish this part

        self.normalization = None
        if self.mlp_config.type_norm == 'pixel_norm':
            self.normalization =
        elif self.mlp_config.type_norm == 'batch_norm'
            self.normalization = nn.BatchNorm2d()
        self.dropout = None

    def forward(self, x):
        pass
