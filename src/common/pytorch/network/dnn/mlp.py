import torch.nn as nn
from src.common.layer.normalize_linear_layer import NormedLinearLayer
from src.common.normalization.pixel_norm import PixelNorm


class MLPConfig:
    def __init__(self, neuron_list, use_norm=True, use_dropout=True, dropout_rate=0.05, type_norm='pixel_norm',
                 type_linear='norm_linear', activation='mish', momentum=0.1):
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
        self.momentum = momentum


class MLP(nn.Module):
    def __init__(self, mlp_config):
        super(MLP, self).__init__()
        self.mlp_config = mlp_config
        self.layers = list()

        for layer in range(len(self.mlp_config.neuron_list) - 2):
            in_neuron = self.mlp_config.neuron_list[layer]
            out_neuron = self.mlp_config.neuron_list[layer + 1]
            if self.mlp_config.type_linear == 'linear':
                self.layers.append((nn.Linear(in_neuron, out_neuron)))
            elif self.mlp_config.type_linear == 'norm_linear':
                self.layers.append(NormedLinearLayer(in_neuron, out_neuron, mlp_config.momentum))
        n_layer = len(self.mlp_config.neuron_list)
        self.final_linear = nn.Linear(self.mlp_config.neuron_list[n_layer-2], self.mlp_config.neuron)
        self.normalization = None
        if self.mlp_config.type_norm == 'pixel_norm':
            self.normalization = PixelNorm()
        elif self.mlp_config.type_norm == 'batch_norm':
            self.normalization = nn.BatchNorm2d()


        self.dropout = None
        if self.mlp_config.use_dropout and self.mlp_config.dropout_rate > 0:
            self.dropout = nn.Dropout(self.self.mlp_config.dropout_rate)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.mlp_config.normalization and i != len(self.layers) - 1:
                x = layer(x)
                x = self.normalization(x)

        if self.dropout:
            x = self.dropout(x)
        return x
